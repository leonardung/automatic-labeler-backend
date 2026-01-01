"""
Invoice OCR pipeline using PaddleOCR components.
Stages:
1) Detection
2) Crop generation
3) Recognition
4) Merge DET+REC (chunking for KIE)
5) Visualize boxes
6) SER/KIE
7) Merge KIE with DET scores
8) Export for PP-OCRLabel

Configuration is provided via a YAML file (see sample at the end).
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import yaml
from django.conf import settings


def split_by_token_budget(
    annotations: List[Dict[str, Any]],
    tokenizer,
    max_seq_len: int,
    min_token_overlap: int = 0,
):
    """
    Chunk 'annotations' (already in reading order) into windows that fit the token budget.
    Each window satisfies: sum(tokens) + 2 <= max_seq_len   (# +2 for [CLS], [SEP]).
    Overlap is measured in TOKENS (not boxes) between consecutive windows.
    """

    if max_seq_len <= 2:
        raise ValueError("max_seq_len must be > 2")
    if min_token_overlap < 0:
        raise ValueError("min_token_overlap must be >= 0")

    tok_counts = [
        token_len(ann.get("transcription", ""), tokenizer) for ann in annotations
    ]
    n = len(annotations)
    if n == 0:
        return []

    budget = max_seq_len - 2  # [CLS], [SEP]
    prefix = [0]
    for t in tok_counts:
        prefix.append(prefix[-1] + t)

    def range_tokens(i, j):
        return prefix[j] - prefix[i]

    out = []
    start = 0
    while start < n:
        end = start
        while end < n and range_tokens(start, end + 1) <= budget:
            end += 1
        if end == start:
            end = min(start + 1, n)

        out.append(annotations[start:end])

        if end >= n:
            break

        if min_token_overlap == 0:
            start = end
        else:
            start_prime = start
            while (
                start_prime < end
                and range_tokens(start_prime, end) >= min_token_overlap
            ):
                start_prime += 1

            if start_prime == start:
                # Not enough tokens to satisfy the overlap requirement; move past this window.
                start = end
            else:
                # start_prime - 1 keeps the overlap condition; ensure we still progress.
                start = max(start_prime - 1, start + 1)
    return out


def sort_annotations(
    annotations: List[Dict[str, Any]], reverse: bool = False
) -> List[Dict[str, Any]]:
    def _key(ann: Dict[str, Any]):
        try:
            return ann["points"][0][1]
        except Exception:
            return 0

    return sorted(annotations, key=_key, reverse=reverse)


PADDLE_ROOT = Path(settings.BASE_DIR) / "submodules" / "PaddleOCR"
sys.path.append(str(PADDLE_ROOT))

from PaddleOCR.tools import infer_det, infer_kie_token_ser, program, infer_rec
from PaddleOCR.tools.infer_det import draw_det_res
from PaddleOCR.paddleocr import TextRecognition
from paddlenlp.transformers import LayoutXLMTokenizer, LayoutLMv2Tokenizer

tokenizer = LayoutLMv2Tokenizer.from_pretrained("layoutlmv2-base-uncased")

# --------------------------------------------------------------------------------------
# Utility: Config dataclass
# --------------------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    # Core paths
    infer_img_det: Path
    save_output: Path

    # Model/resources
    character_dict_path: Path
    rec_model: Path
    det_model: Path
    kie_checkpoint: Path
    class_path: Path
    font_path: Path

    # Options
    use_gpu: bool = True
    thresh: float = 0.3
    box_thresh: float = 0.4
    unclip_ratio: float = 1.7
    max_len_per_part: int = 70  # chunk size for KIE
    min_overlap: int = 20

    # Derived paths (auto if not provided)
    save_res_det_path: Path | None = None
    save_res_rec_path: Path | None = None
    save_res_ser_path: Path | None = None
    output_txt_file: Path | None = None
    crop_im_dir: Path | None = None
    kie_output_path: Path | None = None

    def finalize(self) -> None:
        self.save_output.mkdir(parents=True, exist_ok=True)
        if self.save_res_det_path is None:
            self.save_res_det_path = self.save_output / "predicts_det.txt"
        if self.save_res_rec_path is None:
            self.save_res_rec_path = self.save_output / "predicts_rec.txt"
        if self.save_res_ser_path is None:
            self.save_res_ser_path = self.save_output / "predicts_ser"
        self.save_res_ser_path.mkdir(parents=True, exist_ok=True)
        if self.output_txt_file is None:
            self.output_txt_file = Path(
                str(self.save_res_det_path).replace(".txt", "_with_transcription.txt")
            )
        if self.crop_im_dir is None:
            self.crop_im_dir = self.save_output / "crop_images"
        self.crop_im_dir.mkdir(parents=True, exist_ok=True)
        if self.kie_output_path is None:
            self.kie_output_path = self.save_res_ser_path / "infer_results.txt"


# --------------------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------------------


def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Resolve to Paths and handle relative paths neatly
    def P(x: str | Path) -> Path:
        return Path(x).expanduser().resolve()

    cfg = PipelineConfig(
        infer_img_det=P(raw["infer_img_det"]),
        save_output=P(raw["save_output"]),
        character_dict_path=raw["character_dict_path"],
        rec_model=P(raw["rec_model"]),
        det_model=P(raw["det_model"]),
        kie_checkpoint=P(raw["kie_checkpoint"]),
        class_path=P(raw["class_path"]),
        font_path=raw["font_path"],
        use_gpu=bool(raw.get("use_gpu", True)),
        thresh=float(raw.get("thresh", 0.4)),
        box_thresh=float(raw.get("box_thresh", 0.4)),
        unclip_ratio=float(raw.get("unclip_ratio", 1.7)),
        max_len_per_part=int(raw.get("max_len_per_part", 70)),
        min_overlap=int(raw.get("min_overlap", 20)),
        save_res_det_path=(
            P(raw.get("save_res_det_path", ""))
            if raw.get("save_res_det_path")
            else None
        ),
        save_res_rec_path=(
            P(raw.get("save_res_rec_path", ""))
            if raw.get("save_res_rec_path")
            else None
        ),
        save_res_ser_path=(
            P(raw.get("save_res_ser_path", ""))
            if raw.get("save_res_ser_path")
            else None
        ),
        output_txt_file=(
            P(raw.get("output_txt_file", "")) if raw.get("output_txt_file") else None
        ),
        crop_im_dir=P(raw.get("crop_im_dir", "")) if raw.get("crop_im_dir") else None,
        kie_output_path=(
            P(raw.get("kie_output_path", "")) if raw.get("kie_output_path") else None
        ),
    )
    cfg.finalize()
    return cfg


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """Crop a quadrilateral region from img by perspective transform; normalize orientation."""
    d = 0.0
    for index in range(-1, 3):
        d += (
            -0.5
            * (points[index + 1][1] + points[index][1])
            * (points[index + 1][0] - points[index][0])
        )
    if d < 0:  # counterclockwise
        tmp = np.array(points)
        points[1], points[3] = tmp[3], tmp[1]
    try:
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        h, w = dst_img.shape[:2]
        if h / (w + 1e-6) >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("Crop failed: %s", e)
        return None


def convert_to_horizontal_rectangle(points: List[List[float]]) -> List[List[float]]:
    """Convert quadrilateral to axis-aligned bounding box (xmin,ymin,xmax,ymax)."""
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_min = min(p[1] for p in points)
    y_max = max(p[1] for p in points)
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def add_padding_to_rectangle(
    points: List[List[float]], padding: float = 5.0
) -> List[List[float]]:
    """
    Expand a quadrilateral by padding each point outward.

    Args:
        points: List of four [x, y] points (any order).
        pad: Amount of padding to expand outward.

    Returns:
        New list of four [x, y] points with padding applied.
    """
    return [
        [points[0][0] - padding, points[0][1] - padding],  # top-left
        [points[1][0] + padding, points[1][1] - padding],  # top-right
        [points[2][0] + padding, points[2][1] + padding],  # bottom-right
        [points[3][0] - padding, points[3][1] + padding],  # bottom-left
    ]


# --------------------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------------------


def run_infer_det(cfg: PipelineConfig) -> None:
    """Run detection and write results to predicts_det.txt."""
    sys.argv = [
        "infer_det.py",
        "-c",
        str(PADDLE_ROOT / "configs/det/PP-OCRv5/PP-OCRv5_server_det.yml"),
        "-o",
        f"Global.use_gpu={cfg.use_gpu}",
        f"Global.pretrained_model={cfg.det_model}",
        f"Global.infer_img={cfg.infer_img_det}",
        f"Global.save_res_path={cfg.save_res_det_path}",
        f"PostProcess.thresh={cfg.thresh}",
        f"PostProcess.box_thresh={cfg.box_thresh}",
        f"PostProcess.unclip_ratio={cfg.unclip_ratio}",
    ]
    config, device, logger, vdl_writer = program.preprocess()
    infer_det.main(config, device, logger, vdl_writer)


def preprocess_with_padding(cfg: PipelineConfig):
    """Rewrite detection results with padded rectangles."""
    assert (
        cfg.save_res_det_path and cfg.save_res_det_path.exists()
    ), "Detection output missing."
    updated_lines = []
    with cfg.save_res_det_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Adding padding"):
            img_path_str, annotations_str = line.strip().split("\t")
            annotations = json.loads(annotations_str)

            # for ann in annotations:
            #     ann["points"] = add_padding_to_rectangle(ann["points"])

            updated_lines.append(
                f"{img_path_str}\t{json.dumps(annotations, ensure_ascii=False)}\n"
            )

    # overwrite file with padded version
    with cfg.save_res_det_path.open("w", encoding="utf-8") as f:
        f.writelines(updated_lines)


def create_crops(cfg: PipelineConfig) -> None:
    """Generate crop images from DET results."""
    logger = logging.getLogger(__name__)
    assert (
        cfg.save_res_det_path and cfg.save_res_det_path.exists()
    ), "Detection output missing."

    with cfg.save_res_det_path.open("r", encoding="utf-8") as det_output:
        for line in tqdm(det_output, desc="Cropping"):
            img_path_str, annotations_str = line.strip().split("\t")
            img_path = Path(img_path_str)
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Could not read image: %s", img_path)
                continue

            annotations = json.loads(annotations_str)
            for i, ann in enumerate(annotations):
                pts = np.array(ann["points"], dtype=np.float32)
                crop = get_rotate_crop_image(img, pts)
                if crop is None:
                    continue
                im_name = img_path.stem
                crop_name = f"{im_name}_crop_{i}.jpg"
                cv2.imwrite(str(cfg.crop_im_dir / crop_name), crop)


# def run_infer_rec(cfg: PipelineConfig) -> None:
#     """Run recognition on crop directory."""
#     sys.argv = [
#         "infer_rec.py",
#         "-c",
#         str(PADDLE_ROOT / "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml"),
#         "-o",
#         f"Global.use_gpu={cfg.use_gpu}",
#         f"Global.character_dict_path={PADDLE_ROOT / cfg.character_dict_path}",
#         f"Global.pretrained_model={cfg.rec_model}",
#         f"Global.checkpoints={cfg.rec_model}",
#         f"Global.infer_img={cfg.crop_im_dir}",
#         f"Global.save_res_path={cfg.save_res_rec_path}",
#     ]
#     config, device, logger, vdl_writer = program.preprocess()
#     infer_rec.main(config, device, logger, vdl_writer)


def run_infer_rec(cfg: PipelineConfig, batch_size: int = 16) -> None:
    """
    Run recognition on crop directory using PaddleX TextRecognition.
    Writes a TSV-style file at save_res_rec_path:
        <abs_image_path>\t<rec_text>\t<rec_score>
    """
    logger = logging.getLogger(__name__)
    crop_dir = cfg.crop_im_dir
    out_path = cfg.save_res_rec_path

    # Collect images (same pattern you saved earlier: *_crop_*.jpg)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_paths = sorted([p for p in crop_dir.iterdir() if p.suffix.lower() in exts])
    if not img_paths:
        logger.warning("No crop images found in %s", crop_dir)
        # still create an empty file to keep downstream steps happy
        out_path.write_text("", encoding="utf-8")
        return

    # Initialize recognizer (model_name per your snippet)
    model = TextRecognition(model_name="latin_PP-OCRv5_mobile_rec")
    # model = TextRecognition(model_name="en_PP-OCRv5_mobile_rec")

    # Predict in batches and stream results to disk
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        # for batch in tqdm(list(chunks(img_paths, batch_size)), desc="Recognition"):
        for img_path in tqdm(img_paths, desc="Recognition"):
            res = model.predict(str(img_path))[0]
            rec_text = res.get("rec_text", "") if isinstance(res, dict) else ""
            rec_score = res.get("rec_score", 0.0) if isinstance(res, dict) else 0.0

            # Write exactly: <path>\t<text>\t<score>
            fw.write(f"{img_path}\t{rec_text}\t{rec_score}\n")


def merge_det_rec(cfg: PipelineConfig) -> None:
    """Attach REC transcriptions to DET boxes; chunk to max_len_per_part."""
    assert cfg.save_res_det_path and cfg.save_res_rec_path
    rec_dict: Dict[str, Tuple[str, str]] = {}
    with cfg.save_res_rec_path.open("r", encoding="utf-8") as rec_output:
        for line in rec_output:
            crop_img_path, label, score = line.strip().split("\t")
            crop_img_name = Path(crop_img_path).name
            rec_dict[crop_img_name] = (label, score)

    with cfg.save_res_det_path.open(
        "r", encoding="utf-8"
    ) as det_output, cfg.output_txt_file.open("w", encoding="utf-8") as outfile:
        for line in det_output:
            img_path_str, annotations_str = line.strip().split("\t")
            img_path = Path(img_path_str)
            annotations = json.loads(annotations_str)
            new_annotations: List[Dict[str, Any]] = []

            for i, ann in enumerate(annotations):
                im_name = img_path.stem
                crop_name = f"{im_name}_crop_{i}.jpg"
                if crop_name not in rec_dict:
                    # No recognition result for this crop — keep box but empty text/score
                    transcription, score = "", "0"
                else:
                    transcription, score = rec_dict[crop_name]

                ann = ann.copy()
                ann["transcription"] = transcription
                ann["score"] = score
                ann["label"] = "None"
                new_annotations.append(ann)

            new_annotations = sort_annotations(new_annotations, reverse=False)
            parts = split_by_token_budget(
                new_annotations,
                tokenizer,
                max_seq_len=cfg.max_len_per_part,
                min_token_overlap=cfg.min_overlap,
            )

            for part in parts:
                outfile.write(f"{img_path}\t{json.dumps(part, ensure_ascii=False)}\n")


def visualize_boxes(cfg: PipelineConfig, draw_det_res) -> None:
    """Render detection results over source images for quick QA."""
    logger = logging.getLogger(__name__)
    vis_dir = cfg.save_output / "det_image_w_thresh"
    vis_dir.mkdir(parents=True, exist_ok=True)

    with cfg.output_txt_file.open("r", encoding="utf-8") as infile:
        lines = infile.readlines()  # read all lines first so tqdm knows the length
    with tqdm(lines, desc="Visualize images") as pbar:
        for idx, line in enumerate(pbar):
            image_path_str, annotations_str = line.strip().split("\t")
            image_path = Path(image_path_str)

            # show current file being processed inside the bar
            pbar.set_postfix({"file": image_path.name})

            annotations = json.loads(annotations_str)
            src_img = cv2.imread(str(image_path))
            if src_img is None:
                logger.warning("Could not read image: %s", image_path)
                continue

            boxes = []
            for ann in annotations:
                boxes.append(ann["points"])

            out_name = f"{image_path.name}_{idx}.png"
            draw_det_res(boxes, None, src_img, out_name, str(vis_dir), logger)


def run_infer_ser(cfg: PipelineConfig) -> None:
    """Run KIE/SER model (vi_layoutxlm config by default)."""
    kie_config = Path(cfg.kie_checkpoint).parent / "config.yml"
    sys.argv = [
        "infer_kie_token_ser.py",
        "-c",
        str(kie_config),
        "-o",
        f"Global.use_gpu={cfg.use_gpu}",
        f"Global.class_path={cfg.class_path}",
        f"Global.font_path={PADDLE_ROOT / cfg.font_path}",
        f"Global.infer_img={cfg.output_txt_file}",
        f"Global.save_res_path={cfg.save_res_ser_path}",
        "Global.infer_mode=False",
        f"Architecture.Backbone.checkpoints={cfg.kie_checkpoint}",
        # f"Eval.dataset.data_dir={cfg.save_output}",
        f"Eval.dataset.data_dir=/",
        f"Eval.dataset.label_file_list={cfg.output_txt_file}",
    ]
    config, device, logger, vdl_writer = program.preprocess()
    infer_kie_token_ser.main(config, device, logger, vdl_writer)


def merge_kie_with_det_scores(cfg: PipelineConfig) -> Path:
    """
    Add DET score into KIE results by matching points, then write combined file.
    Returns path to combined KIE output.
    """
    combined_out = cfg.save_res_ser_path / "infer_results_combined.txt"

    # Load KIE (may be split by images)
    path_annotations: Dict[str, List[Dict[str, Any]]] = {}
    with cfg.kie_output_path.open("r", encoding="utf-8") as kout:
        for line in kout:
            path_str, annotations_str = line.strip().split("\t")
            annotations = json.loads(annotations_str)["ocr_info"]
            path_annotations.setdefault(path_str, []).extend(annotations)

    # Load REC+DET (chunked) into dict
    rec_det_annotations: Dict[str, List[Dict[str, Any]]] = {}
    with cfg.output_txt_file.open("r", encoding="utf-8") as rdo:
        for line in rdo:
            path_str, annotations_str = line.strip().split("\t")
            rec_det_annotations[path_str] = json.loads(annotations_str)

    # Merge
    with combined_out.open("w", encoding="utf-8") as out:
        for path_str, ann_list in path_annotations.items():
            det_list = rec_det_annotations.get(path_str, [])
            for ann in ann_list:
                # attach det score by exact points match
                pts = ann.get("points")
                if not pts:
                    continue
                for d in det_list:
                    if d.get("points") == pts:
                        ann["score_det"] = d.get("score")
                        break
            out.write(f"{path_str}\t{json.dumps(ann_list, ensure_ascii=False)}\n")

    return combined_out


def export_for_ppocrlabel(cfg: PipelineConfig, kie_combined_path: Path) -> Path:
    """
    Write PP-OCRLabel cache file in the inference image folder for re-training/labeling.
    Mirrors original behavior and tie-breaks duplicates by score.
    """
    to_load_in_ppocrlabel = cfg.infer_img_det / "Cache.cach"

    # Load KIE combined
    with kie_combined_path.open(
        "r", encoding="utf-8"
    ) as kout, to_load_in_ppocrlabel.open("w", encoding="utf-8") as label_output:
        for line in kout:
            path_str, annotations_str = line.strip().split("\t")
            annotations = json.loads(annotations_str)

            # Transform to new annotation format
            new_annotations: List[Dict[str, Any]] = []
            for ann in annotations:
                if "pred" not in ann:
                    logging.getLogger(__name__).warning(
                        "Warning, 'pred' not in %s in path:\n%s\n", ann, path_str
                    )
                    continue
                new_ann = {
                    "transcription": ann.get("transcription", ""),
                    "points": ann.get("points", []),
                    "key_cls": (ann.get("pred") or "").lower(),
                    "difficult": False,
                    "score": ann.get("score", "0"),
                }
                if ann.get("pred") == "NONE":
                    new_ann["key_cls"] = "None"
                new_annotations.append(new_ann)

            # Deduplicate by points, keep higher score
            accumulated: List[Dict[str, Any]] = []
            for na in new_annotations:
                idx = next(
                    (
                        i
                        for i, ex in enumerate(accumulated)
                        if na.get("points") == ex.get("points")
                    ),
                    None,
                )
                if idx is None:
                    accumulated.append(na)
                else:
                    ex = accumulated[idx]
                    s_new = float(na.get("score", 0) or 0)
                    s_old = float(ex.get("score", 0) or 0)
                    if s_new >= s_old:
                        accumulated[idx] = na

            # PP-OCRLabel expects page path relative to dataset root (replicates original code)
            path = Path(path_str)
            page = f"{path.parent.name}/{path.name}".replace("\\", "/")
            label_output.write(
                f"{page}\t{json.dumps(accumulated, ensure_ascii=False)}\n"
            )

    return to_load_in_ppocrlabel


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoice OCR pipeline")
    parser.add_argument(
        "-c", "--config", required=True, type=Path, help="Path to YAML config"
    )
    parser.add_argument(
        "--kie",
        action="store_true",
        help="Run only KIE (and DET+REC merge). Skips detection, cropping, recognition, and visualization.",
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("invoice_pipeline")

    # Load config
    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    if not args.kie:
        logger.info("1) DETECTION")
        run_infer_det(cfg)

        logger.info("2) CROPS")
        preprocess_with_padding(cfg)
        create_crops(cfg)

        logger.info("3) RECOGNITION")
        run_infer_rec(cfg)

    logger.info("4) MERGE DET+REC")
    merge_det_rec(cfg)

    logger.info("5) VISUALIZE BOXES")
    visualize_boxes(cfg, draw_det_res)

    logger.info("6) SER/KIE")
    run_infer_ser(cfg)

    logger.info("7) MERGE KIE WITH DET SCORES")
    combined = merge_kie_with_det_scores(cfg)

    logger.info("8) EXPORT FOR PP-OCRLABEL")
    cache_path = export_for_ppocrlabel(cfg, combined)

    logger.info("Done. Outputs:")
    logger.info("  DET:            %s", cfg.save_res_det_path)
    logger.info("  CROP DIR:       %s", cfg.crop_im_dir)
    logger.info("  REC:            %s", cfg.save_res_rec_path)
    logger.info("  DET+REC (KIE):  %s", cfg.output_txt_file)
    logger.info("  VIS:            %s", cfg.save_output / "det_image_w_thresh")
    logger.info("  SER RAW:        %s", cfg.kie_output_path)
    logger.info("  SER COMBINED:   %s", combined)
    logger.info("  PPOCRLabel:     %s", cache_path)


if __name__ == "__main__":
    main()
