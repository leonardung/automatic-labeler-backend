import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
from itertools import product
import cv2
import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# basic YAML utilities                                                        #
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_overrides(cfg: Dict, overrides: Dict[Tuple[str, ...], str]) -> Dict:
    """
    Replace values inside ``cfg`` by walking the tuple key path.
    Any override with value ``None`` is ignored.
    """
    for key_path, value in overrides.items():
        if value is None:
            continue
        node = cfg
        for k in key_path[:-1]:
            node = node[k]
        node[key_path[-1]] = value
    return cfg


# --------------------------------------------------------------------------- #
# generic helpers                                                             #
# --------------------------------------------------------------------------- #
def split_file(
    inp: str,
    train_out: str,
    val_out: str,
    test_ratio: float = 0.2,
    seed: int = 0,
):
    with open(inp, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if seed is not None:
        random.seed(seed)
    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - test_ratio))
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, "w", encoding="utf-8") as f:
        f.writelines(lines[:split_idx])
    with open(val_out, "w", encoding="utf-8") as f:
        f.writelines(lines[split_idx:])


# --------------------------------------------------------------------------- #
# recognition dataset preparation                                             #
# --------------------------------------------------------------------------- #
def _rotate_crop(img, pts: np.ndarray):
    d = sum(
        -0.5 * (pts[i + 1][1] + pts[i][1]) * (pts[i + 1][0] - pts[i][0])
        for i in range(-1, 3)
    )
    if d < 0:  # counter-clockwise
        pts[[1, 3]] = pts[[3, 1]]  # swap
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    M = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    dst = cv2.warpPerspective(
        img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    if dst.shape[0] / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return dst


def prepare_rec_dataset(cfg: Dict):
    """
    Build the cropped-text dataset for the recognition model
    and split it into train / val lists.

    Parameters
    ----------
    cfg : Dict
        Loaded YAML config.
    test_ratio : float, optional
        Fraction of samples to send to the validation set, by default 0.2.
    """
    global_cfg = cfg["global"]
    rec_cfg = cfg["models"]["rec"]

    crop_dir = Path(global_cfg["crop_im_dir"])
    crop_dir.mkdir(parents=True, exist_ok=True)

    # collect all crop-lines first -------------------------------------------
    lines: List[str] = []
    with open(global_cfg["raw_dataset_file"], "r", encoding="utf-8") as fin:
        for raw_line in fin:
            img_rel, anns_json = raw_line.strip().split("\t")
            img = cv2.imread(str(Path(global_cfg["images_folder"]) / img_rel))
            anns = json.loads(anns_json)
            base = Path(img_rel).stem
            for idx, ann in enumerate(anns):
                crop_name = f"{base}_crop_{idx}.jpg"
                crop_path = crop_dir / crop_name
                crop_img = _rotate_crop(img, np.array(ann["points"], np.float32))
                cv2.imwrite(str(crop_path), crop_img)
                lines.append(f"{crop_dir.name}/{crop_name}\t{ann['transcription']}\n")

    Path(rec_cfg["dataset_train"]).parent.mkdir(parents=True, exist_ok=True)
    with open(rec_cfg["dataset_total"], "w", encoding="utf-8") as f_tot:
        f_tot.writelines(lines)

    # shuffle and split -------------------------------------------------------
    seed = global_cfg["split_seed"]
    if seed is not None:
        random.seed(seed)
    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - global_cfg["test_ratio"]))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # write out ---------------------------------------------------------------
    with open(rec_cfg["dataset_train"], "w", encoding="utf-8") as f_tot:
        f_tot.writelines(train_lines)
    with open(rec_cfg["dataset_val"], "w", encoding="utf-8") as f_val:
        f_val.writelines(val_lines)


# --------------------------------------------------------------------------- #
# KIE dataset preparation                                                     #
# --------------------------------------------------------------------------- #
def prepare_kie_dataset(cfg: Dict):
    from paddlenlp.transformers import LayoutXLMTokenizer, LayoutLMv2Tokenizer

    tokenizer = LayoutLMv2Tokenizer.from_pretrained("layoutlmv2-base-uncased")
    global_cfg = cfg["global"]
    det_model_cfg = cfg["models"]["det"]
    kie_model_cfg = cfg["models"]["kie"]

    # prepare the classes file
    with open(kie_model_cfg["class_path"], "w", encoding="utf-8") as class_file, open(
        global_cfg["raw_dataset_file"], "r", encoding="utf-8"
    ) as raw_file:
        classes = set()
        for line in raw_file:
            line = line.strip()
            if not line:
                continue
            _, ann_json = line.split("\t", 1)
            anns = json.loads(ann_json)
            for ann in anns:
                classes.add(ann["key_cls"])

        for cls in sorted(classes):
            class_file.write(cls + "\n")

    # make sure det train file exists (needed as source)
    if not Path(det_model_cfg["dataset_train"]).exists():
        split_file(
            global_cfg["raw_dataset_file"],
            det_model_cfg["dataset_train"],
            det_model_cfg["dataset_val"],
            test_ratio=global_cfg["test_ratio"],
        )
    # temporary source (un-augmented)
    kie_train_src = kie_model_cfg["dataset_train"] + ".src"
    kie_val_src = kie_model_cfg["dataset_val"] + ".src"

    split_file(
        global_cfg["raw_dataset_file"],
        kie_train_src,
        kie_val_src,
        test_ratio=global_cfg["test_ratio"],
        seed=global_cfg.get("split_seed"),
    )

    # Read parameter grids (can be scalar or list)
    max_lens_train = kie_model_cfg["train"].get("max_len_per_part", [1000])
    min_overlaps_train = kie_model_cfg["train"].get("min_overlap", [0])

    max_lens_test = kie_model_cfg["test"].get("max_len_per_part", [1000])
    min_overlaps_test = kie_model_cfg["test"].get("min_overlap", [0])
    print(max_lens_train)
    print(min_overlaps_train)
    print(max_lens_test)
    print(min_overlaps_test)
    # Augment TRAIN split into final KIE train file
    _augment_file_per_grid(
        src_file=kie_train_src,
        dst_file=kie_model_cfg["dataset_train"],
        max_lens=max_lens_train,
        min_overlaps=min_overlaps_train,
        tokenizer=tokenizer,
    )

    # Augment VAL split into final KIE val file
    _augment_file_per_grid(
        src_file=kie_val_src,
        dst_file=kie_model_cfg["dataset_val"],
        max_lens=max_lens_test,
        min_overlaps=min_overlaps_test,
        tokenizer=tokenizer,
    )

    # Cleanup temporary raw-split files
    os.remove(kie_train_src)
    os.remove(kie_val_src)


def sort_annotations(
    annotations: List[Dict[str, Any]], reverse=False
) -> List[Dict[str, Any]]:
    return sorted(annotations, key=lambda ann: ann["points"][0][1], reverse=reverse)


def _augment_file_per_grid(
    src_file: str, dst_file: str, max_lens, min_overlaps, tokenizer
):
    """Read src_file (un-augmented), write dst_file (augmented) across all param combos."""
    # normalize grids
    if not isinstance(max_lens, (list, tuple)):
        max_lens = [max_lens]
    if not isinstance(min_overlaps, (list, tuple)):
        min_overlaps = [min_overlaps]
    if not max_lens or not min_overlaps:
        raise ValueError(
            "models.kie.max_len_per_part and models.kie.min_overlap must be provided "
            "as a scalar or list with at least one value."
        )

    with open(src_file, "r", encoding="utf-8") as fin, open(
        dst_file, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            img_path, ann_json = line.split("\t", 1)
            anns = json.loads(ann_json)
            anns = sort_annotations(anns)

            # normalize to expected KIE schema
            new_anns: List[Dict] = [
                {
                    "transcription": a["transcription"],
                    "points": a["points"],
                    "label": a.get("key_cls", ""),
                }
                for a in anns
            ]

            wrote_any = False
            for ml, mo in product(max_lens, min_overlaps):
                if ml <= mo:
                    continue

                parts = split_by_token_budget(
                    new_anns,
                    tokenizer,
                    max_seq_len=ml,
                    min_token_overlap=mo,
                )
                if not parts:
                    continue
                wrote_any = True
                for part in parts:
                    fout.write(f"{img_path}\t{json.dumps(part, ensure_ascii=False)}\n")

            # fallback: keep the full sample if no combo produced parts
            if not wrote_any:
                fout.write(f"{img_path}\t{json.dumps(new_anns, ensure_ascii=False)}\n")


# def split_balanced_with_min_overlap(data, max_len, min_overlap):
#     n = len(data)
#     if max_len <= 0:
#         raise ValueError("max_len must be > 0")
#     if min_overlap < 0:
#         raise ValueError("min_overlap must be >= 0")
#     if min_overlap >= max_len:
#         raise ValueError("min_overlap must be < max_len")
#     if n <= max_len:
#         return [data[:]]

#     # Minimum number of parts to guarantee overlap
#     num_parts = math.ceil(1 + (n - max_len) / (max_len - min_overlap))
#     # Step size balanced across the length
#     step = math.ceil((n - max_len) / (num_parts - 1))
#     # Clamp step so overlap ≥ min_overlap
#     step = min(step, max_len - min_overlap)
#     parts = []
#     for i in range(0, n, step):
#         part = data[i : i + max_len]
#         parts.append(part)
#         if i + max_len >= n:
#             break

#     return parts


def split_balanced_with_min_overlap(data, max_len, min_overlap):
    n = len(data)
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if min_overlap < 0:
        raise ValueError("min_overlap must be >= 0")
    if min_overlap >= max_len:
        raise ValueError("min_overlap must be < max_len")
    if n <= max_len:
        return [data[:]]

    # Minimum parts to guarantee step <= max_len - min_overlap
    parts_count = math.ceil(1 + (n - max_len) / (max_len - min_overlap))

    # Evenly distribute starts so last slice ends at n
    total_shift = n - max_len  # distance from first to last start
    base_step = total_shift // (parts_count - 1)
    remainder = total_shift - base_step * (
        parts_count - 1
    )  # first 'remainder' steps are +1

    starts = [0]
    for t in range(parts_count - 1):
        step = base_step + (1 if t < remainder else 0)
        starts.append(starts[-1] + step)

    return [data[s : s + max_len] for s in starts]


def token_len(text: str, tokenizer) -> int:
    """
    Count subword tokens for a single 'word box' text as the KIE model will see it.
    No special tokens here; they are added at the window level.
    """
    if not text:
        return 0
    # IMPORTANT: add_special_tokens=False to measure only content tokens
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def split_by_token_budget(
    annotations, tokenizer, max_seq_len: int, min_token_overlap: int = 0
):
    """
    Chunk 'annotations' (already in reading order) into windows that fit the token budget.
    Each window satisfies: sum(tokens) + 2 <= max_seq_len   (# +2 for [CLS], [SEP]).
    Overlap is measured in TOKENS (not boxes) between consecutive windows.

    Returns: list of slices (each is a list of annotation dicts).
    """

    if max_seq_len <= 2:
        raise ValueError("max_seq_len must be > 2")
    if min_token_overlap < 0:
        raise ValueError("min_token_overlap must be >= 0")

    # per-box token counts
    tok_counts = [
        token_len(ann.get("transcription", ""), tokenizer) for ann in annotations
    ]
    n = len(annotations)
    if n == 0:
        return []

    budget = max_seq_len - 2  # [CLS], [SEP]
    # prefix sums of token counts (for O(1) range sums)
    prefix = [0]
    for t in tok_counts:
        prefix.append(prefix[-1] + t)

    def range_tokens(i, j):
        # total tokens in boxes [i, j) (i inclusive, j exclusive)
        return prefix[j] - prefix[i]

    out = []
    start = 0
    while start < n:
        # grow 'end' as far as we can within budget
        end = start
        while end < n and range_tokens(start, end + 1) <= budget:
            end += 1
        # If the very next single box already exceeds budget (huge OCR run),
        # force-include it alone to avoid infinite loop.
        if end == start:
            end = min(start + 1, n)

        out.append(annotations[start:end])

        if end >= n:
            break

        if min_token_overlap == 0:
            start = end
        else:
            # move 'start' forward so that the next window shares at least
            # 'min_token_overlap' tokens with the previous one.
            # We pick the largest start' in [start, end) such that
            # tokens in [start', end) >= min_token_overlap.
            # This guarantees overlap measured in tokens.
            start_prime = start
            # shrink from the left until overlap threshold reached
            while (
                start_prime < end
                and range_tokens(start_prime, end) >= min_token_overlap
            ):
                start_prime += 1
            # we went one step too far—step back
            start = max(start_prime - 1, start)
    return out
