import os
from pathlib import Path
import sys
from django.conf import settings

PADDLE_ROOT = Path(settings.BASE_DIR) / "submodules" / "PaddleOCR"
SUBMODULE_ROOT = Path(settings.BASE_DIR) / "submodules"
sys.path.append(str(PADDLE_ROOT))
sys.path.append(str(SUBMODULE_ROOT))
import paddle
from PaddleOCR.tools import program
from PaddleOCR.tools import train as paddle_train

from .helper import split_file


def train(cfg: dict) -> None:
    global_cfg, det_model_cfg = cfg["global"], cfg["models"]["det"]

    # ensure split files exist
    if os.path.exists(det_model_cfg["dataset_train"]) and os.path.exists(
        det_model_cfg["dataset_val"]
    ):
        print("Using existing splitted dataset files.")
    else:
        split_file(
            global_cfg["raw_dataset_file"],
            det_model_cfg["dataset_train"],
            det_model_cfg["dataset_val"],
            test_ratio=global_cfg["test_ratio"],
            seed=global_cfg["split_seed"],
        )

    sys.argv = [
        "train.py",
        "-c",
        det_model_cfg["paddle_cfg"],
        "-o",
        f"Global.use_gpu={global_cfg['use_gpu']}",
        f"Global.epoch_num={det_model_cfg['epoch_num']}",
        f"Global.pretrained_model={det_model_cfg['pretrained_model']}",
        f"Global.print_batch_step={det_model_cfg['print_batch_step']}",
        f"Global.save_model_dir={det_model_cfg['save_model_dir']}",
        f"Global.save_epoch_step={det_model_cfg['save_epoch_step']}",
        f"Global.eval_batch_step={det_model_cfg['eval_batch_step']}",
        f"Train.dataset.data_dir={global_cfg['images_folder']}",
        f"Train.dataset.label_file_list=[{det_model_cfg['dataset_train']}]",
        f"Eval.dataset.data_dir={global_cfg['images_folder']}",
        f"Eval.dataset.label_file_list=[{det_model_cfg['dataset_val']}]",
    ]

    pp_cfg, device, logger, vdl = program.preprocess(is_train=True)
    seed = global_cfg["train_seed"]
    paddle_train.set_seed(seed)
    paddle_train.main(pp_cfg, device, logger, vdl, seed)

    with paddle.no_grad():
        paddle.device.cuda.empty_cache()
