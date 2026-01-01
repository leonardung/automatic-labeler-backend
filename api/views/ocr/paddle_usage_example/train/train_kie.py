from pathlib import Path
import sys
from django.conf import settings
import os

PADDLE_ROOT = Path(settings.BASE_DIR) / "submodules" / "PaddleOCR"
SUBMODULE_ROOT = Path(settings.BASE_DIR) / "submodules"
sys.path.append(str(PADDLE_ROOT))
sys.path.append(str(SUBMODULE_ROOT))

import paddle
from PaddleOCR.tools import program
from PaddleOCR.tools import train as paddle_train

# from PaddleOCR.tools import eval as paddle_train

from .helper import prepare_kie_dataset


def train(cfg: dict) -> None:
    global_cfg, kie_model_cfg = cfg["global"], cfg["models"]["kie"]
    if os.path.exists(kie_model_cfg["dataset_train"]) and os.path.exists(
        kie_model_cfg["dataset_val"]
    ):
        print("Using existing KIE dataset files.")
    else:
        prepare_kie_dataset(cfg)

    with open(kie_model_cfg["class_path"], "r", encoding="utf-8") as f:
        lines = f.readlines()
        num_classes = sum(1 for line in lines if line.strip()) + 1

    sys.argv = [
        "train.py",
        # "eval.py",
        "-c",
        kie_model_cfg["paddle_cfg"],
        "-o",
        f"Global.use_gpu={global_cfg['use_gpu']}",
        f"Global.epoch_num={kie_model_cfg['epoch_num']}",
        f"Global.print_batch_step={kie_model_cfg['print_batch_step']}",
        f"Global.save_model_dir={kie_model_cfg['save_model_dir']}",
        f"Global.save_epoch_step={kie_model_cfg['save_epoch_step']}",
        f"Global.eval_batch_step={kie_model_cfg['eval_batch_step']}",
        f"Global.class_path={kie_model_cfg['class_path']}",
        f"PostProcess.class_path={kie_model_cfg['class_path']}",
        f"Train.dataset.transforms.1.VQATokenLabelEncode.class_path={kie_model_cfg['class_path']}",
        f"Eval.dataset.transforms.1.VQATokenLabelEncode.class_path={kie_model_cfg['class_path']}",
        f"Architecture.Backbone.pretrained={kie_model_cfg['pretrained_model']}",
        f"Architecture.Backbone.num_classes={int(2 * num_classes - 1)}",
        f"Loss.num_classes={int(2 * num_classes - 1)}",
        f"Train.dataset.data_dir={global_cfg['images_folder']}",
        f"Train.dataset.label_file_list={kie_model_cfg['dataset_train']}",
        "Train.dataset.ratio_list=1",
        f"Eval.dataset.data_dir={global_cfg['images_folder']}",
        f"Eval.dataset.label_file_list={kie_model_cfg['dataset_val']}",
        "Eval.dataset.ratio_list=1",
    ]

    pp_cfg, device, logger, vdl = program.preprocess(is_train=True)
    seed = global_cfg["train_seed"]
    paddle_train.set_seed(seed)
    paddle_train.main(pp_cfg, device, logger, vdl, seed)

    with paddle.no_grad():
        paddle.device.cuda.empty_cache()
