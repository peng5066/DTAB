#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import torch
from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer2 import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import gc
gc.collect()
torch.cuda.empty_cache()
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # Adds configuration options specifically for UBTeacher to the configuration object.
    add_ubteacher_config(cfg)
    # The `config_file` specifies the YAML configuration file.
    # The `merge_from_file` function then overrides the default
    # hyperparameter values with those defined in the YAML file.
    cfg.merge_from_file(args.config_file)
    ###
    if cfg.DATASETS.LABEL_NUMS == None:
        print("No label numbers defined")
    # `merge_from_list` serves a similar function to the one described above,
    # but allows for overriding settings via command-line arguments.
    cfg.merge_from_list(args.opts)
    ### eval
    if args.eval_only:
        cfg.MODEL.WEIGHTS = r"output/model_DTAB_COCO10.pth"
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            # Construct a model, specifically a semi-supervised learning network in this case,
            # based on the provided configuration, `cfg`.
            model = Trainer.build_model(cfg)
            # Construct a separate "teacher" model based on an identical configuration, denoted as cfg.
            model_teacher = Trainer.build_model(cfg)
            # model_teacher2 = Trainer.build_model(cfg)
            # Construct an ensemble model named `ensem_ts_model`,
            # incorporating the "teacher" model alongside the previously built models as parameters.
            ensem_ts_model = EnsembleTSModel(model_teacher, model)
            # Instantiate a Checkpointer to manage the saving and loading of model parameters,
            # specifically for the `ensem_ts_model`.
            # `cfg.OUTPUT_DIR` specifies the directory for saving model parameters,
            # while `cfg.MODEL.WEIGHTS` indicates the path to the pre-trained weights file.
            # `args.resume` is a boolean flag that determines whether to resume training from a previous checkpoint.


            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
