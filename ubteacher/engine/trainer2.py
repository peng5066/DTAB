# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from detectron2.structures import  pairwise_iou
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog
import time


from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.engine.hooks import LossEvalHook
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler


# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# Unbiased Teacher Trainer
class UBTeacherTrainer(DefaultTrainer):
    def __init__( self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)
        self.flag1 = True
        self.flag2 = False
        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        self.epoch = 0
        # create an teacher model
        model_teacher = self.build_model(cfg)

        model_teacher2 = self.build_model(cfg)
        self.model_teacher = model_teacher
        model_teacher_model_dict = self.model_teacher.state_dict()
        model_student_model_dict = model.state_dict()
        self.model_teacher2 = model_teacher2

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))
        ###############################
        self.iter_per_epoch = self.cfg.DATASETS.LABEL_NUMS // self.data_loader.batch_size_label
        if start_iter != 0:
            self.epoch = (start_iter - 1) // self.iter_per_epoch + 1
            if self.epoch % 2 == 1:
                self.ema_teacher= self.model_teacher
                self.ema_teacher2 = self.model_teacher2
            else:
                self.ema_teacher = self.model_teacher2
                self.ema_teacher2 = self.model_teacher
        ###############################
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    if self.iter % self.iter_per_epoch == 0:
                        self.epoch += 1
                        if self.epoch % 2 == 1:
                            self.ema_teacher = self.model_teacher
                            self.ema_teacher2 = self.model_teacher2
                        else:
                            self.ema_teacher = self.model_teacher2
                            self.ema_teacher2 = self.model_teacher
                    if self.iter == 60000:
                        self.ema_teacher = self.model_teacher
                        self.ema_teacher2 = self.model_teacher
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def show_image(self,tensor_image):
        image_np = tensor_image.numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        plt.imshow(image_np)
        plt.show()

    def draw_teacherbox(self,image,teacher_proposals,change = False,flip = False):
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        from ubteacher import add_ubteacher_config
        from detectron2.config import get_cfg
        import cv2
        from detectron2.modeling.postprocessing import detector_postprocess
        cfg = get_cfg()
        add_ubteacher_config(cfg)
        cfg.merge_from_file('configs/voc/voc07_voc12.yaml')
        im = cv2.imread(image['file_name'])
        if flip == True:
            im = cv2.flip(im, 1)
        height = image['height']
        width = image['width']
        if change == False:
            result = detector_postprocess(teacher_proposals, height, width)
        else:
            result = teacher_proposals
        dpi = 500
        pred_classes = result.gt_classes
        pred_boxes = result.gt_boxes
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        v = v.draw_instance_predictions(result.to("cpu"))
        plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.imshow(v.get_image())
        plt.axis("off")
        plt.show()

    def build_label_matrix(self, teacher1, teacher2):
        num_teacher1_labels = teacher1.gt_classes.size(0)
        num_teacher2_labels = teacher2.gt_classes.size(0)
        label_matrix = torch.zeros((num_teacher1_labels, num_teacher2_labels), dtype=torch.int)
        for i in range(num_teacher1_labels):
            for j in range(num_teacher2_labels):
                if teacher1.gt_classes[i] == teacher2.gt_classes[j]:
                    label_matrix[i, j] = 1
                else:
                    label_matrix[i, j] = 0
        return label_matrix , num_teacher1_labels, num_teacher2_labels

    def calculate_iou(self, box1, box2):
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection / (area_box1 + area_box2 - intersection)
        return iou

    def build_iou_matrix(self,boxes1, boxes2):
        num_boxes1 = boxes1.size(0)
        num_boxes2 = boxes2.size(0)
        iou_matrix = torch.zeros((num_boxes1, num_boxes2), dtype=torch.float)

        for i in range(num_boxes1):
            for j in range(num_boxes2):
                iou = self.calculate_iou(boxes1[i], boxes2[j])
                iou_matrix[i, j] = iou
        return iou_matrix

    def result_concat(self,result,original,idx,device):

        result_tensor = result.gt_boxes.tensor.to(device)
        new_tensor = original.gt_boxes.tensor[idx]
        empty_tensor = torch.empty((1, 4)).to(device)
        empty_tensor[0] = new_tensor
        result.gt_boxes.tensor = torch.cat((result_tensor, empty_tensor), dim=0)

        result_scores = result.scores.to(device)
        new_scores = original.scores[idx]
        new_scores = new_scores.unsqueeze(0)
        result.scores = torch.cat((result_scores, new_scores), dim=0)

        result_classes = result.gt_classes.to(device)
        new_classes = original.gt_classes[idx]
        new_classes = new_classes.unsqueeze(0)
        result.gt_classes = torch.cat((result_classes, new_classes), dim=0)
        return result

    def compute_match_indices(self, label_matrix, iou_matrix, teacher_num_pre, teacher_num_last, thresh, row2col = False):

        match_indices = []
        if row2col:
            for i in range(teacher_num_pre):
                max_iou = 0
                idx1 = None
                idx2 = None
                for j in range(teacher_num_last):
                    if label_matrix[j][i] and iou_matrix[j][i] > thresh and iou_matrix[j][i] > max_iou:
                        max_iou = iou_matrix[j][i]
                        idx1 = j
                        idx2 = i
                if idx1 is not None and idx2 is not None:
                    match_indices.append((idx1, idx2))
        else:
            for i in range(teacher_num_pre):
                max_iou = 0
                idx1 = None
                idx2 = None
                for j in range(teacher_num_last):
                    if label_matrix[i][j] and iou_matrix[i][j] > thresh and iou_matrix[i][j] > max_iou:
                        max_iou = iou_matrix[i][j]
                        idx1 = i
                        idx2 = j
                if idx1 is not None and idx2 is not None:
                    match_indices.append((idx1, idx2))
        return match_indices

    def result_match_align(self, iou_matrix, lable_matrix, iou_threshold=0.6):
        matched_pairs = []
        active_matrix = iou_matrix.clone()
        while True:
            max_iou, flat_idx = active_matrix.flatten().max(dim=0)
            if max_iou < iou_threshold:
                break
            idx1, idx2 = np.unravel_index(flat_idx.item(), active_matrix.shape)
            if lable_matrix[idx1, idx2]:
                matched_pairs.append((idx1.item(), idx2.item()))

                active_matrix[idx1, :] = -1
                active_matrix[:, idx2] = -1
            else:
                active_matrix[idx1, idx2] = -1
        return matched_pairs
    def cluster_fusion_single(self, instances: Instances, iou_threshold, score_threshold ) :

        empty_inst = Instances(instances.image_size)
        empty_inst.gt_boxes = Boxes(torch.empty((0, 4), device=instances.pred_boxes.device))
        empty_inst.gt_classes = torch.empty(0, device= instances.pred_classes.device, dtype=torch.int64)
        empty_inst.scores = torch.empty(0, device = instances.pred_classes.device)
        empty_inst.objectness_logits = torch.tensor([], device=instances.scores.device)

        keep_mask = instances.scores >= score_threshold
        instances = instances[keep_mask]
        if len(instances) == 0:
            return empty_inst
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        labels = instances.pred_classes
        sorted_idx = torch.argsort(scores, descending=True)

        clusters = []
        fused_boxes = []
        fused_scores = []
        fused_labels = []
        for i in sorted_idx:
            current_box = boxes[i]
            current_label = labels[i]
            matched = False
            for j in range(len(fused_boxes)):
                iou = pairwise_iou(Boxes(current_box.unsqueeze(0)), Boxes(fused_boxes[j].unsqueeze(0))).item()
                if iou > iou_threshold and current_label == fused_labels[j]:
                    clusters[j].append(i)

                    indices = torch.stack(clusters[j])
                    cluster_scores = scores[indices]
                    new_score = cluster_scores.mean()
                    weights = cluster_scores / cluster_scores.sum()
                    new_box = (weights.unsqueeze(1) * boxes[indices]).sum(dim=0)
                    fused_scores[j] = new_score
                    fused_boxes[j] = new_box
                    matched = True
                    break
            if not matched:
                clusters.append([i])
                fused_boxes.append(boxes[i])
                fused_scores.append(scores[i])
                fused_labels.append(labels[i])
        fused_inst = Instances(instances.image_size)
        if len(fused_boxes) > 0:
            fused_inst.gt_boxes = Boxes(torch.stack(fused_boxes))
            fused_inst.scores = torch.stack(fused_scores)
            fused_inst.gt_classes = torch.stack(fused_labels)
        return fused_inst

    def batch_cluster_fusion(self, instances_list, iou_threshold, score_threshold):
        """
        Batch process Instances prediction results for multiple images
        """
        return [self.cluster_fusion_single(image_instances, iou_threshold, score_threshold)
                for image_instances in instances_list]
    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)
        # Generate horizontally flipped images
        import torchvision.transforms as transforms
        transforms = transforms.RandomHorizontalFlip(p=1)
        # Rotate images
        rotate_unlabel_data_k = []
        for i in range(len(unlabel_data_k)):
            my_dict = {'file_name':None,'height': None, 'width': None, 'image': None}
            my_dict['file_name'] = unlabel_data_k[i]['file_name']
            my_dict['image_id'] = unlabel_data_k[i]['image_id']
            my_dict['image'] = transforms(unlabel_data_k[i]['image'])
            my_dict['height'] = unlabel_data_k[i]['height']
            my_dict['width'] = unlabel_data_k[i]['width']
            rotate_unlabel_data_k.append(my_dict)
        # self.show_image(unlabel_data_k[1]['image'])
        # self.show_image(rotate_unlabel_data_k[1]['image'])
        # burn-in stage (supervised training with labeled data) perform supervised learning when iteration is less than burn-up steps
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())
        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model   # when iteration equals SEMISUPNET.BURN_UP_STEP, copy model to Teacher model
                self._update_teacher_model(keep_rate=0.00)
                self._update_teacher_model2(keep_rate=0.00)
            elif (   # when iteration is greater than SEMISUPNET.BURN_UP_STEP
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                # select model_teacher
                # EMA update teacher
                # self._update_teacher_model(   # EMA update teacher
                #     keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well

            # Teacher model generates pseudo boxes, this is inference stage not training stage
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.ema_teacher(unlabel_data_k, branch="unsup_data_weak") # Generate pseudo labels on weakly augmented images, including RPN (Region Proposal Network) and ROI (Region of Interest) heads

                (
                    _,
                    rotate_proposals_rpn_unsup_k2,
                    rotate_proposals_roih_unsup_k2,
                    _,
                ) = self.ema_teacher2(rotate_unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            (  # rpn pseudo labels pesudo_proposals_rpn_unsup_k
                pesudo_proposals_rpn_unsup_k,
                num_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(  # Pseudo labels processed according to threshold
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k

            # Clustering fusion strategy
            pesudo_proposals_roih_unsup_k = self.batch_cluster_fusion(proposals_roih_unsup_k, 0.6, 0.5)
            rotate_pesudo_proposals_roih_unsup_k2 = self.batch_cluster_fusion(rotate_proposals_roih_unsup_k2, 0.6, 0.5)
            # self.draw_teacherbox(rotate_unlabel_data_k[1], pesudo_proposals_roih_unsup_k2[1])
            # Flip the predicted pseudo boxes of the second teacher to make the predicted images of both teachers consistent
            pesudo_proposals_roih_unsup_k2 = [] # Used to save flipped pseudo box Instances
            for i in range(len(rotate_pesudo_proposals_roih_unsup_k2)):
                pesudo_proposals_roih_unsup_k2.append(Instances(
                    image_size= rotate_pesudo_proposals_roih_unsup_k2[i].image_size,
                    gt_boxes= rotate_pesudo_proposals_roih_unsup_k2[i].gt_boxes.clone(),
                    gt_classes =  rotate_pesudo_proposals_roih_unsup_k2[i].gt_classes.clone(),
                    scores =  rotate_pesudo_proposals_roih_unsup_k2[i].scores.clone(),
                )) # Need to clone, otherwise they will change together
                # print(pesudo_proposals_roih_unsup_k2)
                width = rotate_pesudo_proposals_roih_unsup_k2[i].image_size[1]
                pesudo_proposals_roih_unsup_k2[i].gt_boxes.tensor[:,0] = torch.sub(width, rotate_pesudo_proposals_roih_unsup_k2[i].gt_boxes.tensor[:,2])
                pesudo_proposals_roih_unsup_k2[i].gt_boxes.tensor[:,2] = torch.sub(width, rotate_pesudo_proposals_roih_unsup_k2[i].gt_boxes.tensor[:,0])
            # self.draw_teacherbox(unlabel_data_k[0],rotate_pesudo_proposals_roih_unsup_k2[0],flap=True)
            # self.draw_teacherbox(unlabel_data_k[0],pesudo_proposals_roih_unsup_k2[0])
            # self.draw_teacherbox(unlabel_data_k[1], rotate_pesudo_proposals_roih_unsup_k2[1], flip=True)

            # Dual teacher label alignment
            # Used to store final prediction results
            result_pesudo_proposals_roih_unsup_k = []
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # index represents the index-th image in a batch
            for index in range(len(pesudo_proposals_roih_unsup_k)):
                # Collect all boxes
                #     result_pesudo_proposals_roih_unsup_k = []
                result_pesudo_proposals_roih_unsup_k.append(Instances(
                    image_size=pesudo_proposals_roih_unsup_k[index].image_size,
                    gt_boxes=Boxes(torch.empty(0)),
                    gt_classes=torch.empty(0, dtype=torch.int64),
                    scores=torch.empty(0),
                ))
                if len(pesudo_proposals_roih_unsup_k[index]) == 0 or \
                        len(rotate_pesudo_proposals_roih_unsup_k2[index]) == 0:
                    continue
            # Label matrix
                label_matrix, num1, num2 = self.build_label_matrix(pesudo_proposals_roih_unsup_k[index], pesudo_proposals_roih_unsup_k2[index])
            # IoU matrix
                iou_matrix = pairwise_iou(pesudo_proposals_roih_unsup_k[index].gt_boxes,pesudo_proposals_roih_unsup_k2[index].gt_boxes)
        # Dual teacher prediction matching
            # If remaining boxes after matching are 1, treat as separate box without fusion
                teacher1_track_matrix =[1] * num1
                teacher2_track_matrix =[1] * num2
                thresh = 0.6
                # Calculate matching indices, row-to-column and column-to-row comprehensive comparison
                result_match_indices = self.result_match_align(iou_matrix, label_matrix, thresh)

            # # Optimization
            #     result_match_indices = list(set(match_indices) & set(match_indices2))
                extracted_list_idx1 = [tup[0] for tup in result_match_indices]
                teacher1_track_matrix = [0 if i in extracted_list_idx1 else value for i, value in enumerate(teacher1_track_matrix)]
                extracted_list_idx2 = [tup[1] for tup in result_match_indices]
                teacher2_track_matrix = [0 if i in extracted_list_idx2 else value for i, value in
                                     enumerate(teacher2_track_matrix)]
            # # Optimization
            # Collect boxes with matched labels
                for i in range(len(result_match_indices)):
                    # Get corresponding index values of dual teacher gt_boxes
                    # Pass bounding box information
                    result_tensor = result_pesudo_proposals_roih_unsup_k[index].gt_boxes.tensor.to(device)
                    box1 = pesudo_proposals_roih_unsup_k[index].gt_boxes.tensor[result_match_indices[i][0]]
                    box2 = pesudo_proposals_roih_unsup_k2[index].gt_boxes.tensor[result_match_indices[i][1]]
                    score1 = pesudo_proposals_roih_unsup_k[index].scores[result_match_indices[i][0]]
                    score2 = pesudo_proposals_roih_unsup_k2[index].scores[result_match_indices[i][1]]
                    new_tensor = (score1 * box1 + score2 * box2) / (score1 + score2)  # Weighted average
                    empty_tensor = torch.empty((1, 4)).to(device)
                    empty_tensor[0] = new_tensor
                    result_pesudo_proposals_roih_unsup_k[index].gt_boxes.tensor = torch.cat((result_tensor, empty_tensor), dim=0)
                    # Pass scores
                    result_scores = result_pesudo_proposals_roih_unsup_k[index].scores.to(device)
                    new_scores = (pesudo_proposals_roih_unsup_k[index].scores[result_match_indices[i][0]] +
                                  pesudo_proposals_roih_unsup_k2[index].scores[result_match_indices[i][1]]) / 2
                    new_scores = new_scores.unsqueeze(0)
                    result_pesudo_proposals_roih_unsup_k[index].scores = torch.cat((result_scores, new_scores), dim=0)

                    # Pass classes
                    result_classes = result_pesudo_proposals_roih_unsup_k[index].gt_classes.to(device)
                    new_classes = pesudo_proposals_roih_unsup_k[index].gt_classes[result_match_indices[i][0]]
                    new_classes = new_classes.unsqueeze(0)
                    result_pesudo_proposals_roih_unsup_k[index].gt_classes = torch.cat((result_classes, new_classes), dim=0)

                # Collect unmatched boxes and add to result_pesudo_proposals_roih_unsup_k
                # Unmatched boxes in teacher1
                indices1 = [index for index, value in enumerate(teacher1_track_matrix) if value == 1]
                # Unmatched boxes in teacher2
                indices2 = [index for index, value in enumerate(teacher2_track_matrix) if value == 1]

                for i in range(len(indices1)):
                    result_pesudo_proposals_roih_unsup_k[index] =\
                        self.result_concat(result_pesudo_proposals_roih_unsup_k[index], pesudo_proposals_roih_unsup_k[index], indices1[i], device)
                for i in range(len(indices2)):
                    result_pesudo_proposals_roih_unsup_k[index] =\
                        self.result_concat(result_pesudo_proposals_roih_unsup_k[index], pesudo_proposals_roih_unsup_k2[index], indices2[i], device)

            # Teacher1 prediction
            # self.draw_teacherbox(unlabel_data_k[0], pesudo_proposals_roih_unsup_k[0])
            # # Teacher2 predicted flipped image
            # self.draw_teacherbox(unlabel_data_k[0], rotate_pesudo_proposals_roih_unsup_k2[0], flip=True)
            # # Flip the flipped image again
            # self.draw_teacherbox(unlabel_data_k[0], pesudo_proposals_roih_unsup_k2[0], flip=True)
            # # # Pseudo label fusion
            # self.draw_teacherbox(unlabel_data_k[0], result_pesudo_proposals_roih_unsup_k[0], flip=True)

            # Use mixed labels as pseudo labels
            joint_proposal_dict["proposals_pseudo_roih"] = result_pesudo_proposals_roih_unsup_k
            #  add pseudo-label to unlabeled data
            # Add pseudo labels to strongly augmented unlabel_data for supervision
            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q
            # Base trainer learns labeled and unlabeled images
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)
            # Since unlabeled images already include teacher predictions, student model branch learns directly using supervised approach
            record_all_unlabel_data, _, _, _ = self.model(  # Input teacher model proposals to student model ROIHead
                all_unlabel_data, branch="supervised"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # Modified: Add a feature perturbation branch with weakly augmented image input
            record_all_unlabel_data_fp, _, _, _ = self.model(
                unlabel_data_k, branch="supervised", need_fp = True
            )
            new_record_all_unlabel_data_fp = {}
            for key in record_all_unlabel_data_fp.keys():
                new_record_all_unlabel_data_fp[key + "_fp_pseudo"] = record_all_unlabel_data_fp[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data_fp)
            # Modified

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1
            losses = loss_dict['loss_cls'] + loss_dict['loss_box_reg'] + loss_dict['loss_rpn_cls'] + loss_dict[
                'loss_rpn_loc'] + (loss_dict['loss_cls_pseudo'] + loss_dict['loss_cls_fp_pseudo']) / 2 + (
                                 loss_dict['loss_rpn_cls_pseudo'] + loss_dict['loss_rpn_cls_fp_pseudo']) / 2

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)
            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]
            total_losses_reduced = sum(loss for loss in loss_dict.values())
            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()
        # Create a new ordered dictionary new_teacher_dict to save updated teacher model parameters.
        new_teacher_dict = OrderedDict()
        # Iterate through teacher model parameter dictionary, for each parameter key-value pair (key and value), perform the following operations:
        # If this parameter exists in student model parameter dictionary (i.e., key exists in student_model_dict),
        # then perform weighted average update. Calculation is student model parameters multiplied by (1 - keep_rate), plus teacher model parameters multiplied by keep_rate,
        # to get new parameter values and save them to new_teacher_dict.
        # If this parameter does not exist in student model parameter dictionary, throw exception indicating the parameter does not exist in student model.
        for key, value in self.ema_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
                # new_teacher_dict[key] = self.EMA(value, keep_rate, student_model_dict[key])
            else:
                raise Exception("{} is not found in student model".format(key))

        self.ema_teacher.load_state_dict(new_teacher_dict)
    @torch.no_grad()
    def _update_teacher_model2(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()
        # Create a new ordered dictionary new_teacher_dict to save updated teacher model parameters.
        new_teacher_dict = OrderedDict()
        # Iterate through teacher model parameter dictionary, for each parameter key-value pair (key and value), perform the following operations:
        # If this parameter exists in student model parameter dictionary (i.e., key exists in student_model_dict),
        # then perform weighted average update. Calculation is student model parameters multiplied by (1 - keep_rate), plus teacher model parameters multiplied by keep_rate,
        # to get new parameter values and save them to new_teacher_dict.
        # If this parameter does not exist in student model parameter dictionary, throw exception indicating the parameter does not exist in student model.
        for key, value in self.ema_teacher2.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
                # new_teacher_dict[key] = self.EMA(value, keep_rate ,student_model_dict[key])
            # if key in student_model_dict.keys():

            #      _value  = self.EMA(value, keep_rate ,student_model_dict[key])

            #     new_teacher_dict[key] = 2 * ema_value - self.EMA(ema_value, keep_rate, student_model_dict[key])
            else:
                raise Exception("{} is not found in student model".format(key))

        self.ema_teacher2.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():

            print('Starting student model testing')
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            print('Starting teacher model testing')
            self._last_eval_results_teacher = self.test(
                self.cfg, self.ema_teacher)
            return self._last_eval_results_teacher

        # def test_and_save_results_teacher2():
        #     print('Starting teacher model 2 testing')
        #     self._last_eval_results_teacher2 = self.test(
        #         self.cfg, self.model_teacher2)
        #     return self._last_eval_results_teacher2

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
        #            test_and_save_results_teacher2))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

class CustomPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        Trainer = UBTeacherTrainer
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        # model_teacher2 = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)
        checkpointer = DetectionCheckpointer(
            ensem_ts_model,
            save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

        model = ensem_ts_model.modelTeacher # Teacher prediction
        # model = ensem_ts_model.modelStudent # Student prediction
        self.model = model
        self.model.eval()

    def __call__(self, original_image):
        with torch.no_grad():
            original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            # predictions = self.model([inputs])
            return predictions