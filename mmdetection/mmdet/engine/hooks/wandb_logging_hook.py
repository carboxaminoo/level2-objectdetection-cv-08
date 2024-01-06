from typing import Optional, Dict

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmdet.registry import MODELS

import wandb



@HOOKS.register_module()
class WandbLoggingHook(Hook):
    """
    Wandb logging hook.

    This hook regularly logs experimental values in wandb during training and validation.

    Args:
        train_loss (float) : model training에서 사용한 loss, train_loss = loss_rpn_cls + loss_rpn_bbox + loss_cls + loss_bbox
        train_loss_rpn_cls (float) : model training에서 사용한 rpn cls loss
        train_loss_rpn_bbox (float) : model training에서 사용한 rpn bbox loss
        train_loss_cls (float) : model training에서 사용한 cls loss
        train_loss_bbox (float) : model training에서 사용한 bbox loss
        train_acc (float) : model training에서 사용한 acc
        
        val_loss (float) : model validation에서 사용한 loss
        val_acc (float) : model validation에서 사용한 acc
        val_mAP (float) : model validation에서 사용한 mean average precision

    """

    def __init__(self):
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.val_mAP = 0.0
        
    def after_train_iter(self, 
                         runner: Runner, 
                         batch_idx: int,
                         data_batch: dict,
                         outputs: Optional[dict] = None) -> None:
        """
        Train에서 나온 loss, loss_rpn_cls, loss_rpn_bbox, loss_cls, loss_bbox, acc를 iter마다 wandb에 logging

        Args:
            outputs:
                loss
                loss_rpn_cls
                loss_rpn_bbox
                loss_cls
                loss_bbox
                acc            
        """
        wandb.log({
            "train_loss": outputs['loss'],
            "train_loss_rpn_cls": outputs['loss_rpn_cls'],
            "train_loss_rpn_bbox": outputs['loss_rpn_bbox'],
            "train_loss_cls": outputs['loss_cls'],
            "train_loss_bbox": outputs['loss_bbox'],
            "train_acc": outputs['acc']
        })

    def after_val_iter(self, 
                       runner: Runner, 
                       batch_idx: int,
                       data_batch: dict,
                       outputs: Optional[dict] = None) -> None:
        """
        Val에서 나온 loss, acc를 iter마다 wandb에 logging

        Args:
            data_batch:
                data_samples[0]:
                    gt_instances:
                        labels
                        bboxes
            outputs[0]:
                pred_instances:
                    labels
                    scores
                    bboxes
            loss:
                cls_loss
                bbox_loss
        """
        # gt
        gt_labels = outputs[0].gt_instances.labels
        gt_bboxes = outputs[0].gt_instances.bboxes
        # pred
        pred_labels = outputs[0].pred_instances.labels
        pred_scores = outputs[0].pred_instances.scores
        pred_bboxes = outputs[0].pred_instances.bboxes
        # loss
        if runner.cfg['model']['roi_head'] is not None:
            cls_loss_function = runner.cfg['model']['roi_head']['bbox_head']['loss_cls']['type']
            bbox_loss_function = runner.cfg['model']['roi_head']['bbox_head']['loss_bbox']['type']
        elif runner.cfg['model']['bbox_head'] is not None:
            cls_loss_function = runner.cfg['model']['bbox_head']['loss_cls']['type']
            bbox_loss_function = runner.cfg['model']['bbox_head']['loss_bbox']['type']
        else:
            raise NotImplementedError
        cls_loss_function = MODELS.module_dict[cls_loss_function]
        bbox_loss_function = MODELS.module_dict[bbox_loss_function]

        # 근데 생각해보니까 cls loss는 model output logits으로 구해야하는데, pred_labels로 구하고 있음
        # 근데 model output logits을 여기서 찾을 수가 없음
        # 찾으려면 라이브러리를 수정해야할 것 같음(패키지 수정 필요)
        # 어떻게 할지 생각 해야할 듯...
        # val_loss_bbox = bbox_loss_function(pred_labels, gt_labels)

        # 
        val_loss_bbox = bbox_loss_function(pred_bboxes, gt_bboxes)

        # mAP

        # cls acc

        wandb.log({
            "val_loss_bbox": val_loss_bbox,
            # "val_acc": outputs['acc']
        })


    def after_val_epoch(self,
                        runner: Runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Val에서 나온 mAP를 wandb에 logging

        Args:
            metrics:
                bbox_mAP
                bbox_mAP_50
                bbox_mAP_75
                bbox_mAP_s
                bbox_mAP_m
                bbox_mAP_l
        """
        wandb.log({
            "val_mAP": metrics['bbox_mAP'],
            "val_mAP_50": metrics['bbox_mAP_50'],
            "val_mAP_75": metrics['bbox_mAP_75'],
            "val_mAP_s": metrics['bbox_mAP_s'],
            "val_mAP_m": metrics['bbox_mAP_m'],
            "val_mAP_l": metrics['bbox_mAP_l']
        })