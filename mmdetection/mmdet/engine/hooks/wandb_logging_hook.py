from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

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

    def before_val(self, runner: Runner):
        pass

    def after_val_iter(self, 
                       runner: Runner, 
                       batch_idx: int,
                       data_batch: dict,
                       outputs: Optional[dict] = None) -> None:
        pass

    def after_val_epoch(self, runner: Runner):
        pass