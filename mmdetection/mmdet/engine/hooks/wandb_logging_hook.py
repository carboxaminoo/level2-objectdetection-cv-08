from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

@HOOKS.register_module()
class WandbLoggingHook(Hook):
    """
    Wandb logging hook.

    This hook will regularly saveing the experimential values
    on wandb during training and validationing.

    Args:
        train_loss (float) : model training에서 사용한 loss
        train_acc (float) : model training에서 사용한 acc
        train_mAP (float) : model training에서 사용한 mean average precision
        train_iou_score (float) : interest of union과 ground true 사이의
                                  차이를 계산하여 출력
        
        val_loss (float) : model validation에서 사용한 loss
        val_acc (float) : model validation에서 사용한 acc
        val_mAP (float) : model validation에서 사용한 mean average precision
        val_iou_score (float) : interest of union과 ground true 사이의
                                차이를 계산하여 출력

    """

    def __init__(
        self,
        train_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        train_mAP: Optional[float] = None,
        train_iou_score: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        val_mAP: Optional[float] = None,
        val_iou_score: Optional[float] = None,
    ):
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.train_mAP = 0.0
        self.train_iou_score = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.val_mAP = 0.0
        self.val_iou_score = 0.0
        
    def after_train_iter(self, 
                         runner: Runner, 
                         batch_idx: int,
                         data_batch: dict,
                         outputs: Optional[dict] = None) -> None:
        pass

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