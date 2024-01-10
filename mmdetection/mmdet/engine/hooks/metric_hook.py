from typing import Optional, Sequence, Dict

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from mmdet.evaluation.metrics.recycle_metric import RecycleMetric


@HOOKS.register_module()
class MetricHook(Hook):
    def __init__(self) -> None:
        self.metric_class = RecycleMetric()

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[Sequence[DetDataSample]] = None,
    ) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (Sequence[:obj:`DetDataSample`], optional):
                Outputs from model. Defaults to None.
        """
        self.metric_class.calculate_ap50_class(outputs=outputs, data_batch=data_batch)
        self.metric_class.calculate_ap50_bbox_class(
            outputs=outputs, data_batch=data_batch
        )

    def after_val_epoch(
        self, runner: Runner, metrics: Optional[Dict[str, float]] = None
    ) -> None:
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
        class_aps = []
        class_bbox_aps = []
        for class_ap in self.metric_class.ap50_class_list:
            class_aps.append(sum(class_ap) / len(class_ap))
        for bbox_class_ap in self.metric_class.ap50_bbox_class_list:
            class_bbox_aps.append(sum(bbox_class_ap) / len(bbox_class_ap))
        print("Class AP : ", class_aps)
        print("BBOX Class AP : ", class_bbox_aps)
