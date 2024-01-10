from typing import Optional, Sequence, Dict

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from mmdet.evaluation.metrics.recycle_metric import RecycleMetric
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm


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
        # self.metric_class.calculate_ap50_class(outputs=outputs, data_batch=data_batch)
        # self.metric_class.calculate_ap50_bbox_class(
        #     outputs=outputs, data_batch=data_batch
        # )]
        self.metric_class.save_coco_class_data(outputs=outputs, data_batch=data_batch)
        self.metric_class.save_bbox_size_class_data(
            outputs=outputs, data_batch=data_batch
        )
        self.metric_class.save_bbox_count_class_data(
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
        # class_aps = []
        # class_bbox_aps = []
        # for class_ap in self.metric_class.ap50_class_list:
        #     class_aps.append(sum(class_ap) / len(class_ap))
        # for bbox_class_ap in self.metric_class.ap50_bbox_class_list:
        #     class_bbox_aps.append(sum(bbox_class_ap) / len(bbox_class_ap))
        base_metric = MeanAveragePrecision(
            iou_type="bbox", class_metrics=True, iou_thresholds=[0.25, 0.5, 0.75]
        )
        bbox_size_metrics = [
            MeanAveragePrecision(
                iou_type="bbox", class_metrics=True, iou_thresholds=[0.25, 0.5, 0.75]
            )
            for _ in range(self.metric_class.bbox_size_class_num)
        ]
        bbox_count_metrics = [
            MeanAveragePrecision(
                iou_type="bbox", class_metrics=True, iou_thresholds=[0.25, 0.5, 0.75]
            )
            for _ in range(self.metric_class.bbox_count_class_num)
        ]

        for gt_dict, pred_dict in tqdm(self.metric_class.predict_base_list):
            base_metric.update([pred_dict], [gt_dict])
        for idx, bbox_size_dict in enumerate(
            tqdm(self.metric_class.predict_bbox_size_dict)
        ):
            for gt_dict, pred_dict in bbox_size_dict:
                bbox_size_metrics[idx].update([pred_dict], [gt_dict])
        for idx, bbox_count_dict in enumerate(
            tqdm(self.metric_class.predict_bbox_count_dict)
        ):
            for gt_dict, pred_dict in bbox_count_dict:
                bbox_count_metrics[idx].update([pred_dict], [gt_dict])

        box_size_0_map = bbox_size_metrics[0].compute()["map_50"]
        box_size_1_map = bbox_size_metrics[1].compute()["map_50"]
        box_size_2_map = bbox_size_metrics[2].compute()["map_50"]

        box_count_0_map = bbox_count_metrics[0].compute()["map_50"]
        box_count_1_map = bbox_count_metrics[1].compute()["map_50"]
        box_count_2_map = bbox_count_metrics[2].compute()["map_50"]
        box_count_3_map = bbox_count_metrics[3].compute()["map_50"]

        print(base_metric.compute())
        print("Class AP : ", base_metric)
        print("box_size_0_map : ", box_size_0_map)
        print("box_size_1_map : ", box_size_1_map)
        print("box_size_2_map : ", box_size_2_map)
        print("box_count_0_map : ", box_count_0_map)
        print("box_count_1_map : ", box_count_1_map)
        print("box_count_2_map : ", box_count_2_map)
        print("box_count_3_map : ", box_count_3_map)
        # print("BBOX Class AP : ", bbox_metric)
