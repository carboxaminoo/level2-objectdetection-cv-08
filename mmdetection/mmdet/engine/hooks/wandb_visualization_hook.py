# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Dict, Optional, Sequence
import mmcv
import numpy as np
import wandb

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.fileio import get

# from mmengine.visualization import Visualizer

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from torchvision.ops import box_iou
from mmdet.evaluation.functional.bbox_visualization import BboxViz


@HOOKS.register_module()
class WandbVizHook(Hook):
    """
    Visualize the detection results on images and log them to wandb.

    In the testing phase:
        It supports the following visualizations:
            - Ground truth and predicted Bbox visualization
    """

    def __init__(self):
        # self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.class_labels = {
            0: "General trash",
            1: "Paper",
            2: "Paper pack",
            3: "Metal",
            4: "Glass",
            5: "Plastic",
            6: "Styrofoam",
            7: "Plastic bag",
            8: "Battery",
            9: "Clothing",
        }
        self.box_visualizer = BboxViz()
        self.data_root = "data/recycle/"
        self.visualization_dict = dict()

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Optional[Sequence[DetDataSample]] = None,
    ) -> None:
        """
        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.

            class_labels = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
        """
        # runner.cfg.wandb
        image_path = self.data_root + outputs[0].img_path[-14:]
        img_bytes = get(image_path, backend_args=None)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        pred_data = []
        gt_data = []
        pred_data.append(
            self.box_visualizer.visualize_pred_bboxes(
                outputs[0].pred_instances, self.class_labels
            )
        )
        gt_data.append(
            self.box_visualizer.visualize_gt_bboxes(
                outputs[0].gt_instances, self.class_labels
            )
        )

        pred_img = wandb.Image(
            img,
            boxes={
                "predictions": {
                    "box_data": [data for data in pred_data[0]],
                    "class_labels": self.class_labels,
                },
                "ground_truth": {
                    "box_data": [data for data in gt_data[0]],
                    "class_labels": self.class_labels,
                },
            },
        )
        # gt_img = wandb.Image(
        #     image_path,
        #     boxes={
        #         "ground_truth": {"box_data": gt_data},
        #         "class_labels": self.class_labels,
        #     },
        # )
        table = wandb.Table(columns=["image"])
        table.add_data(pred_img)
        self.visualization_dict[f"{outputs[0].img_path[-14:]}_pred"] = table
        

    def after_val_epoch(self, runner, metrics: Dict[str, float] | None = None) -> None:
        wandb.log(self.visualization_dict)

        self.visualization_dict = dict()

    def after_test(self, runner: Runner):
        pass
