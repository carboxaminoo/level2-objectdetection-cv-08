from typing import Optional, Sequence, Dict

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from mmdet.evaluation.metrics.recycle_metric import RecycleMetric
from torchmetrics.detection import MeanAveragePrecision

import torch
import wandb
import os
from tqdm import tqdm
from PIL import Image


@HOOKS.register_module()
class MetricHook(Hook):
    def __init__(self) -> None:
        self.metric_class = RecycleMetric()
        self.count_boundary = self.metric_class.bbox_count_boundary[1:-1]
        self.size_boundary = self.metric_class.bbox_size_boundary[1:-1]
        self.train_loss = {
            "train_loss": [],
            "train_loss_rpn_cls": [],
            "train_loss_rpn_bbox": [],
            "train_loss_cls": [],
            "train_loss_bbox": [],
            "train_acc": [],
        }

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Optional[dict] = None,
    ) -> None:
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
            pre_outs:
                pred_instances:
                    labels
                    scores
                    bboxes
                gt_instances:
                    labels
                    bboxes
            annotations(list[dict]): [img1_ann_dict, img2_ann_dict, …]
                'bboxes': numpy array of shape (n, 4)
                'labels': numpy array of shape (n, )
            results(list[list]): [[cls1_det, cls2_det, …], …]
                cls1_det: numpy array of shape (n, cls, 4)
                cls2_det: numpy array of shape (n, cls, 4)
                …

        """
        self.train_loss["train_loss"].append(outputs["loss"])
        self.train_loss["train_loss_rpn_cls"].append(outputs["loss_rpn_cls"])
        self.train_loss["train_loss_rpn_bbox"].append(outputs["loss_rpn_bbox"])
        self.train_loss["train_loss_cls"].append(outputs["loss_cls"])
        self.train_loss["train_loss_bbox"].append(outputs["loss_bbox"])
        self.train_loss["train_acc"].append(outputs["acc"])

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

        base_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        base_metric50 = MeanAveragePrecision(
            iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
        )
        bbox_size_metrics = [
            MeanAveragePrecision(
                iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
            )
            for _ in range(self.metric_class.bbox_size_class_num)
        ]
        bbox_count_metrics = [
            MeanAveragePrecision(
                iou_type="bbox", class_metrics=True, iou_thresholds=[0.5]
            )
            for _ in range(self.metric_class.bbox_count_class_num)
        ]

        for gt_dict, pred_dict in tqdm(self.metric_class.predict_base_list):
            base_metric.update([pred_dict], [gt_dict])
            base_metric50.update([pred_dict], [gt_dict])
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

        # -----------------------------------------
        # WandB 로깅 부분 추가

        score_dict = {}
        # train Scores
        for key, value in self.train_loss.items():
            score_dict[key] = sum(value) / len(value)

        labels = [
            "General trash",
            "Paper",
            "Paper pack",
            "Metal",
            "Glass",
            "Plastic",
            "Styrofoam",
            "Plastic bag",
            "Battery",
            "Clothing",
        ]
        base_score_names = ["mAP", "mAP50", "mAP75", "mAR_1", "mAR_10", "mAR_100"]
        base_socre_indexs = ["map", "map_50", "map_75", "mar_1", "mar_10", "mar_100"]

        base_metric_score = base_metric.compute()
        base_metric50_score = base_metric50.compute()
        bbox_size_metric_scores = [x.compute() for x in bbox_size_metrics]
        bbox_count_metric_scores = [x.compute() for x in bbox_count_metrics]

        for score_name, score_index in zip(base_score_names, base_socre_indexs):
            score_dict[f"val_{score_name}"] = base_metric_score[score_index]
        for index, label in enumerate(labels):
            score_dict[f"val_{label}_mAP50"] = base_metric50_score["map_per_class"][
                index
            ]
            score_dict[f"val_{label}_mAR100"] = base_metric_score["mar_100_per_class"][
                index
            ]

        bbox_size_map_name = [
            f"val_bbox_size_{self.size_boundary[0]}",
            f"val_bbox_size_{self.size_boundary[0]}_{self.size_boundary[1]}",
            f"val_bbox_size_{self.size_boundary[1]}_{self.size_boundary[2]}",
            f"val_bbox_size_{self.size_boundary[2]}_{self.size_boundary[3]}",
            f"val_bbox_size_{self.size_boundary[3]}",
        ]
        bbox_count_map_name = [
            f"val_bbox_count_{self.count_boundary[0]}",
            f"val_bbox_count_{self.count_boundary[0]}_{self.count_boundary[1]}",
            f"val_bbox_count_{self.count_boundary[1]}_{self.count_boundary[2]}",
            f"val_bbox_count_{self.count_boundary[2]}_{self.count_boundary[3]}",
            f"val_bbox_count_{self.count_boundary[3]}",
        ]
        for index, name in enumerate(bbox_size_map_name):
            score_dict[name + "_mAP50"] = bbox_size_metric_scores[index]["map_50"]
            score_dict[name + "_mAR100"] = bbox_size_metric_scores[index]["mar_100"]
        for index, name in enumerate(bbox_count_map_name):
            score_dict[name + "_mAP50"] = bbox_count_metric_scores[index]["map_50"]
            score_dict[name + "_mAR100"] = bbox_count_metric_scores[index]["mar_100"]

        for metric_index, name in enumerate(bbox_size_map_name):
            for label_index, label in enumerate(labels):
                if label_index in bbox_size_metric_scores[metric_index]["classes"]:
                    score_dict[name + f"_{label}_mAP50"] = bbox_size_metric_scores[
                        metric_index
                    ]["map_per_class"][
                        torch.nonzero(
                            bbox_size_metric_scores[metric_index]["classes"]
                            == label_index
                        ).squeeze()
                    ]
                    score_dict[name + f"_{label}_mAR100"] = bbox_size_metric_scores[
                        metric_index
                    ]["mar_100_per_class"][
                        torch.nonzero(
                            bbox_size_metric_scores[metric_index]["classes"]
                            == label_index
                        ).squeeze()
                    ]
                else:
                    score_dict[name + f"_{label}_mAP50"] = -1
                    score_dict[name + f"_{label}_mAR100"] = -1
        for metric_index, name in enumerate(bbox_count_map_name):
            for label_index, label in enumerate(labels):
                if label_index in bbox_count_metric_scores[metric_index]["classes"]:
                    score_dict[name + f"_{label}_mAP50"] = bbox_count_metric_scores[
                        metric_index
                    ]["map_per_class"][
                        torch.nonzero(
                            bbox_count_metric_scores[metric_index]["classes"]
                            == label_index
                        ).squeeze()
                    ]
                    score_dict[name + f"_{label}_mAR100"] = bbox_count_metric_scores[
                        metric_index
                    ]["mar_100_per_class"][
                        torch.nonzero(
                            bbox_count_metric_scores[metric_index]["classes"]
                            == label_index
                        ).squeeze()
                    ]
                else:
                    score_dict[name + f"_{label}_mAP50"] = -1
                    score_dict[name + f"_{label}_mAR100"] = -1

        # image logging
        image_folder_path = os.path.join(
            runner.work_dir,
            "_".join(runner._experiment_name.split("_")[-2:]),
            "vis_data",
            "vis_image",
        )
        image_path_list = [
            os.path.join(image_folder_path, image_path)
            for image_path in os.listdir(image_folder_path)
        ]
        image_path_list = sorted(image_path_list)
        image_list = [
            wandb.Image(
                Image.open(path), caption=f"{path.split('/')[-1].split('.')[0]}"
            )
            for path in image_path_list
        ]

        score_dict["visual_inference_image"] = image_list

        # wandb log
        wandb.log(score_dict)

        print("box_size_0_map : ", bbox_size_metric_scores[0]["map_50"])
        print("box_size_1_map : ", bbox_size_metric_scores[1]["map_50"])
        print("box_size_2_map : ", bbox_size_metric_scores[2]["map_50"])
        print("box_count_0_map : ", bbox_count_metric_scores[0]["map_50"])
        print("box_count_1_map : ", bbox_count_metric_scores[1]["map_50"])
        print("box_count_2_map : ", bbox_count_metric_scores[2]["map_50"])
        print("box_count_3_map : ", bbox_count_metric_scores[3]["map_50"])

        # clear
        self.train_loss = {
            "train_loss": [],
            "train_loss_rpn_cls": [],
            "train_loss_rpn_bbox": [],
            "train_loss_cls": [],
            "train_loss_bbox": [],
            "train_acc": [],
        }
        self.metric_class.clear_init()

        # wandb.log(
        #     {
        #         "bbox_mAP": base_metric.compute()["map_50"],
        #         "class_AP": sum(base_metric.compute()["map_50"])
        #         / len(base_metric.compute()["map_50"]),
        #     }
        # )

        # # 각 박스 사이즈 별 메트릭 로깅
        # for idx, size_metric in enumerate(bbox_size_metrics):
        #     size_mAP_key = f"box_size_{self.size_boundary[idx]}_mAP"
        #     size_mAP_value = size_metric.compute()["map_50"]
        #     size_class_AP_key = f"box_size_{self.size_boundary[idx]}_class_AP"
        #     size_class_AP_value = sum(size_metric.compute()["map_50"]) / len(
        #         size_metric.compute()["map_50"]
        #     )

        #     wandb.log(
        #         {size_mAP_key: size_mAP_value, size_class_AP_key: size_class_AP_value}
        #     )

        # # 각 박스 카운트 별 메트릭 로깅
        # for idx, count_metric in enumerate(bbox_count_metrics):
        #     count_mAP_key = f"box_count_{idx}_mAP"
        #     count_mAP_value = count_metric.compute()["map_50"]
        #     count_class_AP_key = f"box_count_{idx}_class_AP"
        #     count_class_AP_value = sum(count_metric.compute()["map_50"]) / len(
        #         count_metric.compute()["map_50"]
        #     )

        #     wandb.log(
        #         {
        #             count_mAP_key: count_mAP_value,
        #             count_class_AP_key: count_class_AP_value,
        #         }
        #     )
