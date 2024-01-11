import torch
import numpy as np
import torchvision
from torchmetrics.detection import MeanAveragePrecision


class BboxViz:
    def __init__(self) -> None:
        self.class_num = 10

        self.pred_dict = []
        self.gt_dict = []
        self.predict_base_list = []

    def clear_init(self) -> None:
        self.pred_dict = []
        self.gt_dict = []
        self.predict_base_list = []

    def save_predicted_data(self, output) -> None:
        gt_batch_bboxes = output.gt_instances["bboxes"]
        gt_batch_labels = output.gt_instances["labels"]

        pred_batch_bboxes = output.pred_instances["bboxes"]
        pred_batch_labels = output.pred_instances["labels"]
        pred_batch_scores = output.pred_instances["scores"]

        gt_batch_dict = {
            "bboxes": gt_batch_bboxes,
            "labels": torch.tensor(gt_batch_labels).to(gt_batch_bboxes.device),
        }
        pred_batch_dict = {
            "bboxes": pred_batch_bboxes,
            "labels": torch.tensor(pred_batch_labels).to(pred_batch_bboxes.device),
            "scores": torch.tensor(pred_batch_scores).to(pred_batch_bboxes.device),
        }
        self.predict_base_list.append([gt_batch_dict, pred_batch_dict])

    def visualize_pred_bboxes(self, pred_instances, class_labels):
        for label, score, bbox in zip(
            pred_instances.labels, pred_instances.scores, pred_instances.bboxes
        ):
            bbox = bbox.cpu().numpy()
            position = {
                "minX": bbox[0],
                "maxX": bbox[2],
                "minY": bbox[1],
                "maxY": bbox[3],
            }
            class_id = int(label.cpu())
            box_caption = class_labels[class_id]
            scores = {"acc": float(score.cpu())}
            self.pred_dict.append(
                {
                    "position": position,
                    "class_id": class_id,
                    "box_caption": box_caption,
                    "scores": scores,
                }
            )

        return self.pred_dict

    def visualize_gt_bboxes(self, gt_instances, class_labels):
        for label, bbox in zip(gt_instances.labels, gt_instances.bboxes):
            bbox = bbox.cpu().numpy()
            position = {
                "minX": bbox[0],
                "maxX": bbox[2],
                "minY": bbox[1],
                "maxY": bbox[3],
            }
            class_id = int(label.cpu())
            box_caption = class_labels[class_id]
            self.gt_dict.append(
                {"position": position, "class_id": class_id, "box_caption": box_caption}
            )

        return self.gt_dict
