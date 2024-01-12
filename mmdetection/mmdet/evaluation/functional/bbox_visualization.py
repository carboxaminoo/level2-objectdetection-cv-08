import torch
import numpy as np
import torchvision
from torchmetrics.detection import MeanAveragePrecision


class BboxViz:
    def __init__(self) -> None:
        # self.class_num = 10

        # self.pred_dict = []
        # self.gt_dict = []
        # self.predict_base_list = []
        pass

    # def save_predicted_data(self, output) -> None:
    #     gt_batch_bboxes = output.gt_instances["bboxes"]
    #     gt_batch_labels = output.gt_instances["labels"]

    #     pred_batch_bboxes = output.pred_instances["bboxes"]
    #     pred_batch_labels = output.pred_instances["labels"]
    #     pred_batch_scores = output.pred_instances["scores"]

    #     gt_batch_dict = {
    #         "bboxes": gt_batch_bboxes,
    #         "labels": torch.tensor(gt_batch_labels).to(gt_batch_bboxes.device),
    #     }
    #     pred_batch_dict = {
    #         "bboxes": pred_batch_bboxes,
    #         "labels": torch.tensor(pred_batch_labels).to(pred_batch_bboxes.device),
    #         "scores": torch.tensor(pred_batch_scores).to(pred_batch_bboxes.device),
    #     }
    #     self.predict_base_list.append([gt_batch_dict, pred_batch_dict])
    
    def bbox_info(self, label, bbox, score=None):
        bbox = bbox.cpu().numpy()
        position = {
            "minX": float(bbox[0]),
            "maxX": float(bbox[2]),
            "minY": float(bbox[1]),
            "maxY": float(bbox[3]),
        }
        class_id = int(label.cpu())
        if score is not None:
            scores = {"acc" : float(score.cpu())}
            return position, class_id, scores
        else:
            return position, class_id

    def visualize_pred_bboxes(self, pred_instances, class_labels):
        pred_dict = []
        for label, score, bbox in zip(
            pred_instances.labels, pred_instances.scores, pred_instances.bboxes
        ):
            position, class_id, scores = self.bbox_info(label, bbox, score)
            pred_dict.append(
                {
                    "position": position,
                    "class_id": class_id,
                    "box_caption": class_labels[class_id],
                    "scores": scores,
                }
            )

        return pred_dict

    def visualize_gt_bboxes(self, gt_instances, class_labels):
        gt_dict = []
        for label, bbox in zip(gt_instances.labels, gt_instances.bboxes):
            position, class_id = self.bbox_info(label, bbox)
            gt_dict.append(
                {
                    "position": position,
                    "class_id": class_id,
                    "box_caption": class_labels[class_id]}
            )

        return gt_dict
