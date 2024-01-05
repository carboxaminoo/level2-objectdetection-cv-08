# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet.datasets.samplers import TrackImgSampler
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample, TrackDataSample
import pandas as pd

@HOOKS.register_module()
class SubmissionHook(Hook):
    """
    Hook for submitting results. Saves verification and test process prediction results.

    In the testing phase:

    1. Receives labels, scores, and bboxes from outputs and stores them in prediction_strings.
    2. Get the img_path of outputs and save it in file_names.

    Args:
        prediction_strings (list): [labels + ' ' + scores + ' ' + x_min + ' ' + y_min + ' ' + x_max + ' ' + y_max]를 추가한 list
        file_names (list): img_path를 추가한 list
        test_out_dir (str) : 저장할 경로
    """

    def __init__(self, test_out_dir: Optional[str] = None):
        self.prediction_strings = []
        self.file_names = []
        self.test_out_dir = test_out_dir

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        assert len(outputs) == 1, \
            'only batch_size=1 is supported while testing.'
        
        pred_x_size = data_batch['inputs'][0].shape[1]
        pred_y_size = data_batch['inputs'][0].shape[2]

        for output in outputs:
            prediction_string = ''
            for label, score, bbox in zip(output.pred_instances.labels, output.pred_instances.scores, output.pred_instances.bboxes):
                bbox = bbox.cpu().numpy()
                bbox[0], bbox[2] = bbox[0] / pred_x_size * 1024, bbox[2] / pred_x_size * 1024
                bbox[1], bbox[3] = bbox[1] / pred_y_size * 1024, bbox[3] / pred_y_size * 1024
                prediction_string += str(int(label.cpu())) + ' ' + str(float(score.cpu())) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3])
            self.prediction_strings.append(prediction_string)
            self.file_names.append(output.img_path[13:])

    def after_test(self, runner: Runner):
        """
        Run after testing

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
        """
        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        submission = pd.DataFrame()
        submission['PredictionString'] = self.prediction_strings
        submission['image_id'] = self.file_names
        submission.to_csv(osp.join(self.test_out_dir, 'submission.csv'), index=None)
        print('submission saved to {}'.format(osp.join(self.test_out_dir, 'submission.csv')))
        