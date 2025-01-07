
# Level2-ObjectDetection Competetion
🌟**CV-08**조🌟 **MakeZenerator 팀**
김태양, 김혜지, 신호준, 성주희, 임서현, 정소윤

## Project Structure

```
${PROJECT}
 ┃  
 ┣ Data_EDA
 ┃ ┗ Train_data_EDA.ipynb  
 ┃ ┗ upsampling.ipynb  
 ┃ ┗ Val_data_EDA.ipynb  
 ┃
 ┣ mmdetection
 ┃
 ┣ operating_configs
 ┃ ┗ base_config.py
 ┃
 ┣ .github
 ┣ .gitignore  
 ┣ .gitmessage.txt  
 ┣ .pre-commit-config.yaml  
 ┣ README.md  
 ┗ cuda_test.py
```

- Data_EDA : This folder contains the results of exploratory data analysis (EDA) and various data augmentation methods.
- mmdetection : This folder is set up for a competition using MMdetectionv3, where custom hooks are implemented to calculate mAP at various thresholds (e.g., class-wise, bbox, segmentation) in addition to the metrics provided by MMdetection. A DataVisualization Hook is also created to overlay GT and predicted bboxes for visualization and upload to wandb.
- operating_configs : This file contains an MMDetection configuration file that defines data paths, model, dataloaders, evaluation metrics, visualization, and custom hooks for recycling object detection.
- README.md

