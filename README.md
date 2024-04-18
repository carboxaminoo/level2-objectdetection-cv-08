
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

- Data_EDA : This folder contains..
- mmdetection : This folder contains...
- operating_configs : This folder contains...
- README.md
- requirements.txt : contains the necessary packages to be installed
- train.py : This file used for training the model

## Code Structure Description 

**Data EDA**

 - Use MaskSplitByProfileDataset
 - Downsampling
 - Stratified Kfold

**Model**
 - Ensemble `Soft Voting`
 - Learn additional Fine Tuning based on the public pretrained model
	 `EfficientNet` + `ConvNext` + `ConvNext(Stratified Kfold)`


## Getting Started

### Setting up Vitual Enviornment

1. Initialize and update the server
	```
    su -
    source .bashrc
    ```

2. Create and Activate a virtual environment in the project directory

	```
    conda create -n env python=3.8
    conda activate env
	```

4. To deactivate and exit the virtual environment, simply run:

	```
	deactivate
	```

### ??? Install Requirements

To Insall the necessary packages liksted in `requirements.txt`, run the following command while your virtual environment is activated:
```
pip install -r requirements.txt
```

### Usage
[Description of all arguments]()

#### Training

To train the model with your custom dataset, set the appropriate directories for the training images and model saving, then run the training script.
- **single model**

```
python train.py --data_dir /path/to/images --model_dir /path/to/model --model MODEL_NAME
```

- **single multiple model**

```
python train_single_multiple.py --data_dir /path/to/images --model_dir /path/to/model --model MODEL_NAME
```


#### Inference

For generating predictions with a trained model, provide directories for evaluation data, the trained model, and output, then run the inference script.
- **single model**
```
python inference.py --data_dir /path/to/images --model_dir /path/to/model --output_dir /path/to/model --model MODEL_NAME
```
- **single multiple model**
```
python inference.py --data_dir /path/to/images --model_dir /path/to/model --output_dir /path/to/model --model_mode single_multiple --model MODEL_NAME
```

#### Ensemble
- **ensemble (hard voting)**
```
python hard_voting.py --file_dir ./csv --csv1 file1.csv --csv2 file2.csv --csv3 file3.csv
```

- **ensemble (soft voting)**
```
python soft_voting.py --models MODEL_NAME1 MODEL_NAME2 MODEL_NAME3 --model_dir ./checkpoint --model_files file1.pth file2.pth file3.pth --data_dir ./data/eval
```


### [Arguments](./docs/arguments.md)

## Model
- [ConvNext Tiny](./docs/ConvNext.md)
- [EfficientNet v2 small](./docs/EfficientNet.md)
- [ViT16](./docs/ViT16.md)

## Dataloader
- [oversampling with data augmentation](./docs/data_sampling.md)
