# EALink: An Efficient and Accurate Pre-Trained Framework for Issue-Commit Link Recovery
Source code for the ASE'23 paper ``EALink: An Efficient and Accurate Pre-Trained Framework for Issue-Commit Link Recovery``.

## Folder
- ```Dstill``` folder contains the data class file ```dataset.py``` used in the distillation step, the configuration file ```tiny_bert_config.json``` and the distillation source file  ```bertdistill.py ```.
- ```LinkGenerator``` folder contains the ```parser_lang``` folder for parsing abstract syntax trees and preprocessing steps for raw data.
- ```data``` is used to store the processed datasets (you can get it in the link below).
- ```models```contains training and testing files.

## Environment
* python 3.9.7
* pytorch 1.11.0
* pandas 1.3.4
* numpy 1.21.6
* transformers 4.21.0
* cudatoolkit 11.3.1
* torchaudio 1.11.0
* torchvision 1.12.0
* GPU with CUDA 11.3

## Datasets
We have constructed six large-scale project datasets for evaluating issue-commit link recovery. You can download the *[final dataset](https://drive.google.com/drive/folders/1coZbAtOYGPVQQdjf2MnykMFpfLyhw1JZ)* described in the paper. To generate the dataset used for EALink in our experiments, please follow the data preprocessing steps.

## How to run
 
### 1. Data preprocessing

You can follow the steps in the `LinkGenerator` folder to generate the dataset used for EALink. Or you can directly download the [processed dataset](https://drive.google.com/drive/folders/1c-HkdL7xaKm9OYMlXYyxSVlw3ywnqhXA) for use.

#### Get issue-code links for auxiliary task
In the `LinkGenerator` folder, `0_subdata.py` generates issue-code links. You can run the following command：
```
python 0_subdata.py
```
#### Get issue-commit links after word segmentation processing
```
python 1_splitword.py
```
#### Merge
dataset merging
```
python 2_sub_merge.py
```
### 2. Distill the pre-trained model
```
cd Dstill
python bertdistill.py
```
### 3. Train and test
In the `models` folder, `train.py` and `test.py` enable training and testing of the trained model, respectively.

#### Train
```
cd models
python train.py \
   --tra_batch_size 16 \
   --val_batch_size 16 \
   --end_epoch 400 \
   --output_model <model_save_path> 
```
### Test
```
python test.py \
   --tes_batch_size 16 \
   --model_path <model_path> 
```
