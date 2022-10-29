# TinySSD for AI Experiment work.
TinySSD is a tiny version of the Single Shot Multibox Detection(SSD).
https://arxiv.org/abs/1802.06488
This project is to divide the class code to a clear profile structure and make
some improvement in the code.
## Network frame
An example though the network forward with how did data changing.
![](%E6%96%B0%E5%BB%BA%20Markdown_md_files/7ce362a0-576a-11ed-b4f9-798c562720d8.jpeg?v=1&type=image)
## Requirement
python 3.9.2
torch == 1.11.0 + cu113
opencv-python == 4.5.3.56
pandas == 1.4.2
numpy == 1.21.5
matplotlib == 3.5.1
## Data
The data was created  through combining the nature background picture with the SUN-YAT-SEN logo respectively in the ```background``` file and the ```target``` file.
 The dataset file which includes the ```create_train.py``` and ```data_process.py``` module will be called though the ```data_process``` function to create the new file```sysu_train``` and train data in the ```main.py```.
## Usage
download the project and choose the network parameters name in the ```main.py```, then start code ```main.py```.
There are hyperparameters in the ```main.py```
|Parameter name |  Description of parameter  | 
|  ----  |  ----  | 
|sizes  |the parameters used in the TinySSD.blk_forward.multibox_prior|
|ratios   |the same as sizes|
|num_anchors|the produced anchors number|
|num_classes   |the classes want to class|
|batchsize   |the batchsize of the data|
|num_epochs  |times that you want to train|
|net_name  |the save model parameters by the train network that you want to test|
|is_data_create  |whethter data is created.if the ```sysu_train``` data file exists,set it to False|
|lr|learning rate in SGD|
|weight_decay|the weight decay in SGD|

## Results
![](%E6%96%B0%E5%BB%BA%20Markdown_md_files/82cb12d0-576a-11ed-b4f9-798c562720d8.jpeg?v=1&type=image)
## Improvement in code
student improved the file structure to the below picture.
![](%E6%96%B0%E5%BB%BA%20Markdown_md_files/7151a780-576a-11ed-b4f9-798c562720d8.jpeg?v=1&type=image)
student didn't improve the network performance.
