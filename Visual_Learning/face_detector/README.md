# RVSS 2021 Workshop
The repository contrains the CNN modeule for the fruit detection task.
This is a light-weight fully convolutional network that infers at approximatly 3 fps on a CPU. 

Current input resolution is set to 256x192, the segmentation mask is a quater of the input, i.e. 128x96
## Before you start
```
$ cd rvss2021_detector
$ conda env create -f conda_env_win.yml (For Windows users)
$ conda env create -f conda_env_mac.yml (For Mac users)
$ conda env create -f conda_env_linux.yml (For Linux users)
$ conda activate rvss21
```

## Test the network
change directory to 'scripts', 
```
cd scripts/
jupyter notebook
```
open and run detector_test.ipynb from the browser

## Train the network
1. [Download the sample dataset here](https://anu365-my.sharepoint.com/:u:/g/personal/u5240496_anu_edu_au/Ec3PqU60nk5Amcfznr25XpMBthefcgvu6cqG340p8cDYFQ?e=HKRQ53). Extract the file.

2. Run:
    ```
    $ cd <path_to_folder>/scripts
    $ python main.py --dataset_dir <full_path_to_dataset> --output_folder <folder to save the weights>
    ```
3. Run `python main.py -h` to get a list of configuratble parameters, including network hyper-parameters


## Network Architecture:
![Network Architecture](readme_pics/rvss_arch.png)
*Illustration of the network architecture, 
notations (c1, c2, ect. corresponds to the variable name in res18_skip.py script)*

The network has a auto-encoder, decoder structure.
ResNet18 with pre-trained weights is used as the auto-encoder. 
The decoder is inspired by the "lateral connection" proposed by Lin et al. in _Feature Pyramid Network for Object Detection, 2016_. 

 

ResNet Architecture             |  Lateral Connection
:-------------------------:|:-------------------------:
<img src="readme_pics/resnet.png" width="500">  |  <img src="readme_pics/skip_connection.png" width="500">

