# Workshop setup - before you arrive
In this workshop, we will use Conda. Conda is a package manager for Windows, Mac and Linux - it allows you to install packages similar to `apt`, `homebrew` and `vcpkg`. Conda not only supports Python packages, but also packages in C/C++, FORTRAN, and much more.

Nothing that can be done with conda cannot be achieved otherwise. However, with conda it is usually easier and cross-platform.

Conda handles dependencies seamlessly and makes it easy to set up different environments with different versions of libraries.

Conda is also an environment manager (like `virtualenv`). Therefore, if an environment is ruined beyond repair, you can just remove it and start over with a clean one.

# Guide for setting up Conda/Mamba with conda-forge
- We strongly recommend using the community-driven `conda-forge` channel, instead of the `defaults` channel that is maintained by Anaconda. 
- We also strongly recommend the fast `mamba` drop-in replacement for `conda`.
- The easiest way to install mamba with conda-forge as default channel is with `mambaforge`. We have the executables ready at this [link](https://cloudstor.aarnet.edu.au/plus/s/lR0gyZzyf5bnAMT) or you can download from [here](https://github.com/conda-forge/miniforge#mambaforge).
- If you are on Linux/MacOS, simply type `sh Mambaforge-*.sh` and follow the instructions. On Windows, double click `Mambaforge-Windows-x86_64.exe` and follow the instructions.

# Guide for setting up an RVSS conda environment
- To create an environment with all required packages, we recommend to download the pre-created environment files from [here](https://cloudstor.aarnet.edu.au/plus/s/lR0gyZzyf5bnAMT). You can then extract them to your `~/mambaforge/envs` folder.
- Alternatively, create a new environment with all required packages: `mamba create -n rvss numpy scipy pytorch scikit-learn ipython scikit-image matplotlib tqdm roboticstoolbox-python git ipykernel mediapy py-opencv seaborn gym jupyter spatialmath-python machinevision-toolbox-python ipywidgets plotly torchvision conda-pack tensorboardx`.

## Working with the environment:
- Activating the environment you just created: `conda activate rvss2019`
- Deactivating: `conda deactivate rvss2019`
- Deleting an environment: `conda remove --name FAILED_ENVIRONMENT --all`

# Workshop intro
The workshop for this year aims at giving you the opportunity to deploy a deep neural network on a mobile robot. You will go through all the stages of this process: data collection and training data labelling, network architecture design and implementation, training and testing, deployment on the robot and task execution.

The task you are going to implement is an autonomous steering of a robot to follow a track. The input to the system is an RGB image from a camera onboard the robot and the output is a steering command to keep the robot on track.

The workshop task can be broken down into a number of sub-tasks, as shown below.
- Using your laptop to control the robot
- Collecting data with the robot
- Building a network
- Training your network
- Testing and refining your network

