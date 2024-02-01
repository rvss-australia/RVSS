# Robotic Vision (A) stream

There are four sets of learning resources for each topic that I cover, described in a bit more detail below.

I teach general principles but to put the ideas into practice we need to write code.  There are myriad choices of language
and library/package/toolbox to choose from.  In the past I've done a lot in MATLAB but now I'm working with Python, and Python is
what we will use for the summer school.

## Lectures

The PDFs of my lecture slides are provided in advance.  Feel free to load them into your tablet to annotate as we go along.


## Book

The material that I present is covered in more detail in my book Robotic, Vision & Control, 3rd edition 2023.  There are two versions of this book:

* [Robotic, Vision & Control: Fundamental algorithms in **Python**](https://link.springer.com/book/10.1007/978-3-031-06469-2)
* [Robotic, Vision & Control: Fundamental algorithms in **MATLAB**](https://link.springer.com/book/10.1007/978-3-031-07262-8)

If you are studying at a university it is highly likely that you can download for free the chapters of this book from the links above.
For this course grab the chapters from the latter part of the book

<img src="readme-pix/download1.png" alt="download vision chapters" width="400" />
<img src="readme-pix/download2.png" alt="download visual motion chapters" width="400" />


Feel free to grab any other chapters that might take your fancy.  Chapter 2 is a good introduction to representing position
and orientation in 3D space, Appendix B is a terse refresher on linear algebra and geometry.


## Videos

There are a set of free online video resources (the QUT Robot Academy) that might be useful as a refresher.  

* [Homogeneous coordinates](https://robotacademy.net.au/lesson/homogeneous-coordinates-recap/) (5 mins)
* [Position, orientation & pose in 3D space](https://robotacademy.net.au/masterclass/3d-geometry/) (multiple lessons, 60 mins total)

Code examples in these videos are done with MATLAB, but underneath each video is a code tab, and below that is a tab that allows you to select a 
"translation" of the code used in the video to different languages and toolboxes.

<img src="readme-pix/academy.png" alt="language options for Robot Academy videos" width="400" />

I will mention other, lecture-specific, Robot Academy videos below.


## Jupyter Notebooks

There a selection of Jupyter/Python notebooks that will help to embed knowledge from each lecture.  You can run them on Google Colab, with zero install, by
clicking the <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> button below.  

Alternatively, you can run them locally on your laptop, and that requires that you first install the [Machine Vision Toolbox for Python](https://github.com/petercorke/machinevision-toolbox-python)
```
pip install machinevisiontoolbox
```
Python 3.9 or newer is recommended.

I would highly recommend that you use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest) and create an environment for your RVSS code.
```
conda create -n RVSS python=3.10
conda activate RVSS
pip install machinevisiontoolbox
```

# Lecture resources

## Lecture A1 Image Processing

* [Lecture PDF file](Slides/A1-Image-processing.pdf)

* Robot Academy video masterclasses
  * [Getting images into a computer](https://robotacademy.net.au/masterclass/getting-images-into-a-computer/)
  * [Image processing](https://robotacademy.net.au/masterclass/image-processing/)
  * [Spatial operators](https://robotacademy.net.au/masterclass/spatial-operators/)
  * [Feature extraction](https://robotacademy.net.au/masterclass/feature-extraction/)

* Notebooks

  * `image-features.ipynb`, fundamentals of image processing as discussed in the lecture <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/image-processing.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  * `finding-blobs.ipynb`, extension to blob finding and blob parameters <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/finding-blobs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  * `fiducials.ipynb`, extension to finding ArUco markers (QR-like codes) in an image <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/image-features.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Lecture A2  Camera imaging and geometry

Lecture PDF file

* [`camera_animation.ipynb`](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/camera_animation.ipynb)
* [`camera.ipynb`](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/camera.ipynb
* [`calibration2d.ipynb`](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/calibration2d.ipynb)
* [`fiducials.ipynb`](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/fiducials.ipynb)

## Lecture A3

Lecture PDF file