# Robotic Vision (A) stream

This set of three lectures will introduce you to important foundational concepts in computer vision.  This are "classical" topics
but which we believe are important to understand, even in the modern deep-learning era.

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

The books are very similar in chapter structure and content, the first is based on Python code and open-source packages, the second is based on MATLAB and propietrary toolboxes that you need to licence from MathWorks (most universities will give you the required licences).  It's just a matter of personal preference.

If you are studying at a university it is highly likely that you can download - **for free** - the chapters of this book from the links above.
For this course, just grab the indicated chapters (11, 12, 13, 15) from the latter part of the book

<img src="readme-pix/download1.png" alt="download vision chapters" width="400" />
<img src="readme-pix/download2.png" alt="download visual motion chapters" width="400" />


Feel free to grab any other chapters that might take your fancy.  Chapter 2 is a good (I think) introduction to representing position
and orientation in 3D space, Appendix B is a convise refresher on linear algebra and geometry.


## Videos

There are a set of free online video resources (the QUT Robot Academy) that might be useful as a refresher.  

* [Homogeneous coordinates](https://robotacademy.net.au/lesson/homogeneous-coordinates-recap/) (5 mins)
* [Position, orientation & pose in 3D space](https://robotacademy.net.au/masterclass/3d-geometry/) (multiple lessons, 60 mins total)

Code examples in these videos are done with MATLAB, but underneath each video is a code tab, and below that is a tab that allows you to select a 
"translation" of the code used in the video to different languages and toolboxes.

<img src="readme-pix/academy.png" alt="language options for Robot Academy videos" width="400" />

I will mention other, lecture-specific, Robot Academy videos below.


## Jupyter Notebooks

I provide a selection of Jupyter/Python notebooks that will help to embed knowledge from each lecture.  You can run them on Google Colab, with zero install, by
clicking the <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> button below.  

Alternatively, you can run them locally on your laptop, and that requires that you first install the [Machine Vision Toolbox for Python](https://github.com/petercorke/machinevision-toolbox-python)
```
pip install machinevisiontoolbox
```
Python 3.9 or newer is recommended.  This will install all the required dependencies (including OpenCV) as well as example images for the exercises.

I would highly recommend that you use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest) and create an environment for your RVSS code.
```
conda create -n RVSS python=3.10
conda activate RVSS
pip install machinevisiontoolbox
```

# Lecture resources

## Lecture A1 Image Processing

This lecture introduces the fundamentals of image processing.  Topics include pixels and images, image arithmetic, spatial operations such as convolution, and operations on images to find motion, simple blob objects, and image features.

* <a href="Slides/A1-Image-processing.pdf" target="_blank">Lecture PDF file</a>

* Robotics, Vision & Control: Chapters 11 and 12

* Robot Academy video masterclasses (each is a collection of short videos, ~1h total run time)
  * [Getting images into a computer](https://robotacademy.net.au/masterclass/getting-images-into-a-computer/)
  * [Image processing](https://robotacademy.net.au/masterclass/image-processing/)
  * [Spatial operators](https://robotacademy.net.au/masterclass/spatial-operators/)
  * [Feature extraction](https://robotacademy.net.au/masterclass/feature-extraction/)

* Jupyter/Python Notebooks

  * [`image-processing.ipynb`](image-processing.ipynb), fundamentals of image processing as discussed in the lecture <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/image-processing.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  * [`image-features.ipynb`](image-features.ipynb), fundamentals of corner features as discussed in the lecture <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/image-features.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  * [`finding-blobs.ipynb`](finding-blobs.ipynb), extension to blob finding and blob parameters <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/finding-blobs.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  * [`fiducials.ipynb`](fiducials.ipynb), extension to finding ArUco markers (QR-like codes) in an image <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/fiducials.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Lecture A2  Camera imaging and geometry

This lecture introduces the process of image formation, how the 3D world is projected into a 2D image. Topics include central projection model, homographies, and camera
calibration.

* <a href="Slides/A2-Image-geometry.pdf" target="_blank">Lecture PDF file</a>

* Robotics, Vision & Control: Section 13.1

* Robot Academy video masterclasses (each is a collection of short videos, ~1h total run time)

  * [How images are formed](https://robotacademy.net.au/masterclass/how-images-are-formed/)
  * [The geometry of image formation](https://robotacademy.net.au/masterclass/the-geometry-of-image-formation/)

* Jupyter/Python Notebooks

  * [`camera_animation.ipynb`](camera_animation.ipynb), interactive animation of point projection for central projection model <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/camera_animation.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  * [`camera.ipynb`](camera.ipynb), introducing the Toolbox `CentralCamera` object <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/camera.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
   * [`homogeneous-coords.ipynb`](homogeneous-coords.ipynb), refresher on homogeneous coordinates including an interactive animation<a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/homogeneous-coords.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  * [`calibration2d.ipynb`](calibration2d.ipynb), extension: calibrating a camera using a set of chequerboard images  <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/calibration2d.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  * [`homography.ipynb`](homographies.ipynb),  extension: computing an homography <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/homography.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Lecture A3

This lecture is concerned with the relationship between motion of a camera and the corresponding changes in the image.  This can provide information about
the 3D nature of the world, and can also be inverted to determine how a camera should move in order to create the desired change in an image.

* <a href="Slides/A3-vision-and-motion.pdf" target="_blank">Lecture PDF file</a>

* Robotics, Vision & Control: Chapter 15

* Robot Academy video masterclasses (each is a collection of short videos, ~1h total run time)

  * [Vision and motion](https://robotacademy.net.au/masterclass/vision-and-motion/)

* Jupyter/Python Notebooks
  * [`image-motion.ipynb`](homographies.ipynb),  extension, finding the pose and identity of ArUco (QR codes) in an image <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/image-motion.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  * [`fiducials.ipynb`](homographies.ipynb),  extension, finding the pose and identity of ArUco (QR codes) in an image <a href="https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/homographies.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>