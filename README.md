# Official Repository RVSS 2024

![logo](Pics/RVSS-logo-col.med.jpg)

The material here provided was developed as part of the [Robotic Vision Summer School](https://www.rvss.org.au/).

---
# Workshops
Please see this [separate README](https://github.com/rvss-australia/RVSS_Need4Speed).

---
# Introduction
by Peter Corke, Queensland University of Technology

#### Slides:
Coming soon

---
# Lectorials
You can run the notebooks via Colab: http://colab.research.google.com/github/rvss-australia/RVSS.

Links to slide decks and information about the lectorials will appear below.

---
## Lectorial A: Robotic Vision
Presented by Peter Corke, Queensland University of Technology

### A1: Image Processing Fundamentals
<details>
<summary>Pixels, images, image processing, feature extraction</summary>

#### Slides:
[A1 Slides](Robotic_Vision/Slides/A1-Image-processing.pdf)

#### Coding Session:
* [Finding blobs](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/finding-blobs.ipynb)
* [Image features](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/image_features.ipynb)

#### Supporting Resources:
The following chapters from the [Robotics, Vision and Control](https://link.springer.com/book/10.1007%2F978-3-319-54413-7) Textbook support this session (likely available from your University library as an e-book or individual chapter download):
* Chapters 12.1, 12.2, 12.3, 12.4 and 12.5
* Chapter 13.1
  * Those who are a bit rusty on homogenous transformation matrices, rotation matrices and similar concepts may find Chapters 2.1 and 2.2 - 2D and 3D Geometry useful.

The following masterclasses from the [QUT Robot Academy](https://robotacademy.net.au/) may also be used to help develop your knowledge:
* [Introduction to Robotic Vision](https://robotacademy.net.au/masterclass/robotic-vision/)
* [2D Geometry](https://robotacademy.net.au/masterclass/2d-geometry/) and [3D Geometry](https://robotacademy.net.au/masterclass/3d-geometry/)
* [Getting Images into a Computer](https://robotacademy.net.au/masterclass/getting-images-into-a-computer/)
* [Image Processing](https://robotacademy.net.au/masterclass/image-processing/)
* [Spatial Operators](https://robotacademy.net.au/masterclass/spatial-operators/)
* [Feature Extraction](https://robotacademy.net.au/masterclass/feature-extraction/)

</details>

### A2: Image Formation
<details>
<summary>From light to digital images, the front end of the robotic vision process</summary>

#### Slides:
[A2 Slides](Robotic_Vision/Slides/A2-Image-geometry.pdf)

#### Coding Session:
* [Camera projection basics](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/camera_animation.ipynb)
* [Camera modeling](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/camera.ipynb)
* [Camera calibration](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/calibration.ipynb)
* [Fiducial makers (AprilTags and ArUco markers)](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/fiducuals.ipynb)

#### Supporting Resources:
The following chapters from the [Robotics, Vision and Control](https://link.springer.com/book/10.1007%2F978-3-319-54413-7) Textbook support this session:
* Chapters 11.1 and 11.2

The following masterclasses from the [QUT Robot Academy](https://robotacademy.net.au/) may also be used to help develop your knowledge:
* [How Are Images Formed](https://robotacademy.net.au/masterclass/how-images-are-formed/)
* [The Geometry of Image Formation](https://robotacademy.net.au/masterclass/the-geometry-of-image-formation/) 
  * You may find watching [3D Geometry](https://robotacademy.net.au/masterclass/3d-geometry/) prior to these two will be beneficial 

 </details>
 
### A3: Visual Control
<details>
<summary>Closing the loop from sensing to action</summary>

#### Slides:
[A3 Slides](Robotic_Vision/Slides/A3-vision-and-motion.pdf)

#### Coding Session:
* [Image motion](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Robotic_Vision/ImageMotion.ipynb)
* [Image-based visual servoing (IBVS)](https://githubtocolab.com/rvss-australia/RVSS/blob/main/Robotic_Vision/IBVS.ipynb)

#### Supporting Resources:
The following chapters from the [Robotics, Vision and Control](https://link.springer.com/book/10.1007%2F978-3-319-54413-7) Textbook support this session:
* Chapters 15.2

The following masterclasses from the [QUT Robot Academy](https://robotacademy.net.au/) may also be used to help develop your knowledge:
* [Vision and Motion](https://robotacademy.net.au/masterclass/vision-and-motion/)

</details>
 
---
## Lectorial B: Spatial Awareness
Presented by Teresa Vidal-Calleja, University of Technology Sydney
 
### B1: Poses
<details>
<summary>Describing poses, transformations, and uncertainty</summary>

#### Slides:
[B1 Slides](coming soon)

#### Coding Sessions:
* [Basic Geometry](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Spatial_Awareness/Tutorial_B1_Basic_Geometry/Basic%20Geometry.ipynb)
* [Uncertainty](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Spatial_Awareness/Tutorial_B1_Basic Geometry/2_Uncertainty.ipynb)

</details>
 
### B2: Uncertainty
<details>
<summary>Keeping track of stuff with imperfect measurement; Kalman filters, factor graphs, and batch optimisation</summary>

#### Slides:
[B2 Slides](coming soon)

#### Coding Sessions:
* [Motion Model](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Spatial_Awareness/Tutorial_B2_Robot_Localisation/1_MotionModel.ipynb)
* [Kalman Filter 1D](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Spatial_Awareness/Tutorial_B2_Robot_Localisation/3_KalmanFilter1D.ipynb)
* [Multivariate Gaussian](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Spatial_Awareness/Tutorial_B2_Robot_Localisation/4_MultiVariateGaussian.ipynb)
* [EKF](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Spatial_Awareness/Tutorial_B2_Robot_Localisation/5_EKF.ipynb)
* [SLAM](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Spatial_Awareness/Tutorial_B2_Robot_Localisation/6_SLAM.ipynb)

</details>

### B3: Representations
<details>
<summary>A taxonomy of spatial representations</summary>

#### Slides:
[B3 Slides](coming soon)

</details>

---
## Lectorial C: Visual Learning
Presented by Simon Lucey, University of Adelaide

### C1: Convolutional Neural Networks
<details>
<summary>Introduction to convolutional neural networks</summary>

#### Slides:
[C1 Slides](coming soon)

#### Coding Session:
* [Image classification with multi-layer perceptron](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Visual_Learning/Session1/Classification_MLP_2021.ipynb#scrollTo=lvPV3WzCC6WL)

</details>

### C2: Visual Transformers
<details>
<summary>Visual transformers and their connection to convolutional neural networks</summary>

#### Slides:
[C2](coming soon)

#### Coding Session:
* [Image classification with Convolutional NN](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Visual_Learning/Session2/LeNetClassificationExcercise_2021.ipynb)

</details>

### C3: Point Clouds
<details>
<summary>Point cloud processing with transformers</summary>

#### Slides:
[C3 Slides](coming soon)

</details>


---
## Lectorial D: Learning to Act
Presented by Dana Kulic, Monash University

### D1: Learning from Demonstrations
<details>
<summary>Action selection as supervised learning; behaviour cloning</summary>

#### Slides:
* [D1 Slides](coming soon)

#### Coding Sessions:
* [Introduction to Reinforcement Learning](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/Session%201%20IntroRL.ipynb)

</details>

### D2: Learning from Experience
<details>
<summary>Introduction to reinforcement learning</summary>

#### Slides:
* [D2 Slides](coming soon)

#### Coding Sessions:
* [Introduction to Model-Free Reinforcement Learning](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/Session%202.1%20ModelFreeRL.ipynb)
* [Deep RL - Replay Memory](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/Session%202.2%20-%20DeepRL_ReplayMemory.ipynb)
* [Deep RL - Target Network](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/Session%202.3%20-%20DeepRL_DQNTarget.ipynb)

</details>

### D3: Inverse Reinforcement Learning
<details>
<summary>Introduction to inverse reinforcement learning</summary>

#### Slides:
[D3 Slides](coming soon)

</details>

### Additional Resources
<details>
<summary>If you'd like to know more</summary>

* David Silver's RL [Video Lectures](https://www.davidsilver.uk/teaching/) at UCL 
* Prof. Pascal Poupart's [Video Lectures](https://www.youtube.com/watch?v=KOF_BM-fNPE&t=4s&ab_channel=PascalPoupart) at University of Waterloo, Canada
* Sutton and Barton's [Introduction to Reinforcement Learning](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) book
* Sergey Levine's [Video Lectures](http://rail.eecs.berkeley.edu/deeprlcourse/) on deep reinforcement learning at UCBerkeley

</details>

---
## Deep Dives

### Stephen Gould, The Australian National University
* [Slides](DeepDives/StevenGould/slides.pdf)
* [Isaac Lecture Notes](DeepDives/StevenGould/isaac22-lecture-notes.pdf)

### Richard Hartley, The Australian National University
* [Slides](DeepDives/RichardHartley/Kioloa-2024-small.pdf)
