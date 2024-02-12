# Official Repository RVSS 2024

![logo](Pics/RVSS-logo-col.med.jpg)

The material here provided was developed as part of the [Robotic Vision Summer School](https://www.rvss.org.au/).

---
# Workshops
Please see this [separate README](https://github.com/rvss-australia/RVSS_Need4Speed).

---
# Lectorials
You can run the notebooks via Colab: http://colab.research.google.com/github/rvss-australia/RVSS.

Links to slide decks and information about the lectorials will appear below.

---
## Lectorial A: Robotic Vision
Presented by Peter Corke, Queensland University of Technology

The slides, details and resources for the A stream are here: 
[A Slides](Robotic_Vision/README.md)

 
---
## Lectorial B: Spatial Awareness
Presented by Teresa Vidal-Calleja, University of Technology Sydney
 
### B1: Poses
<details>
<summary>Describing poses, transformations, and uncertainty</summary>

#### Slides:
[B1 Slides](Spatial_Awareness/Slides/RVSS2024-B1-TVC.pdf)

#### Coding Sessions:
* [Basic Geometry](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Spatial_Awareness/Tutorial_B1_Basic_Geometry/Basic%20Geometry.ipynb)
* [Uncertainty](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Spatial_Awareness/Tutorial_B1_Basic_Geometry/2_Uncertainty.ipynb)

</details>
 
### B2: Uncertainty
<details>
<summary>Keeping track of stuff with imperfect measurement; Kalman filters, factor graphs, and batch optimisation</summary>

#### Slides:
[B2 Slides](Spatial_Awareness/Slides/RVSS2024-B2-TVC.pdf)

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

### C1: Shallow and Deep Networks
<details>
<summary>Introduction to visual learning with shallow and deep networks </summary>
 
#### Slides:
[C1 Slides](Visual_Learning/Slides/RVSS-C1-Lucey.pdf)

#### Coding Session:
* [Image classification with multi-layer perceptron](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Visual_Learning/Session1/Classification_MLP_2021.ipynb#scrollTo=lvPV3WzCC6WL)

</details>

### C2: Convolutional Neural Networks
<details>
<summary>Introduction to convolutional neural networks</summary>

#### Slides:
[C2 Slides](Visual_Learning/Slides/RVSS-C2-Lucey.pdf)

#### Coding Session:
* [Image classification with Convolutional NN](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Visual_Learning/Session2/LeNetClassificationExcercise_2021.ipynb)

</details>

### C3: Visual Transformers
<details>
<summary>Visual transformers and their connection to convolutional neural networks</summary>


#### Slides:
[C3 Slides](Visual_Learning/Slides/RVSS-C3-Lucey.pdf)

</details>


---
## Lectorial D: Learning to Act
Presented by Dana Kulic, Monash University

### D1: Learning from Demonstrations
<details>
<summary>Action selection as supervised learning; behaviour cloning</summary>

#### Slides:
* [D1 Slides](Reinforcement_Learning/Slides/Session1.pdf)

#### Coding Sessions:
* [Introduction to Reinforcement Learning](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/LearningToAct.ipynb)

</details>

### D2: Learning from Experience
<details>
<summary>Introduction to reinforcement learning Part 1</summary>

#### Slides:
* [D2 Slides](Reinforcement_Learning/Slides/Session2.pdf)

#### Coding Sessions:
* [Introduction to Model-Free Reinforcement Learning](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/LearningToAct-Session2.ipynb)
<!-- * [Deep RL - Replay Memory](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/Session%202.2%20-%20DeepRL_ReplayMemory.ipynb)
* [Deep RL - Target Network](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/Session%202.3%20-%20DeepRL_DQNTarget.ipynb) -->

</details>

### D3: Learning from Experience
<details>
<summary>Introduction to reinforcement learning part 2</summary>

#### Slides:
[D3 Slides](coming soon)

#### Coding Sessions:
* [Introduction to Model-Free Reinforcement Learning](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/LTASession3-Part1.ipynb)
* [Basic Deep-Q Learning](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/Session-3.2-DeepRL_BasicDQN.ipynb)
* [Target Deep-Q Learning](https://colab.research.google.com/github/rvss-australia/RVSS/blob/main/Reinforcement_Learning/Session-3.3-DeepRL_TargetDQN.ipynb)


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

### Jen Jen Chung, The University of Queensland
* [Slides](DeepDives/JenJenChung/RVSS_JJC.pdf)

### Hanna Kurniawati, The Australian National University
* [Slides](DeepDives/HannaKurniawati/RVSS24DeepDives-hannaKurniawati.pdf)

### Donald Dansereau, The University of Sydney
* [Slides](DeepDives/DonaldDansereau/RVSS2024_Light.pdf)
