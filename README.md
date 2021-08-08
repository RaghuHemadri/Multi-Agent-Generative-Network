# Multi-Agent-Generative-Network (MAGNET)
MAGNET is a multi agnet cooperative learning based architecture for generating new images. MAGNET generate images similar to images in [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.

## Theory
The algorithm of multi-agent cooperative learning is inspired from [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275). 

### Environment
* **Current State:**
A random set of images (episode) sampled from the dataset.
* **New State:**
A random set of images (episode), different from current state, sampled from the dataset.

### Networks
MAGNET consists of 5 networks in total, namely:
* **Generator Actor (G<sub>a</sub>):**
Takes current state (a image) as input, has a series of CNNs and LSTM at the end. It outputs a hidden state vector for next state (z).
* **Generator (G):**
This network is a series of DeConvolution layers, given a hidden state it outputs a image.
* **Generator Critic (G<sub>c</sub>):**
Given the hidden state vector and current state, it outputs the q-value.
* **Discriminator Actor (D<sub>a</sub>):**
Given a image, this network predicts if the image is real (x) or fake (G(x)).
* **Discriminator Critic (D<sub>c</sub>):**
Give the prediction of Discriminator Actor (policy) and the current state (Discriminator Actor input), it outputs the q-value.

### Returns
* **Generator Return:**
log(D<sub>a</sub>(G(z))
* **Discriminator Return:**
log(D<sub>a</sub>(x)) + log(1 – D<sub>a</sub>(G(z)))

### Algorithms
* The Generator Actor-Critic networks use DDPG as the hidden state is continuous.
* The Discriminator Actor-Critic networks use Actor-Critic as the output of D<sub>a</sub> is either '0' or '1'.
* **Multi-Agent DDPG for N-Agents:**
![Multi-Agent DDPG]('MADDPG.png')
