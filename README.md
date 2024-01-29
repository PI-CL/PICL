<h1 style="color: blue;"> PICL: Learning to Incorporate Physical Information when Facing Coarse-Grained Data

## Official Implementation

This repository is the official implementation of "PICL: Learning to Incorporate Physical Information when Facing Coarse-Grained Data". 

## Introduction

We introduce the Physics-Informed Coarse-grained data Learning (PICL) framework. The key innovation of PICL is the reconstruction of a learnable fine-grained state using only physical information. This approach enables the applicability of physics-based loss and significantly enhances the model's generalization capacity for predicting future coarse-grained observations. PICL combines an encoding module for reconstructing learnable fine-grained states with a transition module for predicting future states. This unique approach seamlessly blends data-driven and physics-informed techniques, especially useful when only limited coarse-grained data is available.

![\textbf{PICL.} Base-training period (left): the encoding module is trained with a physics loss without available fine-grained data, and the transition module is trained with a combination of data loss and physics loss. Inference Period (right): given a coarse-grained observation to predict the future coarse-grained observations.](pipline.jpg)

## Experiment

We applied PICL to the Burgers equation (Burgers Eqn.), wave equation (Wave Eqn), Navier-Stokes equation (NSE), linear shallow water equation (LSWE), and nonlinear shallow water equation (NSWE).

## Data Preparation

Before running the experiment:
1. Create a `data` folder in the root directory of this project.
2. Place the dataset files into the `data` folder.

## Usage

To run the experiment:

```bash
./run.sh
