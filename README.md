# PICL: Learning to Incorporate Physical Information when Facing Coarse-Grained Data

## Official Implementation

This repository is the official implementation of "PICL: Learning to Incorporate Physical Information when Facing Coarse-Grained Data". 

## Introduction

We introduce the Physics-Informed Coarse-grained data Learning (PICL) framework. The key innovation of PICL is the reconstruction of a learnable fine-grained state using only physical information. This approach enables the applicability of physics-based loss and significantly enhances the model's generalization capacity for predicting future coarse-grained observations.

PICL combines an encoding module for reconstructing learnable fine-grained states with a transition module for predicting future states. This unique approach seamlessly blends data-driven and physics-informed techniques, especially useful when only limited coarse-grained data is available.

## Key Features

- **Physics-Informed Learning**: Leveraging physical information to enhance learning from coarse-grained data.
- **Improved Generalization**: Significantly better generalization capacity in both single-step and multi-step predictions.
- **Applicability in Various Systems**: Tested and proven in various physical systems.

## Experiment

We applied PICL to the Nonlinear Shallow Water Equation (NSWE) and observed notable improvements.

## Usage

To run the experiment:

```bash
./run.sh
