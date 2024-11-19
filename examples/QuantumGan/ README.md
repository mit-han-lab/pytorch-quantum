# Quantum Generative Adversarial Network (QGAN) Example

This repository contains an example implementation of a Quantum Generative Adversarial Network (QGAN) using PyTorch and TorchQuantum. The example is provided in a Jupyter Notebook for interactive exploration.

## Overview

A QGAN consists of two main components:

1. **Generator:** This network generates fake quantum data samples.
2. **Discriminator:** This network tries to distinguish between real and fake quantum data samples.

The goal is to train the generator to produce quantum data that is indistinguishable from real data, according to the discriminator. This is achieved through an adversarial training process, where the generator and discriminator are trained simultaneously in a competitive manner.

## Repository Contents

- `qgan_notebook.ipynb`: Jupyter Notebook demonstrating the QGAN implementation.
- `qgan_script.py`: Python script containing the QGAN model and a main function for initializing the model with command-line arguments.

## Installation

To run the examples, you need to have the following dependencies installed:

- Python 3
- PyTorch
- TorchQuantum
- Jupyter Notebook
- ipywidgets

You can install the required Python packages using pip:

```bash
pip install torch torchquantum jupyter ipywidgets
```


Running the Examples
Jupyter Notebook
Open the qgan_notebook.ipynb file in Jupyter Notebook.
Execute the notebook cells to see the QGAN model in action.
Python Script
You can also run the QGAN model using the Python script. The script uses argparse to handle command-line arguments.

bash
Copy code
python qgan_script.py <n_qubits> <latent_dim>
Replace <n_qubits> and <latent_dim> with the desired number of qubits and latent dimensions.

Notebook Details
The Jupyter Notebook is structured as follows:

Introduction: Provides an overview of the QGAN and its components.
Import Libraries: Imports the necessary libraries, including PyTorch and TorchQuantum.
Generator Class: Defines the quantum generator model.
Discriminator Class: Defines the quantum discriminator model.
QGAN Class: Combines the generator and discriminator into a single QGAN model.
Main Function: Initializes the QGAN model and prints its structure.
Interactive Model Creation: Uses ipywidgets to create an interactive interface for adjusting the number of qubits and latent dimensions.
Understanding QGANs
QGANs are a type of Generative Adversarial Network (GAN) that operate in the quantum domain. They leverage quantum circuits to generate and evaluate data samples. The adversarial training process involves two competing networks:

The Generator creates fake quantum data samples from a latent space.
The Discriminator attempts to distinguish these fake samples from real quantum data.
Through training, the generator improves its ability to create realistic quantum data, while the discriminator enhances its ability to identify fake data. This process results in a generator that can produce high-quality quantum data samples.


## QGAN Implementation for CIFAR-10 Dataset
This implementation trains a QGAN on the CIFAR-10 dataset to generate fake images. It follows a similar structure to the TorchQuantum QGAN, with the addition of data loading and processing specific to the CIFAR-10 dataset.
Generated images can be seen in the folder

This `README.md` file explains the purpose of the repository, the structure of the notebook, and how to run the examples, along with a brief overview of the QGAN concept for those unfamiliar with it.


## Reference
- [ ] https://arxiv.org/abs/2312.09939
