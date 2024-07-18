# TP2DP2: A Bayesian Mixture Model of Temporal Point Processes with Determinantal Point Process Prior

This repository contains the source code for TP2DP2, a Bayesian mixture model of temporal point processes with a Determinantal Point Process (DPP) prior. The main script for running the neural network-based point process with DPP prior experiments is `RMTPP_dpp_main.py`.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Repository Structure

TP2DP2/<br>
├── hkstools/<br>
│ └── hawkes_utils.py # Utility functions for Hawkes process<br>
├── pp_mix/<br>
│ ├── mixture_model.py # Implementation of mixture models<br>
│ ├── dpp_prior.py # Implementation of DPP prior<br>
│ └── utils.py # Utility functions for point process mixture models<br>
├── HawkesModel.py # Hawkes process model implementation<br>
├── README.md # This README file<br>
├── RMTPP.py # neural network-based point process model implementation<br>
├── RMTPP_dpp.py # neural network-based point process model with DPP prior implementation<br>
├── RMTPP_dpp_main.py # Main script for running neural network-based point process with DPP experiments<br>
├── RMTPP_dpp_train.py # Training script for neural network-based point process with DPP<br>
├── init.py # Package initialization<br>
├── cluster_loader.py # Data loading and preprocessing for clustering<br>
├── data_K2.pkl # Example data file<br>
├── metric.py # Evaluation metrics<br>
└── rmtpp_embed.py # Embedding utilities for RMTPP

## Setup

To set up the environment for running the experiments, please follow the steps below:

1. Clone the repository:
    `
    git clone https://github.com/EthanYDONG/TP2DP2.git`
    `cd TP2DP2`

2. Install the required packages:
  `
    pip install -r requirements.txt
  `

## Usage

To run the main experiments with neural network-based point process and DPP on the provided dataset, use the following command:
`
python RMTPP_dpp_main.py --dataset <dataset_name>
`
Replace `<dataset_name>` with the name of the dataset you wish to use.

## File Descriptions

### Main Scripts

- **RMTPP_dpp_main.py**: The main script for running experiments with neural network-based point process and DPP. This script initializes the model, loads the data, and evaluates the performance.
- **RMTPP_dpp_train.py**: This script handles the training of the neural network-based point process model with DPP.

### Models

- **RMTPP.py**: Implementation of the Recurrent Marked Temporal Point Process (RMTPP) model.
- **RMTPP_dpp.py**: Extension of the neural network-based point process model with a Determinantal Point Process (DPP) prior.
- **HawkesModel.py**: Implementation of the Hawkes process model.

### Utilities

- **hkstools/hawkes_utils.py**: Contains utility functions for the Hawkes process.
- **pp_mix/mixture_model.py**: Contains the implementation of mixture models.
- **pp_mix/dpp_prior.py**: Contains the implementation of the DPP prior.
- **pp_mix/utils.py**: Contains utility functions for point process mixture models.
- **cluster_loader.py**: Contains functions for loading and preprocessing data for clustering.
- **metric.py**: Defines evaluation metrics for model performance.
- **rmtpp_embed.py**: Provides embedding utilities for the neural network-based point process model.

### Data

- **data_K2.pkl**: An example dataset used for running experiments.

## Results

The results of the experiments will be stored in the `results/` directory. This directory will contain logs, model checkpoints, and evaluation metrics.

## License

This project is licensed under the MIT License.
