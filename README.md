# DARI: Code for the paper: Decoupling Forward and Feedback Flows: A Dual-Attention Framework for Relational Inference
This project focuses on inferring and reconstructing graphs from time-series data.
## Abstract
Inferring latent interaction structures from observational time series is a fundamental yet challenging problem in dynamical systems. Existing deep learning methods employ unidirectional information aggregation via incoming edges, failing to identify the reciprocal coupling mechanisms prevalent in real dynamics as well as the feedback effects induced by sampling intervals, which leads to inferential bias. To address this, we propose the Dual-Attention Relational Inference (DARI), a framework designed to learn latent interaction structures from dynamical observations. DARI employs a coupled bidirectional attention mechanism to model forward and feedback dynamics, effectively decoupling information flow from the physical structure.  Extensive synthetic experiments demonstrate consistent improvements in structural recovery across diverse graph topologies, including undirected, directed, and weighted graphs. Experiments on COVID-19 data further show that the inferred transmission structures are consistent with real-world population mobility patterns. In addition, the elimination of costly edge-wise computations in DARI leads to substantial gains in both runtime and memory efficiency. 

## Installation

1. Clone the repository and enter the project directory: 
   ```bash
   cd DARI
   ```

2. Install dependencies:
   Install the following packages manually:
   - numpy
   - pandas
   - torch
   - geopandas
   - matplotlib
   - networkx
   - scipy
   - requests

3. Ensure Python 3.7+ is installed.

## Data Generation and Training

The project supports two main data types: simulated data and real COVID-19 data.

### 1. Simulated Data

#### Data Generation
Use the scripts in the `data/` directory to generate various types of simulated data. Examples include:

- Linear dynamics: `python data/generate_linear.py`
- Kuramoto oscillators: `python data/generate_kuramoto.py` or `python data/generate_kuramoto_weight.py`
- SIS epidemic model: `python data/generate_SIS.py` or `python data/generate_metapop_SIS.py`
- Spring-mass systems: `python data/generate_spring.py` or `python data/generate_spring_weight.py`
- And others (FJ, MM, etc.)

Each script generates adjacency matrices and time-series trajectories for training.

#### Training
- For unweighted graphs: Run `python train.py`
- For weighted graphs: Run `python train_weight.py`

These scripts train models to infer graph structures from the generated time-series data.

### 2. COVID-19 Data

#### Data Generation
Use scripts in the `data/covid19/` directory to process real COVID-19 mobility data:

- Run `python data/covid19/data_hand.ipynb` (Jupyter notebook) to handle and preprocess COVID-19 network data.

The processed data includes mobility graphs (OD matrices) for states like Alaska, Illinois, Maine, Minnesota, Utah, and Washington.

#### Training
Run `python train_covid19_v1.py` to train the model on the COVID-19 dataset.

This trains a model to infer mobility networks from epidemic spread data.

## Model Architecture

The project uses transformer-based architectures for graph inference, implemented in PyTorch.

- `model/model_trans_v1.py`: Model for unweighted graphs
- `model/model_trans_weight_v1.py`: Model for weighted graphs
- `model/any_covid19_v1.py`: Specialized model for COVID data

## Results

Trained models can be evaluated and visualized using notebooks in the `notebooks/` directory, such as `plt_od.ipynb` for plotting inferred vs. ground-truth mobility graphs.

## Configuration

Configuration files are in the `configs/` directory (e.g., `config_covid19_v1.yaml`).

## Citation

If you use this code, please cite the appropriate paper or repository.

## License

See LICENSE file.
