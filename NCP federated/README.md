# Neural Circuit Policy & Federated Learning

This repository contains the code used to train and evaluate a Neural Circuit Policy (NCP) and a Long Short-Term Memory model, both in a Federated Learning framework and in a traditional (centralized) one. The training is done on energy consumption data from four compressors of a tannery in Santa Croce sull'Arno, Tuscany, Italy and each each machine has a stand-alone dataset. In the Federated Learning framework, each client uniquely corresponds to a compressor.

---

## Requirements

- Python ≥ 3.10 and < 3.12 
- Virtual Python environment
- Python package manager (pip)

## Files

    centralized_vs_fed.ipynb
    client.py
    detection.ipynb
    fed_train.py
    lstm_vs_ncp.ipynb
    models.py
    requirements.txt
    server.py
    utils.py
    weights_lstm.npz
    weights_ncp.npz

- [centralized_vs_fed.ipynb](./centralized_vs_fed.ipynb): notebook to compare the performance of a given model in both traditional and federated frameworks
- [client.py](./client.py): instantiates a Flower client
- [detection.ipynb](./detection.ipynb): loads data from a chosen machine and detects anomalies using a given model
- [fed_train.py](./requirements.txt): main script for training
- [lstm_vs_ncp.ipynb](./lstm_vs_ncp.ipynb): comparison of LSTM and NCP in the federated framework (only time-series prediction)
- [models.py](./models.py): script for models and hyperparameters
- [requirements.txt](./requirements.txt): required dependencies
- [server.py](./server.py): script for Flower server
- [utils.py](./utils.py): contains utility functions for data loading, scaling, and splitting
- [weights_lstm.npz](./weights_lstm.npz) and [weights_ncp.npz](./weights_ncp.npz): saved weights from FL training


## Installation 

1. Clone the repository to your local machine
    ```bash
    git clone https://github.com/unisi-lab305/energy-anomaly-detection/tree/main/ncpfl
    cd ncpfl
    ```
2. Create and activate a Python virtual environment:
    - Using `venv`:
        ```bash
        python3 -m ncp venv
        source venv/bin/activate
        ```
    - Using `conda`:
        ```bash
        conda create --name ncp python=3.10.10
        conda activate ncp
        ```
3. Navigate to the project directory and install the required Python packages using ```pip``` or ```conda```:
    ```bash
    pip install -r requirements.txt
    ```
---

## Training

1. Make sure data is stored in a ```datasets``` folder placed in the project directory and each dataset is named ```Compressore n.xlsx```, $n = 1,2,\dots$.
2. Set the desired hyperparameters in [models](./models.py) and choose the model by modifying ```chosen_model()``` accordingly.
3. Run [fed_train](./fed_train.py) file:
   ```bash
   python fed_train.py
   ```
   The script starts the Flower server and instantiates all the clients.

When training is complete, the trained weights are saved in ```weights_lstm.npz``` or ```weights_ncp.npz```, depending on the chosen model.

## Evaluation

Run [detection](./detection.ipynb) to detect anomalies on the test set. The detected anomalies are saved to a multi-sheet XLSX file ```anomalies.xlsx``` reporting timestamps, predicted values, and true values.

The notebooks [centralized_vs_fed](./centralized_vs_fed.ipynb) and [lstm_vs_ncp](./lstm_vs_ncp.ipynb) are intended for comparisons.













