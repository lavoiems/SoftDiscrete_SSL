# Soft-Discretization for Self-Supervised Learning

Official code to reproduce experiments with *MPI3D* and *dSprites*. For ImageNet experiments, please consult the `large_scale` folder.
`Python>=3.8` and the packages listed in `requirements.txt` are required
to start experimenting with this repository.

## Repo structure

 - `src/`: contains the code for this project
 - `src/models/`: contains implementations of the SSL methods
 - `download_<dataset>.sh <path> <K>`: scripts to automatically download and split data to o.o.d. splits
 - `src/common` : Contains code utility shared across the models for running the code such as the data loaders.
 - `src/main.py`: Entry point of training program.
 - `src/evaluate.py`: Entry point of evaluation program.

Each model in `src/models/` contains three files:
  - `__init__.py`: Defines the specific parameters.
  - `model.py`: Defines the network architecture.
  - `train.py` Defines the training procedure.
  - `evaluate/` A folder that contains the evaluation procedures.

## How to run

Download the datasets using the utility script provided. For example, to download `MPI3D`, `K=3` in the `data` folder of the current directory:
```
./download_mpi3d.sh ./data 3
```

A model can be trained by invoking `src/main.py`, which also contains the general parameters shared among all the models.
The syntax for training a model is as follows:

```
python src/main.py [GENERAL PARAMETERS] [MODEL NAME] [SPECIFIC MODEL PARAMETERS]
```

For example, running BYOL with the softmax bottleneck would amount to running:
```
python src/main.py byol --encode_method softmax --dataset mpi3d --dataset_K 3
```

An evaluation script inside the folder `evaluate` of a model can be invoked using `src/evaluate.py`. The general syntax is as follows:
```
python src/evaluate.py [MODEL NAME] [SCRIPT NAME] [SPECIFIC EVALUATION PARAMETERS]
```

For example, running a linear probe for a pre-trained BYOL model can be done as follows
```
python src/evaluate.py byol linear_probe --data_path [PATH TO THE DATA FOLDER] --run_path [PATH TO THE SAVED MODEL]
```

