# Class-Balanced Active Learning for Image Classification

## Description

This repository contains code for the paper Class-Balanced Active Learning for Image Classification.

<div align="center">
  
    <img src="framework.pdf" alt="Logo" width="80" height="80">
</div>  
Bibtex:

### Dependencies
Install Anaconda environment:
https://docs.anaconda.com/anaconda/install/linux/

Install PyTorch :
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
Install CVXPY python package:
https://www.cvxpy.org/install/

Install Gurobi optimizer and its licenese: 
https://www.gurobi.com/gurobi-and-anaconda-for-linux/


## Getting Started

Before starting AL cycles, execute run_cycl_0.py on cifar10/cifar100 dataset to obtain the imbalance dataset (only once) and train the model on initial sampels:
```
CUDA_VISIBLE_DEVICES=0 python run_cycle_0.py --method RandomSampling --dataset cifar100
```
### Executing program
To run Active Learning cycles:
```
python run.py with the following arguments:
--imb_type (To specify the imbalance type of the dataset)
--imb_factor (To specify the imbalance factor)
--dataset (To specify the dataset)
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)
