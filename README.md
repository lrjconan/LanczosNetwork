# LanczosNetwork
This is the PyTorch implementation of [Lanczos Network](https://arxiv.org/pdf/1803.06396.pdf) as described in the following paper:

```
@article{liao2018reviving,
  title={Reviving and Improving Recurrent Back-Propagation},
  author={Liao, Renjie and Xiong, Yuwen and Fetaya, Ethan and Zhang, Lisa and Yoon, KiJung and Pitkow, Xaq and Urtasun, Raquel and Zemel, Richard},
  journal={arXiv preprint arXiv:1803.06396},
  year={2018}
}
```

## Setup
To set up experiments, we need to build our customized operators by running the following scripts:
```
./setup.sh
``` 

## Dependencies
Python 3, PyTorch(0.4.0)


## Run Demos
* To run experiments ```X``` where ```X``` is one of {```hopfield```, ```cora```, ```pubmed```, ```hypergrad```}:

  ```python run_exp.py -c config/X.yaml```
  

**Notes**:
* Most hyperparameters in the configuration yaml file are self-explanatory.
* To switch between BPTT, TBPTT and RBP variants, you need to specify ```grad_method``` in the config file.
* Conjugate gradient based RBP requires support of forward mode auto-differentiation which we only provided for the experiments of Hopfield networks and graph neural networks (GNNs). You can check the comments in ```model/rbp.py``` for more details.

## Cite
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact rjliao@cs.toronto.edu if you have any questions or find any bugs.
