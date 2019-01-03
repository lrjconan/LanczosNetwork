# Lanczos Network
This is the PyTorch implementation of [Lanczos Network](https://openreview.net/pdf?id=BkedznAqKQ) as described in the following paper:

```
@inproceedings{liao2018lanczos,
  title={LanczosNet: Multi-Scale Deep Graph Convolutional Networks},
  author={Liao, Renjie and Zhao, Zhizhen and Urtasun, Raquel and Zemel, Richard},
  booktitle={ICLR},
  year={2019}
}
```

We also provide our own implementation of 9 recent graph neural networks on the [QM8](https://arxiv.org/pdf/1504.01966.pdf) benchmark: 

* [graph convolution networks for fingerprint](https://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints.pdf) (GCNFP)
* [gated graph neural networks](https://arxiv.org/pdf/1511.05493.pdf) (GGNN)
* [diffusion convolutional neural networks](https://arxiv.org/pdf/1511.02136.pdf) (DCNN) 
* [Chebyshev networks](https://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf) (ChebyNet)
* [graph convolutional networks](https://arxiv.org/pdf/1609.02907.pdf) (GCN)
* [message passing neural networks](https://arxiv.org/pdf/1704.01212.pdf) (MPNN)
* [graph sample and aggregate](https://www-cs-faculty.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) (GraphSAGE)
* [graph partition neural networks](https://arxiv.org/pdf/1803.06272.pdf) (GPNN)
* [graph attention networks](https://arxiv.org/pdf/1710.10903.pdf) (GAT)


## Setup
To set up experiments, we need to download the [preprocessed QM8 data](http://www.cs.toronto.edu/~rjliao/data/qm8.zip) and build our customized operators by running the following scripts:

```
./setup.sh
```

**Note**:

* We also provide the script ```dataset/get_qm8_data.py``` to preprocess the [raw QM8](http://quantum-machine.org/datasets/) data which requires the installation of [DeepChem](https://github.com/deepchem/deepchem). 
It will produce a different train/dev/test split than what we used in the paper due to the randomness of DeepChem.
Therefore, we suggest using our preprocessed data for a fair comparison.


## Dependencies
Python 3, PyTorch(1.0), numpy, scipy, sklearn


## Run Demos

### Train
* To run the training of experiment ```X``` where ```X``` is one of {```qm8_lanczos_net```, ```qm8_ada_lanczos_net```, ...}:

  ```python run_exp.py -c config/X.yaml```
  

**Note**:

* Please check the folder ```config``` for a full list of configuration yaml files.
* Most hyperparameters in the configuration yaml file are self-explanatory.

### Test

* After training, you can specify the ```test_model``` field of the configuration yaml file with the path of your best model snapshot, e.g.,

  ```test_model: exp/qm8_lanczos_net/LanczosNet_chemistry_2018-Oct-02-11-55-54_25460/model_snapshot_best.pth```	

* To run the test of experiments ```X```:

  ```python run_exp.py -c config/X.yaml -t```


## Cite
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact rjliao@cs.toronto.edu if you have any questions or find any bugs.
