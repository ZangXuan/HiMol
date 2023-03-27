# HiMol
# Hierarchical Molecular Graph Self-supervised Learning for Property Prediction
Official Pytorch implementation of HiMol model in the paper "Zang, Xuan., Zhao, Xianbing. & Tang, Buzhou. Hierarchical Molecular Graph Self-Supervised Learning for property prediction. Commun Chem 6, 34 (2023)." https://doi.org/10.1038/s42004-023-00825-5. 

## Environment Setup


python
rdkit
scipy 
torch
torch-geometric
torch-sparse
tqdm
networkx
numpy
pandas


## Training
You can pretrain the model by
```
mkdir saved_model
python pretrain.py
```

## Evaluation
You can evaluate the pretrained model by finetuning on downstream tasks

Download the downstream data from https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function, and save the .csv files in the ./finetune/dataset/[dataset_name]/raw/, where [dataset_name] is replaced by the downstream dataset name. 
For example, bace.csv is saved in './finetune/dataset/bace/raw/bace.csv'.

```
cd finetune
mkdir model_checkpoints
python finetune.py
```

Please cite our paper as follows. Thank you.
"
@article{zang2023hierarchical,
  title={Hierarchical Molecular Graph Self-Supervised Learning for property prediction},
  author={Zang, Xuan and Zhao, Xianbing and Tang, Buzhou},
  journal={Communications Chemistry},
  volume={6},
  number={1},
  pages={34},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
"
