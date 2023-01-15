# HiMol
# Hierarchical Molecular Graph Self-supervised Learning for Property Prediction
Official Pytorch implementation of HiMol model in the paper "Xuan Zang, Xianbing Zhao, Buzhou Tang. Hierarchical Molecular Graph Self-supervised Learning for Property Prediction". 

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
```
cd finetune
mkdir model_checkpoints
python finetune.py
```

