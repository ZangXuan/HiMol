import argparse
from cmath import inf

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging, time

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
# from shishi2 import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import pearsonr

from splitters import scaffold_split, random_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(logger, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()

def train_reg(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        loss = torch.sum((pred-y)**2)/y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    #Whether y is non-null or not.
    y = batch.y.view(pred.shape).to(torch.float64)
    is_valid = y**2 > 0
    #Loss matrix
    loss_mat = criterion(pred.double(), (y+1)/2)
    #loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
    loss = torch.sum(loss_mat)/torch.sum(is_valid)


    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    eval_roc = sum(roc_list)/len(roc_list) #y_true.shape[1]

    return eval_roc, loss

def eval_reg(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()
    print(y_true.shape, y_scores.shape)
    mse = mean_squared_error(y_true, y_scores)
    cor = pearsonr(y_true, y_scores)[0]
    rmse=np.sqrt(mean_squared_error(y_true,y_scores))
    # print(mse, cor)
    return mse, cor, rmse

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help='gnn_type (gat, gin, gcn, graphsage)')
    parser.add_argument('--dataset', type=str, default = 'clintox', 
                        help='[bbbp, bace, sider, clintox, hiv, sider,tox21, toxcast, muv,esol,freesolv,lipophilicity]')
    parser.add_argument('--input_model_file', type=str, default = '../saved_model/model_motif_pretrain_fenhuanbn_simdrop_gin3_300_100epoch.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    # parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    # parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--GNN_para', type=bool, default = True, help='if the parameter of pretrain update')
    parser.add_argument('--early_stop', type=int, default = 100, help='number of patient')
    args = parser.parse_args()

    # # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    time_now = time.time()
    fh = logging.FileHandler('finetune_motif_log/{}.log'.format(str(time_now)))
    #fh = logging.FileHandler('nopretrain_motif_log/{}.log'.format(str(time_now)))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag']:
        task_type = 'cls'
    else:
        task_type = 'reg'

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'lipophilicity':
        num_tasks = 1
    elif args.dataset == 'mutag':
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    # if args.dataset == 'freesolv':
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)
    # else:
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    # val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)


    #set up model
    # model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, logger, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    if args.GNN_para:
        logger.info('GNN update')
        model_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    else:
        logger.info('No GNN update')
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    finetune_model_save_path = './model_checkpoints/motif_' + args.dataset + '_' + str(time_now) + '.pth'
    logger.info('====finetune_model_save_path {}'.format(finetune_model_save_path))

    # # best_epoch_val = float(inf)
    # best_epoch_val = 0
    # patient = 0

    # for epoch in range(1, args.epochs+1):
    #     logger.info('====epoch {}'.format(epoch))
    #     # print("====epoch " + str(epoch))
        
    #     train(logger, model, device, train_loader, optimizer)

    #     logger.info('====Evaluation')
    #     # print("====Evaluation")
    #     if args.eval_train:
    #         train_auc, train_loss = eval(args, model, device, train_loader)
    #     else:
    #         logger.info('omit the training accuracy computation')
    #         # print("omit the training accuracy computation")
    #         train_auc = 0
    #     val_auc, val_loss = eval(args, model, device, val_loader)
    #     test_auc, test_loss = eval(args, model, device, test_loader)

    #     #set early-stop
    #     # if val_loss < best_epoch_val:
    #     if val_auc > best_epoch_val:
    #         best_epoch_val = val_auc
    #         # torch.save(model.state_dict(), './model_checkpoints/motif_' + args.dataset + '_' + str(time_now) + '.pth')
    #         torch.save(model.state_dict(), finetune_model_save_path)
    #         patient = 0
    #     else:
    #         patient += 1
    #         if patient > args.early_stop:
    #             break
    #     logger.info("best_epoch_val: {}, patient: {}".format(best_epoch_val, patient))
    #     logger.info("train_loss: {}, val_loss: {}, test_loss: {}".format(train_loss, val_loss, test_loss))
    #     logger.info("train: {}, val: {}, test: {}".format(train_auc, val_auc, test_auc))
    #     #print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

    # # model.load_state_dict(torch.load('./model_checkpoints/motif_'+args.dataset+'_'+ str(time_now) + '.pth'))
    # model.load_state_dict(torch.load(finetune_model_save_path))

    # logger.info('====Final Test')
    # test_auc_final = 0
    # for i in range(10):
    #     test_auc,_ = eval(args, model, device, test_loader)
    #     logger.info("Epoch: {}, test: {}".format(i, test_auc))
    #     test_auc_final += test_auc/10
    # logger.info("Average test: {}".format(test_auc_final))


    # training based on task type
    if task_type == 'cls':
        best_epoch_val = 0
        patient = 0
        for epoch in range(1, args.epochs+1):
            logger.info('====epoch {}'.format(epoch))
            
            train(logger, model, device, train_loader, optimizer)

            logger.info('====Evaluation')
            if args.eval_train:
                train_auc, train_loss = eval(args, model, device, train_loader)
            else:
                logger.info('omit the training accuracy computation')
                train_auc = 0
            val_auc, val_loss = eval(args, model, device, val_loader)
            test_auc, test_loss = eval(args, model, device, test_loader)

            # Early stopping
            if np.greater(val_auc, best_epoch_val):  # change for train loss
                best_epoch_val = val_auc
                patient = 0
                torch.save(model.state_dict(), finetune_model_save_path)
            else:
                patient += 1
                if patient >= args.early_stop:
                    break
            
            logger.info("best_epoch_val: {}, patient: {}".format(best_epoch_val, patient))
            logger.info("train_loss: {}, val_loss: {}, test_loss: {}".format(train_loss, val_loss, test_loss))
            logger.info("train: {}, val: {}, test: {}".format(train_auc, val_auc, test_auc))
        model.load_state_dict(torch.load(finetune_model_save_path))

    elif task_type == 'reg':
        for epoch in range(1, args.epochs+1):
            logger.info('====epoch {}'.format(epoch))
            
            train_reg(logger, model, device, train_loader, optimizer)

            logger.info('====Evaluation')
            if args.eval_train:
                train_mse, train_cor, train_rmse = eval_reg(args, model, device, train_loader)
            else:
                logger.info('omit the training accuracy computation')
                train_mse, train_cor = 0, 0
            val_mse, val_cor, val_rmse = eval_reg(args, model, device, val_loader)
            test_mse, test_cor, test_rmse = eval_reg(args, model, device, test_loader)

            # print("train: %f val: %f test: %f" %(train_mse, val_mse, test_mse))
            # print("train: %f val: %f test: %f" %(train_cor, val_cor, test_cor))
            logger.info("train_mse: {}, val_mse: {}, test_mse: {}".format(train_mse, val_mse, test_mse))
            logger.info("train_cor: {}, val_cor: {}, test_cor: {}".format(train_cor, val_cor, test_cor))
            logger.info("train_rmse: {}, val_rmse: {}, test_rmse: {}".format(train_rmse, val_rmse, test_rmse))


if __name__ == "__main__":
    main()
