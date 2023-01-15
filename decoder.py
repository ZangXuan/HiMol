import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,precision_score

MAX_NB = 8
MAX_DECODE_LEN = 100
MAX_BOND_TYPE = 4
MAX_ATOM_TYPE = 118
MAX_ATOM_CHIRALITY = 4
MAX_BOND_DIR = 3

def create_var(tensor, device, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor).to(device)
    else:
        return Variable(tensor, requires_grad=requires_grad).to(device)


class Model_decoder(nn.Module):

    def __init__(self, hidden_size, device, dropout=0.2):
        super(Model_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
    
        self.loss_linear = nn.Linear(5, 1)

        self.bond_if_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))  

        self.bond_if_s = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1))  

        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x

        self.bond_type_s = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, MAX_BOND_TYPE))  

        self.atom_type_s = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, MAX_ATOM_TYPE))  

        self.atom_num_s = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//4),
            nn.Softplus(),
            nn.Linear(hidden_size//4, 1)
            )

        self.bond_num_s = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//4),
            nn.Softplus(),
            nn.Linear(hidden_size//4, 1)
            )

        self.bond_pred_loss = nn.BCEWithLogitsLoss()
        self.bond_type_pred_loss = nn.CrossEntropyLoss()
        self.atom_type_pred_loss = nn.CrossEntropyLoss()
        self.atom_num_pred_loss = nn.SmoothL1Loss(reduction="mean")
        self.bond_num_pred_loss = nn.SmoothL1Loss(reduction="mean")


    def super_node_rep(self, mol_batch, node_rep):
        super_group = []
        for mol_index, mol in enumerate(mol_batch):     
            super_group.append(node_rep[mol_index][-1, :]).to(self.device)
        super_rep = torch.stack(super_group, dim=0)
        return super_rep
    
    def topo_pred(self, mol_batch, node_rep, super_node_rep):
        bond_if_loss, bond_if_auc, bond_if_ap = 0, 0, 0
        bond_type_loss, bond_type_acc = 0, 0
        atom_type_loss, atom_type_acc = 0, 0
        atom_num_loss, bond_num_loss = 0, 0
        
        atom_num_target, bond_num_target = [], []
        for mol in mol_batch:     
            num_atoms = mol.size_atom() 
            atom_num_target.append(num_atoms) 
            num_bonds = mol.size_bond() 
            bond_num_target.append(num_bonds)

        ###predict atom number, bond number
        super_rep = torch.stack(super_node_rep, dim=0).to(self.device)
        atom_num_pred = self.atom_num_s(super_rep).squeeze(-1)
        atom_num_target = torch.tensor(np.array(atom_num_target)).to(atom_num_pred.dtype).to(self.device)
        atom_num_loss += self.atom_num_pred_loss(atom_num_pred, atom_num_target) / len(mol_batch)
        atom_num_rmse = torch.sqrt(torch.sum((atom_num_pred - atom_num_target) ** 2)).item() / len(mol_batch)

        super_rep = torch.stack(super_node_rep, dim=0).to(self.device)
        bond_num_pred = self.bond_num_s(super_rep).squeeze(-1)
        bond_num_target = torch.tensor(np.array(bond_num_target)).to(bond_num_pred.dtype).to(self.device)
        bond_num_loss += self.bond_num_pred_loss(bond_num_pred, bond_num_target) / len(mol_batch)
        bond_num_rmse = torch.sqrt(torch.sum((bond_num_pred - bond_num_target) ** 2)).item() / len(mol_batch)

        ###predict atom type, bond type
        mol_num = len(mol_batch)
        for mol_index, mol in enumerate(mol_batch):     
            num_atoms = mol.size_atom()
            num_bonds = mol.size_bond()
            if num_bonds<1:
                mol_num -= 1
            else:   
                mol_rep = node_rep[mol_index].to(self.device)
                mol_atom_rep_proj = self.feat_drop(self.bond_if_proj(mol_rep))
            
                bond_if_input = torch.cat([mol_atom_rep_proj.repeat(1, num_atoms).view(num_atoms * num_atoms, -1), 
                                       mol_atom_rep_proj.repeat(num_atoms, 1)], dim=1).view(num_atoms, num_atoms, -1) 
                bond_if_pred = self.bond_if_s(bond_if_input).squeeze(-1)
                a = torch.zeros(num_atoms, num_atoms)
                bond_if_target = a.index_put(indices=[mol.edge_index_nosuper[0,:], mol.edge_index_nosuper[1,:]], values=torch.tensor(1.)).to(self.device)
                bond_if_loss += self.bond_pred_loss(bond_if_pred, bond_if_target) 
                bond_if_auc += roc_auc_score(bond_if_target.flatten().cpu().detach(), torch.sigmoid(bond_if_pred.flatten().cpu().detach())) 
                bond_if_ap += average_precision_score(bond_if_target.cpu().detach(), torch.sigmoid(bond_if_pred.cpu().detach())) 

                start_rep = mol_atom_rep_proj.index_select(0, mol.edge_index_nosuper[0,:].to(self.device))
                end_rep = mol_atom_rep_proj.index_select(0, mol.edge_index_nosuper[1,:].to(self.device))
            
                bond_type_input = torch.cat([start_rep, end_rep], dim=1)
                bond_type_pred = self.bond_type_s(bond_type_input)

                bond_type_target = mol.edge_attr_nosuper[:,0].to(self.device)
                bond_type_loss += self.bond_type_pred_loss(bond_type_pred, bond_type_target) 

                _, preds = torch.max(bond_type_pred, dim=1)
                pred_acc = torch.eq(preds, bond_type_target).float()
                bond_type_acc += (torch.sum(pred_acc) / bond_type_target.nelement())


                mol_rep = node_rep[mol_index].to(self.device)
                atom_type_pred = self.atom_type_s(mol_rep)

                atom_type_target = mol.x_nosuper[:,0].to(self.device)
                atom_type_loss += self.atom_type_pred_loss(atom_type_pred, atom_type_target) 

                _, preds = torch.max(atom_type_pred, dim=1)
                pred_acc = torch.eq(preds, atom_type_target).float()
                atom_type_acc += (torch.sum(pred_acc) / atom_type_target.nelement()) 
        loss_tur = [bond_if_loss/mol_num, bond_type_loss/mol_num, atom_type_loss/mol_num, atom_num_loss, bond_num_loss]
        results = [bond_if_auc/mol_num, bond_if_ap/mol_num, bond_type_acc/mol_num, atom_type_acc/mol_num, atom_num_rmse, bond_num_rmse]

        return loss_tur, results

    def forward(self, mol_batch, node_rep, super_node_rep):
        loss_tur, results = self.topo_pred(mol_batch, node_rep, super_node_rep)
        loss = 0
        loss_weight = create_var(torch.rand(5),self.device, requires_grad=True)
        loss_wei = torch.softmax(loss_weight, dim=-1)
        for index in range(len(loss_tur)):
            loss += loss_tur[index] * loss_wei[index]
        return loss, results[0], results[1], results[2], results[3], results[4], results[5]
        

