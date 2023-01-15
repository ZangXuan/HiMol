import torch
from torch.utils.data import Dataset
# from mol_tree import MolTree
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from torch_geometric.data import Data
from chemutils import get_mol, get_clique_mol


class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]
            # print('data',self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        # print('smiles',smiles)
        mol_graph = MolGraph(smiles)  #motif
        # mol_tree.recover()
        # mol_tree.assemble()
        return mol_graph


# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),  #元素周期表序号
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],    #原子的手性
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_inring': [None, False, True]
}


# class MolGraph(object):

#     def __init__(self, smiles):
#         self.smiles = smiles
#         self.mol = get_mol(smiles)

#         '''
#         #Stereo Generation
#         mol = Chem.MolFromSmiles(smiles)
#         self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
#         self.smiles2D = Chem.MolToSmiles(mol)
#         self.stereo_cands = decode_stereo(self.smiles2D)
#         '''
#         num_atom_features = 2  # atom type,  chirality tag
#         atom_features_list = []
#         for atom in self.mol.GetAtoms():
#             # atom_feature = [allowable_features['possible_atomic_num_list'].index(
#             #     atom.GetAtomicNum())] + [allowable_features[
#             #                              'possible_chirality_list'].index(atom.GetChiralTag())]
#             atom_feature = [allowable_features['possible_atomic_num_list'].index(
#                 atom.GetAtomicNum())] + [allowable_features[
#                                          'possible_degree_list'].index(atom.GetDegree())]
#             atom_features_list.append(atom_feature)
#         self.x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)
#         #两个atom_feature（atom在周期表的索引，atom手性）[num_nodes, num_node_features]

#         # bonds
#         num_bond_features = 2  # bond type, bond direction
#         if len(self.mol.GetBonds()) > 0:  # mol has bonds
#             edges_list = []
#             edge_features_list = []
#             for bond in self.mol.GetBonds():
#                 i = bond.GetBeginAtomIdx()
#                 j = bond.GetEndAtomIdx()
#                 # edge_feature = [allowable_features['possible_bonds'].index(
#                 #  bond.GetBondType())] + [allowable_features[
#                 # 'possible_bond_dirs'].index(
#                 #     bond.GetBondDir())]
#                 edge_feature = [allowable_features['possible_bonds'].index(
#                  bond.GetBondType())] + [allowable_features['possible_bond_inring'].index(
#                  bond.IsInRing())]
#                 edges_list.append((i, j))
#                 edge_features_list.append(edge_feature)
#                 edges_list.append((j, i))
#                 edge_features_list.append(edge_feature)

#             # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
#             self.edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)

#             # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
#             self.edge_attr_nosuper = torch.tensor(np.array(edge_features_list),
#                                  dtype=torch.long)  #两个edge_feature（键类型，键手性？）
#         else:  # mol has no bonds
#             self.edge_index_nosuper = torch.empty((2, 0), dtype=torch.long) #edgeCOO索引，[[row索引],[col索引]]
#             self.edge_attr_nosuper = torch.empty((0, num_bond_features), dtype=torch.long)

#         # add super node and edge
#         num_atoms = self.x_nosuper.size(0)
#         # super_x = torch.tensor([[119, 3]]).to(self.x_nosuper.device)
#         super_x = torch.tensor([[119, 0]]).to(self.x_nosuper.device)
#         self.x = torch.cat((self.x_nosuper, super_x), dim=0)

#         super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
#         super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
#         self.edge_index = torch.cat((self.edge_index_nosuper, super_edge_index), dim=1)

#         super_edge_attr = torch.zeros(num_atoms, 2)
#         super_edge_attr[:,0] = 5 #bond type for self-loop edge
#         super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
#         self.edge_attr = torch.cat((self.edge_attr_nosuper, super_edge_attr), dim = 0)

#         # self.data = Data(node_attr=self.node_attr, edge_index=self.edge_index, edge_attr=self.edge_attr)

#     def size_node(self):
#         return self.x.size()[0]

#     def size_edge(self):
#         return self.edge_attr.size()[0]

#     def size_atom(self):
#         return self.x_nosuper.size()[0]

#     def size_bond(self):
#         return self.edge_attr_nosuper.size()[0]




class MolGraph(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        '''
        #Stereo Generation
        mol = Chem.MolFromSmiles(smiles)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)
        '''
        atom_features_list = []
        for atom in self.mol.GetAtoms():
            atom_feature = [allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())] + [allowable_features[
                                         'possible_degree_list'].index(atom.GetDegree())]

            
            atom_features_list.append(atom_feature)
        self.x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2  # bond type, bond direction
        if len(self.mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in self.mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = [allowable_features['possible_bonds'].index(
                 bond.GetBondType())] + [allowable_features['possible_bond_inring'].index(
                 bond.IsInRing())]

                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            self.edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            self.edge_attr_nosuper = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long) 
        else:  
            self.edge_index_nosuper = torch.empty((2, 0), dtype=torch.long) #edgeCOO索引，[[row索引],[col索引]]
            self.edge_attr_nosuper = torch.empty((0, num_bond_features), dtype=torch.long)

        # add super node
        num_atoms = self.x_nosuper.size(0)
        super_x = torch.tensor([[119, 0]]).to(self.x_nosuper.device)  

        #add motif 
        cliques = motif_decomp(self.mol)
        num_motif = len(cliques)
        if num_motif > 0:
            motif_x = torch.tensor([[120, 0]]).repeat_interleave(num_motif, dim=0).to(self.x_nosuper.device)
            self.x = torch.cat((self.x_nosuper, motif_x, super_x), dim=0)

            motif_edge_index = []
            for k, motif in enumerate(cliques):
                motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
            motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)

            super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

            motif_edge_attr = torch.zeros(motif_edge_index.size()[1], 2)
            motif_edge_attr[:,0] = 6 #bond type for self-loop edge
            motif_edge_attr = motif_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)

            super_edge_attr = torch.zeros(num_motif, 2)
            super_edge_attr[:,0] = 5 #bond type for self-loop edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim = 0)

            self.num_part = (num_atoms, num_motif, 1)

        else:
            self.x = torch.cat((self.x_nosuper, super_x), dim=0)

            super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, super_edge_index), dim=1)

            super_edge_attr = torch.zeros(num_atoms, 2)
            super_edge_attr[:,0] = 5 #bond type for self-loop edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, super_edge_attr), dim = 0)

            self.num_part = (num_atoms, 0, 1)


    def size_node(self):
        return self.x.size()[0]

    def size_edge(self):
        return self.edge_attr.size()[0]

    def size_atom(self):
        return self.x_nosuper.size()[0]

    def size_bond(self):
        return self.edge_attr_nosuper.size()[0]


   

def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(x=mol.x, edge_index=mol.edge_index, edge_attr=mol.edge_attr, num_part=mol.num_part)
        graph_data_batch.append(data)
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch



def motif_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []  
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])  

    res = list(BRICS.FindBRICSBonds(mol))  
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]]) 


    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0: 
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms> len(c) > 0]


    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr)>1:
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
                    cliques[i] = list(set(cliques[i]) - set(list(ring)))
    
    cliques = [c for c in cliques if n_atoms> len(c) > 0]



    return cliques
