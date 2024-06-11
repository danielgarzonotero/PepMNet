
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, ARMAConv
from torch_geometric.nn import aggr
from torch_scatter import scatter


#Hierarchical Graph Neural Network
class amp_GCN_Geo(torch.nn.Module):
    def __init__(self,
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                hidden_dim_nn_2,

                hidden_dim_gat_1,
                
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3,
                dropout):
        super(amp_GCN_Geo, self).__init__()

        self.nn_conv_1 = NNConv(initial_dim_gcn, hidden_dim_nn_1,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, initial_dim_gcn * hidden_dim_nn_1)), 
                                aggr='add' )
        
        self.nn_conv_2 = NNConv(hidden_dim_nn_1, hidden_dim_nn_2,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, hidden_dim_nn_1 * hidden_dim_nn_2)), 
                                aggr='add')
        
        self.readout_atom = AttentionReadoutAtom(in_dim=hidden_dim_nn_2)
        
        #The 7 and 24 comes from the four amino acid features and blosum62 matrix that were concatenated,  95+24
        self.nn_gat_1 = ARMAConv(hidden_dim_nn_2+95, hidden_dim_gat_1, num_stacks = 3, dropout=0.4, num_layers=6, shared_weights = False ) 
        self.readout_aminoacid = AttentionReadoutAminoAcid(in_dim=hidden_dim_gat_1)
        
        #The 7 comes from the four peptides features that were concatenated, +7
        self.linear1 = nn.Linear(hidden_dim_gat_1, hidden_dim_fcn_1)
        self.linear2 = nn.Linear(hidden_dim_fcn_1, hidden_dim_fcn_2 )
        self.linear3 = nn.Linear(hidden_dim_fcn_2, hidden_dim_fcn_3) 
        self.linear4 = nn.Linear(hidden_dim_fcn_3, 1)
        
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x,
                edge_index,
                edge_attr,
                idx_batch,
                cc,
                monomer_labels,
                aminoacids_features,
                amino_index
                ): 
        
        x = self.nn_conv_1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        x = self.nn_conv_2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        results_list = []
        
        for i in range(len(cc)): 
            
            cc_i = cc[i].item()
            mask = idx_batch == i
            xi = x[mask]
            monomer_labels_i = monomer_labels[mask]
            
            aminoacid_ft_tupla = [tupla for tupla in aminoacids_features if tupla[0] == cc_i]
            aminoacids_features_i = aminoacid_ft_tupla[0][1]
            
            # getting amino acids representation from atom features
            xi = self.readout_atom(xi, monomer_labels_i)
            xi = torch.cat((xi, aminoacids_features_i), dim=1)

            amino_index_tupla = [tupla for tupla in amino_index if tupla[0] == cc_i]
            amino_index_i = amino_index_tupla[0][1]

            # Graph convolution amino acid level
            xi = self.nn_gat_1(xi, amino_index_i) 
            xi = F.relu(xi)
            
            # Readout for peptide representation
            xi = self.readout_aminoacid(xi)
            
            results_list.append(xi)
            
        p = torch.cat(results_list, dim=0)
        
        p = self.dropout(p)    
        p = self.linear1(p)
        p = F.relu(p)
        
        p = self.dropout(p)
        p = self.linear2(p)
        p = F.relu(p)
        
        p = self.dropout(p) 
        p = self.linear3(p)
        p = F.relu(p)
        
        p = self.linear4(p)
        
        return p.view(-1)

class AttentionReadoutAminoAcid(nn.Module):
    def __init__(self, in_dim):
        super(AttentionReadoutAminoAcid, self).__init__()
        self.attention = nn.Linear(in_dim, 1)
        self.readout = aggr.SumAggregation()

    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=0)
        weighted_x = x * attn_weights
        return self.readout(weighted_x)
    
class AttentionReadoutAtom(nn.Module):
    def __init__(self, in_dim):
        super(AttentionReadoutAtom, self).__init__()
        self.attention = nn.Linear(in_dim, 1)
        
    def forward(self, x, monomer_labels_i):
        attn_weights = F.softmax(self.attention(x), dim=0)
        weighted_x = x * attn_weights
        
        return scatter(weighted_x, monomer_labels_i, dim=0, reduce="sum")

# %%


