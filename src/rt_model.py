
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, ARMAConv
from torch_geometric.nn import aggr
from torch_scatter import scatter

#Hierarchical Graph Neural Network
class rt_pepmnet(torch.nn.Module):
    def __init__(self,
                initial_dim_gcn, # Input feature dimension for the GCN layers
                edge_dim_feature, # Edge feature dimension
                hidden_dim_nn_1, # Hidden layer dimension for the first NNConv
                hidden_dim_nn_2, # Hidden layer dimension for the second NNConv
                hidden_dim_nn_3, # Hidden layer dimension for the third NNConv
                
                hidden_dim_gat_1, # Hidden layer dimension for the ARMAConv layer
                
                hidden_dim_fcn_1, # Hidden layer dimension for the first fully connected layer
                hidden_dim_fcn_2, # Hidden layer dimension for the second fully connected layer
                hidden_dim_fcn_3, # Hidden layer dimension for the third fully connected layer
                dropout):
        super(rt_pepmnet, self).__init__()

        self.nn_conv_1 = NNConv(initial_dim_gcn, hidden_dim_nn_1,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, initial_dim_gcn * hidden_dim_nn_1)),
                                aggr='add')
        
        self.nn_conv_2 = NNConv(hidden_dim_nn_1, hidden_dim_nn_2,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, hidden_dim_nn_1 * hidden_dim_nn_2)),
                                aggr='add')
        
        self.nn_conv_3 = NNConv(hidden_dim_nn_2, hidden_dim_nn_3,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, hidden_dim_nn_2 * hidden_dim_nn_3)),
                                aggr='add')
        
        self.readout_atom = read_out_atom(in_dim=hidden_dim_nn_3)
        
        # ARMAConv layer for graph convolution at the amino acid level
        # The "8" is added to the feature dimension when amino acid features are concatenated
        self.nn_gat_1 = ARMAConv(hidden_dim_nn_3+8, hidden_dim_gat_1, num_stacks = 3, dropout=0, num_layers=7, shared_weights = False ) 
        
        self.readout_aminoacid = read_out_amino_acid(in_dim=hidden_dim_gat_1)
        
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
        
        x = self.nn_conv_3(x, edge_index, edge_attr)
        x = F.relu(x)
        
        results_list = []
        
        for i in range(len(cc)): 
            
            cc_i = cc[i].item() # Get the current component ID
            mask = idx_batch == i # Get the mask for the current batch
            xi = x[mask] # Select the corresponding atom features
            monomer_labels_i = monomer_labels[mask] # Get monomer labels for the current batch
            
            # Getting amino acids representation from atom features
            xi = self.readout_atom(xi, monomer_labels_i)
            
            # Concatenate amino acid features if they exist
            # Comment out the following 3 lines if you are not using amino acid feature concatenation
            aminoacid_ft_tupla = [tupla for tupla in aminoacids_features if tupla[0] == cc_i]
            aminoacids_features_i = aminoacid_ft_tupla[0][1]
            xi = torch.cat((xi, aminoacids_features_i), dim=1)
            
            amino_index_tupla = [tupla for tupla in amino_index if tupla[0] == cc_i]
            amino_index_i = amino_index_tupla[0][1]
            
            # Graph convolution amino acid level
            xi = self.nn_gat_1(xi, amino_index_i) 
            xi = F.relu(xi)
            
            # Readout for peptide representation
            xi = self.readout_aminoacid(xi)
            
            results_list.append(xi)
        
        # Concatenate all peptide representations into a single tensor    
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


class read_out_amino_acid(nn.Module):
    def __init__(self, in_dim):
        super(read_out_amino_acid, self).__init__()
        self.readout = aggr.SumAggregation()

    def forward(self, x):
        return self.readout(x)
    
class read_out_atom(nn.Module):
    def __init__(self, in_dim):
        super(read_out_atom, self).__init__()
        
    def forward(self, x, monomer_labels_i):
        return scatter(x, monomer_labels_i, dim=0, reduce="sum")




# %%
