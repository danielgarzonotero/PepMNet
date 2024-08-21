#%%
import time
start_time = time.time()
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from src.data import GeoDataset_2
from src.rt_process import rt_tester
from src.rt_model import rt_GCN_Geo
from src.device import device_info

device_information = device_info()
device = device_information.device
batch_size = 100
threshold = 0.5

# Cargar los datos de prueba
indep_testing_dataset = GeoDataset_2(root='',
                                    raw_name='data/test.csv',
                                    index_x=0
                                    )
finish_time = time.time()
final_time = (finish_time - start_time)/60
print('//Processing Time = ', final_time)
indep_testing_dataloader = DataLoader(indep_testing_dataset, batch_size, shuffle=False)

# Set up model:
# Initial Inputs
initial_dim_gcn = indep_testing_dataset.num_features
edge_dim_feature = indep_testing_dataset.num_edge_features

print(initial_dim_gcn)
print(edge_dim_feature)

hidden_dim_nn_1 = 500
hidden_dim_nn_2 = 250  
hidden_dim_nn_3 = 100

hidden_dim_gat_0 = 15

hidden_dim_fcn_1 = 100
hidden_dim_fcn_2 = 50
hidden_dim_fcn_3 = 10

dropout = 0


model = rt_GCN_Geo(
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                hidden_dim_nn_2,
                hidden_dim_nn_3,
                
                hidden_dim_gat_0,
                
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3,
                dropout
            ).to(device)

weights_file="weights_RT/yeast_unmod_best_model_weights.pth"

input_all_inde, pred_prob_all_inde = rt_tester(model, indep_testing_dataloader, device, weights_file)

#Saving a CSV file with prediction values
prediction_independent_set =   {
                                'Sequence':input_all_inde,
                                'Prediction (s)':  pred_prob_all_inde.cpu().numpy()
                                }

df = pd.DataFrame(prediction_independent_set)
df.to_excel('results/prediction_independet_set.xlsx', index=False)

#TODO ANADIR LAS GRAFICAS ACA, ESTA FUNCIONANDO HASTA EL MOMENTO

# %%