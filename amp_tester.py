#%%
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from src.data import GeoDataset_2
from src.amp_process import indep_test
from src.amp_model import amp_pepmnet
from src.device import device_info
from src.amp_evaluation_metrics import amp_evaluate_model

#TODO REVISAR SI SIRVE
device_information = device_info()
device = device_information.device
batch_size = 100
threshold = 0.5

# Cargar los datos de prueba
indep_testing_dataset = GeoDataset_2(raw_name='data/dataset/d8_options_p_2_5_6_7.csv') #TODO ver si se puede tener una sola clase
indep_testing_dataloader = DataLoader(indep_testing_dataset, batch_size, shuffle=False)

# Set up model:
# Initial Inputs
initial_dim_gcn = indep_testing_dataset.num_features
edge_dim_feature = indep_testing_dataset.num_edge_features

hidden_dim_nn_1 = 20
hidden_dim_nn_2 = 10

hidden_dim_gat_0 = 10

hidden_dim_fcn_1 = 10
hidden_dim_fcn_2 = 5
hidden_dim_fcn_3 = 3 

model = amp_pepmnet(
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                hidden_dim_nn_2,
                hidden_dim_gat_0,
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3,
                ).to(device)

weights_file="weights/best_model_weights_p_Chung_n_Chung_Xiao_epochs100_batch100.pth"

# Ejecutar la función de predicción en el conjunto de datos de prueba utilizando el modelo cargado
indep_testing_input,indep_testing_pred, indep_testing_pred_csv, indep_testing_scores = indep_test(model, indep_testing_dataloader, device, weights_file, threshold, type_dataset='testing')

# Guardar un archivo CSV con los valores de predicción
indep_prediction_test_set = {
    'Sequence': indep_testing_input,
    'Scores' : indep_testing_scores,
    'Prediction': indep_testing_pred_csv, 
}

df = pd.DataFrame(indep_prediction_test_set)
df.to_excel('results/indep_testing_prediction.xlsx', index=False)

print('\n ///// Independent_set_prediction.xlsx File Created /////')


# %%
