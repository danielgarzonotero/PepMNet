#%%
import time
start_time = time.time()
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from src.data import GeoDataset_1
from src.rt_process import  rt_predict_test
from src.rt_model import rt_pepmnet
from src.device import device_info
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt


device_information = device_info()
device = device_information.device
batch_size = 100
threshold = 0.5
has_targets = True #TODO

# Cargar los datos de prueba
indep_testing_dataset = GeoDataset_1(
                                    raw_name='data/test_1.csv',
                                    root='',
                                    index_x=0,
                                    index_y=1,
                                    has_targets=has_targets
                                    )

finish_time = time.time()
# Tiempo en segundos redondeado a 4 decimales
final_time_seconds = round(finish_time - start_time, 3)
# Tiempo en minutos
final_time_minutes = round(final_time_seconds / 60, 3)
print(f'//// Processing Time //// = {final_time_seconds} seconds ({final_time_minutes} minutes)')

indep_testing_dataloader = DataLoader(indep_testing_dataset, batch_size, shuffle=False)

# Set up model:
# Initial Inputs
initial_dim_gcn = indep_testing_dataset.num_features
edge_dim_feature = indep_testing_dataset.num_edge_features

print('Node Features:',initial_dim_gcn)
print('Edge Features:',edge_dim_feature)

hidden_dim_nn_1 = 500
hidden_dim_nn_2 = 250  
hidden_dim_nn_3 = 100

hidden_dim_gat_0 = 15

hidden_dim_fcn_1 = 100
hidden_dim_fcn_2 = 50
hidden_dim_fcn_3 = 10

dropout = 0

model = rt_pepmnet(
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

weights_file="weights/RT/a1/scx_best_model_weights.pth"


if has_targets:
    input_all_inde, target_all, pred_prob_all_inde =  rt_predict_test(  model,
                                                        indep_testing_dataloader,
                                                        device,
                                                        weights_file,
                                                        has_targets=has_targets
                                                        )
    
    #Saving a CSV file with prediction values
    prediction_independent_set =   {
                                    'Sequence':input_all_inde,
                                    'Targets': target_all.cpu().numpy(),
                                    'Prediction (time unit)':  pred_prob_all_inde.cpu().numpy() #TODO time units
                                    }
    
    df = pd.DataFrame(prediction_independent_set)
    df.to_excel('results/Independent/RT/rt_prediction_independet_set.xlsx', index=False)
    
    r2 = r2_score(target_all.cpu(), pred_prob_all_inde.cpu())
    r_, _ = pearsonr(target_all.cpu(), pred_prob_all_inde.cpu()) 
    mae = mean_absolute_error(target_all.cpu(), pred_prob_all_inde.cpu())
    mse = mean_squared_error(target_all.cpu(), pred_prob_all_inde.cpu(), squared=False)
    rmse = sqrt(mse)
    
    # Save the evaluation metrics using pandas
    evaluation_metrics = {
                            'Metric': ['R²', "Pearson's r", 'MAE', 'MSE', 'RMSE'],
                            'Value': [r2, r_, mae, mse, rmse]
                        }
    
    # Convert to a DataFrame
    df_metrics = pd.DataFrame(evaluation_metrics)
    
    # Save as a CSV file
    df_metrics.to_csv('results/Independent/RT/evaluation_metrics.csv', index=False)
    
    print(  "\n///// Evaluation Metrics: /////\n"
            f"R²: {r2:.4f}\n"
            f"Pearson's r: {r_:.4f}\n"
            f"MAE: {mae:.4f}\n"
            f"MSE: {mse:.4f}\n"
            f"RMSE: {rmse:.4f}")
    
    
    # Visualization: Scatter Plot for Training Set:
    # Plot a scatter plot of true vs. predicted values on the training set. 
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(target_all.cpu(), pred_prob_all_inde.cpu(), alpha=0.3, color='steelblue')
    plt.plot([min(target_all.cpu()), max(pred_prob_all_inde.cpu())], [min(target_all.cpu()), max(target_all.cpu())], color='steelblue', ls="-", alpha=0.7, linewidth=3.5)
    plt.xlim([min(target_all.cpu()), max(pred_prob_all_inde.cpu())])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Scatter Plot Training Set\nIndependet Dataset', fontsize=18, pad=30)
    plt.xlabel(f"True Retention Time (min)", fontsize=18, labelpad=10)
    plt.ylabel(f"Predicted Retention Time (min)", fontsize=18, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'results/Independent/RT/rt_scatter_independent_set.png', format="png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    input_all_inde, pred_prob_all_inde =  rt_predict_test(  model,
                                                        indep_testing_dataloader,
                                                        device,
                                                        weights_file,
                                                        has_targets=has_targets
                                                        )
    
    #Saving a CSV file with prediction values
    prediction_independent_set =   {
                                    'Sequence':input_all_inde,
                                    'Prediction (s)':  pred_prob_all_inde.cpu().numpy()
                                    }
    
    df = pd.DataFrame(prediction_independent_set)
    df.to_excel('results/Independent/RT/rt_prediction_independet_set.xlsx', index=False)

# %%