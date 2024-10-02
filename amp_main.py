#%%
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.explain import CaptumExplainer, Explainer
from sklearn.model_selection import KFold
from torch.utils.data import  Subset
import numpy as np


import pandas as pd
from src.device import device_info
from src.data import GeoDataset_1
from src.amp_model import amp_pepmnet
from src.amp_process import amp_train, amp_validation, amp_predict_test
from src.amp_evaluation_metrics import amp_evaluate_model

device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

## SET UP DATALOADERS: 

# Build starting dataset: 
datasets = {
            'test_ruiz': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Test_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=1,
                                            has_targets= True
                                            ),
            'train_ruiz': GeoDataset_1(raw_name='data/AMP/dataset Ruiz/Train_clean.csv',
                                            root='',
                                            index_x=0,
                                            index_y=1,
                                            has_targets= True
                                            )
            }

testing_datataset = datasets['test_ruiz']

print('Number of NODES features: ', testing_datataset.num_features)
print('Number of EDGES features: ', testing_datataset.num_edge_features)

finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - start_time) / 60 

# Train with a random seed to initialize weights:

# Set up model:
initial_dim_gcn = testing_datataset.num_features
edge_dim_feature = testing_datataset.num_edge_features

hidden_dim_nn_1 = 20
hidden_dim_nn_2 = 10

hidden_dim_gat_0 = 50

hidden_dim_fcn_1 = 200
hidden_dim_fcn_2 = 100
hidden_dim_fcn_3 = 10
dropout = 0

def initialize_model(initial_dim_gcn,
                    edge_dim_feature,
                    hidden_dim_nn_1,
                    hidden_dim_nn_2,
                    hidden_dim_gat_0,
                    hidden_dim_fcn_1,
                    hidden_dim_fcn_2,
                    hidden_dim_fcn_3,
                    dropout,

                    learning_rate,
                    weight_decay,
                    seed,
                    ):
    
    torch.manual_seed(seed)

    model = amp_pepmnet(
                    initial_dim_gcn,
                    edge_dim_feature,
                    hidden_dim_nn_1,
                    hidden_dim_nn_2,
                    
                    hidden_dim_gat_0,
                    
                    hidden_dim_fcn_1,
                    hidden_dim_fcn_2,
                    hidden_dim_fcn_3,
                    dropout
                    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return model, optimizer



#/////////////////// Training /////////////////////////////
# Set up optimizer:
learning_rate = 1E-3 
weight_decay = 1E-5 
batch_size = 100

ruiz_testing_datataset = datasets['test_ruiz']
ruiz_test_dataloader = DataLoader(ruiz_testing_datataset , batch_size, shuffle=False)

start_time_training = time.time()
ruiz_training_datataset = datasets['train_ruiz']
number_of_epochs = 500
k_folds = 5  # Número de particiones para K-Fold Cross-Validation
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=24)

# List to store validation losses for each fold
fold_val_losses = []
threshold = 0.5
seeds = [0,3,9,1,21]

# Iterate over the K-Folds
for fold, (train_ids, val_ids) in enumerate(kfold.split(ruiz_training_datataset)):
    print(f'//////// Fold {fold+1}/{k_folds}: ///////////')
    print(f'  Train IDs: {len(train_ids)} samples')
    print(f'  Val IDs:   {len(val_ids)} samples')

    seed = seeds[fold]
    
    model, optimizer = initialize_model(initial_dim_gcn,
                                        edge_dim_feature,
                                        hidden_dim_nn_1,
                                        hidden_dim_nn_2,
                                        hidden_dim_gat_0,
                                        hidden_dim_fcn_1,
                                        hidden_dim_fcn_2,
                                        hidden_dim_fcn_3,
                                        dropout,
                                        
                                        learning_rate,
                                        weight_decay,
                                        seed = seed
                                        )

    # Crear subsets y dataloaders
    train_subsampler = Subset(ruiz_training_datataset, train_ids)
    val_subsampler = Subset(ruiz_training_datataset, val_ids)
    tun_train_dataloader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    tun_val_dataloader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=True)
    
    # Listas para almacenar pérdidas
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_fold = None
    
    # Entrenamiento y validación por épocas
    for epoch in range(1, number_of_epochs+1):
        print(f'  Epoch {epoch}/{number_of_epochs}:')
        
        # Entrenamiento
        model.train()
        train_loss = amp_train(model, device, tun_train_dataloader, optimizer, epoch)
        train_losses.append(train_loss)
        
        # Validación
        model.eval()
        val_loss = amp_validation(model, device, tun_val_dataloader, epoch)
        val_losses.append(val_loss)
        
        # Guardar el mejor modelo según la pérdida de validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_fold = model.state_dict()
            best_fold_index = fold  # Actualizar el índice del fold con mejor pérdida de validación
    
    # Guardar el modelo con la mejor pérdida de validación para este fold
    torch.save(best_model_fold, f"weights_AMP/best_model_weights_fold_{fold+1}.pth")
    print(f'  Best model in this fold: best_model_weights_fold_{fold+1}.pth')
    
    # Almacenar la mejor pérdida de validación para este fold
    fold_val_losses.append((fold, best_val_loss))
    print(f'  Lowest validation loss in this fold: {best_val_loss:.4f}\n')
    
    # Graficar la curva de pérdida para este fold
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss', color='darkorange')
    plt.plot(val_losses, label='Validation loss', color='seagreen')
    plt.title(f'Training and Validation Loss\nFold {fold+1}\nBest Validation Loss: {best_val_loss:.5f}', fontsize=17)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=14)
    plt.savefig(f'results/AMP/loss_curve_fold_{fold+1}.png', dpi=216)
    plt.show()
    
    weights_file = f"weights_AMP/best_model_weights_fold_{fold+1}.pth"
    
    # ------------------------------------////////// Training set /////////////---------------------------------------------------
    training_sequences, training_target, training_logits, training_pred_csv, training_scores = amp_predict_test(
                                                                                                    model,
                                                                                                    tun_train_dataloader,
                                                                                                    device,
                                                                                                    weights_file,  # Usar el modelo guardado del fold actual
                                                                                                    threshold,
                                                                                                    has_targets = True
                                                                                                    )
    
    # Guardar un archivo CSV con los valores de predicción
    prediction_train_set = {
                            'Sequence': training_sequences,
                            'Target': training_target,
                            'Scores': training_scores,
                            'Prediction': training_pred_csv
                            }
    
    df = pd.DataFrame(prediction_train_set)
    df.to_excel(f'results/AMP/training_prediction_fold_{fold+1}.xlsx', index=False)
    
    # Evaluar el modelo en el conjunto de entrenamiento y obtener métricas
    TP_training, TN_training, FP_training, FN_training, ACC_training, PR_training, \
    SN_training, SP_training, F1_training, mcc_training, roc_auc_training, ap_training = \
    amp_evaluate_model(
                        prediction=training_scores,
                        target=training_target,
                        dataset_type=f'Training_fold_{fold+1}',
                        threshold=threshold,
                        device=device
                        
                        )
    
    # -------------------------------------------- ////////// Validation Set //////////-------------------------------------------------
    validation_sequences, validation_target, validation_pred, validation_pred_csv, validation_scores = amp_predict_test(
                                                                                                                        model,
                                                                                                                        tun_val_dataloader,
                                                                                                                        device,
                                                                                                                        weights_file,  # Usar el modelo guardado del fold actual
                                                                                                                        threshold,
                                                                                                                        has_targets = True
                                                                                                                        )
    
    # Guardar un archivo CSV con los valores de predicción
    prediction_validation_set = {
                                'Sequence': validation_sequences,
                                'Target': validation_target,
                                'Scores': validation_scores,
                                'Prediction': validation_pred_csv
                                }
    
    df = pd.DataFrame(prediction_validation_set)
    df.to_excel(f'results/AMP/validation_prediction_fold_{fold+1}.xlsx', index=False)
    
    # Evaluar el modelo en el conjunto de validación y obtener métricas
    TP_validation, TN_validation, FP_validation, FN_validation, ACC_validation, PR_validation, \
    SN_validation, SP_validation, F1_validation, mcc_validation, roc_auc_validation, ap_validation = \
    amp_evaluate_model( 
                        prediction= validation_scores,
                        target=validation_target,
                        dataset_type=f'Validation_fold_{fold+1}',
                        threshold=threshold,
                        device=device
                        )
    
    print(f'///// Fold {fold+1}-Training Metrics: //////')
    print(f'  Accuracy: {ACC_training:.4f}')
    print(f'  Precision: {PR_training:.4f}')
    print(f'  Recall: {SN_training:.4f}')
    print(f'  F1-score: {F1_training:.4f}')
    print(f'  MCC: {mcc_training:.4f}')
    print(f'  ROC AUC: {roc_auc_training:.4f}')
    print(f'  Average Precision: {ap_training:.4f}\n')
    
    print(f'///// Fold {fold+1}-Validation Metrics: /////')
    print(f'  Accuracy: {ACC_validation:.4f}')
    print(f'  Precision: {PR_validation:.4f}')
    print(f'  Recall: {SN_validation:.4f}')
    print(f'  F1-score: {F1_validation:.4f}')
    print(f'  MCC: {mcc_validation:.4f}')
    print(f'  ROC AUC: {roc_auc_validation:.4f}')
    print(f'  Average Precision: {ap_validation:.4f}\n')


finish_time_training = time.time()
time_training= (finish_time_training - start_time) / 60 

# Extract only the validation losses from the tuples
val_losses = [val_loss for _, val_loss in fold_val_losses]

# Calculate the average and standard deviation of the validation losses
avg_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)

# Display the average and standard deviation of the validation losses
print(f'Average validation loss across all folds: {avg_val_loss:.7f} ± {std_val_loss:.7f}')

# --------------------------------------------////////// Test Set //////////---------------------------------------------------
print('//////// Testing ///////')
# Realizar predicciones usando los cinco modelos
folds = [1, 2, 3, 4, 5]
sequences, targets = [], []
all_scores = []
start_time_testing = time.time()

for fold in folds:
    weights_file = f"weights_AMP/best_model_weights_fold_{fold}.pth"
    
    # Realizar la predicción usando el modelo cargado
    test_sequences, test_target, _, test_pred_csv, test_scores = amp_predict_test(
                                                                                    model,
                                                                                    ruiz_test_dataloader,
                                                                                    device,
                                                                                    weights_file,
                                                                                    threshold,
                                                                                    has_targets = True
                                                                                )
    
    # Almacenar secuencias y targets (solo una vez)
    if not sequences:
        sequences = test_sequences
        targets = test_target
        
    # Almacenar las puntuaciones del modelo
    all_scores.append(test_scores)

# Convertir las predicciones en un DataFrame con scores de cada fold
df = pd.DataFrame({
    'Sequence': sequences,
    'Target': targets,
    'Scores model 1': all_scores[0],
    'Scores model 2': all_scores[1],
    'Scores model 3': all_scores[2],
    'Scores model 4': all_scores[3],
    'Scores model 5': all_scores[4],
})

# Calcular el promedio y la desviación estándar de las predicciones
df['Average score'] = df[['Scores model 1', 'Scores model 2', 'Scores model 3', 'Scores model 4', 'Scores model 5']].mean(axis=1)
df['Standard deviation'] = df[['Scores model 1', 'Scores model 2', 'Scores model 3', 'Scores model 4', 'Scores model 5']].std(axis=1)

# Generar predicción redondeada con el threshold
df['Prediction'] = (df['Average score'] >= threshold).astype(int)

# Guardar el DataFrame en un archivo Excel
df.to_excel('results/AMP/testing_prediction.xlsx', index=False)

finish_time_testing = time.time()
time_prediction = (finish_time_testing - start_time_testing) / 60

# Calcular métricas de evaluación
TP_test, TN_test, FP_test, FN_test, ACC_test, PR_test, \
SN_test, SP_test, F1_test, mcc_test, roc_auc_test, ap_test = amp_evaluate_model(
                                                                        prediction=df['Average score'].values,
                                                                        target=df['Target'].values,
                                                                        dataset_type='Testing',
                                                                        threshold=threshold,
                                                                        device=device
                                                                    )

print(f'///// Testing Metrics: //////')
print(f'  Accuracy: {ACC_test:.4f}')
print(f'  Precision: {PR_test:.4f}')
print(f'  Recall: {SN_test:.4f}')
print(f'  F1-score: {F1_test:.4f}')
print(f'  MCC: {mcc_test:.4f}')
print(f'  ROC AUC: {roc_auc_test:.4f}')
print(f'  Average Precision: {ap_test:.4f}\n')

finish_time = time.time()
total_time = (finish_time - start_time) / 60

#--------------------------------///////////Result DataFrame////////////---------------------------------------
data = {
    "Metric": [
    "node_features",
    "edge_features",
    "initial_dim_gcn",
    "edge_dim_feature",
    "hidden_dim_nn_1",
    "hidden_dim_nn_2",
    "hidden_dim_gat_0",
    "hidden_dim_fcn_1",
    "hidden_dim_fcn_2",
    "hidden_dim_fcn_3",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "number_of_epochs",
    "threshold",
    "TP_training",
    "TN_training",
    "FP_training",
    "FN_training",
    "ACC_training",
    "PR_training",
    "SN_training",
    "SP_training",
    "F1_training",
    "mcc_training",
    "roc_auc_training",
    "TP_validation",
    "TN_validation",
    "FP_validation",
    "FN_validation",
    "ACC_validation",
    "PR_validation",
    "SN_validation",
    "SP_validation",
    "F1_validation",
    "mcc_validation",
    "roc_auc_validation",
    "TP_test",
    "TN_test",
    "FP_test",
    "FN_test",
    "ACC_test",
    "PR_test",
    "SN_test",
    "SP_test",
    "F1_test",
    "mcc_test",
    "roc_auc_test",
    "time_preprocessing",
    "time_training",
    "time_prediction",
    "total_time"
    ],
    "Value": [
        testing_datataset.num_features,
        testing_datataset.num_edge_features,
        initial_dim_gcn,
        edge_dim_feature ,
        hidden_dim_nn_1 ,
        hidden_dim_nn_2 ,
        hidden_dim_gat_0,
        hidden_dim_fcn_1 ,
        hidden_dim_fcn_2 ,
        hidden_dim_fcn_3 ,
        batch_size,
        learning_rate,
        weight_decay,
        number_of_epochs,
        threshold,
        TP_training,
        TN_training,
        FP_training,
        FN_training,
        ACC_training, 
        PR_training, 
        SN_training, 
        SP_training,
        F1_training,
        mcc_training,
        roc_auc_training,
        TP_validation,
        TN_validation,
        FP_validation,
        FN_validation,
        ACC_validation, 
        PR_validation, 
        SN_validation, 
        SP_validation,
        F1_validation,
        mcc_validation,
        roc_auc_validation,
        TP_test,
        TN_test,
        FP_test,
        FN_test,
        ACC_test, 
        PR_test, 
        SN_test, 
        SP_test,
        F1_test,
        mcc_test,
        roc_auc_test,
        time_preprocessing, 
        time_training,
        time_prediction,
        total_time
    ],
    
}


df = pd.DataFrame(data)
df.to_csv('results/AMP/results_training_validation_test.csv', index=False)

print('/////////////// Ready ////////////////////')


# %%
