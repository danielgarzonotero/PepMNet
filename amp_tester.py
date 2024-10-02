#%%
import time
start_time = time.time()
import torch
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from src.data import GeoDataset_1
from src.amp_process import amp_predict_test
from src.amp_model import amp_pepmnet
from src.device import device_info
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import matthews_corrcoef
import numpy as np



# Configuración del dispositivo y batch size
device_information = device_info()
device = device_information.device
batch_size = 100
threshold = 0.5
has_targets = True  #TODO  True or None

# Cargar los datos de prueba
indep_testing_dataset = GeoDataset_1(
                                    raw_name='data/test_3.csv',
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

# Configuración del modelo
initial_dim_gcn = indep_testing_dataset.num_features
edge_dim_feature = indep_testing_dataset.num_edge_features

hidden_dim_nn_1 = 20
hidden_dim_nn_2 = 10
hidden_dim_gat_0 = 50
hidden_dim_fcn_1 = 200
hidden_dim_fcn_2 = 100
hidden_dim_fcn_3 = 10
dropout = 0

# Inicialización del modelo
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

# Listas para almacenar las secuencias, targets y las predicciones de los modelos
sequences = []
targets = []
all_scores = []

# Realizar predicciones usando los cinco modelos
folds = [1, 2, 3, 4, 5]

for fold in folds:
    weights_file = f"weights_AMP/model_0_3_9_1_21/best_model_weights_fold_{fold}.pth"
    
    if has_targets:
        # Realizar la predicción usando el modelo cargado
        test_sequences, test_target, test_logits, test_pred_csv, test_scores = amp_predict_test(
                                                                                        model,
                                                                                        indep_testing_dataloader,
                                                                                        device,
                                                                                        weights_file,
                                                                                        threshold,
                                                                                        has_targets
                                                                                    )
        
        # Almacenar secuencias y targets (solo una vez)
        if not sequences:
            sequences = test_sequences
            targets = test_target
            
        # Almacenar las puntuaciones del modelo
        all_scores.append(test_scores)
    else:
        test_sequences, test_logits, test_pred_csv, test_scores = amp_predict_test(
                                                                                        model,
                                                                                        indep_testing_dataloader,
                                                                                        device,
                                                                                        weights_file,
                                                                                        threshold,
                                                                                        has_targets
                                                                                    )
        
        # Almacenar secuencias y targets (solo una vez)
        if not sequences:
            sequences = test_sequences
            
        # Almacenar las puntuaciones del modelo
        all_scores.append(test_scores)

# Convertir las predicciones en un DataFrame
if has_targets:
    df = pd.DataFrame({
                        'Sequence': sequences,
                        'Target': targets,
                        'Scores model 1': all_scores[0],
                        'Scores model 2': all_scores[1],
                        'Scores model 3': all_scores[2],
                        'Scores model 4': all_scores[3],
                        'Scores model 5': all_scores[4],
                    })
else:
    df = pd.DataFrame({
                        'Sequence': sequences,
                        'Scores model 1': all_scores[0],
                        'Scores model 2': all_scores[1],
                        'Scores model 3': all_scores[2],
                        'Scores model 4': all_scores[3],
                        'Scores model 5': all_scores[4],
                    })

# Calcular el promedio y la desviación estándar de las predicciones
df['Average score'] = df[['Scores model 1', 'Scores model 2', 'Scores model 3', 'Scores model 4', 'Scores model 5']].mean(axis=1)
df['Standard deviation'] = df[['Scores model 1', 'Scores model 2', 'Scores model 3', 'Scores model 4', 'Scores model 5']].std(axis=1)

# Guardar el DataFrame en un archivo CSV
df.to_excel('results/Independent/AMP/amp_prediction_independet_set.xlsx', index=False)

print('\n ///// File Created with Predictions, Average Score, and Standard Deviation /////')

if has_targets:
    # Evaluar el rendimiento usando el promedio de los scores
    y_true = df['Target'].values
    y_scores = df['Average score'].values
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calcular métricas de desempeño
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    conf_mat = confusion_matrix(y_true, y_pred)
    
    # Calcular la curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    tn, fp, fn, tp = conf_mat.ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # (Aquí van todos tus prints de métricas)
    print(f'\n///////// Evaluation Metrics using Average Scores ////////')
    print(f"- AUC using roc_auc_score: {auc_roc:.4f}")
    print(f"- AUC using auc: {roc_auc:.4f}")
    print(f"- Accuracy: {acc:.4f}")
    print(f"- Precision: {prec:.4f}")
    print(f"- Recall: {rec:.4f}")
    print(f"- F1-score: {f1:.4f}")
    print(f"- Average Precision: {ap:.4f}")
    print(f"- Specificity: {specificity:.4f}")
    print(f"- MCC: {mcc:.4f}")
    
    # Save the evaluation metrics using pandas
    evaluation_metrics_avg = {
                                'Metric': ['AUC (roc_auc_score)', 'AUC (auc)', 'Accuracy', 'Precision', 'Recall', 
                                        'F1-score', 'Average Precision', 'Specificity', 'MCC'],
                                'Value': [auc_roc, roc_auc, acc, prec, rec, f1, ap, specificity, mcc]
                            }
    
    # Convert to a DataFrame
    df_metrics_avg = pd.DataFrame(evaluation_metrics_avg)
    
    # Save as a CSV file
    df_metrics_avg.to_csv('results/Independent/AMP/evaluation_metrics.csv', index=False)
    
    
    # Guardar la curva ROC
    roc_curve_path = 'results/Independent/AMP/roc_curve.png'
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='green', lw=2, label=f'AUC-ROC = {auc_roc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate', fontsize=26, labelpad=15)
    plt.ylabel('True Positive Rate', fontsize=26, labelpad=15)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=20, pad=30)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(roc_curve_path)  # Guardar la figura
    plt.show()
    
    # Guardar la matriz de confusión
    conf_matrix_path = 'results/Independent/AMP/confusion_matrix.png'
    cmap = plt.get_cmap('YlGn')
    plt.figure(figsize=(10, 7))
    plt.imshow(conf_mat, cmap=cmap, interpolation='nearest')
    plt.title('Confusion Matrix - Testing', fontsize=20, pad=20)
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'], fontsize=16)
    plt.yticks([0, 1], ['Target 0', 'Target 1'], fontsize=16)
    
    # Añadir valores en las celdas
    for i in range(2):
        for j in range(2):
            plt.text(j, i, conf_mat[i, j], horizontalalignment='center', color='black', fontsize=24)
            
    plt.xlabel('Predicted', fontsize=26, labelpad=15)
    plt.ylabel('Target', fontsize=26, labelpad=15)
    plt.grid(False)
    plt.savefig(conf_matrix_path)  # Guardar la figura
    plt.show()


# %%
