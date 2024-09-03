#%% ///////////////// Correlation Plots ////////////////////////////
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.linear_model import LinearRegression

# Define la función de hidrofobicidad
def hydrophobicity(peptide_sequence):
    peptide_analysis = ProteinAnalysis(peptide_sequence)
    return peptide_analysis.gravy()

# Función para limpiar la secuencia
def clean_sequence(peptide_sequence):
    sequence = peptide_sequence.replace("(ac)", "[ac]").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    return sequence

# Lee el archivo CSV para AMP
file_path_amp = 'indep_testing_predictions.csv'
df_amp = pd.read_csv(file_path_amp)

# Limpia las secuencias en el DataFrame
df_amp['cleaned_sequence'] = df_amp['Sequence'].apply(clean_sequence)

# Calcula la carga de las secuencias a pH 7 y agrega la nueva columna 'charge'
df_amp['charge'] = df_amp['cleaned_sequence'].apply(lambda x: ProteinAnalysis(x).charge_at_pH(7))

# Calcula la hidrofobicidad de las secuencias y agrega la nueva columna 'hydrophobicity'
df_amp['hydrophobicity'] = df_amp['cleaned_sequence'].apply(hydrophobicity)

# Calcula la longitud de las secuencias y agrega la nueva columna 'length'
df_amp['length'] = df_amp['Sequence'].apply(len)

# Lee los archivos CSV para SCX, RPLC, y HILIC
file_path_scx = 'SCX_prediction_amp.csv'
df_scx = pd.read_csv(file_path_scx)

file_path_rplc = 'yeast_RPLC_prediction_amp.csv'
df_rplc = pd.read_csv(file_path_rplc)

file_path_hilic = 'xbridge_HILIC_prediction_amp.csv'
df_hilic = pd.read_csv(file_path_hilic)

# Extraer RT para cada dataset
rt_scx = df_scx['RT(min)']
rt_rplc = df_rplc['RT(s)']
rt_hilic = df_hilic['RT(min)']

# Extraer columnas necesarias
scores = df_amp['Average score']
charge = df_amp['charge']
hydrophobicity_values = df_amp['hydrophobicity']
length = df_amp['length']
targets = df_amp['Target']

# Calcular las curvas ROC para cada conjunto de datos
fpr_scores, tpr_scores, _ = roc_curve(targets, scores)
roc_auc_scores = roc_auc_score(targets, scores)

fpr_scores_charge, tpr_scores_charge, _ = roc_curve(targets, charge)
roc_auc_scores_charge = roc_auc_score(targets, charge)

fpr_scores_hydrophobicity, tpr_scores_hydrophobicity, _ = roc_curve(targets, hydrophobicity_values)
roc_auc_scores_hydrophobicity = roc_auc_score(targets, hydrophobicity_values)

fpr_scores_length, tpr_scores_length, _ = roc_curve(targets, length)
roc_auc_scores_length = roc_auc_score(targets, length)

fpr_scx, tpr_scx, _ = roc_curve(targets, rt_scx)
roc_auc_scx = roc_auc_score(targets, rt_scx)

fpr_rplc, tpr_rplc, _ = roc_curve(targets, rt_rplc)
roc_auc_rplc = roc_auc_score(targets, rt_rplc)

fpr_hilic, tpr_hilic, _ = roc_curve(targets, rt_hilic)
roc_auc_hilic = roc_auc_score(targets, rt_hilic)

# Graficar las curvas ROC
plt.figure(figsize=(8, 7), dpi=300)  

plt.plot(fpr_scores, tpr_scores, color='mediumseagreen', lw=2, label=f'AMP Score = {roc_auc_scores:.3f}')
plt.plot(fpr_scores_charge, tpr_scores_charge, color='orange', lw=2, label=f'Charge = {roc_auc_scores_charge:.3f}')
plt.plot(fpr_scores_hydrophobicity, tpr_scores_hydrophobicity, color='tomato', lw=2, label=f'Hydrophobicity = {roc_auc_scores_hydrophobicity:.3f}')
plt.plot(fpr_scores_length, tpr_scores_length, color='lightseagreen', lw=2, label=f'Length = {roc_auc_scores_length:.3f}')
plt.plot(fpr_scx, tpr_scx, color='khaki', lw=2, label=f'RT Predicted - SCX model = {roc_auc_scx:.3f}')
plt.plot(fpr_rplc, tpr_rplc, color='steelblue', lw=2, label=f'RT Predicted - RPLC model = {roc_auc_rplc:.3f}')
plt.plot(fpr_hilic, tpr_hilic, color='lightgrey', lw=2, label=f'RT Predicted - HILIC model = {roc_auc_hilic:.3f}')

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=22, labelpad=5)
plt.ylabel('True Positive Rate', fontsize=22)
plt.title('ROC Curves: Scores, Charge, Hydrophobicity, Length, and Retention Times', fontsize=16, pad=30)

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22, title='AUC Values', title_fontsize='22')

plt.grid(True)

# Guardar la figura con DPI 300
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight' asegura que todo esté dentro del cuadro

# Mostrar la figura
plt.show()


# %%

import pandas as pd

# Leer el archivo CSV con el orden deseado
order_df = pd.read_csv('SCX_prediction_amp.csv')  # Este es el archivo con el orden deseado
order = order_df['Sequence'].tolist()  # Obtener la lista de secuencias en el orden deseado

# Leer el archivo CSV que necesitas reorganizar
data_df = pd.read_excel('indep_testing_predictions.xlsx')  # Este es el archivo que quieres organizar

# Asegurarse de que todas las secuencias del archivo a reorganizar estén en el archivo de orden
# Para evitar errores si hay secuencias faltantes en el archivo a reorganizar
data_df = data_df[data_df['Sequence'].isin(order)]

# Reordenar el dataframe de acuerdo al orden del archivo de referencia
data_df = data_df.set_index('Sequence').reindex(order).reset_index()

# Guardar el resultado en un nuevo archivo CSV
data_df.to_csv('indep_testing_predictions.csv', index=False)

# %%
