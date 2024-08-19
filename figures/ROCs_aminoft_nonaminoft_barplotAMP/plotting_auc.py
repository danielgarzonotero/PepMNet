#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import re

file_paths = [
    'egconv_testing_prediction (0).xlsx',
    'egconv_testing_prediction (1).xlsx',
    'egconv_testing_prediction (2).xlsx',
    'gat_testing_prediction (0).xlsx',
    'gat_testing_prediction (1).xlsx',
    'gat_testing_prediction (2).xlsx',
    'gcnconv_testing_prediction (0).xlsx',
    'gcnconv_testing_prediction (1).xlsx',
    'gcnconv_testing_prediction (2).xlsx',
    'sageconv_testing_prediction (0).xlsx',
    'sageconv_testing_prediction (1).xlsx',
    'sageconv_testing_prediction (2).xlsx',
    'transformerconv_testing_prediction (0).xlsx',
    'transformerconv_testing_prediction (1).xlsx',
    'transformerconv_testing_prediction (2).xlsx',
    'arma_testing_prediction (0).xlsx',
    'arma_testing_prediction (1).xlsx',
    'arma_testing_prediction (2).xlsx'
] 


# Función para leer datos de un archivo Excel
def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Diccionario para almacenar resultados
results = {}

# Leer datos de cada archivo y calcular AUC
for file_path in file_paths:
    df = read_excel_data(file_path)
    
    # Suponiendo que 'Target' es la columna con las etiquetas verdaderas
    y_true = df['Target']
    
    # Suponiendo que 'Scores' es la columna con las puntuaciones predichas
    y_scores = df['Scores']
    
    # Calcular la curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Calcular el AUC
    roc_auc = auc(fpr, tpr)
    
    # Extraer y formatear el label del archivo
    file_name = os.path.basename(file_path)
    label_match = re.match(r'^(.*?)_', file_name)  # Extraer la palabra antes del guión bajo
    if label_match:
        label = label_match.group(1).upper()
        label = re.sub(r'CONV', 'Conv', label)
        label = re.sub(r'TRANSFORMER', 'Transformer', label)  # Reemplazar 'CON' con 'Con'
        label = re.sub(r'NOAMINOFT-', '*No_ft-', label)  # Reemplazar 'NOAMINOFT' con ''
    else:
        label = file_name.split('_')[0].upper()
        label = re.sub(r'CONV', 'Conv', label)
        label = re.sub(r'TRANSFORMER', 'Transformer', label)  # Reemplazar 'CON' con 'Con'
        label = re.sub(r'NOAMINOFT-', '*No_ft-', label)  # Reemplazar 'NOAMINOFT' con ''
    
    # Almacenar resultados en el diccionario
    if label not in results:
        results[label] = {'fprs': [], 'tprs': [], 'aucs': []}
    
    results[label]['fprs'].append(fpr)
    results[label]['tprs'].append(tpr)
    results[label]['aucs'].append(roc_auc)

# Ajustar el tamaño de los ejes y etiquetas
plt.rc('axes', labelsize=18)  # Tamaño de las etiquetas de los ejes
plt.rc('xtick', labelsize=18)  # Tamaño de las etiquetas del eje X
plt.rc('ytick', labelsize=18)  # Tamaño de las etiquetas del eje Y
plt.rc('legend', fontsize=16)  # Tamaño de las etiquetas de la leyenda

# Crear la figura
fig, ax = plt.subplots(figsize=(9, 7))  # Aumentar el tamaño de la figura

# Plotear las curvas ROC
for label, data in results.items():
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(data['fprs'], data['tprs'])], axis=0)
    mean_auc = np.mean(data['aucs'])
    std_auc = np.std(data['aucs'])
    ax.plot(mean_fpr, mean_tpr, linewidth=2.5, label=f'{label}')
    #ax.plot(mean_fpr, mean_tpr, linewidth=2.5, label=f'{label} (AUC = {mean_auc:.2f} ± {std_auc:.3e})')

# Configuración del gráfico
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)  # Línea diagonal de referencia
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=24, labelpad=20)  # Aumentar el espacio con labelpad
ax.set_ylabel('True Positive Rate', fontsize=24, labelpad=20)  # Aumentar el espacio con labelpad
#ax.set_title('ROC Curve for Antimicrobial Classification: Comparing Graph Convolution Layers at the Amino Acid Level', fontsize=20, pad=30)  # Aumentar el espacio con pad
ax.grid()
ax.legend(loc='lower right', fontsize=18)

# Crear una leyenda personalizada
#legend_texts = [f'{label} = {np.mean(data["aucs"]):.4f} ± {np.std(data["aucs"]):.4f}' for label, data in results.items()]

# Crear la leyenda personalizada
''' legend = ax.legend(
    legend_texts,
    loc='center left',  # Colocar la leyenda a la izquierda
    frameon=False,      # No añadir un marco alrededor de la leyenda
    #title='AUC:',       # Título de la leyenda
    title_fontsize='18',  # Tamaño del título de la leyenda
    fontsize='20',  # Tamaño de la letra en la leyenda
    borderpad=1.5,  # Espacio entre la leyenda y el cuadro
    labelspacing=1.5,  # Espacio entre cada línea de texto en la leyenda
    bbox_to_anchor=(1.05, 0.5),  # Ajustar la posición de la leyenda fuera de la figura
    handletextpad=1.2  # Espacio entre los cuadros de la leyenda y el texto
) '''

# Ajustar los márgenes para que la leyenda no se superponga
plt.subplots_adjust(right=0.7)

# Mostrar el gráfico
plt.savefig('roc_amp.png', dpi=300, bbox_inches='tight', facecolor='white')
  # Guardar el gráfico como archivo PNG
plt.show()

# %%
