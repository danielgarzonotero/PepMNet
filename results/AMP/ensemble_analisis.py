#%%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Cargar el archivo Excel (asegúrate de tener instalada la librería openpyxl si tu archivo es .xlsx)
df = pd.read_excel('indep_testing_predictions.xlsx')

# Calcular el error absoluto entre Target y Average score
df['Absolute Error'] = abs(df['Target'] - df['Average score'])

# Calcular correlaciones de Pearson y Spearman
pearson_corr, _ = pearsonr(df['Standard deviation'], df['Absolute Error'])
spearman_corr, _ = spearmanr(df['Standard deviation'], df['Absolute Error'])

# Crear el scatter plot con ajustes de etiquetas y tamaños
plt.figure(figsize=(8, 7))  # Aumenta el tamaño de la figura
scatter = plt.scatter(df['Standard deviation'], df['Absolute Error'], color='steelblue', alpha=0.7, label='Data Points')

# Ajustar etiquetas de los ejes con tamaños de fuente más grandes
plt.xlabel('Standard Deviation of Ensamble Scores', fontsize=18)
plt.ylabel('Absolute Error', fontsize=18)
plt.title('Scatter Plot of Absolute Error vs. Standard Deviation', fontsize=18, pad=12)

# Ajustar el tamaño de los números en los ejes
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Añadir leyenda con las correlaciones de Pearson y Spearman y ajustar el tamaño de la fuente
plt.legend(title=f'Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}', fontsize=18, title_fontsize=18, loc='lower right')

# Mostrar grid
plt.grid(True)

# Mostrar el gráfico
plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo Excel (reemplaza 'indep_testing_predictions.xlsx' con la ruta correcta)
df = pd.read_excel('indep_testing_predictions.xlsx')

# Extraer los puntajes de los modelos (columnas 3 a 7)
model_scores = df[['Scores model 1', 'Scores model 2', 'Scores model 3', 'Scores model 4', 'Scores model 5']]

# Calcular la correlación entre los modelos
correlation_matrix = model_scores.corr()

# Definir el mapa de colores
cmap = plt.get_cmap('YlGn')

# Crear el heatmap usando matplotlib
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(correlation_matrix, cmap=cmap, vmin=0.8, vmax=1) #TODO

# Añadir barra de color que representa los coeficientes de correlación
cbar = plt.colorbar(heatmap)
cbar.set_label('Correlation Coefficient', fontsize=14)

# Configurar los ejes con etiquetas para los modelos
plt.xticks(ticks=np.arange(5), labels=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], fontsize=12)
plt.yticks(ticks=np.arange(5), labels=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], fontsize=12)

# Etiquetas y título del gráfico
plt.xlabel('Models', fontsize=14)
plt.ylabel('Models', fontsize=14)
plt.title('Heatmap of Model Prediction Correlations', fontsize=16, pad=12)

# Mostrar los valores de correlación dentro del heatmap
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=10) 

plt.grid(False)
plt.show()



#%%
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV (reemplaza 'indep_testing_predictions.xlsx' con la ruta correcta)
df = pd.read_excel('indep_testing_predictions.xlsx')

# Extraer la columna de desviación estándar
std_devs = df['Standard deviation']

# Calcular el promedio y la desviación estándar
mean_std_dev = std_devs.mean()
std_dev_std_dev = std_devs.std()

# Crear el histograma
plt.figure(figsize=(5, 6))
plt.hist(std_devs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Standard Deviation', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title('Histogram of Standard Deviations')
plt.grid(False)

# Añadir la leyenda con el promedio y la desviación estándar
plt.legend([f'Standard Deviation Ensembles\nMean: {mean_std_dev:.4f}\nStd Dev: {std_dev_std_dev:.4f}'],
           loc='upper right', fontsize=12, frameon=False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Mostrar el gráfico
plt.show()




# %%
