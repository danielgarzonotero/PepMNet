#%%
import matplotlib.pyplot as plt
import numpy as np

# Nombres de los métodos
methods = ['TransformerConv', 'SAGEConv', 'GCNConv', 'NNConv', 'EGConv', 'GATConv', 'ARMA', 'PepMNet']

# AUC y desviación estándar para el primer conjunto de datos
auc_data1 = [0.9087, 0.9236, 0.9249, 0.9253, 0.9266, 0.9329, 0.9492, 0.9619]
std_data1 = [0.0083, 0.0047, 0.0090, 0.0010, 0.0026, 0.0024, 0.0008, 0.0010]

# AUC y desviación estándar para el segundo conjunto de datos (alineado con los métodos)
auc_data2 = [0.9439, 0.9482, 0.9087, 0, 0.9343, 0.9375, 0.9297, 0]
std_data2 = [0.0012, 0.0015, 0.0428, 0, 0.0030, 0.0044, 0.0133, 0]

# Ordenar las listas en función de auc_data1 de menor a mayor
sorted_indices = np.argsort(auc_data1)  # Índices ordenados de menor a mayor

methods = [methods[i] for i in sorted_indices]
auc_data1 = [auc_data1[i] for i in sorted_indices]
std_data1 = [std_data1[i] for i in sorted_indices]
auc_data2 = [auc_data2[i] for i in sorted_indices]
std_data2 = [std_data2[i] for i in sorted_indices]

# Colores para las barras, destacando "HierGraph"
colors1 = ['coral'] * (len(methods) - 1) + ['yellowgreen']  # Color para barras
colors2 = ['steelblue'] * (len(methods) - 1) + ['yellowgreen']  # Color para barras

# Gráfico de Barras Horizontales
y = np.arange(len(methods))  # Localización de las etiquetas en el eje Y
height = 0.35  # Ancho de las barras

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)  # Aumentar calidad de la figura con dpi

# Barras para el primer conjunto de datos
rects1 = ax.barh(y - height/2, auc_data1, height, label='Non-HierGraph: Atomic Level', xerr=std_data1, capsize=5, color=colors1)

# Barras para el segundo conjunto de datos
rects2 = ax.barh(y + height/2, auc_data2, height, label='Non-HierGraph: Amino Acid Level', xerr=std_data2, capsize=5, color=colors2)

# Añadir etiquetas y título
ax.set_xlabel('AUC-ROC', fontsize=24)
ax.set_yticks(y)
ax.set_yticklabels(methods, fontsize=20)
ax.set_xticklabels(ax.get_xticks(), fontsize=20 )
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=24, frameon=False) # Leyenda sin contorno

# Formato del eje X para mostrar 4 cifras decimales
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

# Ajustar los límites del eje X
ax.set_xlim(0.85, 1)

# Ajustar la figura para que haya más espacio para la leyenda
plt.subplots_adjust(bottom=0.30)  # Aumentar espacio en la parte inferior

# Guardar la figura como PNG
plt.savefig('bar_plot.png', bbox_inches='tight', dpi=300)

# Mostrar gráfico
plt.show()

# %%
