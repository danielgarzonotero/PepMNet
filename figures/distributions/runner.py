#%%
from distribution import distribution_amp, distribution

print(distribution_amp(
                    property=1,
                    path='/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/AMP/dataset Ruiz/Train_clean.csv',
                    bins=10
                    )
    )

print(distribution_amp(
                    property=2,
                    path='/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/AMP/dataset Ruiz/Train_clean.csv',
                    bins=10
                    )
    )

print(distribution_amp(
                    property=8,
                    path='/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/AMP/dataset Ruiz/Train_clean.csv',
                    bins=10
                    )
    )


    # 0 : aa average concentration
    # 1: Length             # 2: Charge              # 3: Amphiphilicity           
    # 4: Hydropathy         # 5: Secondary Structure # 6: FAI
    # 7: Molecular Weight   # 8: Hydrophobicity      # 9: Aromaticity
    # 10: Isoelectric Point # 11: Instability Index  

# %% Scatter Plots RT vs Charge, Hydrophobicity and Length
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define la función de hidrofobicidad
def hydrophobicity(peptide_sequence):
    peptide_analysis = ProteinAnalysis(peptide_sequence)
    hydrophobicity = peptide_analysis.gravy()
    return hydrophobicity

# Función para limpiar la secuencia
def clean_sequence(peptide_sequence):
    sequence = peptide_sequence.replace("(ac)", "[ac]").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    return sequence

# Lee los archivos CSV y almacena los DataFrames en una lista
file_paths = [
    ('hela', '/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/RT/hela_mod.csv'),
    ('scx', '/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/RT/scx.csv'),
    ('xbridge', '/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/RT/xbridge_amide.csv'),
    ('misc', '/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/RT/misc_dia.csv')
]

dataframes = []

for name, file_path in file_paths:
    df = pd.read_csv(file_path)

    # Limpia las secuencias en el DataFrame
    df['cleaned_sequence'] = df['sequence'].apply(clean_sequence)

    # Calcula la carga de las secuencias a pH 7 y agrega la nueva columna 'charge'
    df['charge'] = df['cleaned_sequence'].apply(lambda x: ProteinAnalysis(x).charge_at_pH(7))

    # Calcula la hidrofobicidad de las secuencias y agrega la nueva columna 'hydrophobicity'
    df['hydrophobicity'] = df['cleaned_sequence'].apply(hydrophobicity)

    # Calcula la longitud de las secuencias y agrega la nueva columna 'length'
    df['length'] = df['cleaned_sequence'].apply(len)

    # Agrega el nombre del dataset al DataFrame para identificación
    df['dataset'] = name
    dataframes.append(df)

# Función para crear un scatter plot con R²
def scatter_plot_with_r2(df, property_name, color, y_label, title):
    X = df[property_name].values.reshape(-1, 1)
    y = df['RT'].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = reg.score(X, y)

    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(df[property_name], df['RT'], alpha=0.2, color=color)

    # Agregar el R^2 como texto en la gráfica con un cuadro alrededor
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', bbox=bbox_props)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(title, fontsize=20, pad=30)
    plt.xlabel(property_name.capitalize(), fontsize=20, labelpad=10)
    plt.ylabel(y_label, fontsize=20, labelpad=10)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.show()

# Genera las gráficas para cada DataFrame
plot_configs = [
    ('hela', 'hydrophobicity', "grey", "Retention Time (min)"),
    ('scx', 'charge', "goldenrod", "Retention Time (min)"),
    ('xbridge', 'hydrophobicity', "darkcyan", "Retention Time (min)"),
    ('misc', 'hydrophobicity', "grey", "Retention Time (s)")
]

for name_dataset, property_to_plot, color, y_label in plot_configs:
    df = next(df for df in dataframes if df['dataset'].iloc[0] == name_dataset)
    scatter_plot_with_r2(df, property_to_plot, color, y_label, f'Scatter Plot RT vs {property_to_plot.capitalize()} \n{name_dataset.capitalize()} Dataset')


# %%
