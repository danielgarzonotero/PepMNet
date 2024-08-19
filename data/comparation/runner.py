#%%
from distribution import distribution_amp, distribution

print(distribution_amp(
                    property=1,
                    path='/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/AMP/dataset Ruiz/Train_clean.csv',
                    bins=10
                    )
    )

print(distribution(
                    property=2,
                    path='/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/RT/hela_mod.csv',
                    bins=50
                    )
    )

    # 0 : aa average concentration
    # 1: Length             # 2: Charge              # 3: Amphiphilicity           
    # 4: Hydropathy         # 5: Secondary Structure # 6: FAI
    # 7: Molecular Weight   # 8: Hydrophobicity      # 9: Aromaticity
    # 10: Isoelectric Point # 11: Instability Index  
    
    
# %%
import pandas as pd

def top_n_longest_and_shortest_sequences_with_labels(csv_path, n):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    
    # Asegurarse de que las columnas 'Sequence' y 'Label' existan
    if 'Sequence' not in df.columns or 'Label' not in df.columns:
        raise ValueError("Las columnas 'Sequence' y/o 'Label' no se encuentran en el archivo CSV.")
    
    # Calcular las longitudes de las secuencias y añadirlas como una nueva columna
    df['Length'] = df['Sequence'].apply(len)
    
    # Obtener las n secuencias más largas y sus índices
    top_n_longest = df.nlargest(n, 'Length')
    
    # Obtener las n secuencias más cortas y sus índices
    top_n_shortest = df.nsmallest(n, 'Length')
    
    # Obtener las filas originales (índices), longitudes, secuencias y etiquetas para las más largas
    top_n_longest_indices = top_n_longest.index.tolist()
    top_n_longest_lengths = top_n_longest['Length'].tolist()
    top_n_longest_sequences = top_n_longest['Sequence'].tolist()
    top_n_longest_labels = top_n_longest['Label'].tolist()
    
    # Obtener las filas originales (índices), longitudes, secuencias y etiquetas para las más cortas
    top_n_shortest_indices = top_n_shortest.index.tolist()
    top_n_shortest_lengths = top_n_shortest['Length'].tolist()
    top_n_shortest_sequences = top_n_shortest['Sequence'].tolist()
    top_n_shortest_labels = top_n_shortest['Label'].tolist()
    
    return (top_n_longest_sequences, top_n_longest_lengths, top_n_longest_indices, top_n_longest_labels), \
           (top_n_shortest_sequences, top_n_shortest_lengths, top_n_shortest_indices, top_n_shortest_labels)

# Ejemplo de uso
csv_path = '/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/AMP/dataset Ruiz/Test_clean.csv'
n = 100
(top_n_longest_sequences, top_n_longest_lengths, top_n_longest_indices, top_n_longest_labels), \
(top_n_shortest_sequences, top_n_shortest_lengths, top_n_shortest_indices, top_n_shortest_labels) = top_n_longest_and_shortest_sequences_with_labels(csv_path, n)

print(f"Top {n} secuencias más largas, sus filas y etiquetas (AMP/nonAMP):")
for i in range(n):
    label = "AMP" if top_n_longest_labels[i] == 1 else "nonAMP"
    print(f"Fila: {top_n_longest_indices[i]}, Longitud: {top_n_longest_lengths[i]}, Etiqueta: {label}, Secuencia: {top_n_longest_sequences[i]}")

print(f"\nTop {n} secuencias más cortas, sus filas y etiquetas (AMP/nonAMP):")
for i in range(n):
    label = "AMP" if top_n_shortest_labels[i] == 1 else "nonAMP"
    print(f"Fila: {top_n_shortest_indices[i]}, Longitud: {top_n_shortest_lengths[i]}, Etiqueta: {label}, Secuencia: {top_n_shortest_sequences[i]}")

# %%
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Ejemplo de secuencia proteica
sequence = "APVDNRDHNEEMVTRCIIEVLSNALSKSSVPTITPECRQVLKKSGKEVKGEEKGENQNSKFEVRLLRDPADASGTRWASSREDAGAPVEDSQGQTKVGNEKWTEGGGHSREGVDDQESLRPSNQQASKEAKIYHSEERVGKEREKEEGKIYPMGEHREDAGEEKKHIEDSGEKPNTFSNKRSEASAKKKDESVARADAHSMELEEKTHSREQSSQESGEETRRQEKPQELTDQDQSQEESQEGEEGEEGEEGEEGEEDSASEVTKRRPRHHHGRSGSNKSSYEGHPLSEERRPSPKESKEADVATVRLGEKRSHHLAHYRASEEEPEYGEESRSYRGLQYRGRGSEEDRAPRPRSEESQEREYKRNHPDSELESTANRHGEETEEERSYEGANGRQHRGRGREPGAHSALDTREEKRLLDEGHYPVRESPIDTAKRYPQSKWQEQEKNYLNYGEEGDQGRWWQQEEQLGPEESREEVRFPDRQYEPYPITEKRKRLGALFNPYFDPLQWKNSDFEKRGNPDDSFLEDEGEDRNGVTLTEKNSFPEYNYDWWERRPFSEDVNWGYEKRSFARAPQLDLKRQYDGVAELDQLLHYRKKADEFPDFYDSEEQMGPHQEANDEKARADQRVLTAEEKKELENLAAMDLELQKIAEKFSQRG"

# Crear un objeto ProteinAnalysis
analizador = ProteinAnalysis(sequence)

# Calcular la carga a pH 7
carga_pH7 = analizador.charge_at_pH(7)

# Imprimir el resultado
print(f"Carga neta a pH 7: {carga_pH7:.2f}")


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

# Lee el archivo CSV
file_path = '/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/RT/hela_mod.csv'
#file_path = '/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/RT/scx.csv'
#file_path = '/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/RT/xbridge_amide.csv'
#file_path = '/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/Hierarchical-Graph-Neural-Network/data/RT/misc_dia.csv'

df = pd.read_csv(file_path)

# Limpia las secuencias en el DataFrame
df['cleaned_sequence'] = df['sequence'].apply(clean_sequence)

# Calcula la carga de las secuencias a pH 7 y agrega la nueva columna 'charge'
df['charge'] = df['cleaned_sequence'].apply(lambda x: ProteinAnalysis(x).charge_at_pH(7))

# Calcula la hidrofobicidad de las secuencias y agrega la nueva columna 'hydrophobicity'
df['hydrophobicity'] = df['cleaned_sequence'].apply(hydrophobicity)

# Calcula la longitud de las secuencias y agrega la nueva columna 'length'
df['length'] = df['cleaned_sequence'].apply(len)

# Función para crear un scatter plot con R²
def scatter_plot_with_r2(df, property_name,color, y_label, title):
    X = df[property_name].values.reshape(-1, 1)
    y = df['RT'].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = reg.score(X, y)

    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(df[property_name], df['RT'], alpha=0.2, color=color)

    # Agregar el R^2 como texto en la gráfica con un cuadro alrededor
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    plt.text(0.05, 0.95, f'R-Pearson = {r2:.3f}', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', bbox=bbox_props)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(title, fontsize=20, pad=30)
    plt.xlabel(property_name.capitalize(), fontsize=20, labelpad=10)
    plt.ylabel(y_label, fontsize=20, labelpad=10)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.show()

# Ejemplo de uso de la función
name_dataset = 'hela'
property_to_plot = 'hydrophobicity'  # Cambia esto a 'hydrophobicity' o 'length' según la propiedad que quieras graficar
scatter_plot_with_r2(df, property_to_plot, "grey" ,"Retention Time (min)", f'Scatter Plot RT vs {property_to_plot.capitalize()} \n{name_dataset} Dataset')

""" name_dataset = 'scx'
property_to_plot = 'charge'  # Cambia esto a 'hydrophobicity' o 'length' según la propiedad que quieras graficar
scatter_plot_with_r2(df, property_to_plot, "goldenrod", "Retention Time (min)", f'Scatter Plot RT vs {property_to_plot.capitalize()} \n{name_dataset} Dataset') """

""" name_dataset = "xbridge"
property_to_plot = 'hydrophobicity'  # Cambia esto a 'hydrophobicity' o 'length' según la propiedad que quieras graficar
scatter_plot_with_r2(df, property_to_plot, "darkcyan", "Retention Time (min)", f'Scatter Plot RT vs {property_to_plot.capitalize()} \n{name_dataset} Dataset') """

""" name_dataset = "misc"
property_to_plot = 'hydrophobicity'  # Cambia esto a 'hydrophobicity' o 'length' según la propiedad que quieras graficar
scatter_plot_with_r2(df, property_to_plot, "grey", "Retention Time (s)", f'Scatter Plot RT vs {property_to_plot.capitalize()} \n{name_dataset} Dataset') """
# %%
