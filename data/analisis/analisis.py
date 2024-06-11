
#%%# %% /////////////////////Lenghts Distributions​////////////////////////////////
import pandas as pd
import matplotlib.pyplot as plt

# AMP Dataset
df_amp = pd.read_csv('/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/AMP-Peptide-Hierarchical-Graph-NN/data/AMP_labeled.csv', names=['sequence', 'activity'])

# Add a column with the length of each sequence
df_amp['len'] = df_amp['sequence'].apply(len)

# Calculate mean and standard deviation
mean_amp = df_amp['len'].mean()
std_amp = df_amp['len'].std()
df_amp.to_csv('AMP_analisis.csv', sep=',', index=False)

# Histogram of sequence lengths
values_amp, edges_amp, _ = plt.hist(df_amp['len'], bins=100, color="g") 
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of AMP Sequence Lengths')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_amp:.2f}\nStd: {std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

# ------------------------------Non-AMP Dataset------------------------------
df_nonamp = pd.read_csv('/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/AMP-Peptide-Hierarchical-Graph-NN/data/nonAMP_labeled.csv', names=['sequence', 'activity'])

# Add a column with the length of each sequence
df_nonamp['len'] = df_nonamp['sequence'].apply(len)

# Calculate mean and standard deviation
mean_nonamp = df_nonamp['len'].mean()
std_nonamp = df_nonamp['len'].std()
df_nonamp.to_csv('nonAMP_analisis.csv', sep=',', index=False)

# Histogram of sequence lengths
values_nonamp, edges_nonamp, _ = plt.hist(df_nonamp['len'], bins=100, color="firebrick") 
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Non-AMP Sequence Lengths')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_nonamp:.2f}\nStd: {std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

# %% /////////////////////Amino Acid Frequency in the Sequences – Length No Included​////////////////////////////////
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Lista de los 20 aminoácidos
aminoacidos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
# ------------------------------AMP Dataset------------------------------
df_amp = pd.read_csv('AMP_analisis.csv')

# Crear columnas para cada aminoácido y contar su frecuencia en cada secuencia
for aa in aminoacidos:
    df_amp[aa] = df_amp['sequence'].apply(lambda x: x.count(aa))

df_amp.to_csv('AMP_analisis.csv', sep=',', index=False)

# ------------------------------NOnAMP Dataset------------------------------
df_namp = pd.read_csv('nonAMP_analisis.csv')

# Crear columnas para cada aminoácido y contar su frecuencia en cada secuencia
for aa in aminoacidos:
    df_namp[aa] = df_namp['sequence'].apply(lambda x: x.count(aa))

df_namp.to_csv('nonAMP_analisis.csv', sep=',', index=False)

#----------------------------Distributions-------------------------------------

def plot_aminoacid_distribution(df, dataset_name, aminoacidos, color):
    # Crear un DataFrame para facilitar el trazado
    df_plot = df[aminoacidos]

    # Calcular el promedio y la desviación estándar para cada aminoácido
    means = df_plot.mean().round(4)
    stds = df_plot.std().round(4)

    # Encontrar los 5 aminoácidos con la mayor y menor media
    top5_means = means.nlargest(5)
    bottom5_means = means.nsmallest(5)
    
    # Ordenar los datos de mayor a menor
    top5_means = top5_means.sort_values(ascending=False)
    bottom5_means = bottom5_means.sort_values(ascending=False)

    # Trazar gráfico de barras con barras de error
    plt.bar(aminoacidos, means, yerr=stds, capsize=5, alpha=0.7, color=color)

    # Añadir tabla con los 5 aminoácidos con mayor media
    table_data_top5 = pd.DataFrame(top5_means, columns=['Top 5 Max Mean'])
    table(ax=plt.gca(), data=table_data_top5, loc='bottom', bbox=[0, -0.65, 0.3, 0.4])

    # Añadir tabla con los 5 aminoácidos con menor media
    table_data_bottom5 = pd.DataFrame(bottom5_means, columns=['Bottom 5 Min Mean'])
    table(ax=plt.gca(), data=table_data_bottom5, loc='bottom', bbox=[0.7, -0.65, 0.3, 0.4])

    plt.xlabel('Amino Acid')
    plt.ylabel('Mean')
    plt.title(f'Amino Acid Mean({dataset_name})')
    plt.show()

# AMP Dataset
print('////////// Aminoacid Frecuecy in the sequences - Length NO Included /////////')

df_amp = pd.read_csv('AMP_analisis.csv')
plot_aminoacid_distribution(df_amp, 'AMP', aminoacidos, 'g')

# Non-AMP Dataset
df_nonamp = pd.read_csv('nonAMP_analisis.csv')
plot_aminoacid_distribution(df_nonamp, 'Non-AMP', aminoacidos, 'firebrick')


# %% # %% /////////////////////Amino Acid Frequency in the Sequences – Length Included​////////////////////////////////
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Lista de los 20 aminoácidos
aminoacidos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
# ------------------------------AMP Dataset------------------------------
df_amp = pd.read_csv('nonAMP_analisis13.csv')

# Crear columnas para cada aminoácido y contar su frecuencia en cada secuencia
for aa in aminoacidos:
    df_amp[aa] = df_amp['sequence'].apply(lambda x: x.count(aa))/(df_amp['sequence'].apply(len))

df_amp.to_csv('new_nonAMP_analisis13.csv', sep=',', index=False)

# ------------------------------NOnAMP Dataset------------------------------
''' df_namp = pd.read_csv('nonAMP_analisis.csv')

# Crear columnas para cada aminoácido y contar su frecuencia en cada secuencia
for aa in aminoacidos:
    df_namp[aa] = df_namp['sequence'].apply(lambda x: x.count(aa))/df_amp['sequence'].apply(len)

df_namp.to_csv('nonAMP_analisis2.csv', sep=',', index=False) '''

#----------------------------Distributions-------------------------------------

def plot_aminoacid_distribution(df, dataset_name, aminoacidos, color):
    # Crear un DataFrame para facilitar el trazado
    df_plot = df[aminoacidos]

    # Calcular el promedio y la desviación estándar para cada aminoácido
    means = df_plot.mean().round(4)
    stds = df_plot.std().round(4)

    # Encontrar los 5 aminoácidos con la mayor y menor media
    top5_means = means.nlargest(5)
    bottom5_means = means.nsmallest(5)
    # Ordenar los datos de mayor a menor
    top5_means = top5_means.sort_values(ascending=False)
    bottom5_means = bottom5_means.sort_values(ascending=False)

    # Trazar gráfico de barras con barras de error
    plt.bar(aminoacidos, means, yerr=stds, capsize=5, alpha=0.7, color=color)

    # Añadir tabla con los 5 aminoácidos con mayor media
    table_data_top5 = pd.DataFrame(top5_means, columns=['Top 5 Max Mean'])
    table(ax=plt.gca(), data=table_data_top5, loc='bottom', bbox=[0, -0.65, 0.3, 0.4])

    # Añadir tabla con los 5 aminoácidos con menor media
    table_data_bottom5 = pd.DataFrame(bottom5_means, columns=['Bottom 5 Min Mean'])
    table(ax=plt.gca(), data=table_data_bottom5, loc='bottom', bbox=[0.7, -0.65, 0.3, 0.4])

    plt.xlabel('Amino Acid')
    plt.ylabel('Mean/len(Sequence)')
    plt.title(f'Amino Acid Mean/len(Sequence) ({dataset_name})')
    plt.show()

# AMP Dataset
''' print('////////// Aminoacid Frecuecy in the sequences - Length Included /////////')
df_amp = pd.read_csv('AMP_analisis2.csv')
plot_aminoacid_distribution(df_amp, 'AMP', aminoacidos, 'g') '''

# Non-AMP Dataset
df_nonamp = pd.read_csv('new_nonAMP_analisis13.csv')
plot_aminoacid_distribution(df_nonamp, 'Non-AMP', aminoacidos, 'firebrick')



# %%# %% /////////////////////Charge Distribution at pH 7​////////////////////////////////

from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ AMP Dataset-------------------------------------------
df_amp = pd.read_csv('AMP_analisis13.csv')

# Add a column with the length of each sequence
df_amp['net_charge@pH_7'] = df_amp['sequence'].apply(lambda x: ProteinAnalysis(x).charge_at_pH(7))

# Calculate mean and standard deviation
mean_amp = df_amp['net_charge@pH_7'].mean()
std_amp = df_amp['net_charge@pH_7'].std()
df_amp.to_csv('AMP_analisis13.csv', sep=',', index=False)

# Histogram of sequence lengths
values_amp, edges_amp, _ = plt.hist(df_amp['net_charge@pH_7'], bins=100, color="g") 
plt.xlabel('Sequences Charge')
plt.ylabel('Frequency')
plt.title('Distribution of AMP Sequence Charge')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_amp:.2f}\nStd: {std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

# ------------------------------Non-AMP Dataset------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis13.csv')

# Add a column with the length of each sequence
df_nonamp['net_charge@pH_7'] = df_nonamp['sequence'].apply(lambda x: ProteinAnalysis(x).charge_at_pH(7))

# Calculate mean and standard deviation
mean_nonamp = df_nonamp['net_charge@pH_7'].mean()
std_nonamp = df_nonamp['net_charge@pH_7'].std()
df_nonamp.to_csv('nonAMP_analisis13.csv', sep=',', index=False)

# Histogram of sequence lengths
values_nonamp, edges_nonamp, _ = plt.hist(df_nonamp['net_charge@pH_7'], bins=100, color="firebrick") 
plt.xlabel('Sequences Charge')
plt.ylabel('Frequency')
plt.title('Distribution of nonAMP Sequence Charge')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_nonamp:.2f}\nStd: {std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

#%%//////////////// Concentrations of amino acids possess a charge at neutral pH ///////////////////////////
import pandas as pd

df_amp = pd.read_csv('AMP_analisis13.csv')
df_nonamp = pd.read_csv('new_nonAMP_analisis13.csv')
aminoacidos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# ------------------------------Print and Save Average Concentrations------------------------------
amp_avg_concentrations = df_amp[aminoacidos].mean().sort_values(ascending=False)
nonamp_avg_concentrations = df_nonamp[aminoacidos].mean().sort_values(ascending=False)

# Crear DataFrames con las concentraciones promedio
amp_avg_concentrations_df = pd.DataFrame({
    'Amino Acid': amp_avg_concentrations.index,
    'AMP Average Concentration': amp_avg_concentrations.round(4)
})

nonamp_avg_concentrations_df = pd.DataFrame({
    'Amino Acid': nonamp_avg_concentrations.index,
    'Non-AMP Average Concentration': nonamp_avg_concentrations.round(4)
})

# Imprimir y guardar tablas como archivos CSV
amp_avg_concentrations_df.to_csv('amp_average_concentrations.csv', index=False)
nonamp_avg_concentrations_df.to_csv('nonamp_average_concentrations.csv', index=False)

df_amp_avg_concentrations = pd.read_csv('amp_average_concentrations.csv')
df_nonamp_avg_concentrations = pd.read_csv('nonamp_average_concentrations.csv')

# Calcular y imprimir la varianza de la segunda columna de los DataFrames
variance_amp = df_amp_avg_concentrations.iloc[:, 1].std()
variance_nonamp = df_nonamp_avg_concentrations.iloc[:, 1].std()

print(f"\nσ AMP: {variance_amp}")
print(f"σ Non-AMP: {variance_nonamp}")

# %% /////////// Correlation of amino acids and Net Charge /////////////////////

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Especificar el aminoacido
aminoacido = 'E'

# ------------------AMP Dataset---------------------
df_amp = pd.read_csv('AMP_analisis13.csv')

r2 = r2_score(df_amp[aminoacido], df_amp['net_charge@pH_7'])
r, _ = pearsonr(df_amp[aminoacido], df_amp['net_charge@pH_7'])

# Graficar scatter plot
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}".format(r2, r)
plt.scatter(df_amp[aminoacido], df_amp['net_charge@pH_7'], color='g')
plt.xlabel('[{}]'.format(aminoacido))
plt.ylabel('Net Charge at pH 7')
plt.title('[{}] vs Net Charge at pH 7 - AMP'.format(aminoacido))

# Añadir línea de tendencia
coefficients = np.polyfit(df_amp[aminoacido], df_amp['net_charge@pH_7'], 1)
polynomial = np.poly1d(coefficients)
plt.plot(df_amp[aminoacido], polynomial(df_amp[aminoacido]), color='blue', linestyle='--')

plt.legend([legend_text], loc="lower right")
plt.show()

# ------------------nonAMP Dataset---------------------
df_amp = pd.read_csv('nonAMP_analisis13.csv')
r2 = r2_score(df_amp[aminoacido], df_amp['net_charge@pH_7'])
r, _ = pearsonr(df_amp[aminoacido], df_amp['net_charge@pH_7'])

# Graficar scatter plot
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}".format(r2, r)
plt.scatter(df_amp[aminoacido], df_amp['net_charge@pH_7'], color='firebrick')
plt.xlabel('[{}]'.format(aminoacido))
plt.ylabel('Net Charge at pH 7')
plt.title('[{}] vs Net Charge at pH 7 - nonAMP'.format(aminoacido))

# Añadir línea de tendencia
coefficients = np.polyfit(df_amp[aminoacido], df_amp['net_charge@pH_7'], 1)
polynomial = np.poly1d(coefficients)
plt.plot(df_amp[aminoacido], polynomial(df_amp[aminoacido]), color='blue', linestyle='--')

plt.legend([legend_text], loc="lower right")
plt.show()


#%%////////////// To create a csv file with Hydropathy and Amphiphilicity //////////////////
import pandas as pd

# Datos
data = [
    ["Letter", "Hydropathy", "Amphiphilicity"],
    ["A", 1.8, 0],
    ["R", -4.5, 2.45],
    ["N", -3.5, 0],
    ["D", -3.5, 0],
    ["C", 2.5, 0],
    ["Q", -3.5, 1.25],
    ["E", -3.5, 1.27],
    ["G", -0.4, 0],
    ["H", -3.2, 1.45],
    ["I", 4.5, 0],
    ["L", 3.8, 0],
    ["K", -3.9, 3.67],
    ["M", 1.9, 0],
    ["F", 2.8, 0],
    ["P", -1.6, 0],
    ["S", -0.8, 0],
    ["T", -0.7, 0],
    ["W", -0.9, 6.93],
    ["Y", -1.3, 5.06],
    ["V", 4.2, 0],
]

# Crear un DataFrame de Pandas
df = pd.DataFrame(data[1:], columns=data[0])

# Guardar el DataFrame en un archivo CSV
df.to_csv('index.csv', sep=',', index=False)



# %%//////////////////// To add a column with the sum of Hydropathy and Amphiphilicity based od the amino acid sequence ///////////////////
import pandas as pd

# Leer el archivo CSV con las secuencias
df_secuencias = pd.read_csv('AMP_analisis5.csv')  # Reemplaza con la ruta correcta

# Leer el DataFrame original con la tabla de valores
df_valores = pd.read_csv('index.csv')  # Reemplaza con la ruta correcta

# Función para calcular la suma de valores para una secuencia dada
def calcular_suma(secuencia):
    return sum(df_valores.loc[df_valores['Letter'].isin(list(secuencia)), 'Amphiphilicity'])

# Agregar una nueva columna al DataFrame de secuencias con las sumas calculadas
df_secuencias['Sum_Amphiphilicity'] = df_secuencias['sequence'].apply(calcular_suma)

# Guardar el resultado en un nuevo archivo CSV
df_secuencias.to_csv('AMP_analisis6.csv', index=False)  # Reemplaza con la ruta deseada


#%% ///////////////////// Amphiphilicity Distributions​////////////////////////////////
import pandas as pd
import matplotlib.pyplot as plt

# AMP Dataset
df_amp = pd.read_csv('AMP_analisis6.csv')


# Calculate mean and standard deviation
mean_amp = df_amp['Sum_Amphiphilicity'].mean()
std_amp = df_amp['Sum_Amphiphilicity'].std()

# Histogram of sequence lengths
values_amp, edges_amp, _ = plt.hist(df_amp['Sum_Amphiphilicity'], bins=100, color="g") 
plt.xlabel('Sequence Amphiphilicity')
plt.ylabel('Frequency')
plt.title('Distribution of Sequence Amphiphilicity - AMP')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_amp:.2f}\nStd: {std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

# ------------------------------Non-AMP Dataset------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis6.csv')

# Calculate mean and standard deviation
mean_nonamp = df_nonamp['Sum_Amphiphilicity'].mean()
std_nonamp = df_nonamp['Sum_Amphiphilicity'].std()

# Histogram of sequence lengths
values_nonamp, edges_nonamp, _ = plt.hist(df_nonamp['Sum_Amphiphilicity'], bins=100, color="firebrick") 
plt.xlabel('Sequence Amphiphilicity')
plt.ylabel('Frequency')
plt.title('Distribution of Sequence Amphiphilicity - nonAMP')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_nonamp:.2f}\nStd: {std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()
#%% ///////////////////// Hydropathy Distributions​////////////////////////////////

import pandas as pd
import matplotlib.pyplot as plt

# AMP Dataset
df_amp = pd.read_csv('AMP_analisis6.csv')


# Calculate mean and standard deviation
mean_amp = df_amp['Sum_Hydropathy'].mean()
std_amp = df_amp['Sum_Hydropathy'].std()

# Histogram of sequence lengths
values_amp, edges_amp, _ = plt.hist(df_amp['Sum_Hydropathy'], bins=100, color="g") 
plt.xlabel('Sequence Hydropathy')
plt.ylabel('Frequency')
plt.title('Distribution of Sequence Hydropathy - AMP')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_amp:.2f}\nStd: {std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

# ------------------------------Non-AMP Dataset------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis6.csv')

# Calculate mean and standard deviation
mean_nonamp = df_nonamp['Sum_Hydropathy'].mean()
std_nonamp = df_nonamp['Sum_Hydropathy'].std()

# Histogram of sequence lengths
values_nonamp, edges_nonamp, _ = plt.hist(df_nonamp['Sum_Hydropathy'], bins=100, color="firebrick") 
plt.xlabel('Sequence Hydropathy')
plt.ylabel('Frequency')
plt.title('Distribution of Sequence Hydropathy - nonAMP')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_nonamp:.2f}\nStd: {std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

# %% //////////////////////// Secondary Structure Prediction //////////////////////////////////////////
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt

def predict_secondary_structure(peptide_sequence):
    protein_analysis = ProteinAnalysis(peptide_sequence)
    sheet, turn, helix = protein_analysis.secondary_structure_fraction()
    return sheet, turn, helix

# ------------------------ AMP Dataset-------------------------------------------
df_amp = pd.read_csv('AMP_analisis6.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_amp['helix'] = df_amp['sequence'].apply(lambda x: predict_secondary_structure(x)[2])
df_amp['turn'] = df_amp['sequence'].apply(lambda x: predict_secondary_structure(x)[1])
df_amp['sheet'] = df_amp['sequence'].apply(lambda x: predict_secondary_structure(x)[0])

# Calcular medias y desviaciones estándar
helix_mean_amp = df_amp['helix'].mean()
helix_std_amp = df_amp['helix'].std()
turn_mean_amp = df_amp['turn'].mean()
turn_std_amp = df_amp['turn'].std()
sheet_mean_amp = df_amp['sheet'].mean()
sheet_std_amp = df_amp['sheet'].std()

# Guardar el DataFrame modificado en un nuevo archivo CSV
df_amp.to_csv('AMP_analisis7.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values_helix, edges_helix, _ = plt.hist(df_amp['helix'], bins=100, color="g", alpha=0.9, label="Helix") 
values_turn, edges_turn, _ = plt.hist(df_amp['turn'], bins=100, color="b", alpha=0.5, label="Turn") 
values_sheet, edges_sheet, _ = plt.hist(df_amp['sheet'], bins=100, color="r", alpha=0.2, label="Sheet") 

plt.xlabel('Secondary Structure Fraction ',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Secondary Structure of AMP ',size= 15)

# Agregar leyenda
plt.legend(fontsize='20')

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.055, f"Helix: Mean={helix_mean_amp:.2f}, Std={helix_std_amp:.2f}\nTurn: Mean={turn_mean_amp:.2f}, Std={turn_std_amp:.2f}\nSheet: Mean={sheet_mean_amp:.2f}, Std={sheet_std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()

# ------------------------ nonAMP Dataset-------------------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis6.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_nonamp['helix'] = df_nonamp['sequence'].apply(lambda x: predict_secondary_structure(x)[2])
df_nonamp['turn'] = df_nonamp['sequence'].apply(lambda x: predict_secondary_structure(x)[1])
df_nonamp['sheet'] = df_nonamp['sequence'].apply(lambda x: predict_secondary_structure(x)[0])

# Calcular medias y desviaciones estándar
helix_mean_nonamp = df_nonamp['helix'].mean()
helix_std_nonamp = df_nonamp['helix'].std()
turn_mean_nonamp = df_nonamp['turn'].mean()
turn_std_nonamp = df_nonamp['turn'].std()
sheet_mean_nonamp = df_nonamp['sheet'].mean()
sheet_std_nonamp = df_nonamp['sheet'].std()

# Guardar el DataFrame modificado en un nuevo archivo CSV
df_nonamp.to_csv('nonAMP_analisis7.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values_helix, edges_helix, _ = plt.hist(df_nonamp['helix'], bins=100, color="g", alpha=0.9, label="Helix") 
values_turn, edges_turn, _ = plt.hist(df_nonamp['turn'], bins=100, color="b", alpha=0.5, label="Turn") 
values_sheet, edges_sheet, _ = plt.hist(df_nonamp['sheet'], bins=100, color="r", alpha=0.2, label="Sheet") 

plt.xlabel('Secondary Structure Fraction ',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Secondary Structure of nonAMP ',size= 15)

# Agregar leyenda
plt.legend(fontsize='20')

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.055, f"Helix: Mean={helix_mean_nonamp:.2f}, Std={helix_std_nonamp:.2f}\nTurn: Mean={turn_mean_nonamp:.2f}, Std={turn_std_nonamp:.2f}\nSheet: Mean={sheet_mean_nonamp:.2f}, Std={sheet_std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()



# %%
import Bio
print('Biopython version:', Bio.__version__)

# %% ////////////////////////// FAI /////////////////////////////////////////////////
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt

def fai(sequence):
    helm = peptide_to_helm(sequence)
    mol = Chem.MolFromHELM(helm)
    num_anillos = rdMolDescriptors.CalcNumRings(mol)
    
    charged_amino_acids = {'R': 2, 'H': 1, 'K': 2}
    cationic_charges = sum(sequence.count(aa) * charge for aa, charge in charged_amino_acids.items())

    # para evitar un error matemático
    if num_anillos == 0:
        return 0

    return (cationic_charges / num_anillos)


# Convertir a notación HELM para usar RDKit
def peptide_to_helm(sequence):
    polymer_id = 1
    sequence_helm = "".join(sequence)
    
    sequence_helm = ''.join([c + '.' if c.isupper() else c for i, c in enumerate(sequence_helm)])
    sequence_helm = sequence_helm.rstrip('.')
    
    sequence_helm = f"PEPTIDE{polymer_id}{{{sequence_helm}}}$$$$"
    
    return sequence_helm


# ------------------------ AMP Dataset-------------------------------------------
df_amp = pd.read_csv('AMP_analisis7.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_amp['FAI'] = df_amp['sequence'].apply(lambda x: fai(x))


# Calcular medias y desviaciones estándar
fai_mean_amp = df_amp['FAI'].mean()
fai_std_amp = df_amp['FAI'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_amp.to_csv('AMP_analisis8.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values_fai, edges_fai, _ = plt.hist(df_amp['FAI'], bins=100, color="g", alpha=0.9) 
plt.xlabel('FAI',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of FAI - AMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={fai_mean_amp:.2f}\nStd={fai_std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()


# ------------------------ nonAMP Dataset-------------------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis7.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_nonamp['FAI'] = df_nonamp['sequence'].apply(lambda x: fai(x))


# Calcular medias y desviaciones estándar
fai_mean_nonamp = df_nonamp['FAI'].mean()
fai_std_nonamp = df_nonamp['FAI'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_nonamp.to_csv('nonAMP_analisis8.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values_fai, edges_fai, _ = plt.hist(df_nonamp['FAI'], bins=100, color="r", alpha=0.9) 
plt.xlabel('FAI',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of FAI - nonAMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={fai_mean_nonamp:.2f}\nStd={fai_std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()

# %%//////////////////////////// Molecular Weight //////////////////////////////////
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt

def mw_peptide(peptide_sequence):
    peptide_analysis = ProteinAnalysis(peptide_sequence)
    mw_peptide = peptide_analysis .molecular_weight()
    
    return mw_peptide

# ------------------------ AMP Dataset-------------------------------------------
df_amp = pd.read_csv('AMP_analisis8.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_amp['molecular_weight'] = df_amp['sequence'].apply(lambda x: mw_peptide(x))


# Calcular medias y desviaciones estándar
mw_mean_amp = df_amp['molecular_weight'].mean()
mw_std_amp = df_amp['molecular_weight'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_amp.to_csv('AMP_analisis9.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values_fai, edges_fai, _ = plt.hist(df_amp['molecular_weight'], bins=100, color="g", alpha=0.9) 
plt.xlabel('Molecular Weight [Da]',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Molecular Weight - AMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={mw_mean_amp:.2f}\nStd={mw_std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()


# ------------------------ nonAMP Dataset-------------------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis8.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_nonamp['molecular_weight'] = df_nonamp['sequence'].apply(lambda x: mw_peptide(x))


# Calcular medias y desviaciones estándar
mw_mean_nonamp = df_nonamp['molecular_weight'].mean()
mw_std_nonamp = df_nonamp['molecular_weight'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_nonamp.to_csv('nonAMP_analisis9.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values_fai, edges_fai, _ = plt.hist(df_nonamp['molecular_weight'], bins=100, color="r", alpha=0.9) 
plt.xlabel('Molecular Weight [Da]',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Molecular Weight - nonAMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={mw_mean_nonamp:.2f}\nStd={mw_std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()



# %%
# %%//////////////////////////// Hydrophobicity //////////////////////////////////
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt

def hidrophobicity(peptide_sequence):
    peptide_analysis = ProteinAnalysis(peptide_sequence)
    hidrophobicity = peptide_analysis.gravy()
    
    return hidrophobicity

# ------------------------ AMP Dataset-------------------------------------------
df_amp = pd.read_csv('AMP_analisis9.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_amp['hidrophobicity'] = df_amp['sequence'].apply(lambda x: hidrophobicity(x))


# Calcular medias y desviaciones estándar
hidro_mean_amp = df_amp['hidrophobicity'].mean()
hidro_std_amp = df_amp['hidrophobicity'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_amp.to_csv('AMP_analisis10.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values, edges, _ = plt.hist(df_amp['hidrophobicity'], bins=100, color="g", alpha=0.9) 
plt.xlabel('Hydrophobicity',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Hydrophobicity - AMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={hidro_mean_amp:.2f}\nStd={hidro_std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()


# ------------------------ nonAMP Dataset-------------------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis9.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_nonamp['hidrophobicity'] = df_nonamp['sequence'].apply(lambda x: hidrophobicity(x))


# Calcular medias y desviaciones estándar
hidro_mean_nonamp = df_nonamp['hidrophobicity'].mean()
hidro_std_nonamp = df_nonamp['hidrophobicity'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_nonamp.to_csv('nonAMP_analisis10.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values, edges, _ = plt.hist(df_nonamp['hidrophobicity'], bins=100, color="r", alpha=0.9) 
plt.xlabel('Hydrophobicity',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Hydrophobicity - nonAMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={hidro_mean_nonamp:.2f}\nStd={hidro_std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()

# %%//////////////////////////// Aromaticity//////////////////////////////////
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt

def aromaticity(peptide_sequence):
    peptide_analysis = ProteinAnalysis(peptide_sequence)
    aromaticity = peptide_analysis.aromaticity()
    
    return aromaticity

# ------------------------ AMP Dataset-------------------------------------------
df_amp = pd.read_csv('AMP_analisis10.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_amp['aromaticity'] = df_amp['sequence'].apply(lambda x: aromaticity(x))


# Calcular medias y desviaciones estándar
aro_mean_amp = df_amp['aromaticity'].mean()
aro_std_amp = df_amp['aromaticity'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_amp.to_csv('AMP_analisis11.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values_fai, edges_fai, _ = plt.hist(df_amp['aromaticity'], bins=100, color="g", alpha=0.9) 
plt.xlabel('Aromaticity',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Aromaticity- AMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={aro_mean_amp:.2f}\nStd={aro_std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()


# ------------------------ nonAMP Dataset-------------------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis10.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_nonamp['aromaticity'] = df_nonamp['sequence'].apply(lambda x: aromaticity(x))


# Calcular medias y desviaciones estándar
aro_mean_nonamp = df_nonamp['aromaticity'].mean()
aro_std_nonamp = df_nonamp['aromaticity'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_nonamp.to_csv('nonAMP_analisis11.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values, edges, _ = plt.hist(df_nonamp['aromaticity'], bins=100, color="r", alpha=0.9) 
plt.xlabel('Aromaticity',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Aromaticity- nonAMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={aro_mean_nonamp:.2f}\nStd={aro_std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()


# %%//////////////////////////// Isoelectric_Point//////////////////////////////////
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt

def isoelectric_point(peptide_sequence):
    peptide_analysis = ProteinAnalysis(peptide_sequence)
    isoelectric_point = peptide_analysis.isoelectric_point()
    
    return isoelectric_point

# ------------------------ AMP Dataset-------------------------------------------
df_amp = pd.read_csv('AMP_analisis11.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_amp['isoelectric_point'] = df_amp['sequence'].apply(lambda x: isoelectric_point(x))


# Calcular medias y desviaciones estándar
iso_mean_amp = df_amp['isoelectric_point'].mean()
iso_std_amp = df_amp['isoelectric_point'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_amp.to_csv('AMP_analisis12.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values_iso, edges_iso, _ = plt.hist(df_amp['isoelectric_point'], bins=100, color="g", alpha=0.9) 
plt.xlabel('Isoelectric Point',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Isoelectric Point - AMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={iso_mean_amp:.2f}\nStd={iso_std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()


# ------------------------ nonAMP Dataset-------------------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis11.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_nonamp['isoelectric_point'] = df_nonamp['sequence'].apply(lambda x: isoelectric_point(x))


# Calcular medias y desviaciones estándar
iso_mean_nonamp = df_nonamp['isoelectric_point'].mean()
iso_std_nonamp = df_nonamp['isoelectric_point'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_nonamp.to_csv('nonAMP_analisis12.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values_iso, edges_iso, _ = plt.hist(df_nonamp['isoelectric_point'], bins=100, color="r", alpha=0.9) 
plt.xlabel('Isoelectric Point',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Isoelectric Point - nonAMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={iso_mean_nonamp:.2f}\nStd={iso_std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()


# %%//////////////////////////// Instability Index //////////////////////////////////
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt

def instability_index(peptide_sequence):
    peptide_analysis = ProteinAnalysis(peptide_sequence)
    instability_index= peptide_analysis.instability_index()
    
    return instability_index

# ------------------------ AMP Dataset-------------------------------------------
df_amp = pd.read_csv('AMP_analisis12.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_amp['instability_index'] = df_amp['sequence'].apply(lambda x: instability_index(x))


# Calcular medias y desviaciones estándar
inex_mean_amp = df_amp['instability_index'].mean()
inex_std_amp = df_amp['instability_index'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_amp.to_csv('AMP_analisis13.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values, edges, _ = plt.hist(df_amp['instability_index'], bins=100, color="g", alpha=0.9) 
plt.xlabel('Instability Index',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Instability Index- AMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={inex_mean_amp:.2f}\nStd={inex_std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()


# ------------------------ nonAMP Dataset-------------------------------------------
df_nonamp = pd.read_csv('nonAMP_analisis12.csv')

# Añadir columnas con la longitud y la estructura secundaria de cada secuencia
df_nonamp['instability_index'] = df_nonamp['sequence'].apply(lambda x: instability_index(x))


# Calcular medias y desviaciones estándar
inex_mean_nonamp = df_nonamp['instability_index'].mean()
inex_std_nonamp = df_nonamp['instability_index'].std()


# Guardar el DataFrame modificado en un nuevo archivo CSV
df_nonamp.to_csv('nonAMP_analisis13.csv', sep=',', index=False)

# Histograma de las longitudes de las secuencias
plt.figure(figsize=(10, 6))

values, edges, _ = plt.hist(df_nonamp['instability_index'], bins=100, color="r", alpha=0.9) 
plt.xlabel('Instability Index',size= 15)
plt.ylabel('Frequency',size= 15)
plt.title('Distribution of Instability Index- nonAMP ',size= 15)

# Añadir texto con medias y desviaciones estándar
plt.text(0.95, 0.85, f"Mean={inex_mean_nonamp:.2f}\nStd={inex_std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

plt.show()


