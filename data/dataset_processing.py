
#%% /////////////// Comberting 1 fasta file of amp or nonamp //////////////////////
import pandas as pd

def processing(path, activity, path_saving):
    df = pd.read_csv(path, header=None)
    
    df = df[~(df.index % 2 == 0)]
    df = df.reset_index(drop=True)

    df['Activity'] = activity
    
    #Checking:
    filas, columnas = df.shape
    print(f"-amp dataset has {filas} rows and {columnas} columns.")
    
    # Renombra las columnas como "Sequence" y "Activity"
    df.columns = ['Sequence', 'Activity']
    
    df.to_csv(path_saving, index=False, quoting=None)

    return

path = 'dataset Siu/Siu_test_amp.fasta'
path_saving= 'TESTING.csv'
activity = 1
processing(path, activity, path_saving)

#%% /////////////// Combining two fasta files of amp and nonamp //////////////////////
import pandas as pd

def processing(path_amp, path_nonamp, path_saving):
    df_amp = pd.read_csv(path_amp, header=None)
    df_nonamp = pd.read_csv(path_nonamp, header=None)
    
    df_amp = df_amp[~(df_amp.index % 2 == 0)]
    df_amp = df_amp.reset_index(drop=True)

    df_nonamp = df_nonamp[~(df_nonamp.index % 2 == 0)]
    df_nonamp = df_nonamp.reset_index(drop=True)
    
    df_amp['Activity'] = 1
    df_nonamp['Activity'] = 0
    
    #Checking:
    filas, columnas = df_amp.shape
    print(f"\n-amp dataset has {filas} rows and {columnas} columns.")
    filas, columnas = df_nonamp.shape
    print(f"-nonamp dataset has {filas} rows and {columnas} columns.")
    
    df_combined = pd.concat([df_amp, df_nonamp], ignore_index=True)

    # Renombra las columnas como "Sequence" y "Activity"
    df_combined.columns = ['Sequence', 'Activity']
    
    df_shuffled = df_combined.sample(frac=1, random_state=1).reset_index(drop=True)
    filas, columnas = df_shuffled .shape
    print(f"-The datasets combined have {filas} rows and {columnas} columns.")
    
    df_shuffled.to_csv(path_saving, index=False, quoting=None)

    return

path_amp = 'dataset_Chung/fasta/Chung_1593_all_training_c09n3g2.fasta'
path_nonamp= 'datasets_Xiao/Xiao_nonAMP_train.fasta'
path_saving= 'Chung_Xiao_balanced_all_training.csv'
processing(path_amp, path_nonamp, path_saving)

path_amp = 'dataset_Chung/fasta/Chung_454_all_validation_c09n3g2.fasta'
path_nonamp= 'datasets_Xiao/Xiao_nonAMP_validation.fasta'
path_saving= 'Chung_Xiao_balanced_all_validation.csv'
processing(path_amp, path_nonamp, path_saving)

path_amp = 'dataset_Chung/fasta/Chung_226_all_test_c09n3g2.fasta'
path_nonamp= 'datasets_Xiao/Xiao_nonAMP_test.fasta'
path_saving= 'Chung_Xiao_balanced_all_testing.csv'
processing(path_amp, path_nonamp, path_saving)

# %% ///////////// csv to fasta //////////////////
import csv

def convert_to_fasta(input_csv, output_fasta):
    with open(input_csv, 'r', newline='') as csv_file, open(output_fasta, 'w') as fasta_file:
        reader = csv.reader(csv_file)
        next(reader)  # Saltar la primera fila que contiene los encabezados

        i = 1
        for row in reader:
            sequence = row[0]
            fasta_file.write(f">P{i}\n{sequence}\n")
            i += 1

# Reemplaza 'entrada.csv' y 'salida.fasta' con los nombres de tu archivo CSV de entrada y archivo FASTA de salida, respectivamente
convert_to_fasta('90_Xiao_AMP_train.csv', '90_Xiao_AMP_train.fasta')
convert_to_fasta('90_Xiao_nonAMP_train.csv', '90_Xiao_nonAMP_train.fasta')
print("El archivo CSV se ha convertido a formato FASTA exitosamente.")


#%% /////////////// Extraer un numero X de filas aleatoriamente //////////////////
import pandas as pd
import random

df_shuffled = pd.read_csv('Xiao_nonAMP_train.csv' )
df_target = pd.read_csv('Xiao_AMP_trainc09n3g2d1.csv' )

total_rows = len(df_shuffled)

target_rows = len(df_target)

if total_rows > target_rows:
    
    rows_to_drop = total_rows - target_rows

    
    rows_to_drop_indices = random.sample(range(total_rows), rows_to_drop)

    df_shuffled = df_shuffled.drop(rows_to_drop_indices)

df_shuffled.to_csv('Xiao_nonAMP_09train.csv', index=False, quoting=None)
df_shuffled = pd.read_csv('Xiao_nonAMP_09train.csv')
filas, columnas = df_shuffled.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")

#%% ////////////// Removing duplicates ////////////////////
import pandas as pd

# Lee el archivo CSV
df_shuffled = pd.read_csv('datasets/Xiao_AMP_train_final_3.csv')

# Imprime la cantidad de duplicados encontrados
cantidad_duplicados = df_shuffled.duplicated(subset=[df_shuffled.columns[0]]).sum()
print(f'Se encontraron {cantidad_duplicados} duplicados.')

# Elimina las filas duplicadas basadas en el valor de la columna 1
df_sin_duplicados = df_shuffled.drop_duplicates(subset=[df_shuffled.columns[0]])

# Guarda el DataFrame resultante en un nuevo archivo CSV
df_sin_duplicados.to_csv('datasets/Xiao_AMP_train_final_4.csv', index=False)


# %% //////////// Histogram /////////////
import pandas as pd
import matplotlib.pyplot as plt

df_shuffled = pd.read_csv('SCX.csv')
print(df_shuffled.shape)
RT_values = df_shuffled.iloc[:, 1]  

# Imprimir el promedio de RT
mean_RT = RT_values.mean()
print(f"Mean RT: {mean_RT} minutes")

# Distribution Dataset
plt.hist(RT_values, bins=10)  
plt.title("SCX RT Distribution")
plt.xlabel("RT Values (min)")
plt.ylabel("Frequency")
plt.show()


# %% ////// Para buscar secuencias presentes en un dataset en otro //////////
import pandas as pd

def buscar_y_crear_datasets(dataset_principal, dataset_busqueda):
    
    df_principal = pd.read_csv(dataset_principal)
    df_busqueda = pd.read_csv(dataset_busqueda)

    presentes = []
    no_presentes = []

    # Iterar sobre las secuencias del dataset principal
    for secuencia in df_principal.iloc[:, 0]:  # Utilizar iloc para seleccionar la primera columna por posición
        # Verificar si la secuencia está presente en el dataset de búsqueda
        if secuencia in df_busqueda.iloc[:, 0].values:  # Utilizar iloc para seleccionar la primera columna por posición
            presentes.append(secuencia)
        else:
            no_presentes.append(secuencia)

    # Crear datasets con las secuencias presentes y no presentes
    df_presentes = df_principal[df_principal.iloc[:, 0].isin(presentes)]  # Utilizar iloc para seleccionar la primera columna por posición
    df_no_presentes = df_principal[df_principal.iloc[:, 0].isin(no_presentes)]  # Utilizar iloc para seleccionar la primera columna por posición

    # Guardar los nuevos datasets en archivos CSV numerados
    df_presentes.to_csv('comparation/presentes.csv', index=False)
    print('The number of Sequences in the dataset is:', len(df_presentes.iloc[:, 0]))
    print(df_presentes)
    df_no_presentes.to_csv('comparation/no_presentes.csv', index=False)
    print('The number of Sequences that are not: ', len(df_no_presentes.iloc[:, 0]))

# Ejemplo de uso
buscar_y_crear_datasets('datasets Xiao/Xiao_AMP_test.csv', 'Chung_Antibacterial_4685.csv')

# %% ////////////// Eliminar secuencias de un dataset en otro ///////////

import pandas as pd

# Lee los conjuntos de datos desde archivos CSV
df2 = pd.read_csv('presentes.csv')  # Reemplaza 'dataset1.csv' con el nombre de tu primer archivo CSV
filas, columnas = df2.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")

df1 = pd.read_csv('datasets/Jing_all_amp_nonamp_suffled.csv')  # Reemplaza 'dataset2.csv' con el nombre de tu segundo archivo CSV
filas, columnas = df1.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")

# Elimina filas de df1 que tienen IDs en común con df2
result_df = df1[~df1['sequence'].isin(df2['sequence'])]
filas, columnas = result_df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")

# Guarda el nuevo conjunto de datos en un nuevo archivo CSV
result_df.to_csv('Jing_all_amp_nonamp_duplicated_removed.csv', index=False)  # Reemplaza 'nuevo_dataset.csv' con el nombre que desees para el nuevo archivo

#%%
import pandas as pd

# Lee el archivo CSV con los títulos de las columnas especificados
df_shuffled = pd.read_csv("/home/vvd9fd/Documents/Bilodeau Group/Codes/0.Research/AMP-Peptide-Hierarchical-Graph-NN/data/dataset Chung/multiAMP_train.csv", usecols=["Antibacterial","MammalianCells","Antifungal","Antiviral","Anticancer","Antiparasitic"])

# Obtén el número total de filas en el DataFrame
total_filas = len(df_shuffled)

# Suma las filas de cada columna
sum_columnas = df_shuffled.sum()

# Contar cuántos valores son iguales a 1 en cada columna
conteo_unos = (df_shuffled == 1).sum()

# Imprimir los resultados en el formato especificado
for columna, suma, conteo in zip(df_shuffled.columns, sum_columnas, conteo_unos):
    porcentaje = (conteo / total_filas) * 100  # Calcula el porcentaje
    print(f'{columna}: {conteo} ({porcentaje:.3f}%)')
    
print('Total de sequencias', total_filas)


# %%////////////// SubDataset Proccesing ////////////////////////
import pandas as pd
import random

def filtrar_concatenar_revolver_y_guardar(csv_file, actividad):
    # Leer el archivo CSV
    df = pd.read_csv(csv_file)
    
    # Filtrar el DataFrame original
    df_1 = df[df[actividad] == 1]
    df_0 = df[df[actividad] == 0]
    
    # Tomar una muestra aleatoria del DataFrame filtrado con valores de 0 del mismo tamaño que el DataFrame filtrado con valores de 1,
    # o tomar toda la muestra disponible en df_0 si no hay suficientes datos.
    if len(df_0) >= len(df_1):
        df_0_random_sample = df_0.sample(n=len(df_1), random_state=24)
    else:
        df_0_random_sample = df_0
    
    # Concatenar los DataFrames
    df_concatenado = pd.concat([df_1, df_0_random_sample])
    
    # Revolver aleatoriamente las filas
    df_concatenado = df_concatenado.sample(frac=1, random_state=24).reset_index(drop=True)
    
    # Construir el nombre del archivo
    total_secuencias = len(df_concatenado)
    num_1 = len(df_1)
    num_0 = len(df_0_random_sample)
    nombre_archivo = f"Chung_all_training_{actividad}_{total_secuencias}_p{num_1}_n{num_0}.csv"
    
    # Guardar el DataFrame concatenado y revuelto en un archivo CSV
    df_concatenado.to_csv(nombre_archivo, index=False)
    
    print(f"Archivo guardado como '{nombre_archivo}'")
    
    # Imprimir el número de muestras con valor 1 y 0 en el DataFrame final
    print(f"Número de muestras con valor 1 en '{actividad}': {len(df_1)}")
    print(f"Número de muestras con valor 0 en '{actividad}': {len(df_0_random_sample)}")


# Ejemplo de uso:
actividad = "Antibacterial"  # Puedes cambiar esto según la actividad de interés
filtrar_concatenar_revolver_y_guardar("dataset Chung (AMP)/Chung_6160_training_all.csv",
                                    actividad)


#%%
#%% /////////////// Combining 3 csv  //////////////////////
import pandas as pd

def processing(path_1, path_2, path_3):
    df_1 = pd.read_csv(path_1 )
    df_2 = pd.read_csv(path_2)
    df_3 = pd.read_csv(path_3)
    
    df_combined = pd.concat([df_1, df_2, df_3], ignore_index=True)

    # Renombra las columnas como "Sequence" y "Activity"
    df_combined.columns = ['Sequence', 'Activity']
    
    # Verifica duplicados y elimina las filas duplicadas
    df_combined = df_combined.drop_duplicates()

    df_shuffled = df_combined.sample(frac=1, random_state=1).reset_index(drop=True)
    filas, columnas = df_shuffled.shape
    print(f"-The datasets combined have {filas} rows and {columnas} columns.")
    path_saving = f'Chung_Xiao_NonAntibacterial_{filas}.csv'
    
    df_shuffled.to_csv(path_saving, index=False, quoting=None)

    return


processing('Chung_nonAntibacterial_1475.csv', 'Xiao_nonAMP_train_copy.csv', 'Xiao_nonAMP_validation_copy.csv')



#%%
import pandas as pd

def combine_and_save_datasets(path_1, path_2):
    """
    Combina dos conjuntos de datos CSV, elimina duplicados y guarda el resultado en un nuevo archivo CSV.

    Args:
        path_1 (str): Ruta al primer archivo CSV.
        path_2 (str): Ruta al segundo archivo CSV.
    """
    try:
        df_1 = pd.read_csv(path_1)
        df_2 = pd.read_csv(path_2)
        
        df_combined = pd.concat([df_1, df_2], ignore_index=True)

        # Renombra las columnas como "Sequence" y "Activity"
        df_combined.columns = ['Sequence', 'Activity']
        
        # Elimina filas duplicadas
        df_combined.drop_duplicates(inplace=True)

        # Reordena aleatoriamente las filas
        df_shuffled = df_combined.sample(frac=1, random_state=1).reset_index(drop=True)
        
        # Guarda el nuevo archivo CSV
        combined_rows, combined_columns = df_shuffled.shape
        print(f"Los conjuntos de datos combinados tienen {combined_rows} filas y {combined_columns} columnas.")
        saving_path = f'Combined_dataset_{combined_rows}.csv'
        df_shuffled.to_csv(saving_path, index=False, quoting=None)
        print(f"El archivo combinado se ha guardado como '{saving_path}'.")
    except FileNotFoundError as e:
        print("Error: Archivo no encontrado. Asegúrate de que las rutas de archivo sean correctas.")
    except Exception as e:
        print(f"Error inesperado: {e}")

# Ejemplo de uso:
combine_and_save_datasets('Chung_Xiao_NonAntibacterial_3521.csv', 'Chung_Antibacterial_4685.csv')



# %%

import pandas as pd
import numpy as np

def split_dataset(input_file, train_ratio=0.75, val_ratio=0.2, test_ratio=0.05, random_state=None):
    """
    Divide un archivo CSV en tres conjuntos de datos para entrenamiento, validación y prueba.

    Args:
        input_file (str): Ruta al archivo CSV de entrada.
        train_ratio (float): Proporción del conjunto de entrenamiento (por defecto 0.75).
        val_ratio (float): Proporción del conjunto de validación (por defecto 0.2).
        test_ratio (float): Proporción del conjunto de prueba (por defecto 0.05).
        random_state (int): Semilla para la generación de números aleatorios (por defecto None).

    Returns:
        None
    """
    try:
        # Leer el archivo CSV
        df = pd.read_csv(input_file)

        # Obtener el número de filas
        num_rows = len(df)

        # Calcular el número de filas para cada conjunto
        train_size = int(train_ratio * num_rows)
        val_size = int(val_ratio * num_rows)
        test_size = int(test_ratio * num_rows)

        # Crear un índice aleatorio para las filas
        np.random.seed(random_state)
        indices = np.random.permutation(num_rows)

        # Dividir los índices en las tres partes
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Crear los conjuntos de datos
        df_train = df.iloc[train_indices]
        df_val = df.iloc[val_indices]
        df_test = df.iloc[test_indices]

        # Guardar los conjuntos de datos como archivos CSV
        train_filename = f'Chung_Xiao_train_AB_{train_size}.csv'
        val_filename = f'Chung_Xiao_val_AB_{val_size}.csv'
        test_filename = f'Chung_Xiao_test_AB_{test_size}.csv'

        df_train.to_csv(train_filename, index=False)
        df_val.to_csv(val_filename, index=False)
        df_test.to_csv(test_filename, index=False)

        print("Los conjuntos de datos se han dividido y guardado correctamente.")
    except FileNotFoundError as e:
        print("Error: Archivo no encontrado. Asegúrate de que la ruta de archivo sea correcta.")
    except Exception as e:
        print(f"Error inesperado: {e}")

# Ejemplo de uso:
split_dataset('Combined_dataset_8204.csv', train_ratio=0.75, val_ratio=0.2, test_ratio=0.05, random_state=42)


# %%
# %% ///////////// Playing with peptides sequences ////////////////////////
import pandas as pd

# Función para intercambiar aminoácidos en una posición específica
def swap_aminoacids(sequence, position, new_aminoacid):
    sequence_list = list(sequence)
    sequence_list[position-1] = new_aminoacid
    return ''.join(sequence_list)

# Péptido inicial

d8 ='RTVRCTCI'

aminoacids_set = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

# Intercambiar aminoácidos en posiciones específicas
new_sequences = []
df = pd.DataFrame(columns=["sequence_name", "sequence"])
for aminoacid_5 in aminoacids_set:
    position_5 =5
    new_sequence = swap_aminoacids(d8, position_5, aminoacid_5 )
    for aminoacid_7 in aminoacids_set:
        position_7 =7
        new_sequence = swap_aminoacids(new_sequence, position_7, aminoacid_7)
        for aminoacid_2 in aminoacids_set:
            position_2 =2
            new_sequence = swap_aminoacids(new_sequence, position_2, aminoacid_2 )
            for aminoacid_6 in aminoacids_set:
                position_6 =6
                new_sequence = swap_aminoacids(new_sequence, position_6, aminoacid_6 )
                new_sequences.append({"sequence_name": f"p_2{aminoacid_2}5{aminoacid_5}6{aminoacid_6}7{aminoacid_7}", "sequence": new_sequence})


# Agregar las nuevas secuencias al DataFrame
df = df.append(new_sequences, ignore_index=True)
df['nueva_columna'] = 0 


# Guardar el DataFrame en un archivo CSV
df.to_csv('d8_options_p_2_5_6_7.csv', index=False)

print("Nuevas secuencias añadidas al archivo sequences.csv")
# %%
