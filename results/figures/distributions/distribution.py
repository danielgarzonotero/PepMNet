

# %%
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def distribution_amp(property, path, bins):
    # 0 : aa average concentration
    # 1: Length             # 2: Charge              # 3: Amphiphilicity           
    # 4: Hydropathy         # 5: Secondary Structure # 6: FAI
    # 7: Molecular Weight   # 8: Hydrophobicity      # 9: Aromaticity
    # 10: Isoelectric Point # 11: Instability Index  
    
    #Filtering the result excel depending the condition
    if path.endswith('.csv'):
        # Leyendo el archivo CSV
        df_new = pd.read_csv(path).copy()
    else:
        # Leyendo el archivo Excel
        df_new = pd.read_excel(path).copy()
        
    condition_filter_amp = ((df_new['Label'] == 1) )
    condition_filter_nonamp = ((df_new['Label'] == 0) )

    df_new_amp = df_new[condition_filter_amp].copy()
    df_new_nonamp = df_new[condition_filter_nonamp].copy()

    
    if property == 0:
        
        aminoacidos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        for aa in aminoacidos:
            df_new_amp[aa] = df_new_amp['Sequence'].apply(lambda x: x.count(aa) / len(x))
            df_new_nonamp[aa] = df_new_nonamp['Sequence'].apply(lambda x: x.count(aa) / len(x))
            
        avg_concentrations_amp = df_new_amp[aminoacidos].mean().sort_values(ascending=False)
        avg_concentrations_nonamp = df_new_nonamp[aminoacidos].mean().sort_values(ascending=False)
        
        avg_amp_df = pd.DataFrame({'Amino Acid': avg_concentrations_amp.index,
                               'AMP Average Concentration': avg_concentrations_amp.round(4)})
        avg_nonamp_df = pd.DataFrame({'Amino Acid': avg_concentrations_nonamp.index,
                               'nonAMP Average Concentration': avg_concentrations_nonamp.round(4)})
        
        avg_amp_df.to_csv('amp_average_concentrations.csv', index=False)
        avg_nonamp_df.to_csv('nonamp_average_concentrations.csv', index=False)
        pass
    
    elif property == 1:
        property = 'Length'
        df_new_amp[property] = df_new_amp['Sequence'].apply(len)
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(len)
        # Calculate mean and standard deviation
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()
        pass
        
    elif property == 2:
        property = 'Charge'
        df_new_amp[property] = df_new_amp['Sequence'].apply(lambda x: ProteinAnalysis(x).charge_at_pH(7))
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(lambda x: ProteinAnalysis(x).charge_at_pH(7))
        # Calculate mean and standard deviation
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()
        pass
    
    elif property == 3:
        property = 'Amphiphilicity'
        # Leer el DataFrame original con la tabla de valores
        df_valores = pd.read_csv('index.csv')  # Reemplaza con la ruta correcta
        
        # Función para calcular la suma de valores para una secuencia dada
        def calcular_suma(secuencia):
            return sum(df_valores.loc[df_valores['Letter'].isin(list(secuencia)), 'Amphiphilicity'])
        
        # Agregar una nueva columna al DataFrame de secuencias con las sumas calculadas
        df_new_amp[property] = df_new_amp['Sequence'].apply(calcular_suma)
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(calcular_suma)
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()
        pass
    
    elif property == 4:
        property = 'Hydropathy'
        df_valores = pd.read_csv('index.csv')  
        
        def calcular_suma(secuencia):
            return sum(df_valores.loc[df_valores['Letter'].isin(list(secuencia)), 'Hydropathy'])
    
        df_new_amp[property] = df_new_amp['Sequence'].apply(calcular_suma)
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(calcular_suma)
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()
        pass
    
    elif property == 5:
        property = 'Secondary Structure'
        
        def predict_secondary_structure(peptide_sequence):
            protein_analysis = ProteinAnalysis(peptide_sequence)
            sheet, turn, helix = protein_analysis.secondary_structure_fraction()
            return sheet, turn, helix
        
        # Añadir columnas con la longitud y la estructura secundaria de cada secuencia
        df_new_amp['helix'] = df_new_amp['Sequence'].apply(lambda x: predict_secondary_structure(x)[2])
        df_new_amp['turn'] = df_new_amp['Sequence'].apply(lambda x: predict_secondary_structure(x)[1])
        df_new_amp['sheet'] = df_new_amp['Sequence'].apply(lambda x: predict_secondary_structure(x)[0])
        
        df_new_nonamp['helix'] = df_new_nonamp['Sequence'].apply(lambda x: predict_secondary_structure(x)[2])
        df_new_nonamp['turn'] = df_new_nonamp['Sequence'].apply(lambda x: predict_secondary_structure(x)[1])
        df_new_nonamp['sheet'] = df_new_nonamp['Sequence'].apply(lambda x: predict_secondary_structure(x)[0])
        # Calcular medias y desviaciones estándar
        helix_mean_amp = df_new_amp['helix'].mean()
        helix_std_amp = df_new_amp['helix'].std()
        turn_mean_amp = df_new_amp['turn'].mean()
        turn_std_amp = df_new_amp['turn'].std()
        sheet_mean_amp = df_new_amp['sheet'].mean()
        sheet_std_amp = df_new_amp['sheet'].std()
        
        helix_mean_nonamp = df_new_nonamp['helix'].mean()
        helix_std_nonamp = df_new_nonamp['helix'].std()
        turn_mean_nonamp = df_new_nonamp['turn'].mean()
        turn_std_nonamp = df_new_nonamp['turn'].std()
        sheet_mean_nonamp = df_new_nonamp['sheet'].mean()
        sheet_std_nonamp = df_new_nonamp['sheet'].std()        
        pass
    
    elif property == 6:
        property = 'FAI'
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
        
        df_new_amp[property] = df_new_amp['Sequence'].apply(lambda x: fai(x))
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(lambda x: fai(x))
        
        # Calcular medias y desviaciones estándar
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()
        pass
    
    elif property == 7:
        property = 'Molecular Weight'
        
        def mw_peptide(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            mw_peptide = peptide_analysis .molecular_weight()
            return mw_peptide
        
        df_new_amp[property] = df_new_amp['Sequence'].apply(lambda x: mw_peptide(x))
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(lambda x: mw_peptide(x))
        # Calcular medias y desviaciones estándar
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()
        pass
    
    elif property == 8:
        property = 'Hydrophobicity'
        
        def hidrophobicity(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            hidrophobicity = peptide_analysis.gravy()
            return hidrophobicity
        
        df_new_amp[property] = df_new_amp['Sequence'].apply(lambda x: hidrophobicity(x))
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(lambda x: hidrophobicity(x))
        # Calcular medias y desviaciones estándar
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()
        pass
    
    elif property == 9:
        property = 'Aromaticity'
        def aromaticity(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            aromaticity = peptide_analysis.aromaticity()
            return aromaticity
        
        df_new_amp[property] = df_new_amp['Sequence'].apply(lambda x: aromaticity(x))
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(lambda x: aromaticity(x))
        # Calcular medias y desviaciones estándar
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()     
        pass
    
    elif property == 10:
        property = 'Isoelectric Point'
        def isoelectric_point(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            isoelectric_point = peptide_analysis.isoelectric_point()
            return isoelectric_point
        
        df_new_amp[property] = df_new_amp['Sequence'].apply(lambda x: isoelectric_point(x))
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(lambda x: isoelectric_point(x))
        # Calcular medias y desviaciones estándar
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()
        pass
    
    elif property == 11:
        property = 'Instability Index'
        def instability_index(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            instability_index= peptide_analysis.instability_index()
            return instability_index
        
        df_new_amp[property] = df_new_amp['Sequence'].apply(lambda x: instability_index(x))
        df_new_nonamp[property] = df_new_nonamp['Sequence'].apply(lambda x: instability_index(x))
        # Calcular medias y desviaciones estándar
        mean_amp = df_new_amp[property].mean()
        std_amp = df_new_amp[property].std()
        mean_nonamp = df_new_nonamp[property].mean()
        std_nonamp = df_new_nonamp[property].std()       
        pass
        
    elif property == 12:
      
        pass

    #Plottig
        
    
    if property != 0:
        if property == 'Secondary Structure':
            plt.figure(figsize=(10, 6), dpi=260)

            plt.hist(df_new_amp['sheet'], bins=bins, color="yellowgreen", alpha=0.2, label="Sheet") 
            plt.hist(df_new_amp['turn'], bins=bins, color="slateblue", alpha=0.5, label="Turn") 
            plt.hist(df_new_amp['helix'], bins=bins, color="teal", alpha=0.9, label="Helix") 

            plt.xlabel('Secondary Structure Fraction ',size= 17)
            plt.ylabel('Frequency',size= 15)
            plt.title(f"Distribution of {property} - Antimicrobial Peptides",size= 17)

            # Agregar leyenda
            plt.legend(fontsize='20')

            # Añadir texto con medias y desviaciones estándar
            plt.text(0.95, 0.055, f"Helix Mean={helix_mean_amp:.2f}±{helix_std_amp:.2f}\nTurn Mean={turn_mean_amp:.2f}±{turn_std_amp:.2f}\nSheet Mean={sheet_mean_amp:.2f}±{sheet_std_amp:.2f}", 
                    transform=plt.gca().transAxes, ha='right', color='black',
                    bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 16)
            plt.axvline(helix_mean_amp, color='yellowgreen', linestyle='dashed', linewidth=2)
            plt.axvline(turn_mean_amp, color='slateblue', linestyle='dashed', linewidth=2)
            plt.axvline(sheet_mean_amp, color='teal', linestyle='dashed', linewidth=2)
            plt.xticks(fontsize=18)  # Set font size for x-axis tick labels
            plt.yticks(fontsize=18)  # Set font size for y-axis tick labels 
            
            plt.show()
            print(f"Helix Mean={helix_mean_amp:.3f}±{helix_std_amp:.3f}\nTurn Mean={turn_mean_amp:.3f}±{turn_std_amp:.3f}\nSheet Mean={sheet_mean_amp:.3f}±{sheet_std_amp:.3f}")
            
            plt.figure(figsize=(10, 6))

            plt.hist(df_new_nonamp['sheet'], bins=bins, color="yellowgreen", alpha=0.2, label="Sheet") 
            plt.hist(df_new_nonamp['turn'], bins=bins, color="slateblue", alpha=0.5, label="Turn") 
            plt.hist(df_new_nonamp['helix'], bins=bins, color="teal", alpha=0.9, label="Helix") 


            plt.xlabel('Secondary Structure Fraction ',size= 17)
            plt.ylabel('Frequency',size= 15)
            plt.title(f"Distribution of {property} - nonAntimicrobial Peptides",size= 17)

            # Agregar leyenda
            plt.legend(fontsize='20')

            # Añadir texto con medias y desviaciones estándar
            plt.text(0.95, 0.055, f"Helix Mean={helix_mean_amp:.2f}±{helix_std_amp:.2f}\nTurn Mean={turn_mean_amp:.2f}±{turn_std_amp:.2f}\nSheet Mean={sheet_mean_amp:.2f}±{sheet_std_amp:.2f}", 
                    transform=plt.gca().transAxes, ha='right', color='black',
                    bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 16)
            plt.axvline(helix_mean_nonamp, color='yellowgreen', linestyle='dashed', linewidth=2)
            plt.axvline(turn_mean_nonamp, color='slateblue', linestyle='dashed', linewidth=2)
            plt.axvline(sheet_mean_nonamp, color='teal', linestyle='dashed', linewidth=2)
            plt.xticks(fontsize=12)  # Set font size for x-axis tick labels
            plt.yticks(fontsize=12)  # Set font size for y-axis tick labels 
            

            plt.show()
            
            
        else:
            # Crear la figura y los histogramas
            plt.figure(figsize=(9, 7), dpi=260)
            plt.hist(df_new_amp[property], bins=bins, color='seagreen', alpha=0.9, log=True) 
            plt.hist(df_new_nonamp[property], bins=bins, color='tomato', alpha=0.3, log=True)

            # Etiquetas de los ejes y título
            plt.xlabel(property, size=27)
            plt.ylabel('log(Frequency)', size=27)
            plt.xticks(fontsize=21)  # Set font size for x-axis tick labels
            plt.yticks(fontsize=21)  # Set font size for y-axis tick labels 
            plt.title(f"Distribution of {property}", size=20, pad=20)

            # Definir las leyendas y medias en una sola
            legend_handles = [
                mpatches.Patch(color='seagreen', alpha=0.5, label=f'AMP\nμ={mean_amp:.3f}±{std_amp:.3f}'),
                mpatches.Patch(color='tomato', alpha=0.5, label=f'nonAMP\nμ={mean_nonamp:.3f}±{std_nonamp:.3f}')
            ]
            plt.legend(handles=legend_handles, loc='upper left', fontsize=24, frameon=True)

            # Agregar la media al gráfico
            plt.axvline(mean_amp, color='seagreen', linestyle='dashed', linewidth=1)
            plt.axvline(mean_nonamp, color='tomato', linestyle='dashed', linewidth=1)

            # Mostrar el gráfico
            plt.show()

            # Imprimir medias y desviaciones estándar
            print(f"AMP={mean_amp:.3f}±{std_amp:.3f}")
            print(f"nonAMP={mean_nonamp:.3f}±{std_nonamp:.3f}")



                        
#////////////////////////////////////////////////////////////////////////
# %%
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def distribution(property, path, bins):
    # 0 : aa average concentration
    # 1: Length             # 2: Charge              # 3: Amphiphilicity           
    # 4: Hydropathy         # 5: Secondary Structure # 6: FAI
    # 7: Molecular Weight   # 8: Hydrophobicity      # 9: Aromaticity
    # 10: Isoelectric Point # 11: Instability Index  
    
    # Filtrando el resultado Excel dependiendo de la condición
    if path.endswith('.csv'):
        # Leyendo el archivo CSV
        df_new = pd.read_csv(path).copy()
    else:
        # Leyendo el archivo Excel
        df_new = pd.read_excel(path).copy()
    
    if property == 0:
        aminoacidos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        for aa in aminoacidos:
            df_new[aa] = df_new[:, 0].apply(lambda x: x.count(aa) / len(x))
        
        avg_concentrations = df_new[aminoacidos].mean().sort_values(ascending=False)
        
        avg_df = pd.DataFrame({'Amino Acid': avg_concentrations.index,
                               'Average Concentration': avg_concentrations.round(4)})
        
        avg_df.to_csv('average_concentrations.csv', index=False)
    
    elif property == 1:
        property = 'Length'
        df_new[property] = df_new['sequence'].apply(len)
        # Calculate mean and standard deviation
        mean = df_new[property].mean()
        std = df_new[property].std()
    
    elif property == 2:
        property = 'Charge'
        df_new[property] = df_new['sequence'].apply(lambda x: ProteinAnalysis(x).charge_at_pH(7))
        # Calculate mean and standard deviation
        mean = df_new[property].mean()
        std = df_new[property].std()
    
    elif property == 3:
        property = 'Amphiphilicity'
        df_valores = pd.read_csv('index.csv')  # Reemplaza con la ruta correcta
        
        def calcular_suma(secuencia):
            return sum(df_valores.loc[df_valores['Letter'].isin(list(secuencia)), 'Amphiphilicity'])
        
        df_new[property] = df_new['sequence'].apply(calcular_suma)
        mean = df_new[property].mean()
        std = df_new[property].std()
    
    elif property == 4:
        property = 'Hydropathy'
        df_valores = pd.read_csv('index.csv')  
        
        def calcular_suma(secuencia):
            return sum(df_valores.loc[df_valores['Letter'].isin(list(secuencia)), 'Hydropathy'])
    
        df_new[property] = df_new[:, 0].apply(calcular_suma)
        mean = df_new[property].mean()
        std = df_new[property].std()
    
    elif property == 5:
        property = 'Secondary Structure'
        
        def predict_secondary_structure(peptide_sequence):
            protein_analysis = ProteinAnalysis(peptide_sequence)
            sheet, turn, helix = protein_analysis.secondary_structure_fraction()
            return sheet, turn, helix
        
        df_new['helix'] = df_new['sequence'].apply(lambda x: predict_secondary_structure(x)[2])
        df_new['turn'] = df_new['sequence'].apply(lambda x: predict_secondary_structure(x)[1])
        df_new['sheet'] = df_new['sequence'].apply(lambda x: predict_secondary_structure(x)[0])
        
        helix_mean = df_new['helix'].mean()
        helix_std = df_new['helix'].std()
        turn_mean = df_new['turn'].mean()
        turn_std = df_new['turn'].std()
        sheet_mean = df_new['sheet'].mean()
        sheet_std = df_new['sheet'].std()
    
    elif property == 6:
        property = 'FAI'
        def fai(sequence):
            helm = peptide_to_helm(sequence)
            mol = Chem.MolFromHELM(helm)
            num_anillos = rdMolDescriptors.CalcNumRings(mol)
            
            charged_amino_acids = {'R': 2, 'H': 1, 'K': 2}
            cationic_charges = sum(sequence.count(aa) * charge for aa, charge in charged_amino_acids.items())

            if num_anillos == 0:
                return 0

            return (cationic_charges / num_anillos)

        def peptide_to_helm(sequence):
            polymer_id = 1
            sequence_helm = "".join(sequence)
            sequence_helm = ''.join([c + '.' if c.isupper() else c for i, c in enumerate(sequence_helm)])
            sequence_helm = sequence_helm.rstrip('.')
            sequence_helm = f"PEPTIDE{polymer_id}{{{sequence_helm}}}$$$$"
            return sequence_helm
        
        df_new[property] = df_new['sequence'].apply(lambda x: fai(x))
        mean = df_new[property].mean()
        std = df_new[property].std()
    
    elif property == 7:
        property = 'Molecular Weight'
        
        def mw_peptide(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            mw_peptide = peptide_analysis.molecular_weight()
            return mw_peptide
        
        df_new[property] = df_new['sequence'].apply(lambda x: mw_peptide(x))
        mean = df_new[property].mean()
        std = df_new[property].std()
    
    elif property == 8:
        property = 'Hydrophobicity'
        
        def hidrophobicity(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            hidrophobicity = peptide_analysis.gravy()
            return hidrophobicity
        
        df_new[property] = df_new['sequence'].apply(lambda x: hidrophobicity(x))
        mean = df_new[property].mean()
        std = df_new[property].std()
    
    elif property == 9:
        property = 'Aromaticity'
        def aromaticity(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            aromaticity = peptide_analysis.aromaticity()
            return aromaticity
        
        df_new[property] = df_new['sequence'].apply(lambda x: aromaticity(x))
        mean = df_new[property].mean()
        std = df_new[property].std()
    
    elif property == 10:
        property = 'Isoelectric Point'
        def isoelectric_point(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            isoelectric_point = peptide_analysis.isoelectric_point()
            return isoelectric_point
        
        df_new[property] = df_new['sequence'].apply(lambda x: isoelectric_point(x))
        mean = df_new[property].mean()
        std = df_new[property].std()
    
    elif property == 11:
        property = 'Instability Index'
        def instability_index(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            instability_index = peptide_analysis.instability_index()
            return instability_index
        
        df_new[property] = df_new[:, 0].apply(lambda x: instability_index(x))
        mean = df_new[property].mean()
        std = df_new[property].std()

    # Plottig
    if property != 0:
        if property == 'Secondary Structure':
            plt.figure(figsize=(10, 6), dpi=260)

            plt.hist(df_new['sheet'], bins=bins, color="yellowgreen", alpha=0.2, label="Sheet") 
            plt.hist(df_new['turn'], bins=bins, color="slateblue", alpha=0.5, label="Turn") 
            plt.hist(df_new['helix'], bins=bins, color="teal", alpha=0.9, label="Helix") 

            plt.xlabel('Secondary Structure Fraction ', size=17)
            plt.ylabel('Frequency', size=15)
            plt.title(f"Distribution of {property}", size=17)

            plt.legend(fontsize='20')

            plt.text(0.95, 0.055, f"Helix Mean={helix_mean:.2f}±{helix_std:.2f}\nTurn Mean={turn_mean:.4f}±{turn_std:.4f}\nSheet Mean={sheet_mean:.2f}±{sheet_std:.2f}", 
                     transform=plt.gca().transAxes, ha='right', color='black',
                     bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size=16)
            plt.axvline(helix_mean, color='yellowgreen', linestyle='dashed', linewidth=2)
            plt.axvline(turn_mean, color='slateblue', linestyle='dashed', linewidth=2)
            plt.axvline(sheet_mean, color='teal', linestyle='dashed', linewidth=2)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()
        
        else:
            plt.figure(figsize=(5, 5), dpi=260)

            plt.hist(df_new[property], bins=bins, color="lightblue", alpha=0.75, log=True)
            plt.xlabel(f'{property}', size=17)
            plt.ylabel('log(Frequency)', size=15)
            plt.title(f"Distribution of {property} misc", size=17)

            plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
            plt.text(0.94, 0.94, f'μ={mean:.4f}±{std:.4f}', 
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), fontsize=10)

            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            plt.show()
