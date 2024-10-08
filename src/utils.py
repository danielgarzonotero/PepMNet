from collections import defaultdict
import numpy as np

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

from sklearn.preprocessing import OneHotEncoder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Align import substitution_matrices


def sequences_geodata_1(cc, sequence, y, aminoacids_ft_dict, node_ft_dict, edge_ft_dict, device, has_targets):
    
    #Atoms
    sequence = get_sequence(sequence)
    
    polymer_id = "PEPTIDE1" 
    helm_notation = peptide_to_helm(sequence, polymer_id)
    molecule = Chem.MolFromHELM(helm_notation)
    
    atomic_number = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    aromaticity = [int(atom.GetIsAromatic()) for atom in molecule.GetAtoms()]
    num_bonds = [atom.GetDegree() for atom in molecule.GetAtoms()]
    bonded_hydrogens = [atom.GetTotalNumHs() for atom in molecule.GetAtoms()]
    hybridization = [atom.GetHybridization().real for atom in molecule.GetAtoms()]
    implicit_valence = [atom.GetImplicitValence () for atom in molecule.GetAtoms()]
    
    node_keys_features = [  f"{atomic}_{aromatic}_{bonds}_{hydrogen}_{hybrid}_{impli_vale}" 
                            for atomic, aromatic, bonds, hydrogen, hybrid, impli_vale
                            in zip(atomic_number, aromaticity, num_bonds, bonded_hydrogens, hybridization, implicit_valence )]
    
    edge_key_features = []
    for bond in molecule.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        in_ring = int(bond.IsInRing())
        conjugated = int(bond.GetIsConjugated())
        bond_aromatic = int(bond.GetIsAromatic())
        valence_contribution_i = int(bond.GetValenceContrib(bond.GetBeginAtom()))
        valence_contribution_f = int(bond.GetValenceContrib(bond.GetEndAtom()))
        
        edge_key_features.append(f"{bond_type:.1f}_{in_ring:.1f}_{conjugated:.1f}_{bond_aromatic:.1f}_{valence_contribution_i:.1f}_{valence_contribution_f:.1f}") 
    
    nodes_features = torch.tensor(np.array([node_ft_dict[x] for x in node_keys_features]), dtype=torch.float32)
    edges_features = torch.tensor(np.array([edge_ft_dict[x] for x in edge_key_features]), dtype=torch.float32)  
    graph_edges = get_edge_indices(molecule)[0]
    
    edges_peptidic = get_edge_indices(molecule)[1]
    edges_nonpeptidic = get_non_peptide_idx(molecule)
    labels_aminoacid_atoms = get_label_aminoacid_atoms(edges_peptidic, edges_nonpeptidic, molecule)
    
    #amino acid feature:
    aminoacids_features_dict = aminoacids_ft_dict
    aminoacids =get_aminoacids_2(sequence)
    aminoacids_mol = [get_molecule(amino) for amino in aminoacids]
    aminoacids_biopython = [ProteinAnalysis(amino) for amino in aminoacids]
    wt_amino=[round(Descriptors.MolWt(amino), 4) for amino in aminoacids_mol]
    aromaticity_amino=[round(amino.aromaticity(), 4) for amino in aminoacids_biopython]
    hydrophobicity_amino=[round(amino.gravy(), 4) for amino in aminoacids_biopython]
    net_charge_amino=[round(amino.charge_at_pH(7), 4) for amino in aminoacids_biopython]
    p_iso_amino=[round(amino.isoelectric_point(), 4) for amino in aminoacids_biopython]
    logp_amino=[round(Crippen.MolLogP(amino), 4) for amino in aminoacids_mol]
    atoms_amino=[round(float(amino.GetNumAtoms()), 4) for amino in aminoacids_mol]
    
    aminoacids_keys_features = [f"{wt}_{aromaticity}_{hydrophobicity}_{net_charge}_{p_iso}_{logp}_{atoms}"
                                for wt, aromaticity, hydrophobicity, net_charge, p_iso, logp, atoms
                                in zip(wt_amino, aromaticity_amino, hydrophobicity_amino, net_charge_amino, p_iso_amino, logp_amino, atoms_amino)]
    
    aminoacids_features = (cc,torch.tensor(np.array([aminoacids_features_dict[x] for x in aminoacids_keys_features]), dtype=torch.float32, device = device))
    blosum62 = (cc, torch.tensor(np.array([construir_matriz_caracteristicas(sequence)]), dtype =torch.float32, device = device))
    num_aminoacidos = len(sequence)
    amino_index = (cc, get_amino_indices(num_aminoacidos, device))
    
    if has_targets:
        y = torch.tensor(np.array([y]), dtype=torch.float32, device=device)
        
        geo_dp = Data(          
                        x=nodes_features,
                        y=y,
                        edge_index=graph_edges, 
                        edge_attr=edges_features, 
                        monomer_labels=labels_aminoacid_atoms,
                        aminoacids_features=aminoacids_features , 
                        blosumn=blosum62,
                        cc=cc,
                        sequence = sequence,
                        amino_index=amino_index
                        
                    )
        
        return geo_dp
                
    else: 
        geo_dp = Data(          
                        x=nodes_features,
                        edge_index=graph_edges, 
                        edge_attr=edges_features, 
                        monomer_labels=labels_aminoacid_atoms,
                        aminoacids_features=aminoacids_features , 
                        blosumn=blosum62,
                        cc=cc,
                        sequence = sequence,
                        amino_index=amino_index
                    )
        
        return geo_dp



#---------------------------------- For independet sets with out targets values ---------------------------------------

def get_amino_indices(num_aminoacid, device):
    edges = []
    for i in range(num_aminoacid-1):
        edges.append((i, i + 1))
    
    graph_edges = [[x[0] for x in edges], [x[1] for x in edges]]
    
    return torch.tensor(graph_edges, dtype=torch.long, device = device) 

    
#/////////////////// Atomic Features //////////////////////////////////////

#Creating the dictionaries based on the dataset information

#Convert to Helm notation to use Rdkit
def peptide_to_helm(peptide, polymer_id):
    sequence = peptide.replace("(ac)", "[ac].").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    sequence_helm = "".join(sequence)
    
    sequence_helm = ''.join([c + '.' if c.isupper() else c for i, c in enumerate(sequence_helm)])
    sequence_helm = sequence_helm.rstrip('.')
    
    sequence_helm = f"{polymer_id}{{{sequence_helm}}}$$$$"
    
    return sequence_helm


def get_features():
    
    peptides_list_helm = []
    #Sample sequences to optimize the calculation of nodes and edges features. 
    sequence_list = [
    "WNSLKIDNLDA",
    "RIPVMMNWYW",
    'PQPPVEEEDEHFDDTVVCLDTYNCDLHFK',
    "SCRYSQRPSFYRWELYFNGRMWCP",
    "MNQKHSSDFVVIKAVEDGVNVIGLTRGTDTKFHHSEKLDKGEVIIAQFTEHTSAIKVRGEALIQTAYGEMKSEKK",
    "VLSIVACSSGCGSGKTAASCVATCGNKCFTNVGSLC",
    "MKHFLTYLSTAPVLAAIWMTITAGILIEFNRFYPDLLFHPL",
    "GILGKLWEGFKSIV",
    "FNQWTTWCYHHMVPYCDYCHFKR",
    "GLLALLGELAEHLGSKI",
    "GCKKYRRFRWKFKGKLWLWG",
    "EIEKFDKSKLK",
    "GGTIFDCGETCFLGTCYTPGCSCGNYGFCYGTN",
    "MKVLVLITLAVLGAMFVWTSAAELEERGSDQRDSPAWVKSMERIFQSEERACREWLGGCSKDADCCAHLECRKKWPYHCVWDWTVRK",
    "LKMLGMLFHNIRNILKTV",
    "WRPGRWWRPGRWWRPGRWWRPGRW",
    "GLWQIFSSKEEGKDNSQQKSKGDQAKEL",
    "RWMAWPTHKERNWYMTW",
    "HRILMRARQMMT",
    "RRSRFGRFFKKVRKQLGRVLRHSRITVGGRMRF",
    "ACDEFGHIKLMNPQRSTVWY",
    "HRKIFLWAMPCNVGSQYDET"
    ]
    
    for i, peptide in enumerate(sequence_list):
        peptide = get_sequence(peptide)
        #To use Helm notation and RdKit for nodes and bonds features
        polymer_type = "PEPTIDE"  # Tipo de polímero (en este caso, PEPTIDE)
        polymer_id = f"{polymer_type}{i + 1}"
        simple_polymer_helm = peptide_to_helm(peptide, polymer_id)
        peptides_list_helm.append(simple_polymer_helm) #create the list of peptides in Helm notation
        
    #nodes
    atomic_number = []
    aromaticity = []
    num_bonds = []
    bonded_hydrogens = []
    hybridization = []
    implicit_valence = []
    
    
    #edges
    bond_type = []
    in_ring = []
    conjugated = []
    bond_aromatic =[]
    valence_contribution_i = []
    valence_contribution_f = []
    
    for helm in peptides_list_helm:
        molecule = Chem.MolFromHELM(helm)
        
        atomic_number.extend([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
        aromaticity.extend([int(atom.GetIsAromatic()) for atom in molecule.GetAtoms()])
        num_bonds.extend([atom.GetDegree() for atom in molecule.GetAtoms()])
        bonded_hydrogens.extend([atom.GetTotalNumHs() for atom in molecule.GetAtoms()])
        hybridization.extend([atom.GetHybridization().real for atom in molecule.GetAtoms()])
        implicit_valence.extend([atom.GetImplicitValence () for atom in molecule.GetAtoms()])
    
        
        for bond in molecule.GetBonds():
            bond_type.extend([bond.GetBondTypeAsDouble()])
            in_ring.extend([int(bond.IsInRing())])
            conjugated.extend([int(bond.GetIsConjugated())])
            bond_aromatic.extend([int(bond.GetIsAromatic())])
            valence_contribution_i.extend([int(bond.GetValenceContrib(bond.GetBeginAtom()))])
            valence_contribution_f.extend([int(bond.GetValenceContrib(bond.GetEndAtom()))])

    #nodes
    set_atomic = list(set(atomic_number))
    codificador_atomic = OneHotEncoder()
    codificador_atomic.fit(np.array(set_atomic).reshape(-1,1))
    
    set_aromatic = list(set(aromaticity))
    codificador_aromatic = OneHotEncoder()
    codificador_aromatic.fit(np.array(set_aromatic).reshape(-1,1))
    
    set_bonds = list(set(num_bonds))
    codificador_bonds = OneHotEncoder()
    codificador_bonds.fit(np.array(set_bonds).reshape(-1,1))
    
    set_hydrogen = list(set(bonded_hydrogens))
    codificador_hydrogen = OneHotEncoder()
    codificador_hydrogen.fit(np.array(set_hydrogen).reshape(-1,1))   
    
    set_hybrid = list(set(hybridization))
    codificador_hybrid = OneHotEncoder()
    codificador_hybrid.fit(np.array(set_hybrid).reshape(-1,1))
    
    set_implicit_valence = list(set(implicit_valence))
    codificador_implicit_valence = OneHotEncoder()
    codificador_implicit_valence.fit(np.array(set_implicit_valence).reshape(-1,1))
    
    #edges
    set_bond_type = list(set(bond_type))
    codificador_bond_type = OneHotEncoder()
    codificador_bond_type.fit(np.array(set_bond_type).reshape(-1,1))
    
    set_in_ring = list(set(in_ring))
    codificador_in_ring= OneHotEncoder()
    codificador_in_ring.fit(np.array(set_in_ring).reshape(-1,1))
    
    set_conjugated = list(set(conjugated))
    codificador_conjugated= OneHotEncoder()
    codificador_conjugated.fit(np.array(set_conjugated).reshape(-1,1))
    
    set_aromatic_bond = list(set(bond_aromatic))
    codificador_aromatic_bond = OneHotEncoder()
    codificador_aromatic_bond.fit(np.array(set_aromatic_bond).reshape(-1,1))
    
    set_valence_contribution_i = list(set(valence_contribution_i))
    codificador_valence_contribution_i = OneHotEncoder()
    codificador_valence_contribution_i.fit(np.array(set_valence_contribution_i).reshape(-1,1))
    
    set_valence_contribution_f = list(set(valence_contribution_f))
    codificador_valence_contribution_f = OneHotEncoder()
    codificador_valence_contribution_f.fit(np.array(set_valence_contribution_f).reshape(-1,1))

    node_features_dict = defaultdict(list)
    edge_features_dict = defaultdict(list)
    
    for atom, aromatic, bonds, hydrogen, hybrid, impli_vale in zip(atomic_number, aromaticity, num_bonds, bonded_hydrogens, hybridization,implicit_valence ):
        
        node_key_features_combined = f"{atom}_{aromatic}_{bonds}_{hydrogen}_{hybrid}_{impli_vale}"
        
        atomic_feature  = codificador_atomic.transform([[atom]]).toarray()[0]
        aromatic_feature = codificador_aromatic.transform([[aromatic]]).toarray()[0]
        bonds_feature = codificador_bonds.transform([[bonds]]).toarray()[0]
        hydrogen_feature = codificador_hydrogen.transform([[hydrogen]]).toarray()[0]
        hybrid_feature = codificador_hybrid.transform([[hybrid]]).toarray()[0]
        impli_vale_feature = codificador_implicit_valence.transform([[impli_vale]]).toarray()[0]
        
        feature_node = np.concatenate((atomic_feature, aromatic_feature, bonds_feature, hydrogen_feature, hybrid_feature, impli_vale_feature))
        node_features_dict[node_key_features_combined] = feature_node
    
    for bond, ring, conjugat, aroma, valence_i, valence_f in zip(bond_type, in_ring, conjugated, bond_aromatic, valence_contribution_i, valence_contribution_f):
        edge_key_features_combined = f"{bond:.1f}_{ring:.1f}_{conjugat:.1f}_{aroma:.1f}_{valence_i:.1f}_{valence_f:.1f}" 
        
        bond_feature = codificador_bond_type.transform([[bond]]).toarray()[0]
        ring_feature = codificador_in_ring.transform([[ring]]).toarray()[0]
        conjugated_feature = codificador_conjugated.transform([[conjugat]]).toarray()[0] 
        aroma_feature = codificador_aromatic_bond.transform([[aroma]]).toarray()[0]
        valence_feature_i = codificador_valence_contribution_i.transform([[valence_i]]).toarray()[0]
        valence_feature_f = codificador_valence_contribution_f.transform([[valence_f]]).toarray()[0]
        
        feature_edge = np.concatenate((bond_feature, ring_feature, conjugated_feature, aroma_feature, valence_feature_i, valence_feature_f))
        edge_features_dict[edge_key_features_combined] = feature_edge
        
    
    return node_features_dict, edge_features_dict

#Getting the edges to the molecule
def get_edge_indices(molecule):
    edges_peptidic=[]
    for bond in molecule.GetBonds():
        edges_peptidic.append((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
        
    graph_edges = [[x[0] for x in edges_peptidic],[x[1] for x in edges_peptidic]]
    
    return torch.tensor(graph_edges, dtype=torch.long), edges_peptidic

#Getting the edges of the peptidic bonds
def get_non_peptide_idx(molecule):
    """
    Identifies and returns a list of non-peptide bond indices in a given molecule.
    
    This function iterates through the bonds of the molecule and evaluates various atomic properties
    to determine if each bond is non-peptidic. It checks the atomic numbers, hybridization,
    neighbors, and hydrogen counts of the atoms involved in each bond. Bonds that do not meet the
    criteria for peptide bonds are added to a list, which is returned as a list of tuples
    containing the indices of the atoms forming these bonds.    
    
    Parameters:
        molecule (rdkit.Chem.Mol): The RDKit molecule representation to be analyzed.
    
    Returns:
        list: A list of tuples, where each tuple contains the indices of atoms forming non-peptide bonds.
    """
    edges_nonpeptidic= []
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        atomic_num1 = atom1.GetAtomicNum()
        atomic_num2 = atom2.GetAtomicNum()
        neighbors_1 = list(atom1.GetNeighbors())
        neighbors_1_list = [neighbor.GetAtomicNum() for neighbor in neighbors_1]
        neighbors_2 = list(atom2.GetNeighbors())
        neighbors_2_list = [neighbor.GetAtomicNum() for neighbor in neighbors_2]  
        hibrid_1 = str(atom1.GetHybridization())
        hibrid_2 = str(atom2.GetHybridization())
        hidrog_1 = atom1.GetTotalNumHs()
        hidrog_2 = atom2.GetTotalNumHs()
        bond_type = str(bond.GetBondType())
        conjugated = str(bond.GetIsConjugated())
        #print(atom1.GetAtomicNum(), atom2.GetAtomicNum())

        if not(atomic_num1 == 6 and #C
            atomic_num2 == 7 and #N
            8 in neighbors_1_list and #O is neighbor of C
            hibrid_1 == 'SP2' and
            hibrid_2 == 'SP2' and
            hidrog_1 == 0 and  #C
            (hidrog_2 == 1 or  #N
            hidrog_2 == 0 )and  #N in Proline
            conjugated == 'True' and
            bond_type == 'SINGLE'): #ROC---NHR
            if not(atomic_num1 == 7 and   #N
                    atomic_num2 == 6 and  #C
                    8 in neighbors_2_list and #O is neighbor of C
                    hibrid_1 == 'SP2' and 
                    hibrid_2 == 'SP2' and
                    (hidrog_2 == 1 or  #N
                    hidrog_2 == 0 )and  #N in Proline
                    hidrog_2 == 0 and  #C
                    conjugated == 'True' and
                    bond_type == 'SINGLE'):
                edges_nonpeptidic.append((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
                
    
    
    return  edges_nonpeptidic

#Getting the label of the atoms based on the aminoacids e.g: ABC --> [1,1,1,2,2,2,2,2,3,3,3,3,3,3,3]
def get_label_aminoacid_atoms(edges_peptidic, edges_nonpeptidic, molecule):
    """
    Labels atoms in a molecule based on the provided peptide and non-peptide edges.

    This function identifies the unique bonds between peptide and non-peptide components 
    using symmetric difference set operations. It then fragments the molecule at these 
    unique bonds and assigns an index to each atom based on the fragment it belongs to.

    Parameters:
        edges_peptidic (list of tuples): A list of tuples representing edges in peptide components.
        edges_nonpeptidic (list of tuples): A list of tuples representing edges in non-peptide components.
        molecule (rdkit.Chem.Mol): The RDKit molecule representation to be processed.

    Returns:
        torch.Tensor: A tensor containing the fragment indices for each atom in the molecule, 
        indicating which fragment each atom belongs to.
    """
    set_with = set(edges_peptidic)
    set_witout = set(edges_nonpeptidic)
    tuplas_diferentes = list( set_with.symmetric_difference(set_witout))
    # lista_atoms = [elemento for tupla in tuplas_diferentes for elemento in tupla]
    
    break_idx = []
    for tupla in tuplas_diferentes:
        atom_1, atom_2 = tupla
        bond = molecule.GetBondBetweenAtoms(atom_1, atom_2)
        break_idx.append(bond.GetIdx())
        
    mol_f = Chem.FragmentOnBonds(molecule, break_idx, addDummies=False)
    fragmentos = list(Chem.GetMolFrags(mol_f, asMols=True))
    peptide_idx = np.empty(0)
    
    
    for i, fragme in enumerate(fragmentos):
        atoms_in_fragme = fragme.GetNumAtoms()
        idx_vector = np.ones(atoms_in_fragme)*(i)
        peptide_idx = np.concatenate((peptide_idx, idx_vector ))
        
    
    return  torch.tensor(peptide_idx.tolist(), dtype=torch.long)
    

#Aminoacids Features
def get_aminoacids_2(peptide):
    """
    This function processes a peptide sequence by performing several string manipulations:
    The function returns a list of amino acids or sequence components, preserving the intended structure of the peptide.
    Args:
    peptide (str): The input peptide sequence.
    
    Returns:
    list: A list of processed amino acids or modifications.
    """
    sequence = peptide.replace("(ac)", "[ac]*").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    sequence_helm = "".join(sequence)
    
    sequence_helm = ''.join([c + '.' if c.isupper() else c for c in sequence_helm])
    sequence_helm = sequence_helm.rstrip('.')
    
    # Crear una lista a partir de la cadena resultante separada por puntos
    sequence_list = sequence_helm.split('.')
    
    # Reemplazar "_" por "." en cada elemento de la lista
    sequence_list = [elem.replace('*', '.') for elem in sequence_list]
    
    return sequence_list

#Returns molecule from an aminoacid
def get_molecule(amino):
    polymer_id = "PEPTIDE1" 
    helm_notation = peptide_to_helm(amino, polymer_id)
    molecule = Chem.MolFromHELM(helm_notation)
    
    return molecule 

#BLOSUM
def get_sequence(peptide):
    sequence = peptide.replace("(ac)", "[ac].").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    
    return sequence

def construir_matriz_caracteristicas(sequence):
    secuencia = get_sequence(sequence)
    blosum62 = substitution_matrices.load("BLOSUM62")
    num_aminoacidos = len(blosum62.alphabet)
    longitud_secuencia = len(secuencia)
    
    matriz_caracteristicas = np.zeros((num_aminoacidos, longitud_secuencia))

    for i in range(longitud_secuencia):
        for j, aminoacido in enumerate(blosum62.alphabet[:num_aminoacidos]):
            puntuacion = blosum62[secuencia[i], aminoacido]
            matriz_caracteristicas[j, i] = puntuacion

    return matriz_caracteristicas.T
