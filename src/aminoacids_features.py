from collections import defaultdict
import numpy as np

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

from sklearn.preprocessing import OneHotEncoder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from src.utils import get_aminoacids

import os
import torch

def get_aminoacid_features(): 
    
    
    wt_amino =[]
    aromaticity_amino = []
    hydrophobicity_amino = []
    net_charge_amino = []
    p_iso_amino = []
    logp_amino = []
    atoms_amino = []

    aminoacids_set = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    for amino in aminoacids_set:
        amino_mol = get_molecule(amino)
        amino_biopython = ProteinAnalysis(amino)
        
        #molecular weight:
        wt_amino.extend([round(Descriptors.MolWt(amino_mol), 4)])
        aromaticity_amino.extend([round(amino_biopython.aromaticity(), 4)])
        hydrophobicity_amino.extend([round(amino_biopython.gravy(), 4)])
        net_charge_amino.extend([round(amino_biopython.charge_at_pH(7), 4)])
        p_iso_amino.extend([round(amino_biopython.isoelectric_point(), 4)])
        logp_amino.extend([round(Crippen.MolLogP(amino_mol), 4)])
        atoms_amino.extend([round(float(amino_mol.GetNumAtoms()), 4)])
    
    set_aromaticity_amino = list(set(aromaticity_amino))
    codificador_aromaticity_amino = OneHotEncoder()
    codificador_aromaticity_amino.fit(np.array(set_aromaticity_amino).reshape(-1, 1))
    
    aminoacids_features_dict = defaultdict(list)
    
    for wt, aromati, hydrophobicity, net_charge, p_iso, logp, atoms in zip(wt_amino, aromaticity_amino, hydrophobicity_amino, net_charge_amino, p_iso_amino, logp_amino, atoms_amino):
        aminoacid_key_features_combined = f"{wt}_{aromati}_{hydrophobicity}_{net_charge}_{p_iso}_{logp}_{atoms}"
        
        wt_feature = np.array([wt])
        aromati_feature = codificador_aromaticity_amino.transform([[aromati]]).toarray()[0]
        hydrophobicity_feature = np.array([hydrophobicity])
        net_charge_feature = np.array([net_charge])
        p_iso_feature = np.array([p_iso])
        logp_feature = np.array([logp])
        atoms_feature = np.array([atoms])
        
        aminoacid_feature = np.concatenate((
                                            wt_feature,
                                            aromati_feature,
                                            hydrophobicity_feature,
                                            net_charge_feature,
                                            p_iso_feature,
                                            logp_feature,
                                            atoms_feature
                                            ))
        
        aminoacids_features_dict[aminoacid_key_features_combined] = aminoacid_feature

    return aminoacids_features_dict
    
    
    
def get_molecule(amino):
    polymer_id = "PEPTIDE1" 
    helm_notation = peptide_to_helm(amino, polymer_id)
    molecule = Chem.MolFromHELM(helm_notation)
    
    return molecule 

def peptide_to_helm(peptide, polymer_id):
    sequence = peptide.replace("(ac)", "[ac].").replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
    sequence_helm = "".join(sequence)
    
    sequence_helm = ''.join([c + '.' if c.isupper() else c for i, c in enumerate(sequence_helm)])
    sequence_helm = sequence_helm.rstrip('.')
    
    sequence_helm = f"{polymer_id}{{{sequence_helm}}}$$$$"
    
    return sequence_helm