# PepMNet
![](https://github.com/danielgarzonotero/PepMNet/blob/main/PepMNet.gif)

# 1. Environment Setup

This project includes an `environment.yml` file for setting up the development environment.

### Create the Environment

i- Clone this repository:

```bash
git clone https://github.com/danielgarzonotero/PepMNet.git
```
```bash
cd PepMNet
```
ii- Create the environment:

```bash
conda env create -f environment.yml
```
iii- Activate the environment:

```bash
conda activate pepmnet
```
# 2. Processed Datasets

## AMP

The processed datasets used in this project can be found here:
https://zenodo.org/communities/pepmnet


For each dataset, the 3 .pt files must be saved in the directory:
data/AMP/dataset Ruiz/***"datasetname"***_processed/processed/

<img src="data/AMP_processed.png" alt="Descripción de la imagen" width="200" height="200"/>


**Reference**

The datasets used in this project were sourced from the following reference:

**Ruiz Puentes, P.; Henao, M. C.; Cifuentes, J.; Muñoz-Camargo, C.; Reyes, L. H.; Cruz, J. C.; Arbeláez, P.**  
*Rational Discovery of Antimicrobial Peptides by Means of Artificial Intelligence.* Membranes 2022, 12 (7), 708.  
DOI: [https://doi.org/10.3390/membranes12070708](https://doi.org/10.3390/membranes12070708)

Please ensure to cite this paper if you use these datasets.

## RT

**Processed Datasets**

The processed datasets used in this project can be found here:
https://zenodo.org/communities/pepmnet

For each dataset, the 3 .pt files must be saved in the directory:
data/RT/***"datasetname"***_processed/processed/

<img src="data/RT_processed.png" alt="Descripción de la imagen" width="250" height="280"/>



**References**

| File Name            | HPLC Type | Reference            |
|----------------------|-----------|----------------------|
| hela_mod.csv          | RPLC      | Sharma et al.        |
| yeast_unmod.csv       | RPLC      | Nagaraj et al.       |
| misc_dia.csv          | RPLC      | Rosenberger et al.   |
| SCX.csv               | SCX       | Gussakovsky et al.   |
| Luna_hilic.csv        | HILIC     | Spicer et al.        |
| xbridge_amide.csv     | HILIC     | Spicer et al.        |
| atlantis_silica.csv   | HILIC     | Spicer et al.        |
| luna_silica.csv       | HILIC     | Spicer et al.        |

- **Nagaraj, N.; Kulak, N. A.; Cox, J.; Neuhauser, N.; Mayr, K.; Hoerning, O.; Vorm, O.; Mann, M.**  
  *System-Wide Perturbation Analysis with Nearly Complete Coverage of the Yeast Proteome by Single-Shot Ultra HPLC Runs on a Bench Top Orbitrap.* Mol. Cell. Proteomics 2012, 11 (3), M111.013722 DOI: [10.1074/mcp.M111.013722](https://doi.org/10.1074/mcp.M111.013722)

- **Sharma, K.; D’Souza, R. C. J.; Tyanova, S.; Schaab, C.; Wiśniewski, J. R.; Cox, J.; Mann, M.**  
  *Ultradeep Human Phosphoproteome Reveals a Distinct Regulatory Nature of Tyr and Ser/Thr-Based Signaling.* Cell Rep. 2014, 8 (5), 1583–1594, DOI: [10.1016/j.celrep.2014.07.036](https://doi.org/10.1016/j.celrep.2014.07.036)

- **Rosenberger, G.; Koh, C. C.; Guo, T.; Röst, H. L.; Kouvonen, P.; Collins, B. C.; Heusel, M.; Liu, Y.; Caron, E.; Vichalkovski, A., et al.**  
  *A Repository of Assays to Quantify 10,000 Human Proteins by SWATH-MS.* Sci. Data 2014, 1, 140031, DOI: [10.1038/sdata.2014.31](https://doi.org/10.1038/sdata.2014.31)

- **Gussakovsky, D.; Neustaeter, H.; Spicer, V.; Krokhin, O. V.**  
  *Sequence-Specific Model for Peptide Retention Time Prediction in Strong Cation Exchange Chromatography.* Anal. Chem. 2017, 89 (21), 11795–11802, DOI: [10.1021/acs.analchem.7b03436](https://doi.org/10.1021/acs.analchem.7b03436)

- **Spicer, V.; Krokhin, O. V.**  
  *Peptide Retention Time Prediction in Hydrophilic Interaction Liquid Chromatography. Comparison of Separation Selectivity between Bare Silica and Bonded Stationary Phases.* J. Chromatogr. A 2018, 1534, 75–84, DOI: [10.1016/j.chroma.2017.12.046](https://doi.org/10.1016/j.chroma.2017.12.046)

**Acknowledgements**

All datasets were implemented and curated from:

- **Ma, C.; Ren, Y.; Yang, J.; Ren, Z.; Yang, H.; Liu, S.**  
  *Improved Peptide Retention Time Prediction in Liquid Chromatography through Deep Learning.* Anal. Chem. 2018, 90 (18), 10881–10888. DOI: [10.1021/acs.analchem.8b02386](https://doi.org/10.1021/acs.analchem.8b02386)

Please ensure to cite the respective papers if you use these datasets.

# 3. Pretrained models

Pretrained models used in this project can be found here: https://zenodo.org/communities/pepmnet

The **.pt** files must be saved in the directies: 

**Retention Time Prediction:**
weights_RT/

**Antimicrobial Classification:**
weights_AMP/

![](weigths/weigths.png)


# 4. Testing Independent Datasets

[`amp_tester.py`](https://github.com/danielgarzonotero/PepMNet/blob/main/amp_tester.py): 

To test independent sequences or datasets using the ensemble model for antimicrobial peptide (AMP) classification. It can be used to evaluate CSV files containing either labeled sequences (with associated antimicrobial activity) or sequences without labels.

[`rt_tester.py`](https://github.com/danielgarzonotero/PepMNet/blob/main/rt_tester.py) : 

To test independent sequences or datasets for retention time (RT) prediction. Users can select from 1 of 8 available models from avaliable. This script processes CSV files with or without RT target values.

# 5. Retraining 

To retrain the model or reprocess the datasets, please refer to the scripts [`amp_main.py`](https://github.com/danielgarzonotero/PepMNet/blob/main/amp_main.py) for antimicrobial peptide classification and [`rt_main.py`](https://github.com/danielgarzonotero/PepMNet/blob/main/rt_main.py) for retention time prediction. These scripts handle the full training process, including data preprocessing, model configuration, and training. You can adjust hyperparameters, dataset paths, and other settings within the scripts to suit your specific requirements.


## Citation

If you use this code or refer to this work in your research, please cite the following [paper](https://pubs.rsc.org/en/content/articlelanding/2025/me/d4me00172a) :

```
Garzon Otero, D.; Akbari, O.; Bilodeau, C. PepMNet: A Hybrid Deep Learning Model for
Predicting Peptide Properties Using Hierarchical Graph Representations. Mol. Syst. Des.
Eng. 2025, 10.1039.D4ME00172A. https://doi.org/10.1039/D4ME00172A.
```