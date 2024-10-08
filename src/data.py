import pandas as pd
import os
import torch
from torch_geometric.data import InMemoryDataset

from src.utils import sequences_geodata_1, get_features
from src.device import device_info
from src.aminoacids_features import get_aminoacid_features

#---------------------------------------(Training, Validation, Testing) with target values ---------------------------------------
class GeoDatasetBase_1(InMemoryDataset):
    def __init__(self, root, raw_name, index_x, index_y, has_targets, transform=None, pre_transform=None, **kwargs):
        self.filename = raw_name  # La ruta completa ya se proporciona
        
        self.df = pd.read_csv(self.filename)
        self.x = self.df[self.df.columns[index_x]].values
        self.has_targets = has_targets
        
        if has_targets:
            self.y = self.df[self.df.columns[index_y]].values
        else:
            self.y = None 
        
        #self.df_samples = pd.read_csv('data/RT/sample_sequences.csv', header=None)  # 'header=0' es opcional si ya hay un encabezado
        #self.samples = self.df_samples.iloc[:, index_x].values 
        
        super(GeoDatasetBase_1, self).__init__(root=os.path.join(root, f'{raw_name.split(".")[0]}_processed'), transform=transform, pre_transform=pre_transform, **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location= 'cuda:0' if torch.cuda.is_available() else 'cpu')

    def process(self):
        data_list = []
        cc = 0
        
        node_features_dict, edge_features_dict = get_features()
        aminoacids_ft_dict = get_aminoacid_features()
        
        device_info_instance = device_info()
        device = device_info_instance.device
        
        if self.has_targets:
            for i, (x, y) in enumerate(zip(self.x, self.y)):
                data_list.append(sequences_geodata_1(   cc=cc,
                                                        sequence=x,
                                                        y=y,
                                                        aminoacids_ft_dict=aminoacids_ft_dict,
                                                        node_ft_dict=node_features_dict,
                                                        edge_ft_dict=edge_features_dict,
                                                        device=device,
                                                        has_targets= self.has_targets
                                                        )
                                )
                
                cc += 1
        else:
            for cc, x in enumerate(self.x):
                data_list.append(sequences_geodata_1(   cc=cc,
                                                        sequence=x,
                                                        y=self.y,
                                                        aminoacids_ft_dict=aminoacids_ft_dict,
                                                        node_ft_dict=node_features_dict,
                                                        edge_ft_dict=edge_features_dict,
                                                        device=device,
                                                        has_targets= self.has_targets
                                                        )
                                )
            cc += 1
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GeoDataset_1(GeoDatasetBase_1):
    def __init__(self, root, raw_name, index_x, index_y, has_targets, transform=None, pre_transform=None, **kwargs):
        super(GeoDataset_1, self).__init__(root=root, raw_name=raw_name, index_x=index_x, index_y=index_y, has_targets=has_targets, transform=transform, pre_transform=pre_transform, **kwargs)

    def processed_file_names(self):
        return [f'{os.path.splitext(os.path.basename(self.filename))[0]}.pt']
    