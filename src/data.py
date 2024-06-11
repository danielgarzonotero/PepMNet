import pandas as pd
import os
import torch
from torch_geometric.data import InMemoryDataset

from src.utils import sequences_geodata_1, sequences_geodata_2, get_features
from src.device import device_info
from src.aminoacids_features import get_aminoacid_features

#---------------------------------------(Training, Validation, Testing) with target values ---------------------------------------
class GeoDatasetBase_1(InMemoryDataset):
    def __init__(self, root, raw_name, index_x, index_y, transform=None, pre_transform=None, **kwargs):
        self.filename = raw_name  # La ruta completa ya se proporciona
        self.df = pd.read_csv(self.filename)
        self.x = self.df[self.df.columns[index_x]].values
        self.y = self.df[self.df.columns[index_y]].values 
        
        
        super(GeoDatasetBase_1, self).__init__(root=os.path.join(root, f'{raw_name.split(".")[0]}_processed'), transform=transform, pre_transform=pre_transform, **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data_list = []
        cc = 0
        
        node_features_dict, edge_features_dict = get_features(self.x)
        aminoacids_ft_dict = get_aminoacid_features()
        
        device_info_instance = device_info()
        device = device_info_instance.device
        
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            data_list.append(sequences_geodata_1(   cc=cc,
                                                    sequence=x,
                                                    y=y,
                                                    aminoacids_ft_dict=aminoacids_ft_dict,
                                                    node_ft_dict=node_features_dict,
                                                    edge_ft_dict=edge_features_dict,
                                                    device=device)
                            )
            
            cc += 1
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GeoDataset_1(GeoDatasetBase_1):
    def __init__(self, root, raw_name, index_x, index_y, transform=None, pre_transform=None, **kwargs):
        super(GeoDataset_1, self).__init__(root=root, raw_name=raw_name, index_x=index_x, index_y=index_y, transform=transform, pre_transform=pre_transform, **kwargs)

    def processed_file_names(self):
        return [f'{os.path.splitext(os.path.basename(self.filename))[0]}.pt']
    
    

#--------------------------------------- Independet sequences without target values ---------------------------------------

class GeoDatasetBase_2(InMemoryDataset):
    def __init__(self, root, raw_name, index_x, transform=None, pre_transform=None, **kwargs):
        self.filename = raw_name  # La ruta completa ya se proporciona
        self.df = pd.read_csv(self.filename)
        self.x = self.df[self.df.columns[index_x]].values
        
        super(GeoDatasetBase_2, self).__init__(root=os.path.join(root, f'{raw_name.split(".")[0]}_processed'), transform=transform, pre_transform=pre_transform, **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        node_ft_dict, edge_ft_dict = get_features(self.x)
        data_list = []
        cc = 0
        aminoacids_ft_dict = get_aminoacid_features()
        device_info_instance = device_info()
        device = device_info_instance.device

        for cc, sequence in enumerate(self.x):
            data_list.append(sequences_geodata_2(
                                                cc=cc,
                                                sequence=sequence,
                                                aminoacids_ft_dict=aminoacids_ft_dict,
                                                node_ft_dict=node_ft_dict,
                                                edge_ft_dict=edge_ft_dict,
                                                device=device
                                                ))
            cc += 1
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class GeoDataset_2(GeoDatasetBase_2):
    def __init__(self, root, raw_name, index_x, transform=None, pre_transform=None, **kwargs):
        super(GeoDataset_2, self).__init__(root=root, raw_name=raw_name, index_x=index_x, transform=transform, pre_transform=pre_transform, **kwargs)

    def processed_file_names(self):
        return [f'{os.path.splitext(os.path.basename(self.filename))[0]}.pt']