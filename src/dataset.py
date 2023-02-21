from pathlib import Path
import numpy as np
import polars as pl
import torch
from rdkit import Chem
from torch_geometric.data import Data, Dataset

class CustomDataset(Dataset):
    def __init__(self, root, filename, is_test=False, transform=None, pre_transform=None):
        """
        root: Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        is_test: bool if is test data set or not
        """
        self.root = root
        self._is_test = is_test
        self.filename = filename
        self.data = pl.read_csv(self.raw_paths[0]).lazy()
        super(CustomDataset, self).__init__(root, transform, pre_transform)

    @property
    def is_test(self):
        return self._is_test
    
    @property
    def max_idx(self):
        return self.data.select(pl.max('index')).collect().item()
    
    @property
    def min_idx(self):
        return self.data.select(pl.min('index')).collect().item()
    
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        query = self.data.select(
            [pl.col("index").apply(self._mk_filename).apply(lambda x: x.name)]
        )
        return query.collect().to_dict(as_series=False)["index"]
    
    def get(self, idx):
        return torch.load(self._mk_filename(idx + self.min_idx).as_posix())

    def _iter_data(self):
        current_idx = self.min_idx
        while current_idx < self.max_idx + 1:
            query = self.data.select(['index', 'smiles', 'HIV_active']).filter(
                pl.col("index") == current_idx
            ).collect().to_dict(as_series=False)
            yield query["index"][0], query["smiles"][0], query["HIV_active"][0]
            current_idx += 1


    def process(self):
        for idx, smiles, raw_label in self._iter_data():
            mol_obj = Chem.MolFromSmiles(smiles)
            # Get node features
            node_feats = self._get_node_features(mol_obj)
            # Get edge features
            edge_feats = self._get_edge_features(mol_obj)
            # Get adjacency info
            edge_index = self._get_adjacency_info(mol_obj)
            # Get labels info
            label = self._get_labels(raw_label)

            # Create data object
            data = Data(
                x=node_feats, 
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=label,
                smiles=smiles
            ) 
            
            filename = self._mk_filename(idx).as_posix()
            torch.save(data, filename)

    def _mk_filename(self, idx):
        if self.is_test:
            return (Path(self.processed_dir) / f'data_test_{idx}.pt')
        return (Path(self.processed_dir) / f'data_{idx}.pt')

    def download(self):
        pass
    
    def len(self):
        return self.max_idx - self.min_idx
    def _get_node_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
    
    @property
    def feature_size(self):
        return self[0].x.shape[1]

    @property
    def edge_dim(self):
        return self[0].edge_attr.shape[1]