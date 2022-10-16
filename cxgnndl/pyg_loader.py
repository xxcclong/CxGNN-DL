from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected
import torch


class PyGLoader:
    def __init__(self, config):
        from ogb.nodeproppred import PygNodePropPredDataset
        self.dataset = PygNodePropPredDataset(
            name=config.dataset.name, root=config.dataset.path)
        self.data = self.dataset[0]
        if config.dataset.symmetric:
            self.data.edge_index = to_undirected(self.data.edge_index)
        self.split_idx = self.dataset.get_idx_split()

        def transform_wrapper(func):
            def transform(out):
                batch = func(out)
                batch_size = batch.batch_size
                if batch.y.size(0) != batch_size:
                    batch.y = batch.y[:batch_size].flatten().long()
                batch.to(config.device)
                batch.mask = torch.arange(batch_size, device=config.device)
                return batch
            return transform
        self.train_loader = NeighborLoader(
            self.data, input_nodes=self.split_idx['train'], num_neighbors=config.loader.train.fanouts[::-1],
            shuffle=True, batch_size=config.loader.train.batch_size, replace=config.loader.train.replace)
        self.train_loader.transform_fn = transform_wrapper(
            self.train_loader.transform_fn)
        self.val_loader = NeighborLoader(
            self.data, input_nodes=self.split_idx['valid'], num_neighbors=config.loader.eval.fanouts[::-1],
            shuffle=False, batch_size=config.loader.eval.batch_size, replace=config.loader.eval.replace)
        self.val_loader.transform_fn = transform_wrapper(
            self.val_loader.transform_fn)
        self.test_loader = NeighborLoader(
            self.data, input_nodes=self.split_idx['test'], num_neighbors=config.loader.eval.fanouts[::-1],
            shuffle=False, batch_size=config.loader.eval.batch_size, replace=config.loader.eval.replace)
        self.test_loader.transform_fn = transform_wrapper(
            self.test_loader.transform_fn)
