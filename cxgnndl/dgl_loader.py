import torch
import dgl


class DGLLoader:

    def __init__(self, config):
        from ogb.nodeproppred import DglNodePropPredDataset
        self.dataset = DglNodePropPredDataset(
            name=f"ogbn-{config.dataset.name}",
            root=config.dataset.path.strip(config.dataset.name))
        splitted_idx = self.dataset.get_idx_split()
        graph, labels = self.dataset[0]
        labels = labels[:, 0]
        graph.ndata['features'] = graph.ndata.pop('feat')
        graph.ndata['labels'] = labels
        # in_feats = graph.ndata['features'].shape[1]
        # num_labels = len(
        #     torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
        # Find the node IDs in the training, validation, and test set.
        train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx[
            'valid'], splitted_idx['test']
        train_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
        train_mask[train_nid] = True
        val_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
        val_mask[val_nid] = True
        test_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
        test_mask[test_nid] = True
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        self.graph = graph
        self.labels = labels

        train_sampler = dgl.dataloading.MultiLayerNeighborSampler(
            config.sampler.train.fanouts[::-1])
        val_sampler = dgl.dataloading.MultiLayerNeighborSampler(
            config.sampler.eval.fanouts[::-1])
        self.train_loader = dgl.dataloading.NodeDataLoader(
            self.graph,
            train_nid,
            train_sampler,
            device="cpu",
            batch_size=config.sampler.train.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4)

        self.val_loader = dgl.dataloading.NodeDataLoader(
            self.graph,
            val_nid,
            val_sampler,
            device="cpu",
            batch_size=config.sampler.eval.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4)

        self.test_loader = dgl.dataloading.NodeDataLoader(
            self.graph,
            test_nid,
            val_sampler,
            device="cpu",
            batch_size=config.sampler.eval.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4)