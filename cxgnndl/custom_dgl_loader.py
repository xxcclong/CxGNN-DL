import torch
import dgl
import cxgnndl_backend


class CustomNeighborSampler(dgl.dataloading.Sampler):

    def __init__(self, fanouts: list[int]):
        super().__init__()
        self.fanouts = fanouts

    def sample(self, g, seed_nodes):
        output_nodes = seed_nodes
        _1, _2, full_ptr, full_idx, _3 = g._graph.adjacency_matrix_tensors(
            etype=0, transpose=False, fmt="csr")
        # print(type(full_ptr))
        # print(type(self.fanouts))
        # print(type(seed_nodes))
        ptr, idx, input_nodes, num_node_in_layer, num_edge_in_layer = cxgnndl_backend.neighbor_sample(
            full_ptr.tolist(), full_idx.tolist(), self.fanouts,
            seed_nodes.tolist())
        subgs = []
        num_layer = len(self.fanouts)
        for i in range(len(num_node_in_layer) - 1):
            num_src = num_node_in_layer[num_layer - i]
            num_dst = num_node_in_layer[num_layer - i - 1]
            ptr = ptr[:num_dst + 1]
            idx = idx[:num_edge_in_layer[num_layer - i - 1]]
            subgs.append(
                dgl.create_block(('csc', (ptr, idx, torch.tensor([]))),
                                 int(num_src), int(num_dst)))
        # print(num_node_in_layer)
        return input_nodes, output_nodes, subgs


# class CustomNeighborSampler(dgl.dataloading.Sampler):

#     def __init__(self, fanouts: list[int]):
#         super().__init__()
#         self.fanouts = fanouts

#     def sample(self, g, seed_nodes):
#         output_nodes = seed_nodes
#         subgs = []
#         for fanout in reversed(self.fanouts):
#             # Sample a fixed number of neighbors of the current seed nodes.
#             sg = g.sample_neighbors(seed_nodes, fanout)
#             # Convert this subgraph to a message flow graph.
#             sg = dgl.to_block(sg, seed_nodes)
#             seed_nodes = sg.srcdata[NID]
#             subgs.insert(0, sg)
#             input_nodes = seed_nodes
#         return input_nodes, output_nodes, subgs


class CustomDGLLoader:

    def __init__(self, config):
        print("CustomDGLLoader")
        from ogb.nodeproppred import DglNodePropPredDataset
        self.dataset = DglNodePropPredDataset(
            name=f"ogbn-{config.dataset.name}",
            root=config.dataset.path.strip(config.dataset.name))
        splitted_idx = self.dataset.get_idx_split()
        graph, labels = self.dataset[0]
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
        labels = labels[:, 0]
        if config.loading.feat_mode in ["uvm"]:
            self.feat = dgl.contrib.UnifiedTensor(graph.ndata.pop('feat'),
                                                  device=torch.device(
                                                      config.device))
        else:
            assert False, "Not implemented"
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

        train_sampler = CustomNeighborSampler(
            config.sampler.train.fanouts[::-1])
        val_sampler = CustomNeighborSampler(config.sampler.eval.fanouts[::-1])
        num_thread = config.dgl.num_thread
        assert num_thread >= 0 and num_thread <= 32
        self.train_loader = dgl.dataloading.NodeDataLoader(
            self.graph,
            train_nid,
            train_sampler,
            device="cpu",
            batch_size=config.sampler.train.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_thread)

        self.val_loader = dgl.dataloading.NodeDataLoader(
            self.graph,
            val_nid,
            val_sampler,
            device="cpu",
            batch_size=config.sampler.eval.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_thread)

        self.test_loader = dgl.dataloading.NodeDataLoader(
            self.graph,
            test_nid,
            val_sampler,
            device="cpu",
            batch_size=config.sampler.eval.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_thread)