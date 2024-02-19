import cxgnndl_backend
from .feature_data import UVM
import numpy as np


def load_full_graph_structure(name, undirected=True):
    basedir = ""
    if name in [
            "arxiv", "products", "reddit", "papers100M", "mag240m", "wiki90m"
    ] or "wiki" in name or "papers" in name or "arxiv" in name or "friendster" in name:
        basedir = "/home/huangkz/data/dataset_diskgnn/"
    elif name in ["rmag240m", "twitter", "friendster"]:
        basedir = "/mnt/data/huangkz/"
    else:
        assert 0
    if undirected:
        prop = "undirected"
    else:
        prop = "directed"
    ptr = np.fromfile(f"{basedir}{name}/processed/csr_ptr_{prop}.dat",
                      dtype=np.int64)
    idx = np.fromfile(f"{basedir}{name}/processed/csr_idx_{prop}.dat",
                      dtype=np.int64)
    return ptr, idx


class CXGLoader:

    def __init__(self, config):
        self.backend = cxgnndl_backend.CXGDL("new_config.yaml")
        self.split = 'train'
        self.feat_mode = config["loading"]["feat_mode"]
        print("CXGLoader feat_mode", self.feat_mode)
        if self.feat_mode in ["mmap", "uvm", "random"]:
            self.uvm = UVM(config)
        else:
            self.uvm = None
        # self.uvm = None

    def __len__(self):
        return self.backend.num_iters()

    def __iter__(self):
        return self

    def __next__(self) -> cxgnndl_backend.Batch:
        batch = self.backend.get_batch()
        if not self.uvm is None:
            batch.x = self.uvm.get(batch.sub_to_full)
        return batch

    @property
    def train_loader(self):
        self.split = 'train'
        self.backend.start(self.split)
        return self

    @property
    def val_loader(self):
        self.split = 'valid'
        self.backend.start(self.split)
        return self

    @property
    def test_loader(self):
        self.split = 'test'
        self.backend.start(self.split)
        return self
