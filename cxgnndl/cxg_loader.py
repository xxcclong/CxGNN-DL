import cxgnndl_backend
from .feature_data import UVM


class CXGLoader:

    def __init__(self, config):
        self.backend = cxgnndl_backend.CXGDL("new_config.yaml")
        self.split = 'train'
        self.feat_mode = config["loading"]["feat_mode"]
        # if self.feat_mode in ["mmap", "uvm", "random"]:
        #     self.uvm = UVM(config)
        # else:
        #     self.uvm = None
        self.uvm = None

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
