import cxgnndl_backend


class CXGLoader:
    def __init__(self, config):
        self.backend = cxgnndl_backend.CXGDL("new_config.yaml")
        self.split = 'train'

    def __len__(self):
        return self.backend.num_iters()

    def __iter__(self):
        return self

    def __next__(self) -> cxgnndl_backend.Batch:
        return self.backend.get_batch()

    @ property
    def train_loader(self):
        self.split = 'train'
        self.backend.start(self.split)
        return self

    @ property
    def val_loader(self):
        self.split = 'valid'
        self.backend.start(self.split)
        return self

    @ property
    def test_loader(self):
        self.split = 'test'
        self.backend.start(self.split)
        return self
