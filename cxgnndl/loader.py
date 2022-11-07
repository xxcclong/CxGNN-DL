from .cxg_loader import CXGLoader
from .dgl_loader import DGLLoader
from .pyg_loader import PyGLoader
from .custom_dgl_loader import CustomDGLLoader


def get_loader(config):
    return loader_dict[config.type](config)


loader_dict = {
    "cxg": CXGLoader,
    "pyg": PyGLoader,
    "dgl": DGLLoader,
    "customdgl": CustomDGLLoader,
}
