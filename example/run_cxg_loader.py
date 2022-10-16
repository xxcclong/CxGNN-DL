import time
from tqdm import tqdm
import cxgnndl
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(config: DictConfig):
    s = OmegaConf.to_yaml(config)
    log.info(s)
    new_file_name = "new_config.yaml"
    with open(new_file_name, 'w') as f:
        s = s.replace("-", "  -")
        f.write(s)
    loader = cxgnndl.get_loader(config)
    t0 = time.time()
    for batch in tqdm(loader.train_loader):
        continue
    t1 = time.time()
    log.info(f"Time to load a batch: {t1-t0:.2f} seconds")

if __name__ == '__main__':
    main()