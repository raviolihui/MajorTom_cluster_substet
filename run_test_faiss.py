import yaml
from step2_kmeans_faiss import run

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['test_subset_size'] = 50000

run(cfg)
