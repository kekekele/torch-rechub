from .movielens import prepare_movielens_dataset


def prepare_dataset(config):
    dataset_name = str(config["dataset"]).lower()
    if dataset_name in {"movielens", "ml-1m", "ml1m"}:
        return prepare_movielens_dataset(config)
    raise ValueError(f"Unsupported dataset={config['dataset']}")