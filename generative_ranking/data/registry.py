from .dataset import prepare_three_table_dataset_from_config


def prepare_dataset(config):
    if config.get("source", {}).get("tables"):
        display_name = str(config.get("dataset_display_name", config["dataset"]))
        return prepare_three_table_dataset_from_config(config, dataset_name=display_name, desc=f"build {display_name.lower()} ranking samples")
    raise ValueError(f"Unsupported dataset={config['dataset']}")