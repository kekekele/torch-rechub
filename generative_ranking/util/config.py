import copy
from importlib import import_module


def load_config(config_name):
    alias = str(config_name).strip().lower()
    if alias in {"movielens", "ml-1m", "ml1m"}:
        module_name = "generative_ranking.config.movielens"
    else:
        module_name = str(config_name)
    module = import_module(module_name)
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Config module {module_name} does not define CONFIG")
    return copy.deepcopy(module.CONFIG)