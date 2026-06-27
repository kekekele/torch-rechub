import copy
import json
from pathlib import Path
from importlib import import_module


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as fr:
        return json.load(fr)


def _resolve_config_dir(config_name):
    alias = str(config_name).strip().lower()
    if alias in {"movielens", "ml-1m", "ml1m"}:
        return CONFIG_ROOT / "movielens"
    config_path = Path(str(config_name)).expanduser()
    if config_path.is_dir():
        return config_path
    return None


def _load_from_json_dir(config_dir, profile):
    data_config = _load_json_file(config_dir / "data.json")
    train_config = _load_json_file(config_dir / "train.json")
    merged = _deep_merge(data_config, train_config)
    if profile == "infer":
        infer_path = config_dir / "infer.json"
        if infer_path.exists():
            merged = _deep_merge(merged, _load_json_file(infer_path))
    return merged


def load_config(config_name, profile="train"):
    profile_name = str(profile).strip().lower()
    if profile_name not in {"train", "infer"}:
        raise ValueError(f"Unsupported config profile: {profile}")

    config_dir = _resolve_config_dir(config_name)
    if config_dir is not None:
        required_files = [config_dir / "data.json", config_dir / "train.json"]
        missing = [str(path) for path in required_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing config files under {config_dir}: {missing}")
        return _load_from_json_dir(config_dir, profile_name)

    module_name = str(config_name)
    module = import_module(module_name)
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Config module {module_name} does not define CONFIG")
    return copy.deepcopy(module.CONFIG)