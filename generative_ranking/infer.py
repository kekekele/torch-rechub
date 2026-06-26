import argparse
import csv
import os

import torch

from .train import CTRTrainer
from .util import ensure_dir, load_config, set_seed


def _config_default(config, *keys, fallback=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return fallback
        current = current[key]
    return current


def _default_model_name(config):
    config_model = str(config.get("model_name", "")).lower()
    if config_model in {"dcn_v2", "rankmixer"}:
        return config_model
    return "rankmixer"


def build_parser(default_config):
    parser = argparse.ArgumentParser(
        description="Run inference for a trained DCNv2 or RankMixer checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="movielens",
        help="Config alias or python module path. Built-in alias: movielens.",
    )
    parser.add_argument(
        "--model_name",
        default=_default_model_name(default_config),
        choices=["dcn_v2", "rankmixer"],
        help="Model architecture used to build the network before loading checkpoint weights.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file to load. If omitted, infer.py uses <save_dir>/<model_name>/model.pth.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used for evaluation and optional prediction export.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional CSV path for prediction export. When omitted, only AUC is printed.",
    )
    parser.add_argument(
        "--data_dir",
        default=_config_default(default_config, "data_dir"),
        help="Override dataset root directory. The layout must match the selected config.",
    )
    parser.add_argument(
        "--device",
        default=_config_default(default_config, "device", fallback="cpu"),
        help="Torch device string used for model loading and inference, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_config_default(default_config, "seed", fallback=2026),
        help="Random seed applied before dataset preparation and inference.",
    )
    parser.add_argument(
        "--save_dir",
        default=_config_default(default_config, "save_dir", fallback="./outputs/"),
        help="Output root used to infer the default checkpoint path when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=_config_default(default_config, "dataset_params", "max_seq_len"),
        help="Maximum history length used when rebuilding sequence features for datasets such as MovieLens.",
    )
    return parser


def apply_infer_overrides(config, args):
    config["model_name"] = args.model_name
    config["data_dir"] = args.data_dir
    config["device"] = args.device
    config["save_dir"] = args.save_dir
    config["seed"] = args.seed
    if args.max_seq_len is not None:
        config.setdefault("dataset_params", {})["max_seq_len"] = args.max_seq_len
    return config


def resolve_checkpoint_path(args, config):
    if args.checkpoint:
        return args.checkpoint
    return os.path.join(config["save_dir"], args.model_name, "model.pth")


def main():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default="movielens")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    default_config = load_config(bootstrap_args.config)

    parser = build_parser(default_config)
    args = parser.parse_args()

    config = apply_infer_overrides(default_config, args)
    set_seed(config["seed"])
    from .data import prepare_dataset
    from .models import build_model

    bundle = prepare_dataset(config)
    model, loss_mode = build_model(args.model_name, bundle, config)
    device = torch.device(config["device"])
    checkpoint_path = resolve_checkpoint_path(args, config)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Pass --checkpoint explicitly or make sure training has saved <save_dir>/<model_name>/model.pth."
        )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    trainer = CTRTrainer(model, n_epoch=1, device=config["device"], loss_mode=loss_mode, model_path=ensure_dir(os.path.dirname(checkpoint_path) or "."))
    split_key = f"{args.split}_dl"
    data_loader = bundle[split_key]
    predictions = trainer.predict(trainer.model, data_loader)
    auc = trainer.evaluate(trainer.model, data_loader)
    print(f"{args.model_name} {args.split} auc: {auc:.6f}")
    if args.output_path:
        ensure_dir(os.path.dirname(args.output_path) or ".")
        with open(args.output_path, "w", newline="", encoding="utf-8") as fw:
            writer = csv.writer(fw)
            writer.writerow(["index", "prediction"])
            for index, prediction in enumerate(predictions):
                writer.writerow([index, prediction])
        print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()