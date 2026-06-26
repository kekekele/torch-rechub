import argparse
import csv
import os

from .train import CTRTrainer, apply_overrides
from .util import ensure_dir, load_config, set_seed

try:
    import torch
except ImportError:
    torch = None


def main():
    parser = argparse.ArgumentParser(description="Inference for DCNv2 and RankMixer from the standalone generative_ranking package")
    parser.add_argument("--config", default="movielens")
    parser.add_argument("--model_name", required=True, choices=["dcn_v2", "rankmixer"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_path")
    parser.add_argument("--data_dir")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_seq_len", type=int)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    if torch is None:
        raise ImportError("generative_ranking inference requires torch to be installed")
    set_seed(config["seed"])
    from .data import prepare_dataset
    from .models import build_model

    bundle = prepare_dataset(config)
    model, loss_mode = build_model(args.model_name, bundle, config)
    device = torch.device(config["device"])
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    trainer = CTRTrainer(model, n_epoch=1, device=config["device"], loss_mode=loss_mode, model_path=ensure_dir(os.path.dirname(args.checkpoint) or "."))
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