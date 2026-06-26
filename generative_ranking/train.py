import argparse
import os

import torch
import tqdm
from sklearn.metrics import roc_auc_score

from .util import ensure_dir, load_config, set_seed
from .basic.callback import EarlyStopper


class CTRTrainer(object):
    def __init__(self, model, optimizer_params=None, regularization_params=None, n_epoch=10, earlystop_patience=10, device="cpu", loss_mode=True, model_path="./"):
        from .basic.loss_func import RegularizationLoss

        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5} if optimizer_params is None else optimizer_params
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        regularization_params = {"embedding_l1": 0.0, "embedding_l2": 0.0, "dense_l1": 0.0, "dense_l2": 0.0} if regularization_params is None else regularization_params
        self.loss_mode = loss_mode
        self.criterion = torch.nn.BCELoss()
        self.evaluate_fn = roc_auc_score
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path
        self.reg_loss_fn = RegularizationLoss(**regularization_params)

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0.0
        epoch_loss = 0.0
        batch_count = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            y = y.to(self.device).float()
            if self.loss_mode:
                y_pred = self.model(x_dict)
                loss = self.criterion(y_pred, y)
            else:
                y_pred, other_loss = self.model(x_dict)
                loss = self.criterion(y_pred, y) + other_loss
            loss = loss + self.reg_loss_fn(self.model)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0.0
        return epoch_loss / batch_count if batch_count else 0.0

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            print("epoch:", epoch_i)
            self.train_one_epoch(train_dataloader)
            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                print("epoch:", epoch_i, "validation: auc:", auc)
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f"validation: best auc: {self.early_stopper.best_auc}")
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = [], []
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for x_dict, y in tk0:
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device).float().view(-1, 1)
                if self.loss_mode:
                    y_pred = model(x_dict)
                else:
                    y_pred, _ = model(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts)

    def predict(self, model, data_loader):
        model.eval()
        predicts = []
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for x_dict, _ in tk0:
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                if self.loss_mode:
                    y_pred = model(x_dict)
                else:
                    y_pred, _ = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts


def apply_overrides(config, args):
    config["model_name"] = args.model_name
    config["data_dir"] = args.data_dir
    config["device"] = args.device
    config["save_dir"] = args.save_dir
    config["training"]["epoch"] = args.epoch
    config["training"]["learning_rate"] = args.learning_rate
    config["training"]["batch_size"] = args.batch_size
    config["training"]["weight_decay"] = args.weight_decay
    config["seed"] = args.seed
    if args.max_seq_len is not None:
        config.setdefault("dataset_params", {})["max_seq_len"] = args.max_seq_len
    rankmixer_cfg = config["rankmixer"]
    rankmixer_cfg["d_model"] = args.rankmixer_d_model
    rankmixer_cfg["num_layers"] = args.rankmixer_num_layers
    rankmixer_cfg["num_tokens"] = args.rankmixer_num_tokens
    rankmixer_cfg["use_moe"] = args.rankmixer_use_moe
    rankmixer_cfg["moe_experts"] = args.rankmixer_moe_experts
    rankmixer_cfg["moe_l1_coef"] = args.rankmixer_moe_l1_coef
    rankmixer_cfg["moe_sparsity_ratio"] = args.rankmixer_moe_sparsity_ratio
    rankmixer_cfg["moe_routing_type"] = args.rankmixer_moe_routing_type
    rankmixer_cfg["input_dropout"] = args.rankmixer_input_dropout
    rankmixer_cfg["token_mixing_dropout"] = args.rankmixer_token_mixing_dropout
    rankmixer_cfg["ffn_dropout"] = args.rankmixer_ffn_dropout
    rankmixer_cfg["head_dropout"] = args.rankmixer_head_dropout
    return config


def _config_default(config, *keys, fallback=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return fallback
        current = current[key]
    return current


def build_parser(default_config):
    parser = argparse.ArgumentParser(
        description="Train DCNv2 and RankMixer with the standalone generative_ranking package.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    experiment_group = parser.add_argument_group("Experiment")
    experiment_group.add_argument(
        "--config",
        default="movielens",
        help="Config alias or python module path. Built-in alias: movielens.",
    )
    experiment_group.add_argument(
        "--model_name",
        default=_config_default(default_config, "model_name", fallback="all"),
        choices=["all", "dcn_v2", "rankmixer"],
        help="Choose which model to train. 'all' runs DCNv2 and RankMixer on the same dataset split.",
    )
    experiment_group.add_argument(
        "--data_dir",
        default=_config_default(default_config, "data_dir"),
        help="Override dataset root directory. Expected file layout depends on the selected config.",
    )
    experiment_group.add_argument(
        "--device",
        default=_config_default(default_config, "device", fallback="cpu"),
        help="Torch device string, for example cpu, cuda:0, or npu:0 if your runtime supports it.",
    )
    experiment_group.add_argument(
        "--save_dir",
        default=_config_default(default_config, "save_dir", fallback="./outputs/"),
        help="Directory used to save per-model checkpoints and outputs.",
    )
    experiment_group.add_argument(
        "--seed",
        type=int,
        default=_config_default(default_config, "seed", fallback=2026),
        help="Random seed applied to torch, numpy, and python random.",
    )

    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--epoch", type=int, default=_config_default(default_config, "training", "epoch", fallback=5), help="Number of training epochs.")
    training_group.add_argument("--learning_rate", type=float, default=_config_default(default_config, "training", "learning_rate", fallback=1e-3), help="Adam learning rate.")
    training_group.add_argument("--batch_size", type=int, default=_config_default(default_config, "training", "batch_size", fallback=1024), help="Mini-batch size for train/valid/test dataloaders.")
    training_group.add_argument("--weight_decay", type=float, default=_config_default(default_config, "training", "weight_decay", fallback=1e-5), help="Adam weight decay coefficient.")

    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument(
        "--max_seq_len",
        type=int,
        default=_config_default(default_config, "dataset_params", "max_seq_len"),
        help="Maximum user history length. Used by sequence-style datasets such as MovieLens.",
    )

    rankmixer_group = parser.add_argument_group("RankMixer")
    rankmixer_group.add_argument("--rankmixer_d_model", type=int, default=_config_default(default_config, "rankmixer", "d_model", fallback=128), help="Hidden token dimension used by RankMixer.")
    rankmixer_group.add_argument("--rankmixer_num_layers", type=int, default=_config_default(default_config, "rankmixer", "num_layers", fallback=2), help="Number of RankMixer encoder blocks.")
    rankmixer_group.add_argument("--rankmixer_num_tokens", type=int, default=_config_default(default_config, "rankmixer", "num_tokens", fallback=4), help="Target token count produced by semantic tokenization.")
    rankmixer_group.add_argument(
        "--rankmixer_use_moe",
        dest="rankmixer_use_moe",
        action="store_true",
        help="Enable sparse MoE in RankMixer feed-forward layers.",
    )
    rankmixer_group.add_argument(
        "--rankmixer_no_moe",
        dest="rankmixer_use_moe",
        action="store_false",
        help="Disable sparse MoE and use the plain per-token FFN.",
    )
    rankmixer_group.set_defaults(rankmixer_use_moe=_config_default(default_config, "rankmixer", "use_moe", fallback=True))
    rankmixer_group.add_argument("--rankmixer_moe_experts", type=int, default=_config_default(default_config, "rankmixer", "moe_experts", fallback=4), help="Number of experts used when MoE is enabled.")
    rankmixer_group.add_argument("--rankmixer_moe_l1_coef", type=float, default=_config_default(default_config, "rankmixer", "moe_l1_coef", fallback=0.0), help="L1 regularization coefficient applied to MoE routing.")
    rankmixer_group.add_argument("--rankmixer_moe_sparsity_ratio", type=float, default=_config_default(default_config, "rankmixer", "moe_sparsity_ratio", fallback=1.0), help="Expected routing sparsity ratio used in the MoE penalty.")
    rankmixer_group.add_argument("--rankmixer_moe_routing_type", default=_config_default(default_config, "rankmixer", "moe_routing_type", fallback="relu_dtsi"), help="MoE routing type, for example relu_dtsi or relu.")
    rankmixer_group.add_argument("--rankmixer_input_dropout", type=float, default=_config_default(default_config, "rankmixer", "input_dropout", fallback=0.0), help="Dropout applied before the RankMixer encoder.")
    rankmixer_group.add_argument("--rankmixer_token_mixing_dropout", type=float, default=_config_default(default_config, "rankmixer", "token_mixing_dropout", fallback=0.0), help="Dropout applied inside parameter-free token mixing.")
    rankmixer_group.add_argument("--rankmixer_ffn_dropout", type=float, default=_config_default(default_config, "rankmixer", "ffn_dropout", fallback=0.0), help="Dropout applied inside the per-token FFN or MoE experts.")
    rankmixer_group.add_argument("--rankmixer_head_dropout", type=float, default=_config_default(default_config, "rankmixer", "head_dropout", fallback=0.0), help="Dropout applied in the prediction head after token pooling.")

    return parser


def train_single_model(model_name, bundle, config):
    from .models import build_model

    training_cfg = config["training"]
    model_dir = ensure_dir(os.path.join(config["save_dir"], model_name))
    model, loss_mode = build_model(model_name, bundle, config)
    trainer = CTRTrainer(
        model,
        optimizer_params={"lr": training_cfg["learning_rate"], "weight_decay": training_cfg["weight_decay"]},
        n_epoch=training_cfg["epoch"],
        earlystop_patience=training_cfg["earlystop_patience"],
        device=config["device"],
        loss_mode=loss_mode,
        model_path=model_dir,
    )
    trainer.fit(bundle["train_dl"], bundle["val_dl"])
    auc = trainer.evaluate(trainer.model, bundle["test_dl"])
    print(f"{model_name} test auc: {auc:.6f}")
    return auc


def main():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default="movielens")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    default_config = load_config(bootstrap_args.config)

    parser = build_parser(default_config)
    args = parser.parse_args()

    config = apply_overrides(default_config, args)
    print(f"Loaded config: {config}")
    set_seed(config["seed"])
    from .data import prepare_dataset

    bundle = prepare_dataset(config)
    model_name = config["model_name"]
    auc_results = {}
    if model_name in ("all", "dcn_v2"):
        auc_results["dcn_v2"] = train_single_model("dcn_v2", bundle, config)
    if model_name in ("all", "rankmixer"):
        auc_results["rankmixer"] = train_single_model("rankmixer", bundle, config)
    print("AUC summary:")
    for current_model, auc in auc_results.items():
        print(f"  {current_model}: {auc:.6f}")


if __name__ == "__main__":
    main()