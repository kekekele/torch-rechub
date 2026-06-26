import os
import sys

import numpy as np
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DCNv2, RankMixer, build_rankmixer_semantic_groups, normalize_rankmixer_group_schema
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

sys.path.append("../..")


ALI_CCP_REQUIRED_FILES = (
    "ali_ccp_train_sample.csv",
    "ali_ccp_val_sample.csv",
    "ali_ccp_test_sample.csv",
)

ALI_CCP_DENSE_COLS = ["D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853"]

DEFAULT_RANKMIXER_CONFIG = {
    "d_model": 128,
    "num_layers": 2,
    "num_tokens": 5,
    "use_moe": False,
    "moe_experts": 4,
    "moe_l1_coef": 0.0,
    "moe_sparsity_ratio": 1.0,
    "moe_use_dtsi": True,
    "moe_routing_type": "relu_dtsi",
    "input_dropout": 0.0,
    "token_mixing_dropout": 0.0,
    "ffn_dropout": 0.0,
    "head_dropout": 0.0,
}


def ensure_ali_ccp_data(data_dir):
    required_paths = [os.path.join(data_dir, filename) for filename in ALI_CCP_REQUIRED_FILES]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Ali-CCP sample files are required and must already exist:\n{missing_list}")


def get_ali_ccp_frames(data_dir):
    df_train = pd.read_csv(os.path.join(data_dir, "ali_ccp_train_sample.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "ali_ccp_val_sample.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "ali_ccp_test_sample.csv"))
    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    return df_train, df_val, df_test


def get_ali_ccp_columns(data):
    col_names = data.columns.tolist()
    dense_cols = list(ALI_CCP_DENSE_COLS)
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ["click", "purchase"]]
    return dense_cols, sparse_cols


def build_feature_columns(data):
    dense_cols, sparse_cols = get_ali_ccp_columns(data)
    dense_features = [DenseFeature(col) for col in dense_cols]
    sparse_features = [SparseFeature(col, vocab_size=int(data[col].max()) + 1, embed_dim=16) for col in sparse_cols]
    return dense_features, sparse_features


def build_rankmixer_schema(dense_cols, sparse_cols):
    dense_base_cols = {col[1:] for col in dense_cols if col.startswith("D")}
    sparse_stats_cols = [col for col in sparse_cols if col in dense_base_cols]
    sparse_family_1 = [col for col in sparse_cols if col.startswith("1") and col not in sparse_stats_cols]
    sparse_family_2 = [col for col in sparse_cols if col.startswith("2") and col not in sparse_stats_cols]
    sparse_family_3 = [col for col in sparse_cols if col.startswith("3") and col not in sparse_stats_cols]
    sparse_other = [
        col for col in sparse_cols if col not in sparse_stats_cols and col not in sparse_family_1 and col not in sparse_family_2 and col not in sparse_family_3
    ]
    schema = normalize_rankmixer_group_schema([
        {"name": "sparse_family_1xx", "features": sparse_family_1},
        {"name": "sparse_family_2xx", "features": sparse_family_2},
        {"name": "sparse_family_3xx", "features": sparse_family_3},
        {"name": "sparse_statistics", "features": sparse_stats_cols + sparse_other},
        {"name": "dense_statistics", "features": dense_cols},
    ])
    return [group for group in schema if group["features"]]


def frame_to_model_input(frame, label_col="click"):
    x = frame.copy()
    y = x.pop(label_col).to_numpy(dtype=np.float32)
    return x, y


def build_dataloaders(data_dir, batch_size):
    ensure_ali_ccp_data(data_dir)
    df_train, df_val, df_test = get_ali_ccp_frames(data_dir)
    data = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)
    dense_features, sparse_features = build_feature_columns(data)
    dense_cols, sparse_cols = get_ali_ccp_columns(data)
    semantic_schema = build_rankmixer_schema(dense_cols, sparse_cols)

    train_x, train_y = frame_to_model_input(df_train)
    val_x, val_y = frame_to_model_input(df_val)
    test_x, test_y = frame_to_model_input(df_test)

    generator = DataGenerator(train_x, train_y)
    train_dl, val_dl, test_dl = generator.generate_dataloader(
        x_val=val_x,
        y_val=val_y,
        x_test=test_x,
        y_test=test_y,
        batch_size=batch_size,
    )
    print(f"Ali-CCP CTR samples: train={len(train_y)}, valid={len(val_y)}, test={len(test_y)}")
    return dense_features, sparse_features, semantic_schema, train_dl, val_dl, test_dl


def build_rankmixer_config(args):
    rankmixer_config = dict(DEFAULT_RANKMIXER_CONFIG)
    rankmixer_config.update({
        "d_model": args.rankmixer_d_model,
        "num_layers": args.rankmixer_num_layers,
        "num_tokens": args.rankmixer_num_tokens,
        "use_moe": args.rankmixer_use_moe,
        "moe_experts": args.rankmixer_moe_experts,
        "moe_l1_coef": args.rankmixer_moe_l1_coef,
        "moe_sparsity_ratio": args.rankmixer_moe_sparsity_ratio,
        "moe_routing_type": args.rankmixer_moe_routing_type,
        "input_dropout": args.rankmixer_input_dropout,
        "token_mixing_dropout": args.rankmixer_token_mixing_dropout,
        "ffn_dropout": args.rankmixer_ffn_dropout,
        "head_dropout": args.rankmixer_head_dropout,
    })
    return rankmixer_config


def fit_and_evaluate(model_name, model, loss_mode, train_dl, val_dl, test_dl, epoch, learning_rate, weight_decay, device, save_dir):
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    trainer = CTRTrainer(
        model,
        optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
        n_epoch=epoch,
        earlystop_patience=3,
        device=device,
        model_path=model_dir,
        loss_mode=loss_mode,
    )
    trainer.fit(train_dl, val_dl)
    auc = trainer.evaluate(trainer.model, test_dl)
    print(f"{model_name} test auc: {auc:.6f}")
    return auc


def train_and_eval_dcn_v2(dense_features, sparse_features, train_dl, val_dl, test_dl, epoch, learning_rate, weight_decay, device, save_dir):
    model = DCNv2(
        features=dense_features + sparse_features,
        n_cross_layers=3,
        mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    )
    return fit_and_evaluate(
        model_name="dcn_v2",
        model=model,
        loss_mode=True,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        epoch=epoch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        save_dir=save_dir,
    )


def train_and_eval_rankmixer(dense_features, sparse_features, semantic_schema, rankmixer_config, train_dl, val_dl, test_dl, epoch, learning_rate, weight_decay, device, save_dir):
    use_moe = rankmixer_config["use_moe"]
    model = RankMixer(
        features=dense_features + sparse_features,
        d_model=rankmixer_config["d_model"],
        num_layers=rankmixer_config["num_layers"],
        num_tokens=rankmixer_config["num_tokens"],
        semantic_groups=build_rankmixer_semantic_groups(semantic_schema),
        use_moe=use_moe,
        moe_experts=rankmixer_config["moe_experts"],
        moe_l1_coef=rankmixer_config["moe_l1_coef"],
        moe_sparsity_ratio=rankmixer_config["moe_sparsity_ratio"],
        moe_use_dtsi=rankmixer_config["moe_use_dtsi"],
        moe_routing_type=rankmixer_config["moe_routing_type"],
        input_dropout=rankmixer_config["input_dropout"],
        token_mixing_dropout=rankmixer_config["token_mixing_dropout"],
        ffn_dropout=rankmixer_config["ffn_dropout"],
        head_dropout=rankmixer_config["head_dropout"],
        return_moe_loss=use_moe,
    )
    return fit_and_evaluate(
        model_name="rankmixer",
        model=model,
        loss_mode=not use_moe,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        epoch=epoch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        save_dir=save_dir,
    )


def main(data_dir, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed, rankmixer_config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dense_features, sparse_features, semantic_schema, train_dl, val_dl, test_dl = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
    )

    auc_results = {}
    if model_name in ("all", "dcn_v2"):
        auc_results["dcn_v2"] = train_and_eval_dcn_v2(
            dense_features=dense_features,
            sparse_features=sparse_features,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            epoch=epoch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            save_dir=save_dir,
        )
    if model_name in ("all", "rankmixer"):
        auc_results["rankmixer"] = train_and_eval_rankmixer(
            dense_features=dense_features,
            sparse_features=sparse_features,
            semantic_schema=semantic_schema,
            rankmixer_config=rankmixer_config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            epoch=epoch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            save_dir=save_dir,
        )

    print("AUC summary:")
    for current_model, auc in auc_results.items():
        print(f"  {current_model}: {auc:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DCNv2 vs RankMixer on Ali-CCP CTR")
    parser.add_argument("--data_dir", default="./data/ali-ccp/")
    parser.add_argument("--model_name", default="all", choices=["all", "dcn_v2", "rankmixer"])
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default="./data/ali-ccp/saved_rankmixer/")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rankmixer_d_model", type=int, default=DEFAULT_RANKMIXER_CONFIG["d_model"])
    parser.add_argument("--rankmixer_num_layers", type=int, default=DEFAULT_RANKMIXER_CONFIG["num_layers"])
    parser.add_argument("--rankmixer_num_tokens", type=int, default=DEFAULT_RANKMIXER_CONFIG["num_tokens"])
    parser.add_argument("--rankmixer_use_moe", dest="rankmixer_use_moe", action="store_true")
    parser.add_argument("--rankmixer_no_moe", dest="rankmixer_use_moe", action="store_false")
    parser.set_defaults(rankmixer_use_moe=DEFAULT_RANKMIXER_CONFIG["use_moe"])
    parser.add_argument("--rankmixer_moe_experts", type=int, default=DEFAULT_RANKMIXER_CONFIG["moe_experts"])
    parser.add_argument("--rankmixer_moe_l1_coef", type=float, default=DEFAULT_RANKMIXER_CONFIG["moe_l1_coef"])
    parser.add_argument("--rankmixer_moe_sparsity_ratio", type=float, default=DEFAULT_RANKMIXER_CONFIG["moe_sparsity_ratio"])
    parser.add_argument("--rankmixer_moe_routing_type", default=DEFAULT_RANKMIXER_CONFIG["moe_routing_type"])
    parser.add_argument("--rankmixer_input_dropout", type=float, default=DEFAULT_RANKMIXER_CONFIG["input_dropout"])
    parser.add_argument("--rankmixer_token_mixing_dropout", type=float, default=DEFAULT_RANKMIXER_CONFIG["token_mixing_dropout"])
    parser.add_argument("--rankmixer_ffn_dropout", type=float, default=DEFAULT_RANKMIXER_CONFIG["ffn_dropout"])
    parser.add_argument("--rankmixer_head_dropout", type=float, default=DEFAULT_RANKMIXER_CONFIG["head_dropout"])
    args = parser.parse_args()

    rankmixer_config = build_rankmixer_config(args)
    print(f"RankMixer config: {rankmixer_config}")
    main(
        data_dir=args.data_dir,
        model_name=args.model_name,
        epoch=args.epoch,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        device=args.device,
        save_dir=args.save_dir,
        seed=args.seed,
        rankmixer_config=rankmixer_config,
    )