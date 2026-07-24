import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.ranking import AutoInt, DCNv2, DeepFM, WideDeep
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, TorchDataset

sys.path.append("../..")


def parse_kv_text(text):
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def explode_kv_csv(frame):
    records = []
    all_keys = set()
    for row in frame.itertuples(index=False):
        keys = parse_kv_text(row.key)
        values = parse_kv_text(row.value)
        if len(keys) != len(values):
            raise ValueError(
                f"Mismatched key/value length for user_id={row.user_id}: "
                f"len(key)={len(keys)} len(value)={len(values)}"
            )

        record = {"user_id": str(row.user_id), "label": int(row.label)}
        for key, value in zip(keys, values):
            feature_name = f"f_{key}"
            record[feature_name] = str(value)
            all_keys.add(feature_name)
        records.append(record)

    wide = pd.DataFrame(records)
    feature_columns = ["user_id"] + sorted(all_keys, key=lambda item: int(item[2:]))
    for column in feature_columns:
        if column not in wide.columns:
            wide[column] = "0"
    wide[feature_columns] = wide[feature_columns].fillna("0").astype(str)
    wide["label"] = wide["label"].astype("int32")
    return wide, feature_columns


def fit_sparse_encoders(train_df, test_df, feature_columns):
    encoded_train = train_df.copy()
    encoded_test = test_df.copy()
    feature_defs = []

    for column in feature_columns:
        train_values = encoded_train[column].astype(str)
        known_values = sorted(set(train_values.tolist()))
        value_to_id = {value: index + 1 for index, value in enumerate(known_values)}
        encoded_train[column] = train_values.map(value_to_id).astype("int64")
        encoded_test[column] = (
            encoded_test[column].astype(str).map(value_to_id).fillna(0).astype("int64")
        )
        feature_defs.append(
            SparseFeature(column, vocab_size=len(value_to_id) + 1, embed_dim=16)
        )

    return encoded_train, encoded_test, feature_defs


def dataframe_to_input_dict(frame, feature_columns):
    return {
        column: frame[column].to_numpy(dtype=np.int64) for column in feature_columns
    }


def load_static_feature_data(train_path, test_path, seed, val_ratio):
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    required_columns = {"user_id", "key", "value", "label"}
    missing_train = required_columns - set(train_raw.columns)
    missing_test = required_columns - set(test_raw.columns)
    if missing_train:
        raise ValueError(f"train csv missing columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"test csv missing columns: {sorted(missing_test)}")

    train_wide, train_features = explode_kv_csv(train_raw)
    test_wide, test_features = explode_kv_csv(test_raw)
    feature_columns = sorted(
        set(train_features + test_features),
        key=lambda item: (-1, 0) if item == "user_id" else (0, int(item[2:])),
    )
    for frame in (train_wide, test_wide):
        for column in feature_columns:
            if column not in frame.columns:
                frame[column] = "0"
        frame[feature_columns] = frame[feature_columns].fillna("0").astype(str)

    encoded_train, encoded_test, feature_defs = fit_sparse_encoders(
        train_wide, test_wide, feature_columns
    )

    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    if val_ratio > 0:
        train_part, val_part = train_test_split(
            encoded_train,
            test_size=val_ratio,
            random_state=seed,
            stratify=encoded_train["label"],
        )
    else:
        train_part = encoded_train
        val_part = None

    x_train = dataframe_to_input_dict(train_part, feature_columns)
    y_train = train_part["label"].to_numpy(dtype=np.float32)
    if val_part is not None:
        x_val = dataframe_to_input_dict(val_part, feature_columns)
        y_val = val_part["label"].to_numpy(dtype=np.float32)
    else:
        x_val = None
        y_val = None
    x_test = dataframe_to_input_dict(encoded_test, feature_columns)
    y_test = encoded_test["label"].to_numpy(dtype=np.float32)

    return feature_defs, x_train, y_train, x_val, y_val, x_test, y_test


def build_model(model_name, features):
    model_key = model_name.lower()
    if model_key == "dcn_v2":
        return DCNv2(
            features=features,
            n_cross_layers=3,
            mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
        )
    if model_key == "deepfm":
        return DeepFM(
            deep_features=[],
            fm_features=features,
            mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
        )
    if model_key == "widedeep":
        return WideDeep(
            wide_features=[],
            deep_features=features,
            mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
        )
    if model_key == "autoint":
        return AutoInt(
            dense_features=[],
            sparse_features=features,
            num_layers=3,
            num_heads=2,
            dropout=0.2,
            mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
        )
    raise ValueError(
        f"Unsupported model_name={model_name}. Choose from dcn_v2, deepfm, widedeep, autoint."
    )


def resolve_data_paths(dataset_path, train_path, test_path):
    resolved_train = train_path or os.path.join(dataset_path, "train.csv")
    resolved_test = test_path or os.path.join(dataset_path, "test.csv")
    if not os.path.exists(resolved_train):
        raise FileNotFoundError(f"train csv not found: {resolved_train}")
    if not os.path.exists(resolved_test):
        raise FileNotFoundError(f"test csv not found: {resolved_test}")
    return resolved_train, resolved_test


def build_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    if x_val is not None and y_val is not None:
        dg = DataGenerator(x_train, y_train)
        return dg.generate_dataloader(
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            batch_size=batch_size,
        )

    train_dataset = TorchDataset(x_train, y_train)
    test_dataset = TorchDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, None, test_dataloader


def main(
    dataset_path,
    train_path,
    test_path,
    model_name,
    epoch,
    learning_rate,
    batch_size,
    weight_decay,
    device,
    save_dir,
    seed,
    val_ratio,
):
    torch.manual_seed(seed)
    train_csv, test_csv = resolve_data_paths(dataset_path, train_path, test_path)
    (
        features,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ) = load_static_feature_data(train_csv, test_csv, seed, val_ratio)
    print(
        f"train={len(y_train)} val={0 if y_val is None else len(y_val)} test={len(y_test)} features={len(features)}"
    )

    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size,
    )

    model = build_model(model_name, features)
    trainer = CTRTrainer(
        model,
        optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
        n_epoch=epoch,
        earlystop_patience=5,
        device=device,
        model_path=save_dir,
    )
    trainer.fit(train_dataloader, val_dataloader)
    auc = trainer.evaluate(trainer.model, test_dataloader)
    print(f"test auc: {auc:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", default="./examples/ranking/data/static_int_features"
    )
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--test_path", default=None)
    parser.add_argument(
        "--model_name",
        default="dcn_v2",
        choices=["dcn_v2", "deepfm", "widedeep", "autoint"],
    )
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--save_dir", default="./examples/ranking/outputs/static_int_features"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    args = parser.parse_args()
    main(
        args.dataset_path,
        args.train_path,
        args.test_path,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.device,
        args.save_dir,
        args.seed,
        args.val_ratio,
    )
