import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.ranking import (
    AFM,
    AutoInt,
    DCN,
    DCNv2,
    DeepFFM,
    DeepFM,
    EDCN,
    FatDeepFFM,
    FiBiNet,
    WideDeep,
)
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, TorchDataset

sys.path.append("../..")


NON_SEQUENCE_CTR_MODELS = [
    "dcn",
    "dcn_v2",
    "deepfm",
    "widedeep",
    "fibinet",
    "edcn",
    "autoint",
    "afm",
    "deepffm",
    "fat_deepffm",
]


class VerboseCTRTrainer(CTRTrainer):
    def fit(self, train_dataloader, val_dataloader=None):
        for logger in self._iter_loggers():
            logger.log_hyperparams(
                {
                    "n_epoch": self.n_epoch,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "loss_mode": self.loss_mode,
                }
            )

        for epoch_i in range(self.n_epoch):
            print(f"epoch: {epoch_i}")
            train_loss = self.train_one_epoch(train_dataloader)
            print(f"epoch: {epoch_i} train loss: {train_loss:.6f}")

            for logger in self._iter_loggers():
                logger.log_metrics(
                    {
                        "train/loss": train_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=epoch_i,
                )

            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print(
                        "Current lr : {}".format(
                            self.optimizer.state_dict()["param_groups"][0]["lr"]
                        )
                    )
                self.scheduler.step()

            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                print(f"epoch: {epoch_i} validation: auc: {auc}")

                for logger in self._iter_loggers():
                    logger.log_metrics({"val/auc": auc}, step=epoch_i)

                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f"validation: best auc: {self.early_stopper.best_auc}")
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break

        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))

        for logger in self._iter_loggers():
            logger.finish()


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


def fit_sparse_encoders(train_df, val_df, test_df, feature_columns):
    encoded_train = train_df.copy()
    encoded_val = val_df.copy()
    encoded_test = test_df.copy()
    feature_defs = []
    ffm_linear_feature_defs = []
    ffm_cross_feature_defs = []

    for column in feature_columns:
        train_values = encoded_train[column].astype(str)
        known_values = sorted(set(train_values.tolist()))
        value_to_id = {value: index + 1 for index, value in enumerate(known_values)}
        encoded_train[column] = train_values.map(value_to_id).astype("int64")
        encoded_val[column] = (
            encoded_val[column].astype(str).map(value_to_id).fillna(0).astype("int64")
        )
        encoded_test[column] = (
            encoded_test[column].astype(str).map(value_to_id).fillna(0).astype("int64")
        )
        vocab_size = len(value_to_id) + 1
        feature_defs.append(SparseFeature(column, vocab_size=vocab_size, embed_dim=16))
        ffm_linear_feature_defs.append(
            SparseFeature(column, vocab_size=vocab_size, embed_dim=1)
        )
        ffm_cross_feature_defs.append(
            SparseFeature(
                column,
                vocab_size=vocab_size * len(feature_columns),
                embed_dim=10,
            )
        )

    return (
        encoded_train,
        encoded_val,
        encoded_test,
        feature_defs,
        ffm_linear_feature_defs,
        ffm_cross_feature_defs,
    )


def dataframe_to_input_dict(frame, feature_columns):
    return {
        column: frame[column].to_numpy(dtype=np.int64) for column in feature_columns
    }


def normalize_split_ratios(split_ratio):
    parts = [
        float(item.strip()) for item in str(split_ratio).split(":") if item.strip()
    ]
    if len(parts) != 3:
        raise ValueError(
            f"split_ratio must have three parts like 8:1:1, got {split_ratio}"
        )
    total = sum(parts)
    if total <= 0:
        raise ValueError(f"split_ratio must sum to a positive value, got {split_ratio}")
    return tuple(part / total for part in parts)


def split_frame_by_ratio(frame, seed, split_ratio):
    train_ratio, val_ratio, test_ratio = normalize_split_ratios(split_ratio)
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=frame["label"],
    )
    if val_ratio == 0:
        val_frame = temp_frame.iloc[0:0].copy()
        test_frame = temp_frame.copy()
    elif test_ratio == 0:
        val_frame = temp_frame.copy()
        test_frame = temp_frame.iloc[0:0].copy()
    else:
        val_share_in_temp = val_ratio / (val_ratio + test_ratio)
        val_frame, test_frame = train_test_split(
            temp_frame,
            test_size=1 - val_share_in_temp,
            random_state=seed,
            stratify=temp_frame["label"],
        )
    return (
        train_frame.reset_index(drop=True),
        val_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def load_static_feature_data(data_path, seed, split_ratio):
    source_raw = pd.read_csv(data_path)
    required_columns = {"user_id", "key", "value", "label"}
    missing_columns = required_columns - set(source_raw.columns)
    if missing_columns:
        raise ValueError(f"input csv missing columns: {sorted(missing_columns)}")

    full_wide, full_features = explode_kv_csv(source_raw)
    feature_columns = ["user_id"] + sorted(
        [column for column in full_features if column != "user_id"],
        key=lambda item: int(item[2:]),
    )
    for column in feature_columns:
        if column not in full_wide.columns:
            full_wide[column] = "0"
    full_wide[feature_columns] = full_wide[feature_columns].fillna("0").astype(str)

    train_wide, val_wide, test_wide = split_frame_by_ratio(full_wide, seed, split_ratio)
    for frame in (train_wide, val_wide, test_wide):
        for column in feature_columns:
            if column not in frame.columns:
                frame[column] = "0"
        frame[feature_columns] = frame[feature_columns].fillna("0").astype(str)

    (
        encoded_train,
        encoded_val,
        encoded_test,
        feature_defs,
        ffm_linear_feature_defs,
        ffm_cross_feature_defs,
    ) = fit_sparse_encoders(train_wide, val_wide, test_wide, feature_columns)

    train_part = encoded_train
    x_train = dataframe_to_input_dict(train_part, feature_columns)
    y_train = train_part["label"].to_numpy(dtype=np.float32)
    x_val = dataframe_to_input_dict(encoded_val, feature_columns)
    y_val = encoded_val["label"].to_numpy(dtype=np.float32)
    x_test = dataframe_to_input_dict(encoded_test, feature_columns)
    y_test = encoded_test["label"].to_numpy(dtype=np.float32)

    return (
        feature_defs,
        ffm_linear_feature_defs,
        ffm_cross_feature_defs,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )


def build_model(model_name, features):
    model_key = model_name.lower()
    mlp_params = {"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
    if model_key == "dcn_v2":
        return DCNv2(
            features=features,
            n_cross_layers=3,
            mlp_params=mlp_params,
        )
    if model_key == "dcn":
        return DCN(features=features, n_cross_layers=3, mlp_params=mlp_params)
    if model_key == "deepfm":
        return DeepFM(
            deep_features=features,
            fm_features=features,
            mlp_params=mlp_params,
        )
    if model_key == "widedeep":
        return WideDeep(
            wide_features=features,
            deep_features=features,
            mlp_params=mlp_params,
        )
    if model_key == "fibinet":
        return FiBiNet(features=features, reduction_ratio=3, mlp_params=mlp_params)
    if model_key == "edcn":
        return EDCN(features=features, n_cross_layers=3, mlp_params=mlp_params)
    if model_key == "autoint":
        return AutoInt(
            sparse_features=features,
            dense_features=[],
            num_layers=3,
            num_heads=2,
            dropout=0.2,
            mlp_params=mlp_params,
        )
    if model_key == "afm":
        return AFM(fm_features=features, embed_dim=16)
    raise ValueError(
        f"Unsupported model_name={model_name}. Choose from dcn, dcn_v2, deepfm, widedeep, fibinet, edcn, autoint, afm."
    )


def build_ffm_model(model_name, ffm_linear_features, ffm_cross_features):
    model_key = model_name.lower()
    ffm_mlp_params = {"dims": [1600, 1600], "dropout": 0.5, "activation": "relu"}
    if model_key == "deepffm":
        return DeepFFM(
            linear_features=ffm_linear_features,
            cross_features=ffm_cross_features,
            embed_dim=10,
            mlp_params=ffm_mlp_params,
        )
    if model_key == "fat_deepffm":
        return FatDeepFFM(
            linear_features=ffm_linear_features,
            cross_features=ffm_cross_features,
            embed_dim=10,
            reduction_ratio=1,
            mlp_params=ffm_mlp_params,
        )
    return None


def train_and_evaluate_model(
    model_name,
    features,
    ffm_linear_features,
    ffm_cross_features,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    learning_rate,
    weight_decay,
    epoch,
    device,
    save_dir,
):
    model = build_ffm_model(model_name, ffm_linear_features, ffm_cross_features)
    if model is None:
        model = build_model(model_name, features)

    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    trainer = VerboseCTRTrainer(
        model,
        optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
        n_epoch=epoch,
        earlystop_patience=5,
        device=device,
        model_path=model_save_dir,
    )
    trainer.fit(train_dataloader, val_dataloader)
    auc = trainer.evaluate(trainer.model, test_dataloader)
    print(f"{model_name} test auc: {auc:.6f}")
    return auc


def resolve_data_path(dataset_path, data_path):
    resolved_path = data_path or os.path.join(dataset_path, "data.csv")
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"input csv not found: {resolved_path}")
    return resolved_path


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
    data_path,
    model_name,
    epoch,
    learning_rate,
    batch_size,
    weight_decay,
    device,
    save_dir,
    seed,
    split_ratio,
):
    torch.manual_seed(seed)
    source_csv = resolve_data_path(dataset_path, data_path)
    (
        features,
        ffm_linear_features,
        ffm_cross_features,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ) = load_static_feature_data(source_csv, seed, split_ratio)
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

    if model_name == "all":
        scores = {}
        for current_model in NON_SEQUENCE_CTR_MODELS:
            print(f"\n===== training {current_model} =====")
            scores[current_model] = train_and_evaluate_model(
                current_model,
                features,
                ffm_linear_features,
                ffm_cross_features,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                learning_rate,
                weight_decay,
                epoch,
                device,
                save_dir,
            )
        print("\n===== auc summary =====")
        for current_model, score in sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        ):
            print(f"{current_model}: {score:.6f}")
        return

    train_and_evaluate_model(
        model_name,
        features,
        ffm_linear_features,
        ffm_cross_features,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        learning_rate,
        weight_decay,
        epoch,
        device,
        save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", default="./examples/ranking/data/static_int_features"
    )
    parser.add_argument("--data_path", default=None)
    parser.add_argument(
        "--model_name",
        default="dcn_v2",
        choices=[
            "all",
            "dcn",
            "dcn_v2",
            "deepfm",
            "widedeep",
            "fibinet",
            "edcn",
            "autoint",
            "afm",
            "deepffm",
            "fat_deepffm",
        ],
    )
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--save_dir", default="./examples/ranking/outputs/static_int_features"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split_ratio", default="8:1:1")

    args = parser.parse_args()
    main(
        args.dataset_path,
        args.data_path,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.device,
        args.save_dir,
        args.seed,
        args.split_ratio,
    )
