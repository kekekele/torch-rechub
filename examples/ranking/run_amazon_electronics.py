import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from torch_rechub.basic.metric import log_loss
from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.ranking import DCNv2, DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

sys.path.append("../..")


def get_amazon_data_dict(dataset_path):
    data = pd.read_csv(dataset_path)
    print("========== Start Amazon ==========")
    train, val, test = generate_seq_feature(
        data=data,
        user_col="user_id",
        item_col="item_id",
        time_col="time",
        item_attribute_cols=["cate_id"],
        min_item=5,
        max_len=101,
    )
    print("INFO: Now, the dataframe named: ", train.columns)

    n_users = data["user_id"].max()
    n_items = data["item_id"].max()
    n_cates = data["cate_id"].max()

    features = [
        SparseFeature(
            "target_item_id", vocab_size=n_items + 1, embed_dim=8, padding_idx=0
        ),
        SparseFeature(
            "target_cate_id", vocab_size=n_cates + 1, embed_dim=8, padding_idx=0
        ),
        SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8, padding_idx=0),
    ]
    target_features = features
    din_history_features = [
        SequenceFeature(
            "hist_item_id",
            vocab_size=n_items + 1,
            embed_dim=8,
            pooling="concat",
            shared_with="target_item_id",
            padding_idx=0,
        ),
        SequenceFeature(
            "hist_cate_id",
            vocab_size=n_cates + 1,
            embed_dim=8,
            pooling="concat",
            shared_with="target_cate_id",
            padding_idx=0,
        ),
    ]
    dcn_history_features = [
        SequenceFeature(
            "hist_item_id",
            vocab_size=n_items + 1,
            embed_dim=8,
            pooling="mean",
            shared_with="target_item_id",
            padding_idx=0,
        ),
        SequenceFeature(
            "hist_cate_id",
            vocab_size=n_cates + 1,
            embed_dim=8,
            pooling="mean",
            shared_with="target_cate_id",
            padding_idx=0,
        ),
    ]

    print("========== Generate input dict ==========")
    train = df_to_dict(train)
    val = df_to_dict(val)
    test = df_to_dict(test)
    train_y, val_y, test_y = train["label"], val["label"], test["label"]

    del train["label"]
    del val["label"]
    del test["label"]
    train_x, val_x, test_x = train, val, test
    return (
        features,
        target_features,
        din_history_features,
        dcn_history_features,
        (train_x, train_y),
        (val_x, val_y),
        (test_x, test_y),
    )


def build_model(
    model_name, features, target_features, din_history_features, dcn_history_features
):
    if model_name == "din":
        return DIN(
            features=features,
            history_features=din_history_features,
            target_features=target_features,
            mlp_params={"dims": [256, 128]},
            attention_mlp_params={"dims": [256, 128]},
        )
    if model_name == "dcn_v2":
        return DCNv2(
            features=features + dcn_history_features,
            n_cross_layers=3,
            mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def evaluate_auc_and_logloss(trainer, data_loader, labels):
    predicts = np.asarray(trainer.predict(trainer.model, data_loader), dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    predicts = np.clip(predicts, 1e-12, 1 - 1e-12)
    return roc_auc_score(labels, predicts), log_loss(labels, predicts)


def main(
    dataset_path,
    model_name,
    epoch,
    learning_rate,
    batch_size,
    weight_decay,
    device,
    save_dir,
    seed,
):
    torch.manual_seed(seed)
    (
        features,
        target_features,
        din_history_features,
        dcn_history_features,
        (train_x, train_y),
        (val_x, val_y),
        (test_x, test_y),
    ) = get_amazon_data_dict(dataset_path)
    dg = DataGenerator(train_x, train_y)

    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(
        x_val=val_x,
        y_val=val_y,
        x_test=test_x,
        y_test=test_y,
        batch_size=batch_size,
    )
    model = build_model(
        model_name,
        features,
        target_features,
        din_history_features,
        dcn_history_features,
    )

    ctr_trainer = CTRTrainer(
        model,
        optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
        n_epoch=epoch,
        earlystop_patience=4,
        device=device,
        model_path=save_dir,
    )
    ctr_trainer.fit(train_dataloader, val_dataloader)
    val_auc, val_logloss = evaluate_auc_and_logloss(ctr_trainer, val_dataloader, val_y)
    test_auc, test_logloss = evaluate_auc_and_logloss(
        ctr_trainer, test_dataloader, test_y
    )
    print(f"valid auc: {val_auc:.6f}, valid logloss: {val_logloss:.6f}")
    print(f"test auc: {test_auc:.6f}, test logloss: {test_logloss:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="./data/amazon-electronics/amazon_electronics_sample.csv",
    )
    parser.add_argument("--model_name", default="din", choices=["din", "dcn_v2"])
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default="./")
    parser.add_argument("--seed", type=int, default=2022)

    args = parser.parse_args()
    main(
        args.dataset_path,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.device,
        args.save_dir,
        args.seed,
    )

"""
python run_amazon_electronics.py
python run_amazon_electronics.py --model_name dcn_v2
"""
