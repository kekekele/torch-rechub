import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.ranking import DCNv2, RankMixer, build_rankmixer_semantic_groups, normalize_rankmixer_group_schema
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, pad_sequences

sys.path.append("../..")


MOVIELENS_REQUIRED_FILES = ("ratings.dat", "movies.dat", "users.dat")


def ensure_movielens_data(data_dir):
    required_paths = [os.path.join(data_dir, filename) for filename in MOVIELENS_REQUIRED_FILES]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"MovieLens-1M raw files are required and must already exist:\n{missing_list}")


def load_movielens_frames(data_dir):
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"),
        sep="::",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="utf-8",
    )
    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"),
        sep="::",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip"],
        engine="python",
        encoding="utf-8",
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"),
        sep="::",
        header=None,
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="utf-8",
    )
    movies["genre"] = movies["genres"].fillna("unknown").apply(lambda value: value.split("|")[0] if value else "unknown")
    data = ratings.merge(users, on="user_id", how="left").merge(movies[["movie_id", "genre"]], on="movie_id", how="left")
    data["label"] = (data["rating"] > 3.5).astype("int32")
    return data[["user_id", "movie_id", "gender", "age", "occupation", "zip", "genre", "timestamp", "label"]]


def encode_sparse_features(data, sparse_cols):
    encoded = data.copy()
    encoders = {}
    for col in sparse_cols:
        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(encoded[col].astype(str)) + 1
        encoders[col] = encoder
    encoded["label"] = encoded["label"].astype("int32")
    encoded["timestamp"] = encoded["timestamp"].astype("int64")
    return encoded, encoders


def build_rank_samples(data, max_seq_len=50):
    samples = []
    data = data.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    for _, hist in tqdm(data.groupby("user_id", sort=False), desc="build ranking samples"):
        pos_item_hist = []
        pos_genre_hist = []
        for row in hist.itertuples(index=False):
            if pos_item_hist:
                samples.append({
                    "label": int(row.label),
                    "timestamp": int(row.timestamp),
                    "user_id": int(row.user_id),
                    "gender": int(row.gender),
                    "age": int(row.age),
                    "occupation": int(row.occupation),
                    "zip": int(row.zip),
                    "target_item_id": int(row.movie_id),
                    "target_genre_id": int(row.genre),
                    "hist_item_id": pos_item_hist[-max_seq_len:],
                    "hist_genre_id": pos_genre_hist[-max_seq_len:],
                })
            if int(row.label) == 1:
                pos_item_hist.append(int(row.movie_id))
                pos_genre_hist.append(int(row.genre))
    if not samples:
        raise ValueError("No ranking samples were constructed from MovieLens-1M.")
    return pd.DataFrame(samples).sort_values("timestamp").reset_index(drop=True)


def split_rank_samples(samples):
    n_samples = len(samples)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    train = samples.iloc[:train_end].copy()
    val = samples.iloc[train_end:val_end].copy()
    test = samples.iloc[val_end:].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Train/valid/test split produced an empty partition.")
    return train, val, test


def frame_to_model_input(frame, max_seq_len):
    frame = frame.copy()
    hist_item_id = pad_sequences(frame["hist_item_id"].tolist(), maxlen=max_seq_len, padding="pre", truncating="pre", value=0)
    hist_genre_id = pad_sequences(frame["hist_genre_id"].tolist(), maxlen=max_seq_len, padding="pre", truncating="pre", value=0)
    labels = frame.pop("label").to_numpy(dtype=np.float32)
    frame.drop(columns=["timestamp"], inplace=True)
    x_dict = {
        "user_id": frame["user_id"].to_numpy(dtype=np.int64),
        "gender": frame["gender"].to_numpy(dtype=np.int64),
        "age": frame["age"].to_numpy(dtype=np.int64),
        "occupation": frame["occupation"].to_numpy(dtype=np.int64),
        "zip": frame["zip"].to_numpy(dtype=np.int64),
        "target_item_id": frame["target_item_id"].to_numpy(dtype=np.int64),
        "target_genre_id": frame["target_genre_id"].to_numpy(dtype=np.int64),
        "hist_item_id": hist_item_id.astype(np.int64),
        "hist_genre_id": hist_genre_id.astype(np.int64),
    }
    return x_dict, labels


def build_feature_columns(encoded):
    embed_dim = 16
    user_features = [
        SparseFeature("user_id", vocab_size=int(encoded["user_id"].max()) + 1, embed_dim=embed_dim, padding_idx=0),
        SparseFeature("gender", vocab_size=int(encoded["gender"].max()) + 1, embed_dim=embed_dim, padding_idx=0),
        SparseFeature("age", vocab_size=int(encoded["age"].max()) + 1, embed_dim=embed_dim, padding_idx=0),
        SparseFeature("occupation", vocab_size=int(encoded["occupation"].max()) + 1, embed_dim=embed_dim, padding_idx=0),
        SparseFeature("zip", vocab_size=int(encoded["zip"].max()) + 1, embed_dim=embed_dim, padding_idx=0),
    ]
    item_features = [
        SparseFeature("target_item_id", vocab_size=int(encoded["movie_id"].max()) + 1, embed_dim=embed_dim, padding_idx=0),
        SparseFeature("target_genre_id", vocab_size=int(encoded["genre"].max()) + 1, embed_dim=embed_dim, padding_idx=0),
    ]
    seq_features = [
        SequenceFeature(
            "hist_item_id",
            vocab_size=int(encoded["movie_id"].max()) + 1,
            embed_dim=embed_dim,
            pooling="concat",
            shared_with="target_item_id",
            padding_idx=0,
        ),
        SequenceFeature(
            "hist_genre_id",
            vocab_size=int(encoded["genre"].max()) + 1,
            embed_dim=embed_dim,
            pooling="concat",
            shared_with="target_genre_id",
            padding_idx=0,
        ),
    ]
    semantic_schema = normalize_rankmixer_group_schema([
        {
            "name": "user_profile",
            "features": [feature.name for feature in user_features],
        },
        {
            "name": "target_item",
            "features": [feature.name for feature in item_features],
        },
        {
            "name": "sequence_global",
            "sequence_features": [feature.name for feature in seq_features],
            "pool_modes": ("mean",),
        },
        {
            "name": "sequence_target",
            "sequence_features": [feature.name for feature in seq_features],
            "pool_modes": ("target",),
        },
    ])
    return user_features, item_features, seq_features, semantic_schema


def build_dataloaders(data_dir, max_seq_len, batch_size):
    ensure_movielens_data(data_dir)
    sparse_cols = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "genre"]
    raw = load_movielens_frames(data_dir)
    encoded, _ = encode_sparse_features(raw, sparse_cols)
    samples = build_rank_samples(encoded, max_seq_len=max_seq_len)
    train, val, test = split_rank_samples(samples)
    train_x, train_y = frame_to_model_input(train, max_seq_len=max_seq_len)
    val_x, val_y = frame_to_model_input(val, max_seq_len=max_seq_len)
    test_x, test_y = frame_to_model_input(test, max_seq_len=max_seq_len)
    generator = DataGenerator(train_x, train_y)
    train_dl, val_dl, test_dl = generator.generate_dataloader(
        x_val=val_x,
        y_val=val_y,
        x_test=test_x,
        y_test=test_y,
        batch_size=batch_size,
    )
    user_features, item_features, seq_features, semantic_schema = build_feature_columns(encoded)
    print(f"MovieLens-1M ranking samples: train={len(train_y)}, valid={len(val_y)}, test={len(test_y)}")
    return user_features, item_features, seq_features, semantic_schema, train_dl, val_dl, test_dl


def build_model(model_name, user_features, item_features, seq_features, semantic_schema):
    if model_name == "dcn_v2":
        return DCNv2(
            features=user_features + item_features + seq_features,
            n_cross_layers=3,
            mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
        ), True
    if model_name == "rankmixer":
        seq_pool_modes = ("mean", "target")
        return RankMixer(
            features=user_features + item_features,
            sequence_features=seq_features,
            d_model=128,
            num_layers=2,
            num_tokens=4,
            semantic_groups=build_rankmixer_semantic_groups(semantic_schema, default_seq_pool_modes=seq_pool_modes),
            seq_pool_modes=seq_pool_modes,
            use_moe=True,
            moe_experts=4,
            moe_l1_coef=1e-4,
            moe_sparsity_ratio=0.5,
            moe_use_dtsi=True,
            moe_routing_type="relu_dtsi",
            input_dropout=0.1,
            token_mixing_dropout=0.1,
            ffn_dropout=0.1,
            head_dropout=0.1,
        ), False
    raise ValueError(f"Unsupported model_name: {model_name}")


def train_and_eval(model_name, user_features, item_features, seq_features, semantic_schema, train_dl, val_dl, test_dl, epoch, learning_rate, weight_decay, device, save_dir):
    model, loss_mode = build_model(model_name, user_features, item_features, seq_features, semantic_schema)
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


def main(data_dir, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed, max_seq_len):
    torch.manual_seed(seed)
    np.random.seed(seed)
    user_features, item_features, seq_features, semantic_schema, train_dl, val_dl, test_dl = build_dataloaders(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )

    model_names = [model_name] if model_name != "all" else ["dcn_v2", "rankmixer"]
    auc_results = {}
    for current_model in model_names:
        auc_results[current_model] = train_and_eval(
            model_name=current_model,
            user_features=user_features,
            item_features=item_features,
            seq_features=seq_features,
            semantic_schema=semantic_schema,
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

    parser = argparse.ArgumentParser(description="DCNv2 vs RankMixer on MovieLens-1M")
    parser.add_argument("--data_dir", default="./data/ml-1m/")
    parser.add_argument("--model_name", default="all", choices=["all", "dcn_v2", "rankmixer"])
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default="./data/ml-1m/saved_rankmixer/")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_seq_len", type=int, default=50)
    args = parser.parse_args()
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
        max_seq_len=args.max_seq_len,
    )