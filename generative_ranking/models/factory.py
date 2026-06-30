from .dcn_v2 import DCNv2
from .onetrans import OneTrans
from .rankmixer import RankMixer
from .rankmixer_grouping import build_rankmixer_semantic_groups


def build_model(model_name, bundle, config):
    model_key = str(model_name).lower()
    if model_key == "dcn_v2":
        model = DCNv2(features=bundle["dcn_v2_features"], **config["dcn_v2"])
        return model, True
    if model_key == "rankmixer":
        rankmixer_cfg = config["rankmixer"]
        use_moe = rankmixer_cfg["use_moe"]
        model = RankMixer(
            features=bundle["rankmixer_features"],
            sequence_features=bundle.get("rankmixer_sequence_features", []),
            d_model=rankmixer_cfg["d_model"],
            num_layers=rankmixer_cfg["num_layers"],
            num_tokens=rankmixer_cfg["num_tokens"],
            semantic_groups=build_rankmixer_semantic_groups(bundle["semantic_schema"], default_seq_pool_modes=tuple(rankmixer_cfg.get("seq_pool_modes", ("mean",)))),
            seq_pool_modes=tuple(rankmixer_cfg.get("seq_pool_modes", ("mean",))),
            use_moe=use_moe,
            moe_experts=rankmixer_cfg["moe_experts"],
            moe_l1_coef=rankmixer_cfg["moe_l1_coef"],
            moe_sparsity_ratio=rankmixer_cfg["moe_sparsity_ratio"],
            moe_use_dtsi=rankmixer_cfg["moe_use_dtsi"],
            moe_routing_type=rankmixer_cfg["moe_routing_type"],
            input_dropout=rankmixer_cfg["input_dropout"],
            token_mixing_dropout=rankmixer_cfg["token_mixing_dropout"],
            ffn_dropout=rankmixer_cfg["ffn_dropout"],
            head_dropout=rankmixer_cfg["head_dropout"],
            return_moe_loss=use_moe,
        )
        return model, not use_moe
    if model_key == "onetrans":
        onetrans_cfg = config["onetrans"]
        model = OneTrans(
            features=bundle["rankmixer_features"],
            sequence_features=bundle.get("rankmixer_sequence_features", []),
            max_seq_len=config.get("dataset_params", {}).get("max_seq_len", 50),
            ns_len=onetrans_cfg["ns_len"],
            d_model=onetrans_cfg["d_model"],
            num_heads=onetrans_cfg["num_heads"],
            ffn_hidden=onetrans_cfg["ffn_hidden"],
            multi_num=onetrans_cfg["multi_num"],
            num_pyramid_layers=onetrans_cfg["num_pyramid_layers"],
            pyramid_align=onetrans_cfg["pyramid_align"],
            mask_type=onetrans_cfg["mask_type"],
            use_sep_token=onetrans_cfg["use_sep_token"],
            use_checkpoint=onetrans_cfg["use_checkpoint"],
            head_dropout=onetrans_cfg["head_dropout"],
        )
        return model, True
    raise ValueError(f"Unsupported model_name={model_name}")