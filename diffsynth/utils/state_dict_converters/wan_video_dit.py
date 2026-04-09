from typing import Literal


WanDiTStateKeyGroup = Literal["text_embedding", "text_context", "other"]


class WanDiTStateKeyPolicy:
    _SOURCE_GROUP_FRAGMENTS = (
        ("text_embedding", (
            "text_embedding.",
            "condition_embedder.text_embedder.",
        )),
        ("text_context", (
            ".attn2.to_k.",
            ".attn2.to_v.",
            ".attn2.norm_k.",
            ".cross_attn.k.",
            ".cross_attn.v.",
            ".cross_attn.norm_k.",
        )),
    )
    _TARGET_GROUP_FRAGMENTS = (
        ("text_embedding", (
            "text_embedding.",
        )),
        ("text_context", (
            ".cross_attn.k.",
            ".cross_attn.v.",
            ".cross_attn.norm_k.",
        )),
    )

    @staticmethod
    def normalize_key(key: str) -> str:
        return key[len("model."):] if key.startswith("model.") else key

    @classmethod
    def _classify_key(cls, key: str, group_fragments) -> WanDiTStateKeyGroup:
        key = cls.normalize_key(key)
        for group, fragments in group_fragments:
            if any(fragment in key for fragment in fragments):
                return group
        return "other"

    @classmethod
    def classify_source_key(cls, key: str) -> WanDiTStateKeyGroup:
        return cls._classify_key(key, cls._SOURCE_GROUP_FRAGMENTS)

    @classmethod
    def classify_target_key(cls, key: str) -> WanDiTStateKeyGroup:
        return cls._classify_key(key, cls._TARGET_GROUP_FRAGMENTS)

    @staticmethod
    def build_allowed_groups(use_text_embedding: bool, has_text_input: bool):
        groups = {"other"}
        if use_text_embedding:
            groups.add("text_embedding")
        if has_text_input:
            groups.add("text_context")
        return frozenset(groups)

    @classmethod
    def should_load_source_key(cls, key: str, allowed_groups) -> bool:
        return cls.classify_source_key(key) in allowed_groups

    @classmethod
    def should_load_target_key(cls, key: str, allowed_groups) -> bool:
        return cls.classify_target_key(key) in allowed_groups


def WanVideoDiTFromDiffusers(state_dict, key_filter=None):
    rename_dict = {
        "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
        "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
        "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
        "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
        "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
        "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
        "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
        "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
        "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
        "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
        "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
        "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
        "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
        "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
        "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
        "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
        "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
        "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
        "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
        "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
        "blocks.0.attn2.add_k_proj.bias":"blocks.0.cross_attn.k_img.bias",
        "blocks.0.attn2.add_k_proj.weight":"blocks.0.cross_attn.k_img.weight",
        "blocks.0.attn2.add_v_proj.bias":"blocks.0.cross_attn.v_img.bias",
        "blocks.0.attn2.add_v_proj.weight":"blocks.0.cross_attn.v_img.weight",
        "blocks.0.attn2.norm_added_k.weight":"blocks.0.cross_attn.norm_k_img.weight",
        "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
        "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
        "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
        "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
        "blocks.0.norm2.bias": "blocks.0.norm3.bias",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.scale_shift_table": "blocks.0.modulation",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.image_embedder.ff.net.0.proj.bias":"img_emb.proj.1.bias",
        "condition_embedder.image_embedder.ff.net.0.proj.weight":"img_emb.proj.1.weight",
        "condition_embedder.image_embedder.ff.net.2.bias":"img_emb.proj.3.bias",
        "condition_embedder.image_embedder.ff.net.2.weight":"img_emb.proj.3.weight",
        "condition_embedder.image_embedder.norm1.bias":"img_emb.proj.0.bias",
        "condition_embedder.image_embedder.norm1.weight":"img_emb.proj.0.weight",
        "condition_embedder.image_embedder.norm2.bias":"img_emb.proj.4.bias",
        "condition_embedder.image_embedder.norm2.weight":"img_emb.proj.4.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "scale_shift_table": "head.modulation",
        "proj_out.bias": "head.head.bias",
        "proj_out.weight": "head.head.weight",
    }
    state_dict_ = {}
    for name in state_dict:
        if name in rename_dict:
            target_name = rename_dict[name]
            if key_filter is not None and not key_filter(target_name):
                continue
            state_dict_[target_name] = state_dict[name]
        else:
            name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
            if name_ in rename_dict:
                target_name = rename_dict[name_]
                target_name = ".".join(target_name.split(".")[:1] + [name.split(".")[1]] + target_name.split(".")[2:])
                if key_filter is not None and not key_filter(target_name):
                    continue
                state_dict_[target_name] = state_dict[name]
    return state_dict_


def WanVideoDiTStateDictConverter(state_dict, key_filter=None):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("vace"):
            continue
        if name.split(".")[0] in ["pose_patch_embedding", "face_adapter", "face_encoder", "motion_encoder"]:
            continue
        name_ = name
        if name_.startswith("model."):
            name_ = name_[len("model."):]
        if key_filter is not None and not key_filter(name_):
            continue
        state_dict_[name_] = state_dict[name]
    return state_dict_
