import json
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field, fields


@dataclass
class ModelArguments:
    model_path: str = field(
        default=None, metadata={"help": "Model checkpoint path"}
    )
    model_config_path: Optional[str] = field(
        default=None, metadata={"help": "The path of model config json file"}
    )
    ignore_mismatched_sizes: bool = field(default=False)
    lat_space: Optional[List[float]] = field(default=None)
    lon_space: Optional[List[float]] = field(default=None)
    patch_size: Optional[List[int]] = field(default=None)
    enc_depths: Optional[List[int]] = field(default=None)
    enc_heads: Optional[List[int]] = field(default=None)
    lg_depths: Optional[List[int]] = field(default=None)
    lg_heads: Optional[List[int]] = field(default=None)
    in_chans: Optional[List[int]] = field(default=None)
    out_chans: Optional[List[int]] = field(default=None)
    embed_dim: Optional[int] = field(default=None)
    num_heads: Optional[List[int]] = field(default=None)
    window_size: Optional[List[int]] = field(default=None)
    mlp_ratio: Optional[float] = field(default=None)
    qkv_bias: Optional[bool] = field(default=None)
    qk_scale: Optional[float] = field(default=None)
    drop_rate: Optional[float] = field(default=None)
    attn_drop: Optional[float] = field(default=None)
    drop_path_rate: Optional[float] = field(default=None)
    patch_norm: Optional[bool] = field(default=None)
    hidden_act: Optional[str] = field(default=None)
    layer_norm_eps: Optional[float] = field(default=None)
    use_absolute_embeddings: Optional[bool] = field(default=None)
    loss_type: Optional[str] = field(default=None)
    atmo_dims: Optional[int] = field(default=None)
    use_mask_token: Optional[bool] = field(default=None)
    is_moe: Optional[bool] = field(default=None)
    is_moe_encoder: Optional[bool] = field(default=None)
    is_moe_decoder: Optional[bool] = field(default=None)
    is_moe_atmo: Optional[bool] = field(default=None)
    atmo_embed_dims: Optional[int] = field(default=None)


    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {
            field.name: getattr(self, field.name) for field in fields(self)
            if field.init and (getattr(self, field.name) is not None or
                               field.name in ['model_path', 'model_config_path'])
        }

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)
