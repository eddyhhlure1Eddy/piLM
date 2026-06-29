"""Model config parser: reads config.json into typed dataclass."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class RopeConfig:
    rope_type: str = "default"
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    mrope_interleaved: bool = False
    mrope_section: List[int] = field(default_factory=list)


@dataclass
class TextConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 12288
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    dtype: str = "bfloat16"
    layer_types: List[str] = field(default_factory=list)
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_value_head_dim: int = 128
    linear_num_value_heads: int = 32
    rope: RopeConfig = field(default_factory=RopeConfig)
    tie_word_embeddings: bool = False
    eos_token_id: int = 0


@dataclass
class ModelConfig:
    architectures: List[str] = field(default_factory=list)
    model_type: str = ""
    text_config: TextConfig = field(default_factory=TextConfig)
    tie_word_embeddings: bool = False
    dtype: str = "bfloat16"
    has_vision: bool = False
    has_mtp: bool = False

    @property
    def num_layers(self):
        return self.text_config.num_hidden_layers

    @property
    def hidden_size(self):
        return self.text_config.hidden_size

    @property
    def num_kv_heads(self):
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self):
        return self.text_config.head_dim

    @property
    def vocab_size(self):
        return self.text_config.vocab_size

    @property
    def n_full_attn_layers(self):
        return sum(1 for t in self.text_config.layer_types if t == "full_attention")

    @property
    def n_linear_attn_layers(self):
        return sum(1 for t in self.text_config.layer_types if t == "linear_attention")


def parse_config(model_dir: str) -> ModelConfig:
    cfg_path = Path(model_dir) / "config.json"
    with open(cfg_path) as f:
        raw = json.load(f)

    tc_raw = raw.get("text_config", {})
    rope_raw = tc_raw.get("rope_parameters", {})
    rope = RopeConfig(
        rope_type=rope_raw.get("rope_type", "default"),
        rope_theta=rope_raw.get("rope_theta", 10000.0),
        partial_rotary_factor=rope_raw.get("partial_rotary_factor", 1.0),
        mrope_interleaved=rope_raw.get("mrope_interleaved", False),
        mrope_section=rope_raw.get("mrope_section", []),
    )

    tc = TextConfig(
        hidden_size=tc_raw.get("hidden_size", 4096),
        num_hidden_layers=tc_raw.get("num_hidden_layers", 32),
        num_attention_heads=tc_raw.get("num_attention_heads", 16),
        num_key_value_heads=tc_raw.get("num_key_value_heads", 4),
        head_dim=tc_raw.get("head_dim", 256),
        intermediate_size=tc_raw.get("intermediate_size", 12288),
        vocab_size=tc_raw.get("vocab_size", 248320),
        max_position_embeddings=tc_raw.get("max_position_embeddings", 262144),
        rms_norm_eps=tc_raw.get("rms_norm_eps", 1e-6),
        hidden_act=tc_raw.get("hidden_act", "silu"),
        dtype=tc_raw.get("dtype", "bfloat16"),
        layer_types=tc_raw.get("layer_types", []),
        full_attention_interval=tc_raw.get("full_attention_interval", 4),
        linear_conv_kernel_dim=tc_raw.get("linear_conv_kernel_dim", 4),
        linear_key_head_dim=tc_raw.get("linear_key_head_dim", 128),
        linear_num_key_heads=tc_raw.get("linear_num_key_heads", 16),
        linear_value_head_dim=tc_raw.get("linear_value_head_dim", 128),
        linear_num_value_heads=tc_raw.get("linear_num_value_heads", 32),
        rope=rope,
        tie_word_embeddings=tc_raw.get("tie_word_embeddings", False),
        eos_token_id=tc_raw.get("eos_token_id", 0),
    )

    cfg = ModelConfig(
        architectures=raw.get("architectures", []),
        model_type=raw.get("model_type", ""),
        text_config=tc,
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
        dtype=raw.get("dtype", tc_raw.get("dtype", "bfloat16")),
        has_vision="vision_config" in raw,
        has_mtp="mtp" in str(raw),
    )
    return cfg