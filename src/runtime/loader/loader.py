"""Unified model loader: safetensors weights + config + tokenizer.

Entry point: load_model(model_dir) -> LoadedModel
"""
import json
from pathlib import Path
from typing import Dict, Optional
from .config_parser import parse_config, ModelConfig
from .safetensors_loader import SafetensorsShardCollection


class LoadedModel:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.config: ModelConfig = parse_config(model_dir)
        self.weights: SafetensorsShardCollection = SafetensorsShardCollection(model_dir)
        self.tokenizer_data: Optional[dict] = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        tok_path = self.model_dir / "tokenizer.json"
        if tok_path.exists():
            with open(tok_path) as f:
                self.tokenizer_data = json.load(f)

    def get_weight(self, name: str) -> bytes:
        return self.weights.get_tensor_bytes(name)

    def list_weights(self) -> list:
        return self.weights.list_all_tensors()

    def weight_nbytes(self, name: str) -> int:
        return self.weights.tensor_info(name).nbytes

    @property
    def total_weight_bytes(self) -> int:
        return self.weights.total_bytes()

    @property
    def num_tensors(self) -> int:
        return len(self.list_weights())

    def close(self):
        self.weights.close_all()


def load_model(model_dir: str) -> LoadedModel:
    return LoadedModel(model_dir)