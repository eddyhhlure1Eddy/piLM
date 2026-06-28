"""Model loader: safetensors + GGUF weight loading into Eram buffers.

Design:
  - loader: unified entry (safetensors + GGUF)
  - safetensors_loader: read .safetensors shards via mmap (zero-copy)
  - gguf_loader: parse .gguf file format, dequant on demand
  - weight_mapper: map source weight names to Ecpu layer layout
  - tokenizer: load tokenizer.json (fast tokenizer)

Weight tensors are loaded into eram_buffer_t (mmap, zero-copy where possible).
"""