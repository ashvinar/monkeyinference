"""LoRA on the existing Splash Q4 DFlash draft.

Full fine-tune of 1.80B draft weights needs ~25 GB (fp16 + fp32 master + Adam)
and does not fit in 15 GB free / 24 GB unified. LoRA rank-16 is 8.6M params
(~104 MB Adam). B is zero-init so step 0 equals the working Q4 draft.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from monkeyinference.splash_q4 import Q4Linear

DEFAULT_ADAPTERS = Path.home() / ".monkey/dflash-ft/adapters.safetensors"
LORA_RANK = 16
LORA_SCALE = 2.0


class LoRAQ4(nn.Module):
    """Frozen Q4 linear plus a trainable low-rank residual."""

    def __init__(self, base: Q4Linear, r: int = LORA_RANK, scale: float = LORA_SCALE):
        super().__init__()
        self.base = base
        self.base.freeze()
        self.r = int(r)
        self.scale = float(scale)
        inn, out = base.in_features, base.out_features
        self.lora_a = mx.random.normal((inn, self.r)) * (1.0 / inn) ** 0.5
        self.lora_b = mx.zeros((self.r, out))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        orig = x.shape
        flat = x.reshape(-1, orig[-1]).astype(mx.float32)
        z = (flat @ self.lora_a.astype(mx.float32)) @ self.lora_b.astype(mx.float32)
        z = (self.scale * z).reshape(*orig[:-1], -1)
        return y.astype(mx.float32) + z.astype(y.dtype if y.dtype == mx.float32 else mx.float32)


def wrap_drafter(drafter, *, r: int = LORA_RANK, scale: float = LORA_SCALE) -> nn.Module:
    """Replace every Q4 projection with LoRAQ4. Returns a Module owning the adapters."""

    class AdapterSet(nn.Module):
        def __init__(self):
            super().__init__()

    adapters = AdapterSet()
    names: list[str] = []

    def put(name: str, lin: Q4Linear) -> LoRAQ4:
        mod = LoRAQ4(lin, r=r, scale=scale)
        setattr(adapters, name, mod)
        names.append(name)
        return mod

    for i, layer in enumerate(drafter.w.layers):
        layer.attn_dynamic = put(f"l{i}_attn_dynamic", layer.attn_dynamic)
        layer.qkv = put(f"l{i}_qkv", layer.qkv)
        layer.o_proj = put(f"l{i}_o_proj", layer.o_proj)
        layer.mlp_dynamic = put(f"l{i}_mlp_dynamic", layer.mlp_dynamic)
        layer.gate = put(f"l{i}_gate", layer.gate)
        layer.up = put(f"l{i}_up", layer.up)
        layer.down = put(f"l{i}_down", layer.down)
    drafter.w.context_proj = put("context_proj", drafter.w.context_proj)
    drafter.w.selector = put("selector", drafter.w.selector)
    adapters.names = names
    n = sum(int(v.size) for _, v in tree_flatten(adapters.trainable_parameters()))
    object.__setattr__(adapters, "n_trainable", n)
    return adapters


def save_adapters(adapters: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), dict(tree_flatten(adapters.trainable_parameters())))
    return path


def load_adapters(adapters: nn.Module, path: str | Path) -> None:
    path = Path(path)
    weights = mx.load(str(path))
    adapters.update(tree_unflatten(list(weights.items())))
    mx.eval(adapters.parameters())
