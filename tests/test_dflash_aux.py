from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from monkeyinference.dflash import TARGET_LAYER_IDS, AuxCapture


class _DummyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
        for _ in range(64):
            layer = nn.Identity()
            layer.is_linear = False
            self.layers.append(layer)


def test_aux_keep_and_uncommitted():
    aux = AuxCapture(_DummyText())
    try:
        for idx in TARGET_LAYER_IDS:
            aux.last[idx] = mx.ones((1, 4, 3), dtype=mx.float16) * (idx + 1)
        aux.record_last_forward()
        assert len(aux) == 4
        assert aux.uncommitted(0).shape[0] == 4
        assert aux.uncommitted(4) is None
        aux.keep_last_n(2)
        assert len(aux) == 2
        rows = aux.uncommitted(0)
        assert int(rows.shape[0]) == 2
        assert int(rows.shape[1]) == 3 * len(TARGET_LAYER_IDS)
        # Simulate a second verify chunk of 8, then keep leftover+2 drafts.
        for idx in TARGET_LAYER_IDS:
            aux.last[idx] = mx.ones((1, 8, 3), dtype=mx.float16)
        aux.record_last_forward()
        assert len(aux) == 10
        aux.keep_last_n(3)
        assert len(aux) == 5
        new = aux.uncommitted(2)
        assert int(new.shape[0]) == 3
    finally:
        aux.close()


if __name__ == "__main__":
    test_aux_keep_and_uncommitted()
    print("ok test_aux_keep_and_uncommitted")
