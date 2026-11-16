# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import statistics

import pytest
import torch

import flaggems_vllm as gems
from tests.test_megamhc import EPS, candidate, make_inputs

pytestmark = [
    pytest.mark.megamhc,
    pytest.mark.skipif(
        gems.vendor_name != "thead" or not torch.cuda.is_available(),
        reason="requires T-Head PPU",
    ),
]


def split(args):
    x, residual, post, comb, fn, scale, base, norm = args
    residual = gems.mhc_post(x, residual, post, comb)
    post, comb, y = gems.mhc_pre(residual, fn, scale, base, *EPS)
    return residual, post, comb, gems.mhc_rms_norm(y, norm, 1e-6)


def _capture(call, args, warmup):
    for _ in range(warmup):
        call(args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = call(args)
    return graph, outputs


@pytest.mark.parametrize("tokens", [1, 24, 64, 128, 512, 1028, 2048, 3076, 16384])
def test_mega_mhc(tokens):
    args = make_inputs(tokens, 4096, 4)
    runs = [_capture(op, args, 20) for op in (split, candidate)]
    graphs = [graph for graph, _ in runs]
    samples = [[], []]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(100):
        for j in (i % 2, 1 - i % 2):
            start.record()
            graphs[j].replay()
            end.record()
            end.synchronize()
            samples[j].append(start.elapsed_time(end) * 1000)
    baseline, fused = map(statistics.median, samples)
    print(
        f"MegaMHC CUDA Graph T={tokens} split={baseline:.3f} us fused={fused:.3f} us {baseline / fused:.4f}x"
    )
