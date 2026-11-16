# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm as gems

pytestmark = [
    pytest.mark.megamhc,
    pytest.mark.skipif(
        gems.vendor_name != "thead" or not torch.cuda.is_available(),
        reason="requires T-Head PPU",
    ),
]

EPS = (1e-6, 1e-6, 1e-6, 2.0, 20)


def make_inputs(tokens, hidden, streams, strided=False):
    gen = torch.Generator().manual_seed(20261117 + tokens)

    def randn(shape, dtype=torch.float32):
        return torch.randn(shape, generator=gen, dtype=dtype).cuda()

    mixes = streams * (streams + 2)
    x = randn((tokens, hidden), torch.bfloat16)
    residual = randn((tokens, streams, hidden), torch.bfloat16)
    post = randn((tokens, streams, 1)).sigmoid()
    comb = randn((tokens, streams, streams)).softmax(-1)
    if strided:
        comb = comb.mT
    fn = randn((mixes, streams * hidden)) * 0.01
    scale = randn((3,)) * 0.1
    base = randn((mixes,)) * 0.1
    norm = (randn((hidden,)) * 0.1 + 1).bfloat16()
    return x, residual, post, comb, fn, scale, base, norm


def reference(args, use_norm=True):
    x, residual, post, comb, fn, scale, base, norm = (a.cpu() for a in args)
    streams = residual.shape[1]
    new_residual = (
        x.float().unsqueeze(1) * post
        + torch.einsum("tij,tih->tjh", comb, residual.float())
    ).bfloat16()
    flat = new_residual.flatten(1).float()
    rms_eps, pre_eps, sink_eps, multiplier, repeats = EPS
    projection = (flat.double() @ fn.double().T).float() * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + rms_eps
    )
    p, q, c = projection.split((streams, streams, streams * streams), dim=-1)
    b_pre, b_post, b_comb = base.split((streams, streams, streams * streams))
    pre = (torch.sigmoid(p * scale[0] + b_pre) + pre_eps).unsqueeze(-1)
    new_post = (torch.sigmoid(q * scale[1] + b_post) * multiplier).unsqueeze(-1)
    new_comb = (c * scale[2] + b_comb).reshape(-1, streams, streams).softmax(
        -1
    ) + sink_eps
    new_comb /= new_comb.sum(-2, keepdim=True) + sink_eps
    for _ in range(1, repeats):
        new_comb /= new_comb.sum(-1, keepdim=True) + sink_eps
        new_comb /= new_comb.sum(-2, keepdim=True) + sink_eps
    y = (new_residual.float() * pre).sum(1).bfloat16()
    if use_norm:
        y = y.float()
        y = (
            y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * norm.float()
        ).bfloat16()
    return new_residual, new_post, new_comb, y


def candidate(args, use_norm=True):
    return gems.mega_mhc(*args[:-1], *EPS, norm_weight=args[-1] if use_norm else None)


def verify(result, expected):
    for actual, target in zip(result, expected, strict=True):
        if target.dtype is torch.float32:
            torch.testing.assert_close(actual.cpu(), target, rtol=3e-4, atol=2e-5)
        else:
            torch.testing.assert_close(actual.cpu(), target, rtol=0.016, atol=0.008)


@pytest.mark.parametrize(
    "tokens,hidden,streams",
    [
        (1, 4096, 4),
        (17, 259, 4),
        (3, 513, 2),
        (5, 127, 1),
        (3, 129, 8),
        (1029, 259, 1),
        (4097, 259, 4),
    ],
)
def test_mega_mhc(tokens, hidden, streams):
    args = make_inputs(tokens, hidden, streams, strided=True)
    verify(candidate(args), reference(args))


def test_mega_mhc_graph_replay():
    args = make_inputs(13, 5120, 4)
    candidate(args, use_norm=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = candidate(args, use_norm=False)
    for _ in range(2):
        args[0].mul_(-0.75)
        graph.replay()
        verify(out, reference(args, use_norm=False))


@pytest.mark.parametrize("tokens", [4097, 8193])
def test_mega_mhc_projection_cancellation(tokens):
    hidden, streams = 521, 4
    args = make_inputs(tokens, hidden, streams)
    x, residual, post, comb, fn, scale, base, norm = args
    x.zero_()
    residual.fill_(1)
    post.zero_()
    comb.copy_(torch.eye(streams, device=comb.device))
    fn.zero_()
    scale.fill_(1)
    base.zero_()
    norm.fill_(1)

    # The first post coefficient depends on a small FP32 remainder in each pair.
    # Converting the weights to two BF16 terms can erase the remainder.
    weight = 1.0 + 2**-9
    fn[streams, 0::2] = weight + 2**-19
    fn[streams, 1::2] = -weight

    rows = torch.tensor([0, tokens // 2, tokens - 1], device=x.device)

    def check(outputs):
        sampled = tuple(a[rows] if i < 4 else a for i, a in enumerate(args))
        verify(tuple(t[rows] for t in outputs), reference(sampled))

    check(candidate(args))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = candidate(args)
    for _ in range(2):
        fn.neg_()
        graph.replay()
        check(outputs)


def test_mega_mhc_empty():
    args = make_inputs(0, 127, 4)
    assert [t.shape for t in candidate(args)] == [
        (0, 4, 127),
        (0, 4, 1),
        (0, 4, 4),
        (0, 127),
    ]
