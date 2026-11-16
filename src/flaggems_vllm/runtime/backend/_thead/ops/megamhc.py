# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import triton

from flaggems_vllm.runtime import device, torch_device_fn

from ._megamhc_kernel import mega_mhc_kernel
from ._megamhc_runtime import clear_counters, pipeline_runtime


def mega_mhc(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=1e-6,
):
    """Fuse the normal mHC post-to-pre boundary for BF16 inference on PPU.

    Combination weights index [token, source stream, destination stream].
    Returns the BF16 updated residual, FP32 post and combination coefficients,
    and the BF16 collapsed layer input with optional RMS normalization.
    """
    if device.vendor_name != "thead" or residual.device.type != "cuda":
        raise NotImplementedError("MegaMHC requires T-Head PPU")
    if residual.ndim != 3:
        raise ValueError("residual must have shape [tokens, streams, hidden]")
    tokens, streams, hidden = residual.shape
    if streams not in (1, 2, 4, 8) or not 0 < hidden <= 16384:
        raise NotImplementedError(
            "MegaMHC supports 1/2/4/8 streams and hidden sizes up to 16384"
        )
    if n_splits != 1 or tile_n != 1:
        raise NotImplementedError("MegaMHC selects its tiling internally")
    if not isinstance(sinkhorn_repeat, int) or sinkhorn_repeat < 1:
        raise ValueError("sinkhorn_repeat must be positive")
    if not all(
        math.isfinite(v) and v >= 0
        for v in (rms_eps, hc_pre_eps, hc_sinkhorn_eps, norm_eps)
    ):
        raise ValueError("MegaMHC epsilons must be finite and nonnegative")
    if not math.isfinite(hc_post_mult_value):
        raise ValueError("hc_post_mult_value must be finite")

    mixes = streams * (streams + 2)
    specs = [
        (x, (tokens, hidden), torch.bfloat16),
        (residual, (tokens, streams, hidden), torch.bfloat16),
        (post_layer_mix, (tokens, streams, 1), torch.float32),
        (fn, (mixes, streams * hidden), torch.float32),
        (hc_scale, (3,), torch.float32),
        (hc_base, (mixes,), torch.float32),
    ]
    if norm_weight is not None:
        specs.append((norm_weight, (hidden,), torch.bfloat16))
    for tensor, shape, dtype in specs:
        if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(
                f"MegaMHC expected a contiguous {dtype} tensor of shape {shape}"
            )
        if tensor.device != residual.device or tensor.requires_grad:
            raise ValueError("MegaMHC requires inference tensors on one PPU")
    if (
        comb_res_mix.shape != (tokens, streams, streams)
        or comb_res_mix.dtype != torch.float32
        or comb_res_mix.device != residual.device
        or comb_res_mix.requires_grad
    ):
        raise ValueError("invalid MegaMHC combination weights")

    new_post = torch.empty_like(post_layer_mix)
    new_comb = torch.empty(
        (tokens, streams, streams), device=x.device, dtype=torch.float32
    )
    y = torch.empty_like(x)
    if not tokens:
        return torch.empty_like(residual), new_post, new_comb, y

    block_p = triton.next_power_of_2(mixes)
    block_h = triton.next_power_of_2(hidden)
    properties = torch.cuda.get_device_properties(x.device)
    multiprocessors = properties.multi_processor_count
    norm = norm_weight if norm_weight is not None else x
    inputs = (x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, norm)
    outputs = (new_post, new_comb, y)
    scalars = (
        *comb_res_mix.stride(),
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
    )
    has_norm = norm_weight is not None
    with torch_device_fn.device(x.device):
        k = streams * hidden
        # Small grids and wide projections favor the per-token path. The
        # pipeline keeps its projection width within one narrow AIU tile.
        if (
            (properties.major, properties.minor) == (8, 0)
            and tokens > 8 * multiprocessors
            and tokens < 2**31
            and k >= 256
            and block_p <= 32
        ):
            runtime = pipeline_runtime()
            if runtime is not None:
                new_residual = _launch_pipeline(
                    inputs,
                    outputs,
                    scalars,
                    tokens,
                    streams,
                    hidden,
                    mixes,
                    max(16, block_p),
                    has_norm,
                    multiprocessors,
                    runtime,
                )
                return new_residual, *outputs

        new_residual = torch.empty_like(residual)
        # With only one CTA per token, use a wider K tile at low occupancy.
        # More waves favor fewer registers per CTA. Bound the accumulator for
        # larger stream counts as its size is proportional to the mix width.
        if tokens <= 2 * multiprocessors and block_p <= 32:
            block_k, warps = 256, 8
        elif tokens <= 4 * multiprocessors and block_p <= 32:
            block_k, warps = 256, 4
        else:
            block_k, warps = 128, 8 if block_p > 32 else 2
        mega_mhc_kernel[(tokens,)](
            *inputs,
            new_residual,
            *outputs,
            *scalars,
            hidden,
            streams,
            mixes,
            block_p,
            min(block_k, block_h),
            block_h,
            has_norm,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return new_residual, *outputs


def _launch_pipeline(
    inputs,
    outputs,
    scalars,
    tokens,
    streams,
    hidden,
    mixes,
    q,
    has_norm,
    multiprocessors,
    runtime,
):
    from ._megamhc_pipeline import mega_mhc_pipeline

    memset, bitcode = runtime
    device = inputs[0].device
    m, bk, pack_block = 64, 64, 2048
    groups = triton.cdiv(tokens, m)
    padded = groups * m
    k = streams * hidden
    splits = min(8, triton.cdiv(k, bk))
    # Shorter queues cannot amortize draining the overlapping stages.
    wave, lag = (8, 2) if groups >= 4 * multiprocessors else (min(groups, 32), 0)
    jobs = (triton.cdiv(groups, wave) + 2 * lag) * wave * (2 * (m // 8) + splits)
    jobs += triton.cdiv(k * q, pack_block)

    residual = torch.empty(
        (padded, streams, hidden), dtype=torch.bfloat16, device=device
    )
    packed = torch.empty((3, q, k), dtype=torch.bfloat16, device=device)
    partial = torch.empty((splits, padded, q), dtype=torch.float32, device=device)
    sum_sq = torch.empty((padded,), dtype=torch.float32, device=device)
    # Ticket, packing counter and two counters per group. Allocation in the
    # caller's stream gives eager invocations and captured graphs private storage.
    counters = torch.empty((2 + 2 * groups,), dtype=torch.int64, device=device)
    clear_counters(counters, memset)
    mega_mhc_pipeline[(jobs,)](
        *inputs,
        residual,
        *outputs,
        counters,
        packed,
        partial,
        sum_sq,
        *scalars,
        tokens,
        hidden,
        streams,
        mixes,
        q,
        m,
        bk,
        splits,
        pack_block,
        wave,
        lag,
        has_norm,
        num_warps=2,
        num_stages=1,
        enable_fp_fusion=False,
        extern_libs={"megamhc": bitcode},
    )
    return residual[:tokens]
