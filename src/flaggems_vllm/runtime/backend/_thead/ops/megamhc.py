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

    new_residual = torch.empty_like(residual)
    new_post = torch.empty_like(post_layer_mix)
    new_comb = torch.empty(
        (tokens, streams, streams), device=x.device, dtype=torch.float32
    )
    y = torch.empty_like(x)
    if tokens:
        block_p = triton.next_power_of_2(mixes)
        block_h = triton.next_power_of_2(hidden)
        multiprocessors = torch.cuda.get_device_properties(
            x.device
        ).multi_processor_count
        # With only one CTA per token, use a wider K tile at low occupancy.
        # More waves favor fewer registers per CTA. Bound the accumulator for
        # larger stream counts as its size is proportional to the mix width.
        if tokens <= 2 * multiprocessors and block_p <= 32:
            block_k, warps = 256, 8
        elif tokens <= 4 * multiprocessors and block_p <= 32:
            block_k, warps = 256, 4
        else:
            block_k, warps = 128, 8 if block_p > 32 else 2
        with torch_device_fn.device(x.device):
            mega_mhc_kernel[(tokens,)](
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                norm_weight if norm_weight is not None else x,
                new_residual,
                new_post,
                new_comb,
                y,
                *comb_res_mix.stride(),
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                norm_eps,
                hidden,
                streams,
                mixes,
                block_p,
                min(block_k, block_h),
                block_h,
                norm_weight is not None,
                num_warps=warps,
                enable_fp_fusion=False,
            )
    return new_residual, new_post, new_comb, y
