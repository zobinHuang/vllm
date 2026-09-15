# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused ``out *= sigmoid(x @ gate_w.T)`` for shared-expert output gating."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _sigmoid_gate_scale_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K,
    N,
    stride_xm,
    stride_om,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    nb = tl.program_id(1)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        xv = tl.load(x_ptr + m * stride_xm + offs_k, mask=mask_k, other=0.0)
        wv = tl.load(w_ptr + offs_k, mask=mask_k, other=0.0)
        acc += xv.to(tl.float32) * wv.to(tl.float32)
    gate = tl.sigmoid(tl.sum(acc, axis=0))
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    o = tl.load(out_ptr + m * stride_om + offs_n, mask=mask_n, other=0.0)
    tl.store(
        out_ptr + m * stride_om + offs_n,
        (o.to(tl.float32) * gate).to(o.dtype),
        mask=mask_n,
    )


def _sigmoid_gate_scale_(x: torch.Tensor, gate_weight: torch.Tensor, out: torch.Tensor) -> None:
    M, K = x.shape
    N = out.shape[1]
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _sigmoid_gate_scale_kernel[grid](
        x,
        gate_weight,
        out,
        K,
        N,
        x.stride(0),
        out.stride(0),
        BLOCK_K=1024,
        BLOCK_N=BLOCK_N,
    )


def _sigmoid_gate_scale_fake(x: torch.Tensor, gate_weight: torch.Tensor, out: torch.Tensor) -> None:
    return None


direct_register_custom_op(
    op_name="sigmoid_gate_scale_",
    op_func=_sigmoid_gate_scale_,
    mutates_args=["out"],
    fake_impl=_sigmoid_gate_scale_fake,
)


def can_fuse_sigmoid_gate(x: torch.Tensor, gate: torch.nn.Module, out: torch.Tensor) -> bool:
    weight = getattr(gate, "weight", None)
    return (
        weight is not None
        and getattr(gate, "bias", None) is None
        and x.is_cuda
        and x.dim() == 2
        and out.dim() == 2
        and x.shape[0] == out.shape[0]
        and x.dtype == out.dtype
        and x.dtype in (torch.bfloat16, torch.float16)
        and weight.dtype == x.dtype
        and weight.dim() == 2
        and weight.shape == (1, x.shape[1])
        and x.stride(1) == 1
        and out.stride(1) == 1
        and weight.is_contiguous()
    )


def sigmoid_gate_scale_(x: torch.Tensor, gate_weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """In-place ``out *= sigmoid(x @ gate_weight.T)``; returns ``out``."""
    torch.ops.vllm.sigmoid_gate_scale_(x, gate_weight.view(-1), out)
    return out
