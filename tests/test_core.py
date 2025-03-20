import random
from typing import Tuple

import torch

import deep_gemm
from deep_gemm import (
    bench_kineto,
    calc_diff,
    ceil_div,
    get_col_major_tma_aligned_tensor,
)


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def construct_backward_w(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()
    
    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_token_cast_to_fp8(y)
    return x_fp8, y_fp8, out, ref_out


def construct_dw_grouped(num_groups: int, m: int, k: int, n: int, is_masked: bool) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)


    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float)
    )
    y_fp8 = (
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, n, k // 128), device='cuda', dtype=torch.float) # NOTE: per-token
    )
    
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])  
        y_fp8[0][i], y_fp8[1][i] = per_token_cast_to_fp8(y[i])  # NOTE: per-token
    
    if not is_masked:
        x_fp8 = (
            x_fp8[0].view(-1, k),
            per_token_cast_to_fp8(x.view(-1, k))[1] 
        )
        
        
        out = out.view(-1, n)
        
        ref_out = ref_out.view(-1, n)

    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    y_fp8 = (y_fp8[0], get_col_major_tma_aligned_tensor(y_fp8[1]))
    
    return x_fp8, y_fp8, out, ref_out
    

def construct_dw_grouped_varlen(group_sizes, k: int, n: int) -> Tuple:
    """构造可变长度分组的测试数据"""
    num_groups = len(group_sizes)
    # 生成随机数据，每个组的m维度不同
    x = torch.randn((sum(group_sizes), k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((sum(group_sizes), n), device='cuda', dtype=torch.bfloat16)
    
    # 生成分组索引
    m_indices = []
    for i, size in enumerate(group_sizes):
        m_indices.extend([i] * size)
    m_indices = torch.tensor(m_indices, device='cuda', dtype=torch.int32)
    
    # 转换FP8格式
    x_fp8, x_scales = per_token_cast_to_fp8(x)
    y_fp8, y_scales = per_token_cast_to_fp8(y.view(-1, k))
    
    # 内存对齐处理
    x_scales = get_col_major_tma_aligned_tensor(x_scales)
    y_scales = get_col_major_tma_aligned_tensor(y_scales.view(num_groups, n, -1))
    
    return (x_fp8, x_scales), (y_fp8.view(num_groups, n, k), y_scales), out, m_indices
def test_varlen_grouped_gemm():
    group_sizes = [1024, 1536, 2048]
    x_fp8, y_fp8, out, m_indices = construct_dw_grouped_varlen(group_sizes, k=7168, n=4096)
    
    deep_gemm.m_grouped_gemm_dw_fp8_fp8_bf16_nt_contiguous(
        x_fp8, y_fp8, out, m_indices)
    
    ref_out = torch.cat([
        x_fp8[0][:1024] @ y_fp8[0][0].t(),
        x_fp8[0][1024:2560] @ y_fp8[0][1].t(),
        x_fp8[0][2560:] @ y_fp8[0][2].t()
    ])
    assert calc_diff(out, ref_out) < 0.001



def test_gemm_backward_w() -> None:
    print('Testing GEMM Backward W:')
    for m in (64, 128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
    # for m in (4096, ):
        # for k, n in [(7168, 2112),]:
            x_fp8, y_fp8, out, ref_out = construct_backward_w(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_bw_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
            torch.cuda.synchronize()

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct_backward_w(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_bw_nt(x_fp8, y_fp8, out)

            t = bench_kineto(test_func, 'fp8_gemm_bw', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()

def test_m_grouped_gemm_dw_contiguous()->None:
    print('Testing grouped contiguous GEMM:')

    for num_groups, m, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):
    # for num_groups, m, k, n in ((2, 128, 128, 128),):
        # TODO: make a stronger test
        x_fp8, y_fp8, out, ref_out = construct_dw_grouped(num_groups, m, k, n, is_masked=False)
        m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
        m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
        # print('m_indices', m_indices)
        deep_gemm.m_grouped_gemm_dw_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'
        
        # if diff > 0.001:
        #     print(f'========>m2 fail, m1={m * num_groups}, {k=}, {n=}, {diff:.5f}')
        #     print(f'front 10 elements = {out[m: m + 10]}')
        # else:
        #     print(f'========>m2 pass')
        # diff = calc_diff(out[0:m], ref_out[0:m])
        # if diff > 0.001:
        #     print(f'========>m1 fail, m1={m * num_groups}, {k=}, {n=}, {diff:.5f}')
        #     print(f'front 10 elements = {out[0: 10]}')
        # else:
        #     print(f'========>m1 pass')

        diff = calc_diff(out, ref_out)

        # noinspection PyShadowingNames
        def test_func():
            # Construct new tensors every time to avoid L2 cache acceleration
            x_fp8, y_fp8, out, ref_out = construct_dw_grouped(num_groups, m, k, n, is_masked=False)
            m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
            m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
            deep_gemm.m_grouped_gemm_dw_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
    print()

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_m_grouped_gemm_dw_contiguous()
    test_gemm_backward_w()
