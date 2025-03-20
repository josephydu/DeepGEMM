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
    
def construct_dw_varlen_grouped(num_groups, m_list, k, n, is_masked):
    x = torch.cat([torch.randn((m, k), device='cuda', dtype=torch.bfloat16) for m in m_list], dim=0)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, sum(m_list), n), device='cuda', dtype=torch.bfloat16)
    
    # calc ref_out first, ref out is varlen grouped
    ref_out = torch.zeros_like(out)
    start_idx = 0
    for i, m in enumerate(m_list):
        x_part = x[start_idx:start_idx + m]
        y_part = y[i]
        ref_out[i, start_idx:start_idx + m] = x_part @ y_part.t()
        start_idx += m
        
    assert sum(m_list) % 4 == 0, f'TMA alignment error: {m}'
    
    x_fp8 = (
    torch.empty_like(x, dtype=torch.float8_e4m3fn),
    torch.empty((num_groups, sum(m_list), k // 128), device='cuda', dtype=torch.float)
    )
    y_fp8 = (
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, n, k // 128), device='cuda', dtype=torch.float) # NOTE: per-token
    )
    
    seq_len = torch.Tensor([0] + m_list)
    cu_seq_len = torch.cumsum(seq_len, dim=0).to(torch.int32).to('cuda')
    print(cu_seq_len)
    for i in range(num_groups):
        # x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[cu_seq_len[i]:cu_seq_len[i + 1]])
        o1, o2 = per_token_cast_to_fp8(x[cu_seq_len[i]:cu_seq_len[i + 1]])
        print(o1.shape, o2.shape)
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

def test_m_grouped_gemm_dw_varlen_contiguous()->None:
    print('Testing grouped variable length contiguous GEMM:')
    configs = [
        (2, [4096, 8192], 7168, 4096),  
        (3, [2048, 3072, 4096], 2048, 7168),  
        (4, [1024, 2048, 3072, 4096], 512, 32768)  
    ]
    for num_groups, m_list, k, n in configs:
        x_fp8, y_fp8, out, ref_out = construct_dw_varlen_grouped(num_groups, m_list, k, n, is_masked=False)
        # deep_gemm.gemm_fp8_fp8_bf16_bw_nt(x_fp8, y_fp8, out)
        # diff = calc_diff(out, ref_out)
        # assert diff < 0.001, f'm={sum(m_list) * num_groups}, {k=}, {n=}, {diff:.5f}'
        # torch.cuda.synchronize()

        
        


        # x_fp8 = (
        #     torch.empty_like(x, dtype=torch.float8_e4m3fn),
        #     torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float)
        # )
        # y_fp8 = (
        #     torch.empty_like(y, dtype=torch.float8_e4m3fn),
        #     torch.empty((num_groups, n, k // 128), device='cuda', dtype=torch.float) # NOTE: per-token
        # )
        
        # for i in range(num_groups):
        #     x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])  
        #     y_fp8[0][i], y_fp8[1][i] = per_token_cast_to_fp8(y[i])  # NOTE: per-token
        # m_indices = torch.cat([torch.full((m,), i, device='cuda', dtype=torch.int) for i, m in enumerate(m_list)])
    
    print()
    


def test_m_grouped_gemm_dw_contiguous()->None:
    print('Testing grouped dw contiguous GEMM:')

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

    test_m_grouped_gemm_dw_varlen_contiguous()
    test_m_grouped_gemm_dw_contiguous()
    test_gemm_backward_w()
