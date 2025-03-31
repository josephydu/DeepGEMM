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

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    return x_fp8, y_fp8, out, ref_out

def construct_grouped(num_groups: int, m: int, k: int, n: int, is_masked: bool) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out

def construct_backward_w(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()
    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_token_cast_to_fp8(y)
    #x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    #y_fp8 = (y_fp8[0], get_col_major_tma_aligned_tensor(y_fp8[1]))
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
    out = torch.empty((sum(m_list), n), device='cuda', dtype=torch.bfloat16)
    
    # calc ref_out first, ref out is varlen grouped
    ref_out = torch.zeros_like(out)
    start_idx = 0
    for i, m in enumerate(m_list):
        x_part = x[start_idx:start_idx + m]
        y_part = y[i]
        gemm = x_part @ y_part.t()
        ref_out[start_idx:start_idx + m] = gemm
        start_idx += m
        
    assert sum(m_list) % 4 == 0, f'TMA alignment error: {m}'
    
    x_fp8 = (
    torch.empty_like(x, dtype=torch.float8_e4m3fn),
    torch.empty((sum(m_list), k // 128), device='cuda', dtype=torch.float)
    )
    y_fp8 = (
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, n, k // 128), device='cuda', dtype=torch.float) # NOTE: per-token
    )
    
    seq_len = torch.Tensor([0] + m_list)
    cu_seq_len = torch.cumsum(seq_len, dim=0).to(torch.int32).to('cuda')
    for i in range(num_groups):
        x_fp8[0][cu_seq_len[i]:cu_seq_len[i + 1]], x_fp8[1][cu_seq_len[i]:cu_seq_len[i + 1]] = per_token_cast_to_fp8(x[cu_seq_len[i]:cu_seq_len[i + 1]])
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


def construct_dw_varlen_xy_grouped(num_groups, groups_list, k, is_masked):
    x = torch.cat([torch.randn((m, k), device='cuda', dtype=torch.bfloat16) for m in groups_list], dim=0)
    y = torch.cat([torch.randn((n, k), device='cuda', dtype=torch.bfloat16) for n in groups_list], dim=0)
    
    print(x.shape, y.shape)
    out = torch.empty((sum(groups_list), sum(groups_list)), device='cuda', dtype=torch.bfloat16)
    
    # calc ref_out first, ref out is varlen grouped
    ref_out = torch.zeros_like(out)
    start_idx = 0
    for i, group in enumerate(groups_list):
        x_part = x[start_idx:start_idx + group]
        y_part = y[start_idx:start_idx + group]
        print('x_part.shape:', x_part.shape, 'y_part.shape:', y_part.shape)
        gemm = x_part @ y_part.t()
        ref_out[start_idx:start_idx + group, start_idx:start_idx + group] = gemm
        start_idx += group
        
    assert sum(groups_list) % 4 == 0, f'TMA alignment error: {groups_list}'
    
    x_fp8 = (
    torch.empty_like(x, dtype=torch.float8_e4m3fn),
    torch.empty((sum(groups_list), k // 128), device='cuda', dtype=torch.float)
    )
    y_fp8 = (
    torch.empty_like(y, dtype=torch.float8_e4m3fn),
    torch.empty((sum(groups_list), k // 128), device='cuda', dtype=torch.float)
    )
    
    seq_len = torch.Tensor([0] + groups_list)
    cu_seq_len = torch.cumsum(seq_len, dim=0).to(torch.int32).to('cuda')
    for i in range(num_groups):
        x_fp8[0][cu_seq_len[i]:cu_seq_len[i + 1]], x_fp8[1][cu_seq_len[i]:cu_seq_len[i + 1]] = per_token_cast_to_fp8(x[cu_seq_len[i]:cu_seq_len[i + 1]])
        y_fp8[0][cu_seq_len[i]:cu_seq_len[i + 1]], y_fp8[1][cu_seq_len[i]:cu_seq_len[i + 1]] = per_token_cast_to_fp8(y[cu_seq_len[i]:cu_seq_len[i + 1]])

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
        (4, [4096, 8192, 8192, 2048], 7168, 4096),  
        (3, [8192, 3072, 4096, ], 2048, 7168),  
        (4, [8192, 8192, 8192, 8192], 7168, 4096),
        (8, [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], 7168, 4096),
        
    ]
    for num_groups, m_list, k, n in configs:
        x_fp8, y_fp8, out, ref_out = construct_dw_varlen_grouped(num_groups, m_list, k, n, is_masked=False)
        m_indices = torch.cat([torch.full((m,), i, device='cuda', dtype=torch.int) for i, m in enumerate(m_list)])
        deep_gemm.m_grouped_gemm_dw_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out,m_indices)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'm={sum(m_list) * num_groups}, {k=}, {n=}, {diff:.5f}'
        torch.cuda.synchronize()

        def test_func():
            # Construct new tensors every time to avoid L2 cache acceleration
            x_fp8, y_fp8, out, ref_out = construct_dw_varlen_grouped(num_groups, m_list, k, n, is_masked=False)
            m_indices = torch.cat([torch.full((m,), i, device='cuda', dtype=torch.int) for i, m in enumerate(m_list)])
            deep_gemm.m_grouped_gemm_dw_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        
        # num_groups * (m * k + k * n + m * n * 2)
        
        total_gb = 0
        for m in m_list:
            total_gb += m * k + k * n + m * n * 2
        
        print(f' > Performance ({num_groups=}, m_list_per_group={m_list}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * sum(m_list) * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{total_gb / 1e9 / t:4.0f} GB/s')
    print()



def test_m_grouped_gemm_dw_varlen_xy_contiguous()->None:
    print('Testing grouped variable length for x and y contiguous GEMM:')
    configs = [
        # (4, [4096, 8192, 8192, 2048], 7168),  
        # (3, [8192, 3072, 4096, ], 2048),  
        # (4, [8192, 8192, 8192, 8192], 7168),
        (8, [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], 7168),
        
    ]
    # NOTE: m_list must be the same as n_list
    for num_groups, groups_list, k in configs:
        x_fp8, y_fp8, out, ref_out = construct_dw_varlen_xy_grouped(num_groups, groups_list, k, is_masked=False)
        m_indices = torch.cat([torch.full((m,), i, device='cuda', dtype=torch.int) for i, m in enumerate(groups_list)])
        deep_gemm.m_grouped_gemm_dw_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out,m_indices,num_groups)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'm={sum(groups_list) * num_groups}, {k=}, {diff:.5f}'
        torch.cuda.synchronize()

        def test_func():
            # Construct new tensors every time to avoid L2 cache acceleration
            x_fp8, y_fp8, out, ref_out = construct_dw_varlen_grouped(num_groups, groups_list, k, n, is_masked=False)
            m_indices = torch.cat([torch.full((m,), i, device='cuda', dtype=torch.int) for i, m in enumerate(groups_list)])
            deep_gemm.m_grouped_gemm_dw_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices,num_groups)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        
        # num_groups * (m * k + k * n + m * n * 2)
        
        total_gb = 0
        for m in groups_list:
            total_gb += m * k + k * n + m * n * 2
        
        print(f' > Performance ({num_groups=}, m_list_per_group={groups_list}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * sum(groups_list) * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{total_gb / 1e9 / t:4.0f} GB/s')
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
        torch.cuda.synchronize()

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

def test_m_grouped_gemm_contiguous() -> None:
    print('Testing grouped contiguous GEMM:')

    for num_groups, m, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):
        # TODO: make a stronger test
        x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
        m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
        m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'

        # noinspection PyShadowingNames
        def test_func():
            # Construct new tensors every time to avoid L2 cache acceleration
            x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
            m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
            m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
    print()

def test_m_grouped_gemm_masked() -> None:
    print('Testing grouped masked GEMM:')

    for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168), ):
            # Test correctness
            masked_m_candidates = list(filter(lambda candidate: candidate <= m, (64, 128, 192, 256, 320, 384)))
            for i in range(10):
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
                masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
                for j in range(num_groups):
                    masked_m[j] = random.choice(masked_m_candidates)
                expected_m = min(int(masked_m.float().mean()) + 1, m)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m)
                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{m=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
                masked_m = torch.ones((num_groups, ), device='cuda', dtype=torch.int) * m
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, m)

            # Test performance with fixed shapes
            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
    print()

def test_gemm() -> None:
    print('Testing GEMM:')
    for m in (64, 128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()
if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    test_m_grouped_gemm_dw_varlen_xy_contiguous()
    # test_m_grouped_gemm_dw_varlen_contiguous()
    # test_m_grouped_gemm_dw_contiguous()
    # test_gemm_backward_w()
    # test_gemm()
    # test_m_grouped_gemm_contiguous()
    # test_m_grouped_gemm_masked()
