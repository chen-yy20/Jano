"""
Benchmark: N 次小 cudaHostAlloc vs 1 次大 cudaHostAlloc（总量相同）
+ 新增 Case 5: 直接多维数组索引方案
"""
import time
import torch

# ── 参数（模拟 30 层 × 2 key × s/m kv） ──────────────────────────────────
NUM_KEYS   = 60          # 30 层 × 2
NUM_LAYERS = 30
S_SHAPE    = (1, 13780, 64, 128)   # static  kv，[B, seq, heads, head_dim]
M_SHAPE    = (1,  2380, 64, 128)   # moderate kv

DTYPE      = torch.bfloat16
REPEATS    = 20

# ── 辅助 ──────────────────────────────────────────────────────────────────
def bench(label: str, fn, repeats=REPEATS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeats * 1000
    print(f"  {label:50s}  {elapsed:.3f} ms / call")
    return elapsed

# ── Case 1: N 次独立小分配（原始方案） ───────────────────────────────────
def alloc_many():
    bufs = []
    for _ in range(NUM_KEYS // 2):
        bufs.append(torch.empty(S_SHAPE, dtype=DTYPE, pin_memory=True))
        bufs.append(torch.empty(M_SHAPE, dtype=DTYPE, pin_memory=True))
    del bufs

# ── Case 5a: 多维数组，s/m 分开两个大 tensor ─────────────────────────────
# shape: [NUM_LAYERS, *S_SHAPE]  直接用 layer_idx 索引
# 访问: s_pool[layer_idx]  → view of shape S_SHAPE，零拷贝
def alloc_array():
    s_pool = torch.empty((NUM_LAYERS, *S_SHAPE), dtype=DTYPE, pin_memory=True)
    m_pool = torch.empty((NUM_LAYERS, *M_SHAPE), dtype=DTYPE, pin_memory=True)
    del s_pool, m_pool

# ── Case 5b: 验证 pool[i] 的访问开销（pool 已存在，只做索引） ────────────
_s_pool_global = None
_m_pool_global = None

def index_array_only():
    """pool 已存在，模拟运行时 store/fetch 时的 tensor 取出"""
    bufs = []
    for i in range(NUM_LAYERS):
        bufs.append(_s_pool_global[i])   # 返回 S_SHAPE 的 view，零拷贝
        bufs.append(_m_pool_global[i])
    del bufs

# ── Case 5c: 验证 pool[i] 仍然是 pinned ──────────────────────────────────
def verify_pool_pinned():
    s_pool = torch.empty((NUM_LAYERS, *S_SHAPE), dtype=DTYPE, pin_memory=True)
    view = s_pool[0]
    is_pinned = view.is_pinned()
    print(f"  pool[0].is_pinned() = {is_pinned}  "
          f"({'✓ DMA 可用' if is_pinned else '❌ 不是 pinned，DMA 不可用！'})")
    del s_pool

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    total_bytes = (
        torch.Size(S_SHAPE).numel() * NUM_LAYERS
        + torch.Size(M_SHAPE).numel() * NUM_LAYERS
    ) * 2  # bfloat16 = 2 bytes
    print(f"\n总 pinned 内存量: {total_bytes / 1024**3:.3f} GB")
    print(f"NUM_LAYERS={NUM_LAYERS}, DTYPE={DTYPE}, REPEATS={REPEATS}\n")

    print("=== Benchmark ===")
    t_many  = bench("60 × torch.empty(S/M_SHAPE, pin_memory=True)  [原始]", alloc_many)
    t_array = bench("2  × torch.empty([30,*S/M_SHAPE], pin_memory=True) [新]", alloc_array)

    _s_pool_global = torch.empty((NUM_LAYERS, *S_SHAPE), dtype=DTYPE, pin_memory=True)
    _m_pool_global = torch.empty((NUM_LAYERS, *M_SHAPE), dtype=DTYPE, pin_memory=True)
    t_index = bench("pool[i] 索引 × 60 (no alloc baseline)         [新]", index_array_only)

    print(f"\n=== 结论 ===")
    print(f"  分配: many={t_many:.3f}ms  array={t_array:.3f}ms  "
          f"比值={t_many/t_array:.2f}x")
    print(f"  运行时索引开销: {t_index:.3f} ms（{t_index/t_array*100:.0f}% of alloc）")

    print("\n=== Pinned 正确性验证 ===")
    verify_pool_pinned()