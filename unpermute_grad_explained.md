# `MoeTokenUnpermuteGrad` 功能讲解

> 配套代码：[moe_token_unpermute_grad.py](moe_token_unpermute_grad.py)
> 配套前向：[moe_token_unpermute.py](moe_token_unpermute.py)
> 姊妹篇：[permute_grad_explained.md](permute_grad_explained.md)（强烈建议先读，这里直接复用同一套迷你例子）

## 1. 它在 MoE 里属于哪一步？

MoE 训练的完整数据流：

```
              forward                          backward
            ┌──────────┐                     ┌──────────────────┐
   tokens ──│ permute  │── permuted_tokens ──│ permute_grad     │── input_grad
            └──────────┘        │            └──────────────────┘
                                ▼
                         expert compute
                                │
                                ▼
                      permuted_outputs ──┐
                              │          ▼
                              │      ┌──────────┐
                              │      │ unpermute│── unpermuted_tokens
                              │      └──────────┘        │
                              ▼                          ▼
                         ┌──────────────────┐
                         │  unpermute_grad  │── perm_grad, probs_grad
                         └──────────────────┘
```

- **前向 `unpermute`**：把每个 token 散落在 K 个专家位置上的输出**合回成一份**（with probs）或**只做行重排**（no probs）。
- **反向 `unpermute_grad`**：上游回传给 `unpermuted_tokens` 的梯度，**散播回**`permuted_tokens` 的对应行；如果 forward 用了 `probs` 加权，还要给 `probs` 也算一份梯度。

本文档专讲反向。它有 **两个模式**（has_probs True/False），数学语义差别很大，要分开讲。

---

## 2. 两个模式

### 2.1 `has_probs = False` —— 纯行重排的反向

**Forward 做的事**（不涉及 K 路 reduce）：

```
unpermuted[r, :] = permuted[ sorted_indices[r], : ]      r ∈ [0, E)
```

就是按 `sorted_indices` 把 `permuted` 的行**重新排个顺序**，输出形状仍然 `[E, H]`。

**Backward** 对 `permuted` 的梯度：

```
perm_grad[ sorted_indices[r], : ] = unpermuted_grad[r, :]      r ∈ [0, E)
```

每个目标行**只被写一次**（`sorted_indices` 是 0..E-1 的排列，1-to-1）——纯 **scatter**，没有累加。

### 2.2 `has_probs = True` —— 带概率加权的 K 路 reduce 的反向

**Forward 做的事**（这是真正"合回一份"的版本）：

```
unpermuted[i, :] = Σ_{k=0..K-1}  probs[i, k] · permuted[ sorted_indices[i*K + k], : ]
```

输出形状 `[T, H]`——每个原始 token 一行，是 K 个专家输出的概率加权和。

**Backward** 同时产出 **两个梯度**：

| 输出 | shape | 公式 |
|---|---|---|
| `perm_grad[j, :]` 其中 j = sorted_indices[i*K+k] | `[E, H]` | `probs[i, k] · unpermuted_grad[i, :]` |
| `probs_grad[i, k]` | `[T, K]` (标量) | `<unpermuted_grad[i, :],  permuted[ sorted_indices[i*K+k], : ]>` |

第一条是**带乘法的 scatter**（每个目标行依然只写一次）。第二条是**两个 H 维向量的点积**（reduce 掉 H 轴，剩 1 个标量）。

不熟 scatter / 点积这两个词的话，先看[附录 A](#附录-a-scatter--点积是什么)。

---

## 3. 接口一览

`MoeTokenUnpermuteGrad.__call__` 的签名：

```text
permuted_tokens         : [E,  H]    <-- forward input, also needed by backward (only if has_probs)
unpermuted_tokens_grad  : [E,  H]    <-- when has_probs = False
                          [T,  H]    <-- when has_probs = True
sorted_indices          : [E]        <-- index from forward permute
probs                   : [T,  K]    <-- only if has_probs = True
                                  -->
perm_grad               : [E,  H]    <-- always returned
probs_grad              : [T,  K]    <-- only if has_probs = True
```

记号同 `permute_grad`：`T=num_tokens`，`K=topK`，`H=hidden_size`，`E=T*K`。

---

## 4. 用一个迷你例子把过程画清楚

为了和 `permute_grad_explained.md` 对照，**继续沿用同一组参数和路由**：

```
T = 3         (3 original tokens: t0, t1, t2)
K = 2         (each token picks 2 of 4 experts)
H = 4         (hidden size = 4)
experts = 4   (E0, E1, E2, E3)
E = T*K = 6   (permuted tensor has 6 rows)

routing:  t0 -> (E0, E2)
          t1 -> (E3, E1)
          t2 -> (E0, E3)

sorted_indices (flat) = [0, 3, 4, 2, 1, 5]
```

permuted 张量 6 行各记作 `M0..M5`（每行是 H=4 维向量）：

```
   slot     belongs to     permuted_tokens
   ────    ──────────      ───────────────
    0      t0 -> E0        M0 = [M0_0  M0_1  M0_2  M0_3]
    1      t2 -> E0        M1 = [M1_0  M1_1  M1_2  M1_3]
    2      t1 -> E1        M2 = [M2_0  M2_1  M2_2  M2_3]
    3      t0 -> E2        M3 = [M3_0  M3_1  M3_2  M3_3]
    4      t1 -> E3        M4 = [M4_0  M4_1  M4_2  M4_3]
    5      t2 -> E3        M5 = [M5_0  M5_1  M5_2  M5_3]
```

### 4.1 has_probs = False：纯 scatter

#### Forward（顺便看一下，便于理解 backward）

输出也是 6 行，按 `sorted_indices` 重排 `permuted` 的行：

```
output_fwd[0] = M[sorted_indices[0]] = M[0] = M0
output_fwd[1] = M[sorted_indices[1]] = M[3] = M3
output_fwd[2] = M[sorted_indices[2]] = M[4] = M4
output_fwd[3] = M[sorted_indices[3]] = M[2] = M2
output_fwd[4] = M[sorted_indices[4]] = M[1] = M1
output_fwd[5] = M[sorted_indices[5]] = M[5] = M5
```

#### Backward：上游梯度记作 `g0..g5`，反着写回：

```
   unpermuted_grad         sorted_indices            perm_grad
                                                     ┌──────────┐
   row 0:  g0  ───────────►  to slot 0  ───────────► │  M0 := g0│
   row 1:  g1  ───────────►  to slot 3  ───────────► │  M3 := g1│
   row 2:  g2  ───────────►  to slot 4  ───────────► │  M4 := g2│
   row 3:  g3  ───────────►  to slot 2  ───────────► │  M2 := g3│
   row 4:  g4  ───────────►  to slot 1  ───────────► │  M1 := g4│
   row 5:  g5  ───────────►  to slot 5  ───────────► │  M5 := g5│
                                                     └──────────┘
```

写成等式：

```
perm_grad[sorted_indices[r]] = unpermuted_grad[r]    for r in 0..5
```

注意：**每个 perm_grad 行只被赋值一次**，没有累加，没有 K 维 reduce。这就是为什么这条路径叫"scatter kernel"——把上游 6 行**散播到** perm_grad 的 6 行（顺序变了而已）。

### 4.2 has_probs = True：scatter (带乘) + 点积

#### Forward

每个原始 token 的输出是 K=2 份专家结果的**概率加权和**。比如概率取 `probs[t0]=(0.6, 0.4)`：

```
unpermuted[t0] = 0.6 * M0 + 0.4 * M3              (t0's two copies live at slot 0 and 3)
unpermuted[t1] = 0.7 * M4 + 0.3 * M2              (assume probs[t1] = (0.7, 0.3))
unpermuted[t2] = 0.5 * M1 + 0.5 * M5              (assume probs[t2] = (0.5, 0.5))
```

输出形状 `[T, H] = [3, 4]`。

#### Backward 输出 1：perm_grad

每个 perm_grad 行 = 「原始 token 的梯度」乘上「当时 forward 用的 probs 系数」：

```
                                                          perm_grad
   ∂L/∂M0 = probs[t0, 0] · unp_grad[t0]                  (i.e. 0.6 · gT0)
   ∂L/∂M3 = probs[t0, 1] · unp_grad[t0]                  (0.4 · gT0)
   ∂L/∂M4 = probs[t1, 0] · unp_grad[t1]                  (0.7 · gT1)
   ∂L/∂M2 = probs[t1, 1] · unp_grad[t1]                  (0.3 · gT1)
   ∂L/∂M1 = probs[t2, 0] · unp_grad[t2]                  (0.5 · gT2)
   ∂L/∂M5 = probs[t2, 1] · unp_grad[t2]                  (0.5 · gT2)
```

（`gT0..gT2` 是 unpermuted_tokens_grad 的 3 行，每行 4 维。）

依然是 scatter，区别是写入前先**乘了一个标量 prob**。

#### Backward 输出 2：probs_grad

每个 `probs[i, k]` 的梯度 = **forward 乘上的 H 维向量**（即 `M[sorted_indices[i*K+k]]`）和**上游 H 维梯度** `unp_grad[i]` 的**点积**：

```
probs_grad[t0, 0] = <gT0, M0>      = gT0_0·M0_0 + gT0_1·M0_1 + gT0_2·M0_2 + gT0_3·M0_3
probs_grad[t0, 1] = <gT0, M3>
probs_grad[t1, 0] = <gT1, M4>
probs_grad[t1, 1] = <gT1, M2>
probs_grad[t2, 0] = <gT2, M1>
probs_grad[t2, 1] = <gT2, M5>
```

每条等式把**两个 4 维向量乘起来再求和成 1 个标量**——这就是点积，把 H 这根轴 reduce 掉了。所以 probs_grad 形状 `[T, K] = [3, 2]`，每个位置是个标量。

---

## 5. 与 NPU 并行结构的对应

和 `permute_grad` 一样用 `(cid, vid)` 双轴：

```
parallel axes (with-probs path):

   cid (core id)        --> splits T tokens across cores
                            each core takes tokens_per_core tokens
                            (no-probs path: tile over E axis instead, see below)

   vid (vector pipe id) --> splits H into two halves
                            vid=0 -> left  HALF_H = TILE_H/2 columns
                            vid=1 -> right HALF_H = TILE_H/2 columns


per-core view (one core processing its own tokens, with-probs path):

           |<-- HALF_H -->|<-- HALF_H -->|
           ┌──────────────┬──────────────┐
   token_0 │   vid = 0    │   vid = 1    │
   token_1 │              │              │
    ...    │              │              │
           └──────────────┴──────────────┘
```

**probs_grad 的特殊处理**：因为 vid=0 和 vid=1 各算了一半 H 的点积（点积是对 H 求和，所以两半要加起来），kernel 把 probs_grad 分别写到 `probs_grad_gm[0]` 和 `probs_grad_gm[1]`，**Python 包装层再把这两半相加**才得到最终的 probs_grad（详见 §7）。

with-probs 路径的核内主循环骨架（取 cast 路径）：

```
for ti in 0..tokens_per_core-1:
    i = cid * tokens_per_core + ti
    if i >= num_tokens: continue                         # padding token, skip

    idx_ub <- sorted_idx_gm[0, i*K : i*K+K]              # K slots for this token
    probs_ub <- probs_gm[i, 0:K]                         # K prob weights
    probs_f32 <- cast(probs_ub)
    pg_acc <- 0     # probs_grad accumulator (1 x K, accumulates over H tiles)

    for ht in 0..n_htiles-1:                             # H-axis tile
        h_off = ht*TILE_H + vid*HALF_H
        grad_buf <- out_grad_gm[i, h_off:h_off+HALF_H]   # slice of unpermuted_grad
        grad_f32 <- cast(grad_buf)

        for k in 0..K-1:
            dst = idx_ub[k]
            prob = probs_f32[k]

            # ---- output 1: perm_grad[dst, h_off:h_off+HALF_H] = prob * grad ----
            mul_buf <- grad_f32 * prob                   # scalar * vector (HALF_H wide)
            out_tmp <- cast(mul_buf)                     # fp32 -> low
            perm_grad_gm[dst, h_off] <- out_tmp          # scatter

            # ---- output 2: probs_grad[i, k] += <grad, M[dst]>_HALF_H ----
            perm_buf <- perm_tokens_gm[dst, h_off:h_off+HALF_H]
            perm_f32 <- cast(perm_buf)
            mul_buf <- grad_f32 * perm_f32               # elementwise (HALF_H wide)
            scalar <- reduce_sum(mul_buf)                # sum over H slice -> scalar
            pg_acc[0, k] += scalar                       # accumulate into (i, k) slot

    probs_grad_gm[vid, i, 0:K] <- pg_acc                 # write this vid's partial sum
```

注意 `pg_acc[0, k] += scalar` 是 fp32 累加（避免 H 很大时小数相加掉精度）。最终写出的 `probs_grad_gm` 是 `[2, padded_tokens, K]` 三维张量，`vid=0/1` 各占一份，由包装层求和。

---

## 6. 三个 kernel 变体

`_compile_grad` 根据 `has_probs` 和 `dtype == acc_dtype` 选路：

```
                          has_probs?
                  ┌───────────┴───────────┐
                  │                       │
                false                   true
                  │                       │
                  ▼                       ▼
        scatter_no_probs            dtype == acc_dtype (fp32) ?
        (perm_grad only)                  │
        (tile E axis,             ┌───────┴────────┐
         no fp32 acc,             │                │
         barrier_all sync)       yes              no
                                  │                │
                                  ▼                ▼
                       grad_with_probs_f32   grad_with_probs (cast)
                       (no cast needed,      (low + fp32 middle acc,
                        simple impl)          manual set/wait_flag pipeline)
```

| 变体 | 适用 | 同步策略 | 中间累加 |
|---|---|---|---|
| `_build_scatter_kernel_no_probs` | `has_probs=False` 全部 dtype | `T.barrier_all` 粗粒度 | 不需要（scatter 无 reduce） |
| `_build_grad_kernel_with_probs` (cast) | `has_probs=True`, 非 fp32 | 手工 `set/wait_flag` 细粒度 | fp32 |
| `_build_grad_kernel_with_probs_f32` | `has_probs=True`, fp32 | `AUTO_SYNC=True` 编译器管 | 直接 fp32 不需要 cast |

三种路径的**数学语义完全一致**，区别只在硬件层面的优化策略——和 `permute_grad` 那边是同一种工程取舍：让最常用 case 跑得最快，其它 case 用兜底实现。

---

## 7. Python 包装层做的额外事情

`MoeTokenUnpermuteGrad.__call__` 在 kernel 调用前后处理 pad 和 dtype：

```
caller   permuted_tokens: [E, H]   unpermuted_tokens_grad: [E or T, H]   sorted_indices: [E]   probs: [T, K]?
              │                              │                                   │                  │
              │                              │                                   │                  │
              ▼                              ▼                                   ▼                  ▼
     pad_last_dim                   pad_last_dim                          pad_first_dim       pad_first_dim
     to H_compile                   to H_compile                          to padded_E         to padded_tokens
              │                              │                                   │                  │
              └──────────────┬───────────────┴───────────────────────────────────┴──────────────────┘
                             ▼
                        kernel(...)
                             │
              ┌──────────────┴──────────────┐
              │                             │  (only if has_probs)
              ▼                             ▼
     perm_grad: [E, H_compile]    probs_grad_raw: [2, padded_tokens, K]
              │                             │
   slice off padded H cols        sum over leading dim 2 (vid 0 + vid 1)
              │                             │
              ▼                             ▼
     perm_grad: [E, H]            probs_grad: [padded_tokens, K]
                                            │
                                  slice [:T, :K]; cast back to probs.dtype
                                            │
                                            ▼
                                  probs_grad: [T, K]
```

几点要注意：

- **`H_compile` ≥ `min_compile_h`**：fp32 至少 64，bf16/fp16 至少 32（同 `permute_grad`）。小于这个值的 H 会先 pad 到 `min_compile_h` 再编译。
- **`padded_tokens` 是 T 向上对齐到 TILE_T**：`pg_acc` 写出时按 padded_tokens 长度走，超出 T 的部分是脏数据，最后切回前 T 行。
- **`probs_grad_raw` 的前导维 2** 是 vid 0/1 各贡献的部分和，**包装层做这个 reduce**。

---

## 8. 一句话总结

> `unpermute_grad` 是 `unpermute` 的反向：
>
> - **没 probs 时**：纯 **scatter**——把上游 6 行按 `sorted_indices` 写回 `perm_grad` 的 6 行，没有累加。
> - **有 probs 时**：在 scatter 的基础上**乘了一个标量 prob**；同时还要算 `probs_grad`，那是上游 H 维梯度和当时被乘的 H 维向量的**点积**（reduce 掉 H）。
>
> 它和 `permute_grad` 互补：`permute_grad` 是 **gather + sum**（K 行收回 1 行），`unpermute_grad` 主要是 **scatter**（1 行散到 K 行）+（可选）每对 (i,k) 的**点积**。

---

## 附录 A：scatter / 点积是什么

### A.1 scatter（按索引散播）

`gather` 的反操作。给一组源行 + 一组目标下标，把每个源行**写到**对应的目标位置：

```python
src = [g0, g1, g2, g3, g4, g5]   # source rows (6 rows)
idx = [0, 3, 4, 2, 1, 5]         # destination indices
dst = [None] * 6
for r in range(6):
    dst[idx[r]] = src[r]          # <-- this is scatter
# result: dst = [g0, g4, g3, g1, g2, g5]
```

放回 `unpermute_grad` 上：

```
   sorted_indices = [0, 3, 4, 2, 1, 5]   <-- this is the dest-index list


   step 1: each source row writes to dst[ idx[r] ]

       unpermuted_grad
       ┌─────────────┐
   r=0 │     g0      │ -- writes to dst[0]
   r=1 │     g1      │ -- writes to dst[3]
   r=2 │     g2      │ -- writes to dst[4]
   r=3 │     g3      │ -- writes to dst[2]
   r=4 │     g4      │ -- writes to dst[1]
   r=5 │     g5      │ -- writes to dst[5]
       └─────────────┘


   step 2: final state of dst (= perm_grad), organized by destination index

           ┌─────────────┐
   M0  =   │      g0     │   <- written by r=0
   M1  =   │      g4     │   <- written by r=4
   M2  =   │      g3     │   <- written by r=3
   M3  =   │      g1     │   <- written by r=1
   M4  =   │      g2     │   <- written by r=2
   M5  =   │      g5     │   <- written by r=5
           └─────────────┘
```

特点：

- **gather 是"读乱序、写顺序"**；**scatter 是"读顺序、写乱序"**。两者互为反向。
- 当 `idx` 是 1-to-1 映射（排列）时，scatter **不需要累加**（每个目标只写一次）。
- 当 `idx` 有重复时（gather 的反向通常会出现），就需要原子加（atomic add）。`unpermute_grad` 里 `sorted_indices` 总是排列，所以是无冲突 scatter，硬件实现简单很多。

### A.2 点积（dot product）

「点积」= **两个相同长度的向量逐位置相乘后求和**，结果是一个标量。

```
a = [a0  a1  a2  a3]
b = [b0  b1  b2  b3]

<a, b> = a0*b0 + a1*b1 + a2*b2 + a3*b3   (a scalar)
```

类比 numpy / torch：

```python
import torch
a = torch.tensor([1., 2., 3., 4.])
b = torch.tensor([5., 6., 7., 8.])
torch.dot(a, b)             # = 1*5 + 2*6 + 3*7 + 4*8 = 70.0
# or equivalently: (a * b).sum()
```

特点：

- 输入是两个 H 维向量，输出是 1 个标量 → 把 H 这根轴**完全 reduce 掉**了。
- 在 `unpermute_grad` 里就是 `T.tile.mul(...)` + `T.reduce_sum(..., dim=-1)` 这两步。

### A.3 为什么 `probs_grad` 是点积

回到 forward：

```
unpermuted[i, h] = Σ_k probs[i, k] · permuted[ idx[i*K+k], h ]
```

链式法则求 `∂L/∂probs[i, k]`：

```
∂L/∂probs[i, k] = Σ_h  (∂L/∂unpermuted[i, h]) · (∂unpermuted[i, h]/∂probs[i, k])
                = Σ_h  unp_grad[i, h] · permuted[ idx[i*K+k], h ]
                = <unp_grad[i, :],  permuted[ idx[i*K+k], : ]>
```

物理含义：probs 是个标量权重，**它影响 H 维向量的所有元素**——所以梯度要把 H 上每个位置的贡献**加起来**才完整。这就是 reduce 掉 H 的来源。
