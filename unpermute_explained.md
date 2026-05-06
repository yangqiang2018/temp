# `MoeTokenUnpermute` 功能讲解

> 配套代码：[moe_token_unpermute.py](moe_token_unpermute.py)
> 姊妹算子：[moe_token_permute.py](moe_token_permute.py)、[moe_token_permute_grad.py](moe_token_permute_grad.py)（反向已有专文 [permute_grad_explained.md](permute_grad_explained.md)）

## 1. 它在 MoE 里属于哪一步？

MoE 前向是 **「散开 → 各专家算 → 收回」** 三段：

```
   permute            expert compute             unpermute
  ┌──────────┐       ┌───────────────┐          ┌─────────────────┐
  │ scatter  │       │ each expert   │          │  gather + (opt) │
  │ K copies │  -->  │ runs its FFN  │   -->    │  weighted sum   │
  └──────────┘       └───────────────┘          └─────────────────┘
  [T,H] -> [E,H]      [E,H] (in-place)           [E,H] -> [T,H]
```

- **前向 `permute`**：每个 token 复制 K 份, 按目标专家排序 (`tokens[T,H] -> permuted[E,H]`)
- **expert compute**：每个专家对自己拿到的子集做 FFN, 形状不变, 内容变化
- **前向 `unpermute`**：把 K 份"专家加工过的副本"按 routing probability 加权求和回去 (`permuted[E,H] -> output[T,H]`)

本文档专讲第三步——unpermute 前向。

---

## 2. 一句话定义

> 对每个原始 token `i`, 把它在 `permuted_tokens` 里散落的 `topK` 份按 `probs[i,k]` 加权求和, 写到 `output[i]`.

**With-probs 模式**（典型用法）：

```
output[i, :] = Σ_{k=0..topK-1}  probs[i, k] * permuted_tokens[ sorted_indices[i*K+k], : ]
```

**No-probs 模式**（退化版, 见 §4.3）：不加权, 也不求和, 单纯按 `sorted_indices` 把行重新排回 `(token, k)` 顺序：

```
output[r, :] = permuted_tokens[ sorted_indices[r], : ]   for r in 0..E-1
```

输出形状随模式不同：

| 模式 | 输出 shape | 含义 |
|---|---|---|
| with probs | `[T, H]` | 求和后, 每个原始 token 一行 |
| no probs   | `[E, H]` | 没求和, 把 permuted 按 sorted_indices "回正" |

---

## 3. 接口一览

`MoeTokenUnpermute.__call__` 真值签名：

```text
permuted_tokens : [E, H]    <-- intermediate after expert FFN
sorted_indices  : [E]       <-- index from forward permute (1D)
probs           : [T, K]    <-- optional; required when has_probs=True
                          -->
output (with probs)    : [T, H]
output (no probs)      : [E, H]
```

记号：

- `T = num_tokens`
- `K = topK`
- `H = hidden_size`
- `E = T * K`

---

## 4. 用一个迷你例子把过程画清楚

为了和 `permute_grad_explained.md` 对得上，**沿用同一个迷你局面**：

```
T = 3         (3 original tokens: t0, t1, t2)
K = 2         (each token picks 2 of 4 experts)
H = 4         (hidden size = 4)
experts = 4   (E0, E1, E2, E3)
E = T*K = 6   (permuted tensor has 6 rows)
```

路由表：

```
t0 -> (E0, E2)
t1 -> (E3, E1)
t2 -> (E0, E3)
```

排序后的 6 个 slot（`permuted_tokens` 的行序，和前向 permute 输出一致）：

```
     slot:    0      1      2      3      4      5
            ┌──────┬──────┬──────┬──────┬──────┬──────┐
permuted_x: │  t0  │  t2  │  t1  │  t0  │  t1  │  t2  │
            └──────┴──────┴──────┴──────┴──────┴──────┘
             └─ E0 ──┘    └ E1 ┘ └ E2 ┘ └─── E3 ────┘
```

由此 `sorted_indices`（行优先：第 `i*K+k` 项 = token `i` 第 `k` 份去了哪个 slot）：

```
sorted_indices (flat) = [0, 3, 4, 2, 1, 5]
                          │  │  │  │  │  │
                          │  │  │  │  │  └ t2's k=1  -> slot 5
                          │  │  │  │  └─── t2's k=0  -> slot 1
                          │  │  │  └────── t1's k=1  -> slot 2
                          │  │  └───────── t1's k=0  -> slot 4
                          │  └──────────── t0's k=1  -> slot 3
                          └─────────────── t0's k=0  -> slot 0
```

补一组 routing probability（这是 unpermute 多出来的输入）：

```
probs (T=3, K=2):
            k=0   k=1
   t0  →  [ 0.6   0.4 ]
   t1  →  [ 0.3   0.7 ]
   t2  →  [ 0.5   0.5 ]
```

「专家算完了」之后，`permuted_tokens` 的 6 行内容已经被各自的 expert FFN 改写了，记作 `P[0]..P[5]`（每行仍是长度 4 的向量）：

```
permuted_tokens (post expert compute, E=6, H=4)

   row 0  (was t0, ran E0)  :  P0 = [P0_0  P0_1  P0_2  P0_3]
   row 1  (was t2, ran E0)  :  P1 = [P1_0  P1_1  P1_2  P1_3]
   row 2  (was t1, ran E1)  :  P2 = [P2_0  P2_1  P2_2  P2_3]
   row 3  (was t0, ran E2)  :  P3 = [P3_0  P3_1  P3_2  P3_3]
   row 4  (was t1, ran E3)  :  P4 = [P4_0  P4_1  P4_2  P4_3]
   row 5  (was t2, ran E3)  :  P5 = [P5_0  P5_1  P5_2  P5_3]
```

注意 `P0` 和 `P3` 不再相等了——虽然两行原本都是 t0，但分别经过 E0 和 E2 两个不同专家加工，输出值通常不同。

### 4.0 先把两个 shape 直觉立起来

**(a) `H` 是「每个 token 的向量维度」。** 例子里 `H=4`, 真实模型里通常 4096/8192。

**(b) probs 是 `[T, K]` 的概率（不是 `[E]`！）。** 它跟 token 一一对应，**不**跟 slot 对应——所以 kernel 里要先按 `(i, k)` 顺序读 probs，再用 `sorted_indices[i*K+k]` 间接定位 `permuted_tokens` 的行。

**(c) 输出形状两种模式不同**：

```
with probs:    output [T, H] = [3, 4]   one row per original token (after sum)
no  probs:     output [E, H] = [6, 4]   one row per slot (no sum, just unsort)
```

### 4.1 With-probs 模式：gather + 加权求和

这是 MoE 前向真正用的模式。对每个 token `i`，沿 K 维做加权和：

```
                                    sorted_indices       output (with probs)
                                    (i*K+k -> slot)      [T=3, H=4]

 t0  --gather-->  P[ slot 0 ] *0.6  ┐                    ┌──────────────────────┐
                  P[ slot 3 ] *0.4  ┴── sum -----------► │  0.6*P0 + 0.4*P3     │  row t0
                                                         ├──────────────────────┤
 t1  --gather-->  P[ slot 4 ] *0.3  ┐                    │  0.3*P4 + 0.7*P2     │  row t1
                  P[ slot 2 ] *0.7  ┴── sum -----------► ├──────────────────────┤
                                                         │                      │
 t2  --gather-->  P[ slot 1 ] *0.5  ┐                    │  0.5*P1 + 0.5*P5     │  row t2
                  P[ slot 5 ] *0.5  ┴── sum -----------► └──────────────────────┘
```

写成等式：

```
output[t0] = 0.6 * P0 + 0.4 * P3
output[t1] = 0.3 * P4 + 0.7 * P2
output[t2] = 0.5 * P1 + 0.5 * P5
```

每行还是 H 个数字, 比如：

```
output[t0] = [ 0.6*P0_0 + 0.4*P3_0,
               0.6*P0_1 + 0.4*P3_1,
               0.6*P0_2 + 0.4*P3_2,
               0.6*P0_3 + 0.4*P3_3 ]
```

**与 permute_grad 的关系**：去掉概率（让所有 `probs[i,k] = 1`），公式就退化成 `permute_grad`——同一根「gather + reduce」骨架，只是 reduce 的系数从 1 变成了 probs。详见附录 B。

### 4.2 加权操作的硬件原语：axpy

把"乘上 prob 后累加"叫 **axpy**（"a · x + y"），是 BLAS 经典原语，硬件上就是 fused multiply-accumulate（FMA）：

```
acc += prob * row    <==>    axpy(acc, row, prob)
```

kernel 里这一行（with-probs cast 路径）：

```
T.tile.axpy(acc_buf, row_f32, prob)    # acc_buf  +=  prob * row_f32
```

为什么要 fused：单条指令同时做乘法和加法，**精度高一档**（中间结果保留全精度），且**指令数减半**。详见附录 A。

### 4.3 No-probs 模式：纯 unsort（不求和）

`has_probs=False` 时，kernel 退化成「按 `sorted_indices` 把 permuted 的行**重新排回 (token, k) 顺序**」，输出形状变成 `[E, H]`：

```
                       sorted_indices                output (no probs)
                       (flat[r] -> source slot)      [E=6, H=4]

  r=0  ──pick from──►  slot 0   (= P0) ───────────► row 0 = P0
  r=1  ──pick from──►  slot 3   (= P3) ───────────► row 1 = P3
  r=2  ──pick from──►  slot 4   (= P4) ───────────► row 2 = P4
  r=3  ──pick from──►  slot 2   (= P2) ───────────► row 3 = P2
  r=4  ──pick from──►  slot 1   (= P1) ───────────► row 4 = P1
  r=5  ──pick from──►  slot 5   (= P5) ───────────► row 5 = P5
```

**直观理解**：permute 把行按"哪个专家"分组，no-probs unpermute 把行**按"原始 (token, k) 顺序"重新摆回**。所以 `permute -> no-probs unpermute` 是个"把序号从 expert-major 切回 token-major"的纯打散重排，K 份副本一份不少。

什么时候用这个模式：

- 调试 / round-trip 检验（测试代码里就有 `permute -> unpermute -> 应等于 K 份原 token` 的用例）
- 显式需要看每份梯度而**不立即求和**的场景（比如分析每条专家路径的贡献）

---

## 5. 与 permute_grad 的对照

把两个算子并排看，骨架一模一样，区别只在 reduce 的"系数"和数据流方向：

| 维度 | `permute_grad`（反向） | `unpermute`（前向） |
|---|---|---|
| 输入张量 | `permuted_output_grad [E,H]` | `permuted_tokens [E,H]` |
| 索引 | `sorted_indices [T,K]` | `sorted_indices [T,K]` |
| 加权 | 无（系数恒为 1） | 有（`probs[i,k]`） |
| reduce | sum | sum (axpy) |
| 输出 | `input_grad [T,H]` | `output [T,H]` |
| 内层操作 | `acc += row` | `acc += prob * row` |

数学上：

```
permute_grad :  out[i] = Σ_k          1     * x[ sorted_indices[i*K+k] ]
unpermute    :  out[i] = Σ_k  probs[i,k]    * x[ sorted_indices[i*K+k] ]
                              ─────────
                              ^^ only difference
```

这就是为什么两份 kernel 的内层循环骨架几乎一样，但 unpermute 多出 `probs_ub` / `probs_f32` / `axpy` 这一组件。

---

## 6. 与 NPU 并行结构对应

`with_probs` 的 cast 路径有 cid / vid 双轴并行（和 `permute_grad` 同款）：

```
parallel axes:

   cid (core id)        --> splits T tokens across actual_cores
                            each core takes tokens_per_core tokens

   vid (vector pipe id) --> splits H into two halves
                            vid=0 -> left  HALF_H = TILE_H/2 columns
                            vid=1 -> right HALF_H = TILE_H/2 columns

per-core view (one core processing its tokens):

           |<-- HALF_H -->|<-- HALF_H -->|
           ┌──────────────┬──────────────┐
   token 0 │   vid = 0    │   vid = 1    │
   token 1 │              │              │
    ...    │              │              │
           └──────────────┴──────────────┘
```

`no_probs` 路径是按 `TILE_E` 这一维（输出行）切，不是按 token 切——因为它的输出形状是 `[E, H]`：

```
   cid (core id) --> splits the E output rows across cores
                     each core handles tiles_per_core tiles of TILE_E rows
   vid (vector pipe id) --> same H halving as with-probs
```

每核内的循环骨架（with-probs cast 路径，源码 [moe_token_unpermute.py:156-209](moe_token_unpermute.py#L156)）：

```
for ti in 0..BATCH_T-1:                        # token within current batch
    acc_buf <- 0
    for lane in 0..topK-1:                     # accumulate K rows
        src  = idx_ub  [ti*K + lane]           # gather source slot
        prob = probs_f32[ti*K + lane]          # corresponding prob (already fp32)
        row_buf <- perm_tokens_gm[src, h_off:h_off+HALF_H]
        row_f32 <- cast(row_buf)               # fp16/bf16 -> fp32
        acc_buf += prob * row_f32              # axpy (fused)
    out_buf <- cast(acc_buf)                   # fp32 -> fp16/bf16
    perm_tokens_gm[ti, h_off:h_off+HALF_H] <- out_buf
```

注意几个和 `permute_grad` 的细微差别：

- 多了 `probs_ub` 和它的 fp32 版 `probs_f32`，`probs` 一开始就批量 cast 到 fp32 减少后续逐元素 cast
- 累加用 `axpy(acc, row_f32, prob)` 而不是 `add(acc, acc, row_f32)`

---

## 7. 三个 kernel 变体（自动选路）

`_compile_gather` 按 `(has_probs, dtype)` 选 3 个变体之一：

```
                ┌──────────────┐
                │  has_probs?  │
                └──────────────┘
                  │           │
                 True       False
                  │           │
                  ▼           ▼
              ┌───────┐    no_probs   (one shared kernel for all dtypes)
              │dtype? │
              └───────┘
              │       │
            fp32   fp16/bf16
              │       │
              ▼       ▼
         f32 path   cast path
        (no cast)  (cast + axpy)
```

| 变体 | 适用 | 关键特点 |
|---|---|---|
| `_build_gather_kernel_with_probs`     | has_probs + (fp16/bf16) | fp32 累加器, axpy 累加, 末尾 cast 回 |
| `_build_gather_kernel_with_probs_f32` | has_probs + fp32        | 直接 fp32 累加, 用三元组展开搬运 |
| `_build_gather_kernel_no_probs`       | no probs                | 只 gather + 重排, 无 reduce |

数学语义在 `with_probs` 的两条路径里完全一致；区别只是 **fp32 路径不需要 cast，直接累加就够**，省掉了来回 cast 的开销。

---

## 8. Python 包装层做的事

`MoeTokenUnpermute.__call__` 在调 kernel 前后做几步杂事：

```
caller    permuted_tokens: [E, H']    sorted_indices: [E]      probs: [T, K]
                  │                          │                       │
                  │                          │ pad_first_dim          │
                  │                          │ to padded_E            │
                  │                          │ .unsqueeze(0)          │
                  │                          ▼                       │
                  │              indices_padded_2d: [1, padded_E]    │
                  │                          │                       │
                  │ pad_last_dim                                     │ pad_first_dim
                  │ to H_compile                                     │ to padded_tokens
                  ▼                                                   │ .reshape(1, -1)
        permuted_tokens_in: [E, H_compile]                            ▼
                  │                                       probs_flat: [1, padded_tokens*K]
                  └────────────┬─────────────────────────────────────┘
                               ▼
                       kernel(perm, idx, probs)   (or kernel(perm, idx) if no probs)
                               │
                               ▼
                     output: [T, H_compile]   (or [E, H_compile] if no probs)
                               │
                     if hidden_size != H_compile:
                               │ slice off padded columns
                               ▼
                     output: [T, hidden_size]
```

为何要 pad：

- **`H` 方向 pad**：`min_compile_h = 64` (fp32) / `32` (fp16/bf16)，保证 kernel `alloc_ub` 行宽 ≥ 32B 对齐。`hidden_size = 4` 也能编译跑通，原因是按 32 元素编译，输出再裁回真实宽度。
- **`probs` 第一维 pad**：到 `padded_tokens`（`actual_cores * tokens_per_core` 向上对齐），让最后一个 core 也有完整的 batch 可读。
- **`sorted_indices` pad**：到 `padded_E = padded_tokens * K`，与 probs 的 padded 对齐。

---

## 9. 一句话总结

> `MoeTokenUnpermute` = 把 `permuted_tokens` 当成 K 份散落的 token 副本池, 按 `sorted_indices` 对每个原始 token **拉回 K 份, 用 probs 加权求和**, 得到 `output[T, H]`.
>
> **它和前向 `permute` 互为反向语义**: permute 是「**复制 + 散开**」, unpermute 是「**收回 + 加权求和**」。
>
> 数学上又**和 `permute_grad` 同骨架**: 都是 gather-K-rows-per-token-then-reduce; unpermute 加了 probs 系数。

---

## 附录 A: axpy 是什么？

「**axpy**」= **a · x + y**, BLAS Level-1 的经典原语（也叫"saxpy"/"daxpy"等，按 dtype 起头字母）。

```
y[i] = a * x[i] + y[i]    for all i
```

- `a` 是标量
- `x` 是向量
- `y` 是被累加的向量（结果）

放回 unpermute 的语境：

```
y    = acc_buf      (fp32 accumulator, length HALF_H)
a    = prob         (a single prob scalar)
x    = row_f32      (one row of perm_tokens cast to fp32, length HALF_H)
```

那条 `T.tile.axpy(acc_buf, row_f32, prob)` 等价于：

```python
for i in range(HALF_H):
    acc_buf[i] = prob * row_f32[i] + acc_buf[i]   # elementwise fused multiply-add
```

为什么不写成 `acc += prob * row` 两步：

1. **精度**：fused multiply-add (FMA) 中间结果保留全精度, 不会先把 `prob*row` round 到目标精度再加, 末位误差更小
2. **指令数**：硬件一条指令做完 mul+add, 而非两条
3. **吞吐**：NPU vector pipe 对 axpy 类原语有专门的 throughput 优化

---

## 附录 B: unpermute vs permute_grad 完整对照

| 维度 | `unpermute` (with probs) | `permute_grad` |
|---|---|---|
| 在 MoE 中的位置 | 前向最后一步 | permute 前向的反向 |
| 语义            | gather K + 加权求和 | gather K + 求和 (无权) |
| 数据张量        | `permuted_tokens` (post-expert) | `permuted_output_grad` (上游梯度) |
| 是否有 probs    | 是 (`probs[T,K]`) | 否 |
| reduce 操作     | `axpy` (`acc += prob * row`) | `add` (`acc += row`) |
| 输出形状        | `[T, H]` | `[T, H]` |
| 数据流方向      | 各专家 → 原 token | 上游梯度 → 输入梯度 |
| Kernel 变体数   | 3 (with-probs cast / with-probs fp32 / no-probs) | 4 (group-pipelined / lane-pipelined / plain cast / nocast) |
| 流水线复杂度    | 简单 (barrier_all 同步) | 复杂 (set/wait flag, 双 buffer / 8 stages) |

> 一句话: 把 `permute_grad` 的 reduce 系数从 1 换成 `probs[i,k]`, 就是 `unpermute` (with probs)。
