# `MoeTokenPermuteGrad` 功能讲解

> 配套代码：[moe_token_permute_grad.py](moe_token_permute_grad.py)
> 配套前向：[moe_token_permute.py](moe_token_permute.py)

## 1. 它在 MoE 里属于哪一步？

MoE（Mixture of Experts）训练里的「routing/permute」分两步：

```
   forward                                backward
  ┌─────────┐                            ┌──────────────┐
  │ permute │  -->  expert compute  -->  │ permute_grad │
  └─────────┘                            └──────────────┘
```

- **前向 `permute`**：把每个 token 复制 `topK` 份，按目标专家排好序 → 得到 `permuted_tokens`。
- **反向 `permute_grad`**：把 `permuted_tokens` 上回传的梯度 **按 token 聚合（gather + sum）** 回去 → 得到 `input_grad`。（不熟 gather/sum 这两个词的话，先看[附录 A](#附录-a-gather--sum-到底是什么操作)。）

本文档专讲反向。

---

## 2. 一句话定义

> 对每个原始 token `i`，把它在 `permuted_output_grad` 里散落的 `topK` 份梯度抓回来求和，写到 `input_grad[i]`。

数学上：

```
input_grad[i, :] = Σ_{k=0..topK-1}  permuted_output_grad[ sorted_indices[i*topK + k], : ]
```

`sorted_indices[i*topK + k]` 告诉你「token i 的第 k 份梯度躺在 permuted 张量的哪一行」——它就是前向 permute 留下的「逆映射表」。

---

## 3. 接口一览

`MoeTokenPermuteGrad.__call__` 的真值签名：

```text
permuted_output_grad : [E,  H]   <-- upstream gradient w.r.t. permuted tensor
sorted_indices       : [T,  K]   <-- index from forward permute (viewed as 1D, length E)
                              -->
input_grad           : [T,  H]   <-- gradient w.r.t. original tokens
```

记号：
- `T = num_tokens`
- `K = topK`
- `H = hidden_size`
- `E = T * K`（permuted 张量行数）

---

## 4. 用一个迷你例子把过程画清楚

参数选小一点，方便画图：

```
T = 3         (3 original tokens: t0, t1, t2)
K = 2         (each token picks 2 out of all experts)
H = 4         (hidden size = 4)
experts = 4   (E0, E1, E2, E3)  <-- intentionally > K, to avoid confusion with K
E = T*K = 6   (permuted tensor has 6 rows)
```

路由表（每个 token 从 4 个专家里选 2 个）：

```
t0 → (E0, E2)
t1 → (E3, E1)
t2 → (E0, E3)
```

注意几个有意义的细节：每个专家收到的 token 数量并不均匀——E0 收 2 个（t0, t2）、E1 收 1 个（t1）、E2 收 1 个（t0）、E3 收 2 个（t1, t2）；这正是 MoE 路由的典型景象。

### 4.0 先把两个 shape 直觉立起来

**(a) `H` 是「每个 token 的向量维度」。**
例子里 `H=4` 表示每个 token 是个 4 维向量，比如 `t0 = [0.1, -0.3, 0.7, 0.2]`。真实模型里 `H` 通常是 4096、8192 这种大小，本文取 4 只为画图方便。

**(b) 梯度 shape 永远跟原张量一致。**
反向传播里有条铁律：对张量 `X` 求 `∂L/∂X` 得到的梯度，shape 跟 `X` 完全相同。所以前向产物 `permuted_tokens : [E, H]` 反向回传的梯度 `permuted_output_grad` 也是 `[E, H]`——每行还是 H 个数字，只是含义从「token 值」变成「该位置的梯度」。

把两个张量并排画出来对比：

```
   slot          permuted_tokens (forward)           permuted_output_grad (backward)
                 [E=6, H=4]                          [E=6, H=4]
                 ┌──────────────────────┐            ┌──────────────────────┐
   0  (t0->E0)   │ 0.1  -0.3   0.7  0.2 │            │ g0_0 g0_1 g0_2 g0_3  │  --> denote g0
   1  (t2->E0)   │ 0.9   0.6  -0.4  0.1 │            │ g1_0 g1_1 g1_2 g1_3  │  --> denote g1
   2  (t1->E1)   │ 0.5   0.4  -0.1  0.0 │            │ g2_0 g2_1 g2_2 g2_3  │  --> denote g2
   3  (t0->E2)   │ 0.1  -0.3   0.7  0.2 │ <-same t0  │ g3_0 g3_1 g3_2 g3_3  │  --> denote g3
   4  (t1->E3)   │ 0.5   0.4  -0.1  0.0 │ <-same t1  │ g4_0 g4_1 g4_2 g4_3  │  --> denote g4
   5  (t2->E3)   │ 0.9   0.6  -0.4  0.1 │ <-same t2  │ g5_0 g5_1 g5_2 g5_3  │  --> denote g5
                 └──────────────────────┘            └──────────────────────┘
```

注意 slot 0 和 slot 3 里的 token 值是相同的（都是 t0 被复制了一份），但它们各自的梯度 `g0` 和 `g3` 通常不同——因为 t0 经过两条不同的专家路径，回传的误差当然不一样。**把这两条路径上的梯度加起来**，才是 t0 应该承担的总梯度。这就是反向要做求和的物理动机。

下面后续小节里出现的 `g0 + g3` 其实是两个 4 维向量逐元素相加：

```
input_grad[t0] = g0 + g3
              = [g0_0+g3_0,  g0_1+g3_1,  g0_2+g3_2,  g0_3+g3_3]
```

最终 `input_grad` 也是 `[T, H] = [3, 4]`——和原始输入 `tokens` shape 对齐，每个原始 token 拿回 H 个数字的梯度。

### 4.1 前向 permute（先理清这步，反向才看得懂）

按专家分组排序后的 6 个 slot（同一个专家内部按 token id 升序）：

```
     slot:    0      1      2      3      4      5
            ┌──────┬──────┬──────┬──────┬──────┬──────┐
permuted_x: │  t0  │  t2  │  t1  │  t0  │  t1  │  t2  │
            └──────┴──────┴──────┴──────┴──────┴──────┘
             └─ E0 ──┘    └ E1 ┘ └ E2 ┘ └─── E3 ────┘
```

几个值得品的细节：

- **E1、E2 各只有 1 个 slot**：因为只有 1 个 token 选了这个专家。
- **E0 / E3 各有 2 个 slot**：内部按 token id 升序排，所以 E0 是 `t0, t2`、E3 是 `t1, t2`。
- **t1 选的是 (E3, E1)**：E3 在 E1 后面，但 t1 的「第 0 份」是按它**自己的路由顺序**落到 E3，去 slot 4；「第 1 份」去 E1，落 slot 2 ——所以同一个 token 的两份梯度在 permuted 张量里位置可以**前后倒挂**。

由此前向输出的 `sorted_indices`（行优先：第 i*K+k 项 = token i 的第 k 份去了哪个 slot）：

```
  i=t0 ┌── k=0 ──┐ ┌── k=1 ──┐
       │    0    │ │    3    │   <-- t0 -> (E0, E2)  =>  (slot 0, slot 3)
  i=t1 ├─────────┤ ├─────────┤
       │    4    │ │    2    │   <-- t1 -> (E3, E1)  =>  (slot 4, slot 2)
  i=t2 ├─────────┤ ├─────────┤
       │    1    │ │    5    │   <-- t2 -> (E0, E3)  =>  (slot 1, slot 5)
       └─────────┘ └─────────┘

sorted_indices (flat) = [0, 3, 4, 2, 1, 5]
```

### 4.2 反向 permute_grad：gather + reduce

上游回传给 permuted 张量的梯度 `permuted_output_grad`，6 行各是一个长度 4 的向量，简记 g0…g5：

```
permuted_output_grad  (E=6, H=4)

   row 0 :  g0 = [g0₀ g0₁ g0₂ g0₃]
   row 1 :  g1 = [g1₀ g1₁ g1₂ g1₃]
   row 2 :  g2 = [g2₀ g2₁ g2₂ g2₃]
   row 3 :  g3 = [g3₀ g3₁ g3₂ g3₃]
   row 4 :  g4 = [g4₀ g4₁ g4₂ g4₃]
   row 5 :  g5 = [g5₀ g5₁ g5₂ g5₃]
```

按 `sorted_indices` 把对应行抓出来求和：

```
                          sorted_indices              input_grad
                          (i*K+k -> slot)             (T=3, H=4)

 t0  --gather--> [ slot 0 ] -+                        ┌──────────────┐
                 [ slot 3 ] -+-- sum --------------►  │   g0 + g3    │  row 0
                                                      ├──────────────┤
 t1  --gather--> [ slot 4 ] -+                        │   g4 + g2    │  row 1
                 [ slot 2 ] -+-- sum --------------►  ├──────────────┤
                                                      │              │
 t2  --gather--> [ slot 1 ] -+                        │   g1 + g5    │  row 2
                 [ slot 5 ] -+-- sum --------------►  └──────────────┘
```

写成等式：

```
input_grad[t0] = g0 + g3
input_grad[t1] = g4 + g2
input_grad[t2] = g1 + g5
```

这就是 `permute_grad` 的全部数学语义。

---

## 5. 与 GPU/AI Core 上的并行结构对应

NPU kernel 在 `K.Kernel(actual_cores, is_npu=True)` 下用两个轴并行：

```
parallel axes:

   cid (core id)        --> splits T tokens across cores
                            each core takes tokens_per_core tokens

   vid (vector pipe id) --> splits H into two halves
                            vid=0 -> left  HALF_H = TILE_H/2 columns
                            vid=1 -> right HALF_H = TILE_H/2 columns

per-core view (one core processing its own tokens):

           |<-- HALF_H -->|<-- HALF_H -->|
           ┌──────────────┬──────────────┐
   token 0 │   vid = 0    │   vid = 1    │
   token 1 │              │              │
    ...    │              │              │
           └──────────────┴──────────────┘
```

- `cid`（core id）：把 `T` 个 token 大致平均切到 `actual_cores` 个核，每核负责 `tokens_per_core` 个。
- `vid`（vector id 0/1）：把 `H` 维拆成两半 `HALF_H = TILE_H/2`，两个 vector pipe 各做一半，单核内部仍是「同一行」的两个 H 子块同时算。

每个核内的循环骨架（取 cast 路径，源码 [moe_token_permute_grad.py:115-149](moe_token_permute_grad.py#L115)）：

```
for ti in 0..BATCH_T-1:                # token within current batch
    acc_buf <- 0
    for lane in 0..topK-1:             # accumulate K rows
        src = idx_ub[ti*K + lane]      # pull slot from sorted_indices
        row_buf <- perm_grad_gm[src, h_off : h_off+HALF_H]   # gather
        row_f32 <- cast(row_buf)       # fp16/bf16 -> fp32
        acc_buf += row_f32             # reduce
    out_buf <- cast(acc_buf)           # fp32 -> fp16/bf16
    input_grad_gm[ti, h_off : h_off+HALF_H] <- out_buf       # write
```

- 累加用 fp32 是为了避免低精度 dtype 反复加法掉精度；只有最后写出时再 cast 回原 dtype。
- `BATCH_T` 是把 `tokens_per_core` 再切成的小批次；为了一次性把 `BATCH_T*K` 个索引拷进 UB（unified buffer）。

---

## 6. 三个 kernel 变体（自动选路）

`_compile_gather_reduce` 会按 `dtype/topK/H` 选最合适的 kernel：

```
            ┌─────────────────────────────────┐
            │  is_cast_path = (dtype != fp32) │
            │  single_htile = (H == TILE_H)   │
            │  half_aligned = (HALF_H aligned)│
            └─────────────────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────────────┐
   │  pipelined_eligible = all three conditions hold    │
   └────────────────────────────────────────────────────┘
        │                       │                       │
        │ topK == 8             │ topK <= 8 (not 8)     │ otherwise
        ▼                       ▼                       ▼
 group_pipelined           lane_pipelined           plain
 (stages = 2,             (stages = 8,             (cast or nocast,
  K-group pipeline)        per-lane pipeline)       barrier_all sync)
```

三种变体的差异本质都在「MTE2 搬运 / V 计算 / MTE3 写出」这三个 pipe 之间怎么开窗口、用多少 stages、用 set_flag/wait_flag 还是 barrier_all——**数学语义完全相同**，区别只在重叠访存和计算的程度。

---

## 7. Python 包装层做的额外事情

`MoeTokenPermuteGrad.__call__` 在调 kernel 前后还做了几件杂事，方便理解为何接口和 kernel 签名对得上：

```
caller    permuted_output_grad: [r, c]      sorted_indices: [T, K]
                       │                              │
                       │                              │  .view(-1) -> [E]
                       │                              │  .to(int32)
                       │                              ▼
                       │                     pad to [padded_E]
                       │                     (padded_E = cores * tokens_per_core * K)
                       ▼                              │
            if r < E or c < H_compile:                │
                pad to [E, H_compile]                 │
                       │                              │
                       └───────────┬──────────────────┘
                                   ▼
                              kernel(...)
                                   │
                                   ▼
                         input_grad: [T, H_compile]
                                   │
                         if hidden_size != H_compile:
                                   │  slice off padded columns
                                   ▼
                         input_grad: [T, hidden_size]
```

为何要 pad：
- **`H` 方向 pad**：fp16/bf16 下最少要 32B 对齐（`min_compile_h = 32`），fp32 要 64B（`min_compile_h = 64`）。`hidden_size = 16` 也能跑，是因为 kernel 按 32 元素编译，输出再切回去。
- **`E` 方向 pad**：因为按 `actual_cores * tokens_per_core * K` 这个对齐量编译，多余的位置填零参与累加是安全的。

---

## 8. 一句话总结

> `permute_grad` = 把 `permuted_output_grad` 当成乱序的「topK 份梯度池」，按 `sorted_indices` 给每个原始 token **拉回 K 份并求和**，得到 `input_grad`。
>
> 它和前向 `permute` 互为反向：前向是「**复制 + 散开**」，反向是「**收回 + 求和**」。

---

## 附录 A：gather / sum 到底是什么操作？

### A.1 gather（按索引挑行）

「gather」= **按一组下标去张量里把对应的那些行/元素挑出来**，挑出的顺序由下标决定，不一定连续。

可以类比 Python：

```python
rows = [g0, g1, g2, g3, g4, g5]   # the 6 rows of permuted_output_grad
idx  = [0, 3]                     # indices to pick
picked = [rows[i] for i in idx]   # -> [g0, g3]   <-- this is gather
```

放回 t0 的反向流程上：

```
            permuted_output_grad             sorted_indices[t0] = [0, 3]
            ┌─────────────────┐
   slot 0   │  g0  ━━━━━━━━━━━━━━━━━━━━━━━━┓ <-- pick by idx = 0
   slot 1   │  g1                          ┃
   slot 2   │  g2                          ┃     gather output:
   slot 3   │  g3  ━━━━━━━━━━━━━━━━━━━┓    ┃     ┌─────┐
   slot 4   │  g4                     ┃    ┃     │ g0  │
   slot 5   │  g5                     ┃    ┃     │ g3  │
            └─────────────────┘       ┃    ┃     └─────┘
                                      ┃    ┗━━━━━━━━┓
                                      ┗━━━━━━━━━━━━━┛  <-- pick by idx = 3
```

特点：

- 输入是「索引数组」，不是「连续区间」 → 访存是**随机的**，硬件上贵。
- 输出行数 = 索引个数（这里 K=2）。

### A.2 sum / reduce-sum（逐元素相加）

「sum」（也叫 reduce-sum）= **把若干个相同 shape 的向量逐位置相加，压成一个**。维度上少了一根「行」轴。

```
K rows from gather                       sum result (1 row)

  g0 = [g0_0  g0_1  g0_2  g0_3]
                   +                     elementwise add  ┌──────────────────────────────────────┐
  g3 = [g3_0  g3_1  g3_2  g3_3]    -------------------->  │ g0_0+g3_0  g0_1+g3_1  g0_2+g3_2  g0_3+g3_3 │
                                                          └──────────────────────────────────────┘
                                                                       ( = input_grad[t0] )
```

类比 Python / Numpy：

```python
input_grad_t0 = picked[0] + picked[1]            # = g0 + g3
# or in reduce style:
input_grad_t0 = sum(picked)                      # add up the K vectors
```

特点：

- 输入是 K 个 H 维向量，输出是 1 个 H 维向量 → 把 K 这根轴「reduce 掉」了。
- 在 kernel 里就是循环里的 `acc_buf += row_f32`，K 次累加。

### A.3 合起来：gather + sum 的语义

整个 `permute_grad` 对每个 token i 做的事就两步：

```
        ┌──────────────┐           ┌────────────────┐
  ────► │    gather    │  K rows   │      sum       │  1 row
        │ pick K rows  │---------->│ elementwise    │----------> input_grad[i]
        │ by index     │           │ add (reduce K) │
        └──────────────┘           └────────────────┘
```

为什么要先 gather 再 sum、而不是只用其中一个：

- **只 gather 不 sum**：会得到 `[T, K, H]` 的张量（每个 token 仍然有 K 份独立梯度），但下游需要的是和 `tokens` 同 shape 的 `[T, H]`。
- **只 sum 不 gather**：得对**全部 E 行**求和，没法区分哪些行属于 t0、哪些属于 t1，物理意义不对。

所以两步缺一不可：**gather 解决「找到属于 t_i 的 K 行」，sum 解决「把这 K 份梯度合并成 t_i 应得的那一份」。**
