# `MoeTokenPermute` 功能讲解

> 配套代码：[moe_token_permute.py](moe_token_permute.py)
> 配套反向：[moe_token_permute_grad.py](moe_token_permute_grad.py) ・ [permute_grad_explained.md](permute_grad_explained.md)

## 1. 它在 MoE 里属于哪一步？

```
   forward                                backward
  ┌─────────┐                            ┌──────────────┐
  │ permute │  -->  expert compute  -->  │ permute_grad │
  └─────────┘                            └──────────────┘
```

- **前向 `permute`（本文档）**：把每个 token **复制 `topK` 份**，按目标专家排好序 → 得到 `permuted_tokens` 和「每份去哪儿」的索引表 `sorted_indices`。
- **反向 `permute_grad`**：把 `permuted_tokens` 上回传的梯度按 `sorted_indices` 收回求和 → 得到 `input_grad`。

> 反向那条线在 [permute_grad_explained.md](permute_grad_explained.md) 里讲了；本文档的输出 `sorted_indices` 就是反向需要的那张「逆映射表」。

---

## 2. 一句话定义

> 对每个原始 token `i`，按 `indices[i, k]` 给出的 `topK` 个目标专家，**把这个 token 复制到 `permuted_tokens` 里这 `topK` 个专家组对应的位置**；同时记下「token i 的第 k 份去到了哪一行」，存成 `sorted_indices`。

数学上，permute 输出由两条不变量定义：

```
(1)  permuted_tokens[ sorted_indices[i*K + k], : ]  ==  tokens[i, :]
(2)  permuted_tokens is grouped by expert id ascending,
     and within one expert, by token id ascending
```

这两条是反向 `permute_grad` 能成立的契约——它会用 `sorted_indices[i*K+k]` 到 `permuted_tokens` 的梯度里把 token i 的 K 份梯度都摸回来。

---

## 3. 接口一览

`MoeTokenPermute.__call__` 的签名：

```text
tokens             : [T, H]            <-- the original tokens
indices            : [T, K] or [T*K]   <-- expert id of each (token, k) pair
                              -->
permuted_tokens    : [out_len, H]      <-- tokens grouped by expert
sorted_indices     : [E]               <-- where each (i, k) landed in permuted_tokens
```

记号：

- `T = num_tokens`, `K = topK`, `H = hidden_size`, `E = T*K`
- `out_len = E`（默认）或 `num_out_tokens`（截断模式：只保留前 `num_out_tokens` 行）

---

## 4. 用一个迷你例子把整个流程画清楚

参数和 grad 文档用同一组：

```
T = 3         (3 original tokens: t0, t1, t2)
K = 2         (each token picks 2 of all experts)
H = 4         (hidden size = 4)
experts = 4   (E0, E1, E2, E3)
E = T*K = 6   (permuted tensor has 6 rows)
```

为了把**多核协作**这部分讲透，本节用 **2 个核** 跑这 3 个 token：

```
cid = 0  ->  handles t0, t1
cid = 1  ->  handles t2  (and 1 padding slot, since 3 tokens / 2 cores rounds up)
tokens_per_core = ceil(3/2) = 2
chunk_size      = tokens_per_core * K = 4   (each core's slice of indices array)
```

### 4.0 输入是什么

**tokens** `[T=3, H=4]`：

```
       ┌──────────────────────┐
  t0   │ 0.1  -0.3   0.7  0.2 │
  t1   │ 0.5   0.4  -0.1  0.0 │
  t2   │ 0.9   0.6  -0.4  0.1 │
       └──────────────────────┘
```

**indices** `[T=3, K=2]`：每个 token 选哪 2 个专家：

```
        k=0  k=1
       ┌────┬────┐
  t0   │ E0 │ E2 │
  t1   │ E3 │ E1 │
  t2   │ E0 │ E3 │
       └────┴────┘
```

把 indices 摊平成 1D（`view(-1)`），按 `(token, k)` 顺序排列——这是 kernel 实际看到的样子：

```
flat indices:  [E0, E2, E3, E1, E0, E3]
slot:          (t0,0)(t0,1)(t1,0)(t1,1)(t2,0)(t2,1)
```

切给两个核：

```
              cid=0 chunk            cid=1 chunk + padding
              ┌──────────────┐      ┌──────────────┐
              │ E0 E2 E3 E1  │      │ E0 E3  ?  ?  │   ? = padding (out-of-range, kernel skips)
              └──────────────┘      └──────────────┘
              my_start = 0          my_start = 4
```

### 4.1 Phase 1 — 每个核独立数自己的「直方图」

每个核扫自己 chunk 里的 `chunk_size` 个 expert id，统计每个专家出现了几次。这步**完全无依赖**，纯 SIMD：

```
                cid=0 chunk: [E0, E2, E3, E1]
                ┌──────────────────────────┐
hist_ub  cid=0  │ E0:1  E1:1  E2:1  E3:1   │
                └──────────────────────────┘

                cid=1 chunk: [E0, E3]
                (last 2 padding slots are skipped by `if my_start+i < E`)
                ┌──────────────────────────┐
hist_ub  cid=1  │ E0:1  E1:0  E2:0  E3:1   │
                └──────────────────────────┘
```

代码里就是简单的累加：

```python
for i in T.Pipelined(chunk_size):
    if my_start + i < E:                # skip padding slots
        expert = idx_ub[0, i]
        hist_ub[0, expert] += 1
```

这一步算完后，每个核**只知道自己**那段贡献了什么——还不够，因为后面要算「我应该往 perm_out 的哪个位置写」需要知道**其他核的情况**。

### 4.2 Phase 2 — 把直方图扔到 workspace，跨核共享

每个核把自己的直方图写到一段共享的 GM workspace（位置由 `cid` 决定，互不重叠），然后做一次**全核同步**（`T.sync_all()`），保证所有核都写完之后大家再继续：

```
   workspace_gm  (in GM, shared by all cores)
   ┌────────────────────────────────────┬────────────────────────────────────┐
   │  cid=0 histogram                   │  cid=1 histogram                   │
   │  E0:1  E1:1  E2:1  E3:1            │  E0:1  E1:0  E2:0  E3:1            │
   └────────────────────────────────────┴────────────────────────────────────┘
    offset = cid*num_experts = 0..3       offset = 4..7
                       ▲                                  ▲
                cid=0 writes here             cid=1 writes here
                       └─────── T.sync_all() ─────────┘
                (wait for both sides to finish, then everyone reads)
```

`T.sync_all()` 之后，每个核**读整张 workspace**到自己的 UB（`ws_ub`）。从此每个核都能看到所有核的直方图，可以独立算自己的 offset 了。

### 4.3 Phase 3 — 算 offsets + 第二次扫 chunk → 得到 sio

每个核要回答一个问题：**「我手里这条 chunk 里第 i 个 (token, k)，最终要写到 `permuted_tokens` 的第几行？」**

#### Step A — 算专家的全局起始位置 + 我自己之前的核占了多少

按专家 id 升序遍历，对每个专家 `e`：

```
running   = sum over all (prior expert, all cores)   = global start row of expert e
cpre[e]   = sum over (cores c < cid, expert e)       = e-slots already taken by my predecessors
offset[e] = running + cpre[e]
            ──┬──   ──┬──
              │       └─ e-slots taken by cores before me
              └─ global start row of expert e in perm_out
```

代入数字（每行：全局 total / running / cpre / offset）：

```
                    cid=0  (cpre always 0)            cid=1  (cpre = cid=0 histogram)
             ┌──────────────────────────────┐  ┌──────────────────────────────┐
   E0  total=2, running=0, cpre=0, offset=0    total=2, running=0, cpre=1, offset=1
   E1  total=1, running=2, cpre=0, offset=2    total=1, running=2, cpre=1, offset=3
   E2  total=1, running=3, cpre=0, offset=3    total=1, running=3, cpre=1, offset=4
   E3  total=2, running=4, cpre=0, offset=4    total=2, running=4, cpre=1, offset=5
             └──────────────────────────────┘  └──────────────────────────────┘
```

这告诉每个核「在 perm_out 里写专家 e 的时候，应该从第几行开始填」。

#### Step B — 第二次扫 chunk，逐个分配 wp

每个核再扫一遍**自己的 chunk**，每碰到一个 expert id 就给它分配一个 perm_out 行号 `wp`，然后把 `wp` 记到 `sio_chunk`：

```
for i in chunk:
    e = idx_ub[i]
    wp = offset[e] + counters[e]   # counters[e]: how many e-slots this core has already filled
    counters[e] += 1
    sio_chunk[i] = wp              # destination of this (token, k)
```

**cid=0** 的扫描（counters 从全 0 开始）：

```
   i  expert  wp = offset + counter  counter after  sio_chunk[i]
   ──────────────────────────────────────────────────────────────
   0  E0      0 + 0 = 0              E0:1           0
   1  E2      3 + 0 = 3              E2:1           3
   2  E3      4 + 0 = 4              E3:1           4
   3  E1      2 + 0 = 2              E1:1           2
                                                    ────────────
                                          cid=0 sio_chunk = [0, 3, 4, 2]
```

**cid=1** 的扫描：

```
   i  expert  wp = offset + counter  counter after  sio_chunk[i]
   ──────────────────────────────────────────────────────────────
   0  E0      1 + 0 = 1              E0:1           1
   1  E3      5 + 0 = 5              E3:1           5
                                                    ────────────
                                          cid=1 sio_chunk = [1, 5, ?, ?]
                                                          (?? = padding, unused later)
```

#### Step C — 拼出全局 sorted_indices

每个核把自己的 `sio_chunk` 写到 `sio_out_gm` 的对应段（位置 `cid * chunk_size`）。最终：

```
sio_out_gm  (concat of all cores' sio_chunks):

   cid=0 segment (slots 0..3)   cid=1 segment (slots 4..7)
   ┌──────────────────────┐    ┌──────────────────────┐
   │ 0   3   4   2        │    │ 1   5   ?   ?        │
   └──────────────────────┘    └──────────────────────┘

trim to the first E=6 entries:

  sorted_indices = [0, 3, 4, 2, 1, 5]
                    ▲   ▲   ▲   ▲   ▲   ▲
                    │   │   │   │   │   └── (t2, k=1) -> slot 5
                    │   │   │   │   └────── (t2, k=0) -> slot 1
                    │   │   │   └────────── (t1, k=1) -> slot 2
                    │   │   └────────────── (t1, k=0) -> slot 4
                    │   └────────────────── (t0, k=1) -> slot 3
                    └────────────────────── (t0, k=0) -> slot 0
```

> 这个 `[0, 3, 4, 2, 1, 5]` 就是 [permute_grad_explained.md §4.1](permute_grad_explained.md) 里反向用到的 `sorted_indices`——前向把它生成出来，反向拿它把梯度收回去。

### 4.4 Phase 4 — 把 token 数据真正写到 perm_out

到这一步 `sio_chunk` 已经告诉每个核「我手里每个 (token, k) 的目的地」。剩下就是从 `tokens_gm` 把数据搬过去。每个核处理自己负责的 `tokens_per_core` 个 token，每个 token 复制 K 份分别写到 K 个目的地：

```
for ti in 0 .. tokens_per_core-1:
    cur_src = cid * tokens_per_core + ti      # global token id
    row_buf <- tokens_gm[cur_src, :]          # load one row from GM into UB
    for k in 0 .. K-1:
        wp = sio_chunk[ti * K + k]            # destination of this k-th copy
        if wp < out_len:
            perm_out_gm[wp, :] <- row_buf     # scatter
```

代入我们的例子（cid=0 处理 t0 / t1，cid=1 处理 t2）：

```
cid=0:
   t0 -> row_buf -> perm_out[0]   (sio_chunk[0]=0)
   t0 -> row_buf -> perm_out[3]   (sio_chunk[1]=3)
   t1 -> row_buf -> perm_out[4]   (sio_chunk[2]=4)
   t1 -> row_buf -> perm_out[2]   (sio_chunk[3]=2)

cid=1:
   t2 -> row_buf -> perm_out[1]   (sio_chunk[0]=1)
   t2 -> row_buf -> perm_out[5]   (sio_chunk[1]=5)
```

最终 `permuted_tokens`：

```
   slot          permuted_tokens [E=6, H=4]
                 ┌──────────────────────┐
   0  (t0->E0)   │ 0.1  -0.3   0.7  0.2 │   <-- copy of t0
   1  (t2->E0)   │ 0.9   0.6  -0.4  0.1 │   <-- copy of t2
   2  (t1->E1)   │ 0.5   0.4  -0.1  0.0 │   <-- copy of t1
   3  (t0->E2)   │ 0.1  -0.3   0.7  0.2 │   <-- copy of t0
   4  (t1->E3)   │ 0.5   0.4  -0.1  0.0 │   <-- copy of t1
   5  (t2->E3)   │ 0.9   0.6  -0.4  0.1 │   <-- copy of t2
                 └──────────────────────┘
                  └─ E0 ─┘ E1 │ E2 │ └─ E3 ─┘
```

跟 grad 文档里画的那张完全一致——闭环成立。

---

## 5. 与 NPU 上的并行结构对应

kernel 用 `T.Kernel(actual_cores, is_npu=True)` 启动 `actual_cores` 个核，循环变量 `(cid, vid)`：

- **`cid` (core id)**：把 `T` 个 token **大致平均切到 `actual_cores` 个核**，每核负责 `tokens_per_core` 个。所有 phase 1~4 都按 `cid` 切分。
- **`vid` (vector pipe id, 0/1)**：把 `H` 维**拆成两半**（`HALF_H = TILE_H/2`），两个 vector pipe 各做一半，单核内部「同一行」的左右两个 H 子块**同时**搬运/写出。

```
parallelism:

  cid axis (cores)
  ──────────────────►
   each core handles tokens_per_core tokens; every phase splits along this axis

  vid axis (vector pipe 0/1)
  ──────────────────────►
   each core splits H into two halves; both pipes load/store in parallel
```

per-core 看一个 token 的写出过程：

```
                |<-- HALF_H -->|<-- HALF_H -->|
                ┌──────────────┬──────────────┐
   tokens_gm    │   vid = 0    │   vid = 1    │   (one row of H)
                └──────────────┴──────────────┘
                       │              │
                       ▼              ▼   (mte2 load + mte3 scatter to K destinations)
                ┌──────────────┬──────────────┐
   perm_out_gm  │   vid = 0    │   vid = 1    │   (K writes, one per destination)
                └──────────────┴──────────────┘
```

---

## 6. Phase 4 的双缓冲流水线

Phase 4 是整个 kernel 里**唯一搬大量数据**的阶段（前 3 个 phase 只搬 indices 和直方图，量很小），所以重点流水。结构很简单：`stages = 2` 双缓冲，让「拷贝下一个 token 进 UB」和「把当前 token 写到 K 个目的地」重叠：

```
time ───►
mte2 :  load t0  ──   load t1  ──   load t2  ──   ...    <-- prefetch next token
v    :          ── (idle, only orchestrates)
mte3 :          ──   write t0xK ──   write t1xK ──   ... <-- scatter to K slots
                     ↑─────────────↑
                     overlaps with the next mte2 load
```

代码骨架（[moe_token_permute.py:155-194](moe_token_permute.py#L155)）：

```
for i in 0 .. total_iters-1:
    cur = i % stages           # row_buf slot to consume (0 or 1)
    nxt = (i + 1) % stages     # row_buf slot to prefetch into

    if has_next:
        wait mte3 -> mte2 nxt                 # wait until the next slot is free
        row_buf[nxt] <- tokens_gm[next_src, next_h_off]   # prefetch
        signal mte2 -> v / mte3 nxt

    wait mte2 -> v cur                        # wait until current slot is loaded
    wait mte2 -> mte3 cur                     # mte3 also needs the load to be done
    for k in 0..K-1:                          # K writes, each to its own destination
        wp = sio_chunk[cur_base + k]
        if wp < out_len:                      # truncation mode skips out-of-range wp
            perm_out_gm[wp, :] <- row_buf[cur, :]

    signal v -> mte3 cur
    wait v -> mte3 cur
    signal mte3 -> mte2 cur                   # current slot may be reused next round
```

注意：**这里的 `K` 和反向不一样**。反向 `permute_grad` 是 K 次「读 + 累加」（gather + sum），前向是 K 次「写」（scatter）——一份输入数据复制成 K 份输出。所以前向不需要 `acc_buf` / fp32 累加，整段都是无须 cast 的纯搬运。

`total_iters = tokens_per_core * n_htiles`：每个核要把自己的 tokens_per_core 个 token 都过一遍，每个 token 在 H 方向分 `n_htiles` 个 tile 处理。多数情况下 `n_htiles = 1`（即 `TILE_H == hidden_size`，整行一次搬完）。

---

## 7. Python 包装层做的额外事情

`MoeTokenPermute.__call__` 上下做的胶水活：

```
caller       tokens: [T, H]                      indices: [T, K] or flat
                  │                                   │
                  │                                   │ (kernel takes 1D of length padded_E)
                  │                                   ▼
                  │                          torch.zeros [padded_E]
                  │                          indices_padded[:E] = indices
                  │                                   │
                  ▼                                   │
        pad_last_dim to H_compile                     │
        (fp16/bf16 -> 32, fp32 -> 64)                 │
                  │                                   │
                  └──────────────┬────────────────────┘
                                 ▼
                          fused_func(...)
                                 │
                                 ▼
              perm_out: [out_len, H_compile],  sio_padded: [1, padded_E]
                                 │
                                 │  sio = sio_padded.squeeze(0)[:E]      # drop padding
                                 │  if H_compile != H:
                                 │      perm_out = perm_out[:, :H]       # slice back to real H
                                 ▼
              perm_out: [out_len, H],          sio: [E]
```

- **H 方向 pad（与反向同款）**：fp16/bf16 至少 32 元素（`min_compile_h = 32`），fp32 至少 16（`min_compile_h = 64` 字节）；不够就 pad，输出再裁回去。
- **indices 方向 pad**：按 `padded_E = actual_cores * tokens_per_core * K` 对齐（保证每个核拿到等长的 chunk），多余位置填 0；kernel 里通过 `if my_start+i < E:` 跳过 padding，不会污染直方图。

---

## 8. 一句话总结

> `permute` = 把每个 token **复制 K 份**，按目标专家排好序，写到 `permuted_tokens`；同时记录每份的目的地，存成 `sorted_indices` 留给反向用。
>
> 它和反向 `permute_grad` 互为对称：前向是「**复制 + 散开**」，反向是「**收回 + 求和**」。

---

## 附录 A：为什么前向要走「直方图 → workspace → offsets → 第二次扫」这么大圈？

直觉做法是「单核串行」：维护一个全局 `pos[E]` 数组，每碰到一个 (token, k) 就追加到对应专家组末尾。问题是这种做法**完全不能并行**——所有核要争抢 `pos[E]` 的写指针。

NPU 上 24 核同时跑，必须**先**让每个核独立干一些活、**再**做一次小同步、**再**让每个核独立干完剩下的事。本 kernel 的 4 个 phase 正是这个套路：

| Phase | 各核做什么 | 是否需要其他核的信息 |
|---|---|---|
| 1 (直方图) | 数自己 chunk 内每个专家几次 | 不需要 ✓ 完全并行 |
| 2 (workspace 同步) | 写一段 GM + `sync_all` | — 这是同步本身 |
| 3 (offsets + sio) | 读全局直方图 → 算 offset → 第二次扫 | 只读, 各核独立 ✓ |
| 4 (拷贝) | 把 token 数据写到 perm_out | sio 已经告诉每个核去哪写, 各核独立 ✓ |

只有 phase 2 那一次 `sync_all` 是真正的同步点，其余阶段都是各核闷头干。这种「**两次扫描 + 一次同步**」的范式叫 **two-pass histogram scatter**，是经典的 GPU/NPU 并行 permute 套路。直方图本质是在告诉每个核「我跟其他核之间的相对位置在哪」，offset 计算就是把这个相对位置具体化成 perm_out 行号。

如果只有 1 个核，phase 1~3 看上去很冗余（直方图就是给自己看的），但 kernel 不分单核多核都走同一套——单核只是 `cpre` 永远 0、`acc = hist` 这种特例。
