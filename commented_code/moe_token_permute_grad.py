# =============================================================================
# moe_token_permute_grad.py  (逐行注释版)
# =============================================================================
# 本文件是 MoE (Mixture-of-Experts) 路由 permute 算子的 *反向* 实现。
#
# 配套前向 moe_token_permute.py:
#   forward 输入 : tokens [T, H], indices [T, K]
#   forward 输出 : permuted_tokens [E, H], sorted_indices (1D, 长度 E)
#                  其中 E = T*K, 每个 token 被复制 K 份并按目标专家排好序
#
# 本文件 backward:
#   backward 输入 : permuted_output_grad [E, H]   (上游对 permuted_tokens 的梯度)
#                   sorted_indices       [T, K]   (前向输出的逆映射表)
#   backward 输出 : input_grad           [T, H]   (对原始 tokens 的梯度)
#
# 数学语义 (gather + sum):
#   input_grad[i, :] = Σ_{k=0..K-1}  permuted_output_grad[ sorted_indices[i*K+k], : ]
#
# 详细图示参见同目录上一级的 permute_grad_explained.md。
# =============================================================================

import math  # 用 math.ceil 算 tokens_per_core / n_batches 等向上取整
import tilelang  # tilelang DSL 入口模块, 提供 jit 装饰器与 PassConfigKey
import tilelang.language as T  # tilelang 张量原语, 习惯把 language 起别名为 T
import torch  # PyTorch 主体, 用于宿主侧 (host) 张量操作: pad / view / cast 等
import torch_npu  # 华为 NPU 的 PyTorch 扩展, 提供 NPU 设备及 npu_moe_token_permute 算子

# 从 moe_token_utils 导入共享辅助函数 (跨多个 moe_token_*.py 复用):
#   - is_fp32_dtype: 判断 dtype 字符串是不是 fp32
# try/except 是为了同时支持 "package 模式" (相对导入 .moe_token_utils) 和
# "脚本模式" (直接 python moe_token_permute_grad.py, 走绝对导入)
try:
    from .moe_token_utils import is_fp32_dtype
except ImportError:
    from moe_token_utils import is_fp32_dtype


# -----------------------------------------------------------------------------
# 编译期 Pass 配置
# -----------------------------------------------------------------------------
# tilelang 在 jit 时需要指定一组 pass 行为. 这里准备两套:
#
#   PASS_CONFIGS         —— "普通"  路径 (AUTO_SYNC=True ): 编译器自动插同步指令
#   PASS_CONFIGS_EXPERT  —— "专家"  路径 (AUTO_SYNC=False): 由 kernel 手动 set/wait flag
#
# 共同选项:
#   AUTO_CV_COMBINE  把相邻的 control + vector 指令合并, 减少指令开销
#   MEMORY_PLANNING  启用 UB (Unified Buffer) 的内存复用规划

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # 让编译器替你管同步, 简单但保守
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

PASS_CONFIGS_EXPERT = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,  # 关闭自动同步, 自己写流水线
}

# 计算用 dtype: fp32 是为了 K 次累加不掉精度
# (例子: bf16 加 8 个相同向量, 末位会被反复抹掉, fp32 中间累加再 cast 回 bf16 更稳)
CAL_DTYPE = "float32"

# tile.cast 时的 round 模式:
#   CAST_NONE  低->高精度 不需要 round (例如 bf16 -> fp32 全精度保留)
#   CAST_RINT  高->低精度 用 "round to nearest even" 防止系统性偏置
CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"

# 注: 早期版本本文件曾有 _is_fp32() 和 _pad_last_dim() 两个本地 helper,
# 现在它们已被抽到 moe_token_utils.py 里改名为 is_fp32_dtype / pad_last_dim,
# 由本文件顶部的 try/except 导入. 抽出来的动机是 unpermute / unpermute_grad
# 等姊妹文件也用同样的逻辑, 集中维护比四份各自抄一遍更稳.


# =============================================================================
# Kernel 变体 1: cast + group-pipelined (topK == 8 专用)
# =============================================================================
# 流水结构: stages = 2, 一次按 "topK 个 lane 一组" 推进流水
#   - 拷贝下一组的 8 个 lane (mte2)  和  累加当前组 (v)  重叠
#   - 每个 token 一次写出 (mte3)
# 适用条件: dtype != fp32, hidden_size == TILE_H, topK == 8
# =============================================================================


def _build_gather_reduce_kernel_cast_group_pipelined(
    num_tokens,  # T: 原始 token 总数
    topK,  # K: 每个 token 选 K 个专家 (这里固定 == 8)
    hidden_size,  # H: 每个 token 的向量维度
    E,  # E = T * K, permuted 张量行数
    padded_E,  # 编译期对齐后的 E, 用于 sorted_indices 形状
    actual_cores,  # 实际使用的核数 (cid 维)
    tokens_per_core,  # 每个核负责的 token 数 (向上取整)
    TILE_H,  # 单核内一次处理多大一段 H (这里 == hidden_size)
    HALF_H,  # H 维再切两半给 vid 0/1 用, HALF_H = TILE_H / 2
    BATCH_T,  # 一个 batch 内处理几个 token (一次拷的索引数 = BATCH_T*K)
    n_batches,  # 一个核需要跑几个 batch
    dtype,  # token / 梯度的 dtype (float16 / bfloat16)
    idx_dtype,  # 索引的 dtype (通常 int32)
):
    stages = 2  # 双 buffer: 正在算 stage A 时, 把 stage B 的数据从 GM 搬到 UB

    # 这个 kernel 是被严格特化的 —— 编译期就 assert 死参数, 跑错配置直接报错而不是悄悄出问题
    assert topK == 8, "group-pipelined kernel is specialized for topK == 8"
    assert dtype != CAL_DTYPE, "group-pipelined kernel is for non-fp32 dtypes"
    assert hidden_size == TILE_H, "group-pipelined kernel assumes single h-tile"
    assert HALF_H * 2 == TILE_H, "HALF_H must be TILE_H/2"

    # @tilelang.jit 装饰内层 _build, 触发 JIT 编译
    #   out_idx=[2]    第 3 个参数 (input_grad_gm) 是输出 (其余是输入)
    #   pass_configs   用上面定义的 EXPERT 配置 (手动同步, 流水线 kernel)
    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS_EXPERT)
    def _build(
        # 把所有参数当作编译期常量传进去, 这样 tilelang 会针对具体形状特化 kernel
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        stages,
    ):
        # ------------ NPU 上的实际 kernel 函数 ------------
        @T.prim_func
        def moe_token_permute_grad(
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),  # 输入: 上游梯度
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),  # 输入: 索引 (扩成 2D 是为了对齐)
            input_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),  # 输出: 原 token 梯度
        ):
            # T.Kernel(actual_cores, is_npu=True) 启动 actual_cores 个核并行
            # (cid, vid): cid 是 core id, vid 是 vector pipe id (NPU 每核 2 个 vector pipe)
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                # ---- 在 UB (Unified Buffer) 上分配若干 buffer ----
                # 索引 buffer: 一次性把 BATCH_T*K 个 sorted_indices 拷进来, 后面查表用
                idx_ub = T.alloc_ub([1, BATCH_T * topK], idx_dtype)

                # 数据 buffer (双 stage):
                #   row_buf 总共 stages*topK 行, 每 topK 行是一个 stage 的数据
                #   一个 stage 在被 V 计算消费时, 另一个 stage 可以并行从 GM 搬数 (mte2)
                row_buf = T.alloc_ub([stages * topK, HALF_H], dtype)

                # 计算用临时 buffer (1 行宽, 因为每次只算 1 个 lane):
                row_tmp = T.alloc_ub([1, HALF_H], dtype)  # 从 row_buf 单独拎出 1 行
                row_f32 = T.alloc_ub([1, HALF_H], CAL_DTYPE)  # cast 后的 fp32 版本
                acc_buf = T.alloc_ub([1, HALF_H], CAL_DTYPE)  # K 次累加的 fp32 累加器
                out_buf = T.alloc_ub([1, HALF_H], dtype)  # 写回前 cast 回低精度

                # 当前 vid 负责的 H 子块在全局 H 维上的偏移
                #   vid=0 -> h_off=0       (前半段)
                #   vid=1 -> h_off=HALF_H  (后半段)
                h_off = vid * HALF_H

                # T.Scope("V") 让里面的 ops 默认绑定到 V (vector) 流水
                # (mte2/mte3 是搬运流水, 通过 set_flag/wait_flag 与 v 互相同步)
                with T.Scope("V"):
                    # 外层循环: 一个核要跑 n_batches 个 batch
                    for batch_id in T.serial(n_batches):
                        # 当前 batch 在全局 token 序列里的起始位置
                        batch_base = cid * tokens_per_core + batch_id * BATCH_T

                        # ---- (1) 把当前 batch 的 BATCH_T*K 个索引一次性拷进 idx_ub ----
                        T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                        # 同步: 通知 v 流水 "mte2 已经把索引搬完了, 你可以读 idx_ub 了"
                        # 后面立刻 wait 是为了保证下面用 idx_ub 的代码能看到正确数据
                        T.set_flag("mte2", "v", 10)
                        T.wait_flag("mte2", "v", 10)

                        # 初始化几个 v->mte2 的 flag, 表示 "v 已经放手, mte2 可以覆盖 row_buf 了"
                        # 这些 flag 在每个 batch 开头要重新拉满, 才能让首批拷贝合法启动
                        T.set_flag("v", "mte2", 0)  # stage 0 的 row_buf 可以被 mte2 覆盖
                        T.set_flag("v", "mte2", 1)  # stage 1 的 row_buf 可以被 mte2 覆盖
                        T.set_flag("mte3", "v", 0)  # mte3 已经读完上一次 out_buf, v 可以覆盖

                        # ---- (2) 预取 batch 的第 0 个 token (8 个 lane) 到 stage 0 ----
                        # 这是流水线的 "prologue", 让循环主体进去时已经有一组数据可算
                        if BATCH_T > 0:
                            T.wait_flag("v", "mte2", 0)  # 等 v 放手 stage 0
                            for lane in T.serial(topK):  # 把 8 个 lane 都搬过来
                                src_p = idx_ub[0, lane]  # lane 在 sorted_indices 里的取值 = 源行号
                                T.copy(
                                    perm_grad_gm[src_p, h_off],  # 从 GM 拷贝该行的 HALF_H 段
                                    row_buf[lane, :],  # 写到 stage 0 的 lane 行 (row_buf[0..7])
                                )
                            T.set_flag("mte2", "v", 0)  # 通知 v "stage 0 的 8 行已就位"

                        # ---- (3) 主循环: 每个 ti 处理 batch 内一个 token ----
                        for ti in T.serial(BATCH_T):
                            cur_stage = ti % stages  # 当前要消费哪个 stage 的 row_buf
                            nxt_stage = (ti + 1) % stages  # 下一轮要预取哪个 stage
                            cur_i_tok = batch_base + ti  # 当前 token 在原始序列里的全局 id

                            # ---- (3a) 预取下一个 token 的 8 个 lane 到 nxt_stage ----
                            #     (这一段和当前 stage 的 v 计算并行执行 —— 流水的核心)
                            if ti + 1 < BATCH_T:
                                T.wait_flag("v", "mte2", nxt_stage)  # 等 nxt_stage 被 v 释放
                                nxt_tk_off = (ti + 1) * topK  # 下一 token 的索引偏移
                                for lane in T.serial(topK):
                                    src_n = idx_ub[0, nxt_tk_off + lane]
                                    T.copy(
                                        perm_grad_gm[src_n, h_off],
                                        # 注意写到 nxt_stage 的 [topK 行] 区间, 不会和 cur_stage 冲突
                                        row_buf[nxt_stage * topK + lane, :],
                                    )
                                T.set_flag("mte2", "v", nxt_stage)  # 通知 v "下一组数据准备好了"

                            # ---- (3b) 等当前 stage 的 8 行被 mte2 搬完, 然后开始算 ----
                            T.wait_flag("mte2", "v", cur_stage)

                            # 把累加器清零, 准备做 K 次加法
                            T.tile.fill(acc_buf, 0.0)

                            # 8 次累加: 把 cur_stage 的 8 行依次加到 acc_buf
                            for lane in T.serial(topK):
                                # 把这 1 行从 row_buf 单独拷到 row_tmp (1 行宽的临时 buffer)
                                T.copy(row_buf[cur_stage * topK + lane, :], row_tmp)
                                # 低精度 -> fp32, 准备做精度安全的累加
                                T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
                                # acc_buf += row_f32 (fp32 累加)
                                T.tile.add(acc_buf, acc_buf, row_f32)

                            # 算完后通知 mte2 "cur_stage 的 row_buf 我用完了, 可以覆盖"
                            T.set_flag("v", "mte2", cur_stage)

                            # ---- (3c) 把 fp32 结果 cast 回低精度并写回 GM ----
                            T.wait_flag("mte3", "v", 0)  # 等上一次写出完成 (out_buf 复用)
                            T.tile.cast(out_buf, acc_buf, CAST_HIGH2LOW, HALF_H)
                            T.set_flag("v", "mte3", 0)  # 通知 mte3 "out_buf 已就绪"

                            T.wait_flag("v", "mte3", 0)  # 等上面那个 set_flag 生效
                            # cur_i_tok 可能越过末尾 (因为 padded), 越界的 token 不写
                            if cur_i_tok < num_tokens:
                                T.copy(out_buf, input_grad_gm[cur_i_tok, h_off])
                            T.set_flag("mte3", "v", 0)  # 写完, 通知 v 可以覆盖 out_buf 了

                        # ---- (4) batch 收尾: 把所有未消费的 flag 等掉, 防止 hang ----
                        T.wait_flag("v", "mte2", 0)
                        T.wait_flag("v", "mte2", 1)
                        T.wait_flag("mte3", "v", 0)

        return moe_token_permute_grad

    # 把所有编译期常量传给 _build, 拿到一个针对该形状特化的 kernel
    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        stages,
    )


# =============================================================================
# Kernel 变体 2: cast + lane-pipelined (topK <= 8, 但 topK != 8)
# =============================================================================
# 流水结构: stages = 8, 一次只推进 *1 个 lane*  (粒度比变体 1 更细)
#   - 把 perm_grad 的搬运分摊到 8 级流水, 任意时刻最多有 7 个 lane 正在 prefetch
#   - 每个 lane 处理完后 acc_buf += row_f32; 第 K-1 个 lane 时 cast + 写出
# 适用条件: dtype != fp32, hidden_size == TILE_H, topK <= 8 且 != 8
# =============================================================================


def _build_gather_reduce_kernel_cast_pipelined(
    num_tokens,
    topK,
    hidden_size,
    E,
    padded_E,
    actual_cores,
    tokens_per_core,
    TILE_H,
    HALF_H,
    BATCH_T,
    n_batches,
    dtype,
    idx_dtype,
):
    stages = 8  # 8 级深度的 lane 级流水
    total_iters_per_batch = BATCH_T * topK  # 一个 batch 总共要处理多少个 lane

    assert topK <= 8  # 用了 8 stages, topK 大于 8 会重叠覆盖
    assert dtype != CAL_DTYPE  # 这个 kernel 走 cast 路径
    assert hidden_size == TILE_H
    assert HALF_H * 2 == TILE_H

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS_EXPERT)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        stages,
        total_iters_per_batch,
    ):
        @T.prim_func
        def moe_token_permute_grad(
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            input_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                # 索引 buffer 一次拷一个 batch 的全部 BATCH_T*K 个索引
                idx_ub = T.alloc_ub([1, BATCH_T * topK], idx_dtype)

                # 这次 row_buf 只开 stages 行 (= 8 行), 因为流水粒度是单 lane 而非整组
                row_buf = T.alloc_ub([stages, HALF_H], dtype)
                row_tmp = T.alloc_ub([1, HALF_H], dtype)
                row_f32 = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                acc_buf = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                out_buf = T.alloc_ub([1, HALF_H], dtype)

                h_off = vid * HALF_H

                with T.Scope("V"):
                    for batch_id in T.serial(n_batches):
                        batch_base = cid * tokens_per_core + batch_id * BATCH_T

                        # 拷索引 + 同步
                        T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                        T.set_flag("mte2", "v", 10)
                        T.wait_flag("mte2", "v", 10)

                        # 把所有 8 个 stage 的 v->mte2 flag 拉满, 让 mte2 可以马上启动 8 路 prefetch
                        T.set_flag("v", "mte2", 0)
                        T.set_flag("v", "mte2", 1)
                        T.set_flag("v", "mte2", 2)
                        T.set_flag("v", "mte2", 3)
                        T.set_flag("v", "mte2", 4)
                        T.set_flag("v", "mte2", 5)
                        T.set_flag("v", "mte2", 6)
                        T.set_flag("v", "mte2", 7)
                        T.set_flag("mte3", "v", 0)

                        # ---- prologue: 预取前 (stages-1) = 7 个 lane 到 row_buf[0..6] ----
                        # (注意只预取 7 个, 第 8 个由主循环的 it=0 那一轮的 prefetch 顺手做,
                        #  相当于流水线"开机加热"——填满除了最后一格外的所有 stage)
                        #
                        # 历史背景: 早期版本是把这 7 步完全展开成 7 个独立 if 块, 当时担心
                        # tilelang 在编译期未必能用变量索引区分不同 stage 的事件 ID.
                        # 后来确认主循环里 next_stage = next_it % stages 已经是 runtime 算的,
                        # 事件 ID 用变量没问题——所以改用一个 T.serial 循环, 代码少 35 行,
                        # 行为完全等价 (idx_ub[0, i] 和 row_buf[i, :] 都接受 runtime i).
                        for i in T.serial(stages - 1):
                            if i < total_iters_per_batch:  # 防止 batch 内 lane 数 < 7 时越界
                                T.wait_flag("v", "mte2", i)
                                src_p = idx_ub[0, i]
                                T.copy(perm_grad_gm[src_p, h_off], row_buf[i, :])
                                T.set_flag("mte2", "v", i)

                        # ---- 主循环: 一次推进 1 个 lane (it = 0..total-1) ----
                        for it in T.serial(total_iters_per_batch):
                            cur_stage = it % stages  # 当前消费的 row_buf 行 (0..7 循环)
                            cur_token = it // topK  # 当前 lane 属于第几个 token (batch 内)
                            cur_lane = it % topK  # 当前 lane 是这个 token 的第几份
                            cur_i_tok = batch_base + cur_token  # 当前 token 的全局 id

                            # 下一个待预取 lane 的位置 = it + (stages - 1) = it + 7
                            # 这样保持流水深度恒为 7 (有 1 个 stage 正在被 v 消费)
                            next_it = it + stages - 1
                            next_stage = next_it % stages

                            # 预取下一 lane (与当前 lane 的 v 计算并行)
                            if next_it < total_iters_per_batch:
                                T.wait_flag("v", "mte2", next_stage)
                                src_n = idx_ub[0, next_it]
                                T.copy(
                                    perm_grad_gm[src_n, h_off],
                                    row_buf[next_stage, :],
                                )
                                T.set_flag("mte2", "v", next_stage)

                            # 等当前 lane 的数据搬完
                            T.wait_flag("mte2", "v", cur_stage)

                            # 一个 token 的第一份 (lane 0): 清零累加器
                            if cur_lane == 0:
                                T.tile.fill(acc_buf, 0.0)

                            # 把 cur_stage 这 1 行 cast 成 fp32 加进 acc_buf
                            T.copy(row_buf[cur_stage, :], row_tmp)
                            T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
                            T.tile.add(acc_buf, acc_buf, row_f32)

                            T.set_flag("v", "mte2", cur_stage)  # 释放 cur_stage 给 mte2

                            # 一个 token 的最后一份 (lane K-1): cast 回低精度并写出
                            if cur_lane == topK - 1:
                                T.wait_flag("mte3", "v", 0)
                                T.tile.cast(out_buf, acc_buf, CAST_HIGH2LOW, HALF_H)
                                T.set_flag("v", "mte3", 0)

                                T.wait_flag("v", "mte3", 0)
                                if cur_i_tok < num_tokens:  # 越界 token 不写
                                    T.copy(out_buf, input_grad_gm[cur_i_tok, h_off])
                                T.set_flag("mte3", "v", 0)

                        # batch 收尾: 等所有 8 个 stage 都被释放, 防 hang
                        T.wait_flag("v", "mte2", 0)
                        T.wait_flag("v", "mte2", 1)
                        T.wait_flag("v", "mte2", 2)
                        T.wait_flag("v", "mte2", 3)
                        T.wait_flag("v", "mte2", 4)
                        T.wait_flag("v", "mte2", 5)
                        T.wait_flag("v", "mte2", 6)
                        T.wait_flag("v", "mte2", 7)
                        T.wait_flag("mte3", "v", 0)

        return moe_token_permute_grad

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        stages,
        total_iters_per_batch,
    )


# =============================================================================
# Kernel 变体 3: cast 普通版 (无流水, 用 barrier_all 同步)
# =============================================================================
# 当不满足流水线 kernel 的前提条件时, 走这条 "兜底" 路径:
#   - 一次按 LANES_PER_ITER 个 lane 拷贝 + 累加 (LANES_PER_ITER 在编译期决定)
#   - 用 T.barrier_all() 简单粗暴地把 mte2 与 v 同步, 不做细粒度重叠
#   - 支持 hidden_size > TILE_H (需要在 H 方向多次 tiling)
# 数学语义和前两个完全一样, 只是性能弱一些.
# =============================================================================


def _build_gather_reduce_kernel_cast(
    num_tokens,
    topK,
    hidden_size,
    E,
    padded_E,
    actual_cores,
    tokens_per_core,
    n_htiles,  # H 方向被切成几片 TILE_H (= hidden_size / TILE_H)
    TILE_H,
    BATCH_T,
    n_batches,
    dtype,
    idx_dtype,
):
    assert topK >= 1, "cast kernel requires topK >= 1"
    HALF_H = TILE_H // 2

    # ---- 决定每次内层循环要并行处理几个 lane (LANES_PER_ITER) ----
    dtype_bytes = 4 if dtype in ("float32", "float") else 2  # bf16/fp16 = 2B, fp32 = 4B
    ALIGN_BYTES = 32  # NPU UB 访存对齐粒度
    # 如果 HALF_H 这一行已经够大 (>=32B), 一次拷 8 个 lane 比较划算
    # 否则 (行很短 / dtype 很窄), 退化到一次只搬 1 个 lane, 避免读不齐
    if HALF_H * dtype_bytes >= ALIGN_BYTES:
        LANES_PER_ITER = min(8, topK)
    else:
        LANES_PER_ITER = 1
    n_iters = topK // LANES_PER_ITER  # 完整跑几轮 LANES_PER_ITER
    rem = topK % LANES_PER_ITER  # 剩余的尾巴 (单独处理, 见下方 if rem > 0 分支)

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS_EXPERT)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        n_htiles,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        LANES_PER_ITER,
        n_iters,
        rem,
    ):
        @T.prim_func
        def moe_token_permute_grad(
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            input_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                idx_ub = T.alloc_ub([1, BATCH_T * topK], idx_dtype)

                # row_buf 行数 = LANES_PER_ITER, 一次能装下一组并行搬运的 lane
                row_buf = T.alloc_ub([LANES_PER_ITER, HALF_H], dtype)
                row_tmp = T.alloc_ub([1, HALF_H], dtype)
                row_f32 = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                acc_buf = T.alloc_ub([1, HALF_H], CAL_DTYPE)
                out_buf = T.alloc_ub([1, HALF_H], dtype)

                with T.Scope("V"):
                    for batch_id in T.serial(n_batches):
                        batch_base = cid * tokens_per_core + batch_id * BATCH_T

                        # 拷一个 batch 的索引, 然后用 barrier_all 等 mte2 完成
                        # (barrier_all 是粗粒度同步, 比 set/wait flag 简单但更保守)
                        T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)
                        T.barrier_all()

                        # 三层循环: token (ti) -> H 方向 tile (ht) -> lane 分组 (jj)
                        for ti in T.serial(BATCH_T):
                            i = batch_base + ti  # 全局 token id
                            if i < num_tokens:  # 越界的 padded token 跳过
                                for ht in T.serial(n_htiles):
                                    # H 维 tile + vid 把 H 切到细粒度: ht 选 TILE_H 段, vid 选半段
                                    h_off = ht * TILE_H + vid * HALF_H
                                    tk_off = ti * topK  # 当前 token 的索引在 idx_ub 内的偏移

                                    T.tile.fill(acc_buf, 0.0)

                                    # ----- 完整 lane 分组 -----
                                    for jj in T.serial(n_iters):
                                        base = jj * LANES_PER_ITER  # 这一组的起始 lane
                                        # 串行 issue LANES_PER_ITER 次拷贝 (硬件可能合并)
                                        for lane in T.serial(LANES_PER_ITER):
                                            src = idx_ub[0, tk_off + base + lane]
                                            T.copy(
                                                perm_grad_gm[src, h_off],
                                                row_buf[lane, :],
                                            )
                                        T.barrier_all()  # 等 mte2 完成
                                        # 串行累加这 LANES_PER_ITER 行
                                        for lane in T.serial(LANES_PER_ITER):
                                            T.copy(row_buf[lane, :], row_tmp)
                                            T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
                                            T.tile.add(acc_buf, acc_buf, row_f32)

                                    # ----- 处理 topK 不能整除 LANES_PER_ITER 的尾巴 -----
                                    if rem > 0:
                                        base = n_iters * LANES_PER_ITER  # 尾巴起点
                                        for lane in T.serial(rem):
                                            src = idx_ub[0, tk_off + base + lane]
                                            T.copy(
                                                perm_grad_gm[src, h_off],
                                                row_buf[lane, :],
                                            )
                                        T.barrier_all()
                                        for lane in T.serial(rem):
                                            T.copy(row_buf[lane, :], row_tmp)
                                            T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, HALF_H)
                                            T.tile.add(acc_buf, acc_buf, row_f32)

                                    # cast fp32 -> low + 写出
                                    T.barrier_all()
                                    T.tile.cast(out_buf, acc_buf, CAST_HIGH2LOW, HALF_H)
                                    T.pipe_barrier("v")  # v 内部按发射顺序排空
                                    T.copy(out_buf, input_grad_gm[i, h_off])
                                    T.pipe_barrier("mte3")  # 等本次写出完成

        return moe_token_permute_grad

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        n_htiles,
        TILE_H,
        HALF_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
        LANES_PER_ITER,
        n_iters,
        rem,
    )


# =============================================================================
# Kernel 变体 4: nocast (dtype == fp32, 不需要 cast)
# =============================================================================
# fp32 路径不需要 fp32 累加器 (本身就是 fp32), 所以省掉所有 cast.
# 同时为了稍微提速, 内层用 "三元组" 展开 (三行同时拷, 然后三行依次加), 让 mte2 更并行.
# =============================================================================


def _build_gather_reduce_kernel_nocast(
    num_tokens,
    topK,
    hidden_size,
    E,
    padded_E,
    actual_cores,
    tokens_per_core,
    n_htiles,
    TILE_H,
    BATCH_T,
    n_batches,
    dtype,
    idx_dtype,
):
    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS)  # 注意: 用普通 PASS_CONFIGS (AUTO_SYNC=True)
    def _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        n_htiles,
        TILE_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
    ):
        @T.prim_func
        def moe_token_permute_grad(
            perm_grad_gm: T.Tensor([E, hidden_size], dtype),
            sorted_idx_gm: T.Tensor([1, padded_E], idx_dtype),
            input_grad_gm: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            with T.Kernel(actual_cores, is_npu=True) as (cid, vid):
                # 这里 buffer 用 alloc_shared (和上面的 alloc_ub 相对): shared 内存范围更大
                # (因为 fp32 一行字节数比 fp16/bf16 多 2 倍, 用 shared 能放更宽)
                idx_ub = T.alloc_shared([1, BATCH_T * topK], idx_dtype)
                # 三个独立的 row buffer, 用于一次 issue 三条搬运 (硬件可以并行)
                row_buf0 = T.alloc_shared([1, TILE_H], dtype)
                row_buf1 = T.alloc_shared([1, TILE_H], dtype)
                row_buf2 = T.alloc_shared([1, TILE_H], dtype)
                acc_buf = T.alloc_shared([1, TILE_H], dtype)  # fp32 直接累加, 不需要单独的 fp32 buf

                # 注: 这里没有 `with T.Scope("V"):`, 因为 AUTO_SYNC=True, 编译器自己管 scope
                for batch_id in T.serial(n_batches):
                    batch_base = cid * tokens_per_core + batch_id * BATCH_T

                    T.copy(sorted_idx_gm[0, batch_base * topK], idx_ub)

                    for ti in T.serial(BATCH_T):
                        i = batch_base + ti
                        if i < num_tokens:
                            for ht in T.serial(n_htiles):
                                h_off = ht * TILE_H  # 这里没 vid 切分, 整个 TILE_H 一起算
                                tk_off = ti * topK

                                T.tile.fill(acc_buf, 0.0)

                                # 把 topK 拆成 (n_triples 组三元组) + (1 或 2 个尾巴)
                                n_triples = topK // 3
                                remainder = topK % 3

                                # ---- 三元组: 一次拷 3 行, 然后 3 次累加 ----
                                # 这样硬件 mte2 可以同时 issue 3 条搬运指令 (利用率更高)
                                for j3 in T.serial(n_triples):
                                    j = j3 * 3
                                    src_a = idx_ub[0, tk_off + j]
                                    src_b = idx_ub[0, tk_off + j + 1]
                                    src_c = idx_ub[0, tk_off + j + 2]
                                    T.copy(perm_grad_gm[src_a, h_off], row_buf0)
                                    T.copy(perm_grad_gm[src_b, h_off], row_buf1)
                                    T.copy(perm_grad_gm[src_c, h_off], row_buf2)
                                    T.tile.add(acc_buf, acc_buf, row_buf0)
                                    T.tile.add(acc_buf, acc_buf, row_buf1)
                                    T.tile.add(acc_buf, acc_buf, row_buf2)

                                # 尾巴 == 2: 用 2 个 row_buf 并行
                                if remainder == 2:
                                    base = n_triples * 3
                                    src_a = idx_ub[0, tk_off + base]
                                    src_b = idx_ub[0, tk_off + base + 1]
                                    T.copy(perm_grad_gm[src_a, h_off], row_buf0)
                                    T.copy(perm_grad_gm[src_b, h_off], row_buf1)
                                    T.tile.add(acc_buf, acc_buf, row_buf0)
                                    T.tile.add(acc_buf, acc_buf, row_buf1)

                                # 尾巴 == 1: 单独处理最后 1 行
                                if remainder == 1:
                                    src_last = idx_ub[0, tk_off + topK - 1]
                                    T.copy(perm_grad_gm[src_last, h_off], row_buf0)
                                    T.tile.add(acc_buf, acc_buf, row_buf0)

                                # 直接把 acc_buf 写回 GM (不需要 cast)
                                T.copy(acc_buf, input_grad_gm[i, h_off])

        return moe_token_permute_grad

    return _build(
        num_tokens,
        topK,
        hidden_size,
        E,
        padded_E,
        actual_cores,
        tokens_per_core,
        n_htiles,
        TILE_H,
        BATCH_T,
        n_batches,
        dtype,
        idx_dtype,
    )


# =============================================================================
# Kernel 选路 + 编译入口: _compile_gather_reduce
# =============================================================================
# 根据 (dtype, topK, hidden_size) 选择上面 4 个变体之一, 把所有形状参数算出来,
# 然后把对应的 kernel 编译出来交给上层 MoeTokenPermuteGrad 使用.
# =============================================================================


def _compile_gather_reduce(
    num_tokens: int,  # T
    topK: int,  # K
    hidden_size: int,  # H (实际计算用的 H, 可能已被上层 pad 过)
    E: int,  # E = T*K
    NUM_CORES: int = 24,  # NPU 上可用的核数, Ascend 一般 24
    TILE_H: int = None,  # H 方向的 tile 大小, None 表示自动选
    dtype: str = "float16",
    idx_dtype: str = "int32",
):
    ALIGN_BYTES = 32  # NPU UB 一次访存最少 32B 对齐
    dtype_bytes = 4 if dtype in ("float32", "float") else 2
    align_elems = ALIGN_BYTES // dtype_bytes  # 一行 32B 能装多少个 dtype 元素

    # ---- 自动选 TILE_H ----
    # 优先用整个 hidden_size; 若太大就退到 4096B / 2048B / 1024 / 512 / 最小对齐
    # 选择条件: 要 ≥ align_elems 且能被 hidden_size 整除
    if TILE_H is None:
        for candidate in [
            hidden_size,
            4096 // dtype_bytes,
            2048 // dtype_bytes,
            1024,
            512,
            align_elems,
        ]:
            if candidate > 0 and candidate >= align_elems and hidden_size % candidate == 0:
                TILE_H = candidate
                break
        else:
            TILE_H = align_elems

    # 校验 TILE_H 是否满足 32B 对齐要求 (≥ 32B 且 32B 的整数倍)
    assert TILE_H * dtype_bytes >= ALIGN_BYTES and (TILE_H * dtype_bytes) % ALIGN_BYTES == 0, (
        f"TILE_H={TILE_H} * sizeof({dtype})={dtype_bytes} = {TILE_H * dtype_bytes}B; must be >= 32B and a multiple of 32B"
    )
    # H 必须能被 TILE_H 整除, 否则会有半截 tile 没办法正确处理
    assert hidden_size % TILE_H == 0, f"hidden_size ({hidden_size}) must be divisible by TILE_H ({TILE_H})"

    n_htiles = int(hidden_size // TILE_H)  # H 方向需要的 tile 数

    # 实际核数 = min(可用核数, T) (token 数比核数还少时不用满)
    actual_cores = int(min(NUM_CORES, max(1, num_tokens)))
    tokens_per_core = int(math.ceil(num_tokens / actual_cores))

    # padded_E: 对齐到 actual_cores * tokens_per_core * topK
    # (因为最后一个核可能拿到比别的核少, 编译期给它补足, 越界 token 在 kernel 里不写)
    padded_E = int(actual_cores * tokens_per_core * topK)

    # ---- 判断能否走流水线 (group / lane) kernel ----
    HALF_H_candidate = hidden_size // 2 if hidden_size % 2 == 0 else 0

    is_cast_path = dtype != CAL_DTYPE  # 非 fp32 才需要 cast
    single_htile = hidden_size == TILE_H  # H 整体只有 1 个 tile
    half_aligned = (
        HALF_H_candidate > 0 and HALF_H_candidate * 2 == TILE_H and HALF_H_candidate * dtype_bytes >= ALIGN_BYTES  # HALF_H 行也要 ≥ 32B
    )
    pipelined_eligible = is_cast_path and single_htile and half_aligned

    # 三条件都满足, 再按 topK 决定哪个流水变体
    use_group_pipelined = pipelined_eligible and topK == 8
    use_lane_pipelined = pipelined_eligible and topK <= 8 and not use_group_pipelined

    # ---- 计算 BATCH_T (一个 batch 内处理几个 token) ----
    # 4096 这个数字大致是 idx_ub 大小预算 / topK / 安全裕度
    # 然后再向下找一个能整除 tokens_per_core 的, 让最后一个 batch 不残缺
    if use_group_pipelined:
        HALF_H = HALF_H_candidate
        BATCH_T = min(tokens_per_core, max(1, 4096 // (topK * 10)))
        while BATCH_T > 1 and tokens_per_core % BATCH_T != 0:
            BATCH_T -= 1
        n_batches = int(math.ceil(tokens_per_core / BATCH_T))

        kernel = _build_gather_reduce_kernel_cast_group_pipelined(
            num_tokens,
            topK,
            hidden_size,
            E,
            padded_E,
            actual_cores,
            tokens_per_core,
            TILE_H,
            HALF_H,
            BATCH_T,
            n_batches,
            dtype,
            idx_dtype,
        )
    elif use_lane_pipelined:
        HALF_H = HALF_H_candidate
        BATCH_T = min(tokens_per_core, max(1, 4096 // (topK * 10)))
        while BATCH_T > 1 and tokens_per_core % BATCH_T != 0:
            BATCH_T -= 1
        n_batches = int(math.ceil(tokens_per_core / BATCH_T))

        kernel = _build_gather_reduce_kernel_cast_pipelined(
            num_tokens,
            topK,
            hidden_size,
            E,
            padded_E,
            actual_cores,
            tokens_per_core,
            TILE_H,
            HALF_H,
            BATCH_T,
            n_batches,
            dtype,
            idx_dtype,
        )
    else:
        # 走兜底路径 (cast 普通版 / nocast)
        BATCH_T = min(tokens_per_core, max(1, 4096 // (topK * 10)))
        while BATCH_T > 1 and tokens_per_core % BATCH_T != 0:
            BATCH_T -= 1
        n_batches = int(math.ceil(tokens_per_core / BATCH_T))

        if dtype != CAL_DTYPE:
            # 非 fp32 -> cast 普通版
            kernel = _build_gather_reduce_kernel_cast(
                num_tokens,
                topK,
                hidden_size,
                E,
                padded_E,
                actual_cores,
                tokens_per_core,
                n_htiles,
                TILE_H,
                BATCH_T,
                n_batches,
                dtype,
                idx_dtype,
            )
        else:
            # fp32 -> nocast (省 cast 步骤)
            kernel = _build_gather_reduce_kernel_nocast(
                num_tokens,
                topK,
                hidden_size,
                E,
                padded_E,
                actual_cores,
                tokens_per_core,
                n_htiles,
                TILE_H,
                BATCH_T,
                n_batches,
                dtype,
                idx_dtype,
            )

    return kernel, padded_E  # 把 padded_E 一并返回, 上层用它准备索引 buffer


# =============================================================================
# 高层封装: MoeTokenPermuteGrad
# =============================================================================
# 对外暴露的 Python 类. 构造时编译 kernel, __call__ 时:
#   1. 必要时 pad 输入到 kernel 期望的形状
#   2. 把索引 cast 到 int32 并 pad 到 padded_E
#   3. 调 kernel
#   4. 把结果裁回原始 hidden_size
# =============================================================================


class MoeTokenPermuteGrad:
    def __init__(
        self,
        num_tokens: int,
        topK: int,
        hidden_size: int,
        num_experts: int = 64,
        num_out_tokens: int = 0,
        padded_mode: bool = False,
        NUM_CORES: int = 24,
        TILE_H: int = None,
        dtype: str = "float16",
    ):
        # padded_mode 是 megatron 里的 "把每个专家固定对齐到 capacity" 模式
        # 本实现没支持, 调到的话直接报错 (上游应避免传入)
        if padded_mode:
            raise NotImplementedError("padded_mode=True not supported.")

        # 把构造参数都存下来, 后面 __call__ 要用
        self.num_tokens = num_tokens
        self.topK = topK
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.E = num_tokens * topK  # 派生: permuted 张量行数
        # _out_len 默认是 E, 也可以被 num_out_tokens 覆盖 (用于截断模式)
        self._out_len = num_out_tokens if num_out_tokens > 0 else self.E

        # H 的最小编译值: fp32 至少 64 (= 16 elements * 4B = 64B 对齐), 其余 32
        # 这样保证 kernel 里 alloc_ub 的最小宽度满足 32B 对齐
        # 如果用户传的 hidden_size 比 min_compile_h 小, 编译时按 min_compile_h 走, __call__ 里 pad
        min_compile_h = 64 if is_fp32_dtype(dtype) else 32
        self._compile_hidden_size = max(hidden_size, min_compile_h)
        # TILE_H 同样要被 min_compile_h 兜底 (None 透传, 让 _compile_gather_reduce 自动选)
        compile_tile_h = TILE_H if TILE_H is None else max(TILE_H, min_compile_h)

        # 编译 kernel, 返回 (kernel, padded_E)
        self._kernel, self._padded_E = _compile_gather_reduce(
            num_tokens,
            topK,
            self._compile_hidden_size,
            self.E,
            NUM_CORES=NUM_CORES,
            TILE_H=compile_tile_h,
            dtype=dtype,
        )

        # 缓存的 padded buffer (lazy 创建, 避免每次 __call__ 都 allocate)
        self._sorted_idx_buf = None  # 索引 padded 缓冲
        self._perm_grad_pad_buf = None  # perm_grad padded 缓冲

    def _get_idx_buf(self, device):
        """拿到 (设备绑定的) 索引 padded 缓冲, 第一次或换设备时创建."""
        if self._sorted_idx_buf is None or self._sorted_idx_buf.device != device:
            self._sorted_idx_buf = torch.zeros(
                self._padded_E,  # 编译期对齐过的长度
                dtype=torch.int32,  # kernel 期望 int32
                device=device,
            )
        return self._sorted_idx_buf

    def __call__(self, permuted_output_grad, sorted_indices):
        device = permuted_output_grad.device
        E = self.E
        H = self._compile_hidden_size  # kernel 内部用的 H (可能 > 真实 hidden_size)

        # ---- (1) 必要时把 perm_grad pad 到 [E, H_compile] ----
        # 真实输入 r 行 c 列, 不够就补零
        needs_pad = permuted_output_grad.shape[0] < E or permuted_output_grad.shape[1] < H
        if needs_pad:
            target_shape = (E, H)
            # lazy 创建 / 重建 padded buffer (设备 / dtype / shape 任一不同就重建)
            if (
                self._perm_grad_pad_buf is None
                or self._perm_grad_pad_buf.device != device
                or self._perm_grad_pad_buf.dtype != permuted_output_grad.dtype
                or tuple(self._perm_grad_pad_buf.shape) != target_shape
            ):
                self._perm_grad_pad_buf = torch.zeros(
                    *target_shape,
                    dtype=permuted_output_grad.dtype,
                    device=device,
                )
            perm_grad_padded = self._perm_grad_pad_buf
            r = permuted_output_grad.shape[0]
            c = permuted_output_grad.shape[1]
            # 把真实数据写到 [0:r, 0:c] 区域, 其余保持 0 (零参与累加是安全的, 不影响 sum 结果)
            perm_grad_padded[:r, :c].copy_(permuted_output_grad)
        else:
            perm_grad_padded = permuted_output_grad  # 完全合规, 直接用原张量

        # ---- (2) 把 sorted_indices 摊平 + cast int32 + pad ----
        sorted_idx_padded = self._get_idx_buf(device)
        si = sorted_indices.view(-1)  # [T, K] -> [E]
        if si.dtype != torch.int32:
            si = si.to(torch.int32)  # kernel 期望 int32
        sorted_idx_padded[:E].copy_(si)  # 真实索引写到前 E, 其余位置保持 0

        # ---- (3) 调 kernel: 输入 perm_grad + indices, 输出 input_grad ----
        # unsqueeze(0) 是把 [padded_E] 升维到 [1, padded_E], 因为 kernel 签名是 2D
        input_grad = self._kernel(
            perm_grad_padded,
            sorted_idx_padded.unsqueeze(0),
        )

        # ---- (4) 如果编译时把 H pad 大了, 现在裁回真实 hidden_size ----
        if self.hidden_size != H:
            input_grad = input_grad[:, : self.hidden_size].contiguous()

        return input_grad

    def __repr__(self):
        # 调试时方便看清楚这个算子实例的形状参数
        return f"MoeTokenPermuteGrad(T={self.num_tokens}, K={self.topK}, H={self.hidden_size}, experts={self.num_experts})"


# =============================================================================
# 测试: 与 torch_npu.npu_moe_token_permute 的 backward 对比
# =============================================================================
# 思路: 用 torch_npu 跑一遍 forward + backward, 拿到金标 input_grad,
# 然后用本文件的 MoeTokenPermuteGrad 跑同样的输入, 比对两者是否数值相同.
# =============================================================================


def test_permute_grad_parameterized(pt_dtype, tl_dtype_str):
    print(f"\n{'=' * 65}")
    print(f"Testing MoeTokenPermuteGrad, dtype: {tl_dtype_str.upper()}")
    print(f"{'=' * 65}")

    torch.manual_seed(42)  # 固定随机种子, 让结果可复现

    # 测试用的小形状 (足够触发各条 kernel 路径但跑得快)
    num_tokens = 16
    hidden_size = 16
    topk = 4
    num_experts = 4

    all_passed = True

    # ---- Test 1: 标准 backward ----
    print(">>> Test case 1: Standard Backward gradient alignment test")

    # 构造原始输入: 随机 token + 随机 expert 选择
    tokens = torch.randn(num_tokens, hidden_size, dtype=pt_dtype, device="npu", requires_grad=True)
    indices = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="npu")

    # forward: 拿到 permuted_tokens 和 sorted_indices (供 backward 用)
    npu_permuted, npu_sorted_idx = torch_npu.npu_moe_token_permute(tokens, indices)

    # 模拟一个上游回传的梯度
    grad_permuted_tokens = torch.randn_like(npu_permuted)

    # 用 torch_npu 自带的 backward 算金标
    npu_permuted.backward(grad_permuted_tokens)
    npu_input_grad = tokens.grad.clone()

    tokens.grad.zero_()  # 清空 grad, 避免下面的对比互相干扰

    # 用我们的 kernel 算同样的 input_grad
    tl_grad_op = MoeTokenPermuteGrad(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        num_experts=num_experts,
        dtype=tl_dtype_str,
    )
    tl_input_grad = tl_grad_op(grad_permuted_tokens, npu_sorted_idx)
    torch.npu.synchronize()  # 等 NPU 算完, 不然 assert_close 可能拿到旧值

    # 数值对比 (allclose), 不通过会抛 AssertionError
    try:
        torch.testing.assert_close(tl_input_grad, npu_input_grad)
        print(f"    [PASS] {tl_dtype_str.upper()} Standard Backward precision test passed!")
    except AssertionError as e:
        print(
            f"    [FAILED] {tl_dtype_str.upper()} Standard Backward precision test failed!\n",
            e,
        )
        all_passed = False

    # ---- Test 2: 截断 (clip) backward ----
    # num_out_tokens < E 时, permuted_tokens 只保留前 num_out_tokens 行 (drop tail)
    # 我们的 kernel 也要能正确处理 (上游传进来的 perm_grad 行数 < E, 需要在 __call__ 里 pad)
    print("\n>>> Test case 2: Clip Backward gradient alignment test with truncation")
    num_out_tokens = 10

    tokens_clip = torch.randn(num_tokens, hidden_size, dtype=pt_dtype, device="npu", requires_grad=True)
    indices_clip = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="npu")

    # forward 加 num_out_tokens 参数, 触发截断
    npu_permuted_clip, npu_sorted_idx_clip = torch_npu.npu_moe_token_permute(tokens_clip, indices_clip, num_out_tokens=num_out_tokens)

    grad_permuted_clip = torch.randn_like(npu_permuted_clip)
    npu_permuted_clip.backward(grad_permuted_clip)
    npu_input_grad_clip = tokens_clip.grad.clone()

    # 注意我们这里同样要传 num_out_tokens, 让 _out_len 正确
    tl_grad_op_clip = MoeTokenPermuteGrad(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_out_tokens=num_out_tokens,
        dtype=tl_dtype_str,
    )
    tl_input_grad_clip = tl_grad_op_clip(grad_permuted_clip, npu_sorted_idx_clip)
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(tl_input_grad_clip, npu_input_grad_clip)
        print(f"    [PASS] {tl_dtype_str.upper()} Clip truncation Backward precision test passed!")
    except AssertionError as e:
        print(
            f"    [FAILED] {tl_dtype_str.upper()} Clip truncation Backward precision test failed!\n",
            e,
        )
        all_passed = False

    return all_passed


def test_permute_grad():
    # 跑遍 3 个常用 dtype, 任何一个挂掉就标记 overall fail
    dtypes_to_test = [
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
        (torch.float32, "float32"),
    ]

    overall_passed = True
    for pt_type, tl_type_str in dtypes_to_test:
        passed = test_permute_grad_parameterized(pt_dtype=pt_type, tl_dtype_str=tl_type_str)
        if not passed:
            overall_passed = False

    print(f"\n{'=' * 65}")
    if overall_passed:
        print("Test passed!")
    else:
        print("Test failed! The precision is not correct!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    # 直接 `python moe_token_permute_grad.py` 时跑测试
    test_permute_grad()
