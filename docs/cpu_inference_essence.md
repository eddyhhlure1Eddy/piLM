# CPU 推理优化的底层原理

> 从 llama.cpp 与 vLLM 的 CPU 后端提炼。每条原理讲"为什么这么做能快"，代码引用仅作例证。

---

## 1. 把算力吃满：SIMD、FMA、AMX、VNNI

CPU 的峰值算力几乎全在向量单元。标量循环只用到 1/16（AVX-512）的算力，所以推理内核的第一性原理是**让一条指令处理尽量多的元素**。

**SIMD 宽度决定每拍吞吐。** AVX2 = 8 个 f32，AVX-512 = 16 个 f32，SVE / RVV 是寄存器长度无关的"可变长向量"。`ggml_vec_mad_f32`（`llama.cpp/ggml/src/ggml-cpu/vec.h:319`）里那一长串 `#if __ARM_FEATURE_SVE / __riscv_v / else` 就是为了在不同寄存器宽度上把同一份点积展开到最大宽度。SVE 路径还展开 8 路寄存器（`vec.h:325`），是为了填满 SVE 的寄存器文件、隐藏 FMA 延迟。

**FMA 把乘加合成一条指令。** `a*b+c` 不走两次 round-trip，中间结果不落寄存器堆外，精度也更好（只舍入一次）。GEMM、卷积、attention 的 QK/PV 全是 FMA 的节拍。AVX512-BF16 的 `_mm512_dpbf16_ps` 更进一步：一条指令做 32 个 BF16 乘加、累加到 16 个 FP32——**输入 BF16、累加 FP32**，既省带宽又保精度（`vllm/csrc/cpu/cpu_types_x86.hpp:997`）。

**AMX 把向量 SIMD 升级成矩阵 SIMD。** 一条 `_tile_dpbf16ps` 完成 16×16×32 的 BF16 矩阵乘加，每拍 ~1024 次 MAC，比 AVX-512 DPBF16 高一个数量级。原理上是"硬件实现的外积法 + VNNI"：8 个 tile 寄存器（16 行 × 64 字节）当矩阵寄存器用，K 维的累加在硬件里完成，软件只需 `tile_loadd` 喂数据 + `tile_stored` 取结果（`vllm/csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp:163`）。代价是要先 `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` 申请许可（`llama.cpp/ggml/src/ggml-cpu/amx/amx.cpp:214`）+ `_tile_loadconfig` 配置 tile 形状。

**VNNI 是 INT8 的"矩阵乘加速器"。** 把 4 个 INT8 拼成 INT32 做点积（`_mm512_dpbusd_epi32`），一条指令 = 4 次乘加。这就是**量化不仅省内存、还能直接提升算力**的硬件根基——INT8 吞吐是 BF16 的 ~4 倍。

---

## 2. GEMM 的本质：用寄存器分块降低访存/计算比

GEMM 的时间是 `O(M·N·K)` 计算量，但访存只有 `O(M·K + K·N + M·N)`。**只要把 K 维做最内层循环、用外积法（rank-1 update），每从内存取一个 A 和 B 元素就能做 RM×RN 次乘加**——访存/计算比随之下降，算力利用率上升。

**寄存器分块大小按寄存器预算反推。** `simd_gemm` 在 AVX-512/NEON 上用 4×4（25/32 寄存器），AVX2 上用 6×2（15/16），标量用 2×2（`llama.cpp/ggml/src/ggml-cpu/simd-gemm.h:12-21`）。这个数字不是拍脑袋：`RM` 个累加器 + 1 个 B 广播寄存器要塞进物理寄存器文件，留一两个给标量尾循环和地址计算。

**AMX 是同一原理的硬件版。** `TileGemm122`（M≤16）和 `TileGemm224`（16<M≤32）按 M 切两种 tile 配置，分别用不同 tile 编号做 A/B/C 的寄存器分配（`vllm/csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp:20-37, 110-129`）。K 维两路展开做软件流水，B 用 `_tile_stream_loadd`（非时序，不污染 L2），A 用普通 `_tile_loadd`（复用率高，值得驻 cache）——**复用率导向的 cache 策略**。

**oneDNN 也要懂它的成本模型。** 创建 `dnnl::matmul` primitive 极贵（要选 kernel、分配 scratch），但 vLLM 的 batch M 一直在变。`W8A8MatMulPrimitiveHandler` 用两级 cache：`ClassMatmulCache`（按 N/K/量化策略）→ `MSizeCache`（按 M/bias），只换 memory ptr 不重建 primitive（`vllm/csrc/cpu/dnnl_helper.h:133-167`）。这是"调用库也要管理库的固定成本"。

---

## 3. 访存层级：L2 当 scratchpad、非时序流、权重预打包

CPU 没有 GPU 的片上共享内存，但 L2 是天然 scratchpad。**把 tile 大小反推成"L2 装得下 Q+K+V+logits+partial_out"，让一次 tile 计算的数据不溢出 L2，避免反复跑 DRAM。** `AttentionScheduler::calcu_default_tile_size`（`vllm/csrc/cpu/cpu_attn_impl.hpp:762`）的公式里有一项 `max_num_q_per_iter * logits_buffer_elem_size`——注释明确说："CPU 与 CUDA 不同，Q@K^T 的 FP32 结果也要驻留 cache"。这是 CPU 版 flash-attention 的核心。

**非时序访存留给"写一次不再读"的输出。** `_mm512_stream_ps` / `_tile_stream_loadd` 绕过 cache 不污染层级，给热数据留出 cache 空间（`vllm/csrc/cpu/cpu_types_x86.hpp:1040-1052`）。vLLM 的 AMX tile kernel 给 B 用 stream load、A 用普通 load，就是基于"B 这一 tile 只用一次、A 在 K 内层会复用"的判断。

**权重预打包把转置/交错/补齐的一次性成本摊到无数次推理里。** `pack_weight`（`vllm/csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp:240`）把权重重排成"每 16 个输出通道一组、组内 4 字节交错"的 VNNI 布局；llama.cpp 的 `repack.cpp`（4836 行）在权重加载时按后端（AMX tile / VNNI / KleidiAI）重打包。代价是首次加载慢、内存翻倍——但推理跑几千次后完全回本。

**HBM 是 CPU 的"显存"。** Xeon Max 有片上 HBM，`memkind` 的 `hbw_posix_memalign` 把权重/KV 分到 HBM，**CPU 后端代码不变就吃到几倍带宽**（`llama.cpp/ggml/src/ggml-cpu/hbm.cpp:24`）。原理：访存受限的 GEMM 在 HBM 上带宽翻几倍 → 直接提速。

---

## 4. 量化的本质：带宽 × 算力双收益

量化加速推理不是因为"算的少了"，而是两条硬件收益叠加：

1. **带宽下降**：FP32→INT8 权重体积 1/4，访存受限的 GEMM 直接提速。
2. **算力上升**：VNNI/AMX-INT8 单拍 MAC 数是 BF16 的 ~4 倍（见 §1）。

**W8A8 的 per-token 动态量化用精度换简单。** 每个激活 token 算自己的 scale，比 per-tensor 精度高一个量级。oneDNN 不直接支持 per-token+AZP，就让 primitive 出 FP32，自己写 `dynamic_quant_epilogue` 做 `scale + azp_adj + bias` 的向量化回写（`vllm/csrc/cpu/dnnl_kernels.cpp:194-300`）。**库的能力边界自己补。**

**FP8→BF16 用位操作捷径，不走 FP32 中转。** E4M3 的 7 位尾数左移 7 位就是 BF16 尾数，符号位左移 8 位对齐，一条指令 `_mm512_or_si512` 完成（`vllm/csrc/cpu/cpu_types_x86.hpp:209-217`）。还有 AMX 专用变体把指数 rebias 折进 k/v scale，连那次乘法都省了。在 attention softmax 这种热点上，省一次 FP32 round-trip 很值。

**量化分派靠宏 + 类型 trait。** `GGML_DISPATCH_QTYPES`（`llama.cpp/ggml/src/ggml-cpu/amx/mmq.cpp:78`）按量化类型分派 `block_q4_0/q4_K/q8_0/…`，每种配一个 `vec_dot_type`（Q4_0 配 Q8_0）——权重低比特、激活 INT8 的混合精度点积由此统一。

---

## 5. Kernel 融合与快速超越函数

**每个融合省一次 round-trip 到内存。** `silu_and_mul` 把 `silu(gate) * up` 融成一次 kernel，省中间 FP32 buffer；`rms_norm`、`rotary_embedding`、`fused_add_rms_norm` 同理。vLLM 把这些注册成自定义 op（`vllm/csrc/cpu/torch_bindings.cpp:289-342`），让 inductor 不去拆它们。

**softmax 的 exp 是热点，用尾数多项式替代 libm。** range reduction 把 x 拆成 `n + r`（n 是整数，r∈[-ln2/2, ln2/2]），对 r 做 4~5 阶多项式逼近，再用 bit trick 把 `2^n` 直接塞进浮点数的指数位（`_mm512_slli_epi32` + `or`）。`ggml_v_expf`（`llama.cpp/ggml/src/ggml-cpu/vec.h:1172`）和 vLLM 的 `DEFINE_FAST_EXP`（`vllm/csrc/cpu/cpu_arch_macros.h:8-58`）都是这套，省几十拍。BF16/FP16 路径还能用 3 阶多项式（1 ULP 内）再快一截（`cpu_arch_macros.h:130-146`）。

**GELU 用查表。** 128KB 预计算表 `ggml_table_gelu_f16[1<<16]`，按 FP16 位直接索引（`llama.cpp/ggml/src/ggml-cpu/vec.h:972`）——把超越函数变成一次内存查表，FP32 输入先转 FP16 再查。表常驻 L1，比多项式还快。

**AMX 上权重只加载一次、做多个专家。** vLLM 的 `cpu_fused_moe` 把 `gate→topk→gemm1→act→gemm2→weighted_sum` 融在一起（`vllm/csrc/cpu/cpu_fused_moe.cpp`），权重预打包后一次流式加载喂给 AMX tile，省掉 MoE 标准实现里多次 kernel launch + 多次激活重读。

---

## 6. 并行与任务调度

**lock-free 任务队列用 atomic counter。** `metadata.acquire_counter()` 一个 `atomic_int64_t++` 当全局任务指针，worker 自旋领取（`vllm/csrc/cpu/cpu_attn_impl.hpp:1540`）。比 centralized scheduler 少一次锁竞争，比 OpenMP `parallel for` 更细粒度——因为任务粒度是 workitem（一个 q-tile × 一段 KV），不是整个 batch。

**barrier 要么交给 OpenMP，要么自旋 + seq_cst fence。** 自旋路径 `atomic_fetch_add(n_barrier)` 到 `n_threads-1` 后翻转 `n_barrier_passed`，其余线程 spin 等翻转（`llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:585-609`）。TSAN 不认独立 fence，改用 dummy `fetch_add(0)`（`ggml-cpu.c:605`）。自旋时用 `_mm_pause` / `__riscv_pause` 降争用省电（`vllm/csrc/cpu/cpu_arch_macros.h:6`）。

**false sharing 必须显式规避。** `AttentionMetadata` 强制 `sizeof % 64 == 0` + 起始地址 `% 64 == 0`，把 `counter` 单独放 cacheline 头，后跟 `_padding1[56]` 把只读字段隔离到下一 cacheline（`vllm/csrc/cpu/cpu_attn_impl.hpp:102-142`）。否则多线程 atomic 自旋会互相 invalidate cacheline，性能掉一个数量级。

**GQA fast-path 把 KV 访存摊到多个 q head 上。** decode 阶段 `q_head_per_kv` 通常 ≤ 16，能塞进一组寄存器，就把多个 q head 一次处理，KV 只读一遍（`vllm/csrc/cpu/cpu_attn_impl.hpp:419-426`）。prefill 时退回 MHA 路径。这是 GQA 模型在 decode 阶段提速的关键。

**长序列 KV split + reduction = flash-attention 的 split-K 搬到 CPU。** 一个线程算不完 KV 时，把 KV 切几段分给多线程，每段产出 `(partial_out, max, sum)`，最后在线做 log-sum-exp 合并（`vllm/csrc/cpu/cpu_attn_impl.hpp:592-635`）。归约阶段用 `volatile bool` flag 标记每个 split 是否完成，最后一个完成的线程触发归约（`cpu_attn_impl.hpp:1493-1506`）。

---

## 7. NUMA、线程绑定、内存分配器

**多 socket 系统上跨 NUMA 访问会走 UPI，带宽掉一半。** 探测 `/sys/devices/system/node/nodeN` 拿拓扑（`llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:627-712`），把线程钉在自己 NUMA node 的核上，权重复制或按 node 分片。`pthread_getaffinity_np` 读 `numactl` 设的 cpuset 作为默认约束。

**`numa_balancing` 内核特性反而拖累性能。** 内核被动迁移页面会导致 cache miss 尖刺，llama.cpp 显式检测 `/proc/sys/kernel/numa_balancing` 并告警（`ggml-cpu.c:703-710`）——**NUMA 优化要主动控制布局，不能交给内核自动平衡**。

**分配器锁是隐藏瓶颈。** vLLM `LD_PRELOAD` 自带的 `libtcmalloc`（`vllm/vllm/platforms/cpu.py:271-291`），因为多线程推理里频繁的小对象分配会让 glibc malloc 的全局锁成为热点。tcmalloc 用 thread-local cache 消掉这个锁。

**PyTorch 的 libgomp 必须预加载。** 否则 PyTorch C++ 扩展只能用满 1 个核——因为运行期加载的 OpenMP runtime 和 PyTorch 主程序里的不匹配，线程绑定失效（`vllm/vllm/platforms/cpu.py:229-267`，对应 issue #27369）。这是"用 PyTorch 做 CPU 推理必踩的坑"。

**`TORCHINDUCTOR_CPP_DYNAMIC_THREADS=1` 避免线程数固化。** inductor 默认会生成 `num_thread(N)` 调用把线程数写死，破坏运行期的线程绑定与 NUMA 亲和性（`vllm/vllm/platforms/cpu.py:223`）。

---

## 8. 运行期分派与可移植性

**"一个变体一个库 + 运行期选最优" 是 CPU 多 ISA 的标准答案。** llama.cpp 在 `GGML_CPU_ALL_VARIANTS` 下按架构 flag 编出多个 `ggml-cpu-<tag>` 后端，运行期按 cpuid 评分选最优（`llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt:20`）。vLLM 走 Python 侧分库加载 `_C / _C_AVX2 / _C_AVX512`（`vllm/vllm/platforms/cpu.py:415-434`）。比 `-march=native`（只能编译机跑）通用，比 JIT 简单。

**特性探测代码必须单独编译 + `-fno-lto`。** 否则 LTO 会把新架构指令内联进"评分函数"，在旧 CPU 上加载后端时还没跑到特性检查就 SIGILL 了（`llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt:12-16`，注释里举的例子是 power9 加载 power10 后端崩）。**运行期分派的前提是探测代码本身在所有 CPU 上都能跑。**

**架构差异用 source-level flag 隔离。** KleidiAI 的 SME ukernel 需要 `+sve+sve2+sme2+fp16`，但其它源文件不需要——用 `set_source_files_properties(... COMPILE_OPTIONS ...)` 只给 KleidiAI 源文件加这些 flag（`llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt:694`）。**一份构建里可以混编不同架构要求的源文件。**

**库后端的能力差异就地消化，不向上层暴露。** oneDNN 的 ACL 后端不支持 `matmul + bias`，调用处改写成 `c.copy_(bias); c += matmul(a,b)`（`vllm/csrc/cpu/dnnl_kernels.cpp:531-554`）。上层 API 不变。

**组合爆炸的分派空间用代码生成。** `(head_dim × ISA × kv_cache_dtype)` 是三维笛卡尔积，手写 switch 会爆炸。vLLM 用 `generate_cpu_attn_dispatch.py` 产出 `CPU_ATTN_DISPATCH` 宏体（`vllm/csrc/cpu/generate_cpu_attn_dispatch.py`）。**元编程对付组合爆炸，手写对付核心热点。**

**降级路径也要写满。** SVE 有 SVE2 时 `svmlalb/svmlalt` 一步 f16→f32 widening FMA；无 SVE2 时用 `svtrn1/svtrn2` 拆奇偶再分别 cvt+fmla（`llama.cpp/ggml/src/ggml-cpu/vec.h:18-44`）。一份源码能在多代 ARM 上跑，靠的是把降级路径当一等公民而非 afterthought。

---

## 9. 一句话浓缩

CPU 推理优化只有两件事：**让算力单元每拍做更多有效计算**（SIMD/FMA/AMX/VNNI/量化），和**让数据在被算力单元需要时已经在最近的存储层级里**（L2 tile/预打包/非时序/融合/HBM/NUMA）。剩下的一切——线程调度、分派、库选型、代码生成——都是为了把这两件事在真实硬件上不打折扣地兑现。