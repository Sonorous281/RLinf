配置 RPent 多模态训练
============================

先完成 :doc:`单机 smoke <../../examples/agentic/rpent_libero_rl>`，再按需要调整重算、缓存和历史处理。默认路径与实验性共享多模态预处理路径使用独立依赖环境。

启用 Logprob 重算与 TIS
--------------------------

SAO 默认关闭训练端 Logprob 重算和截断重要性采样（TIS）。试用这两个选项时，在 SAO 配置中加入：

.. code-block:: yaml

   algorithm:
     recompute_logprobs: true
     importance_sampling_fix: true
     importance_sampling_clip: 2.0
   rollout:
     return_logprobs: true
     max_weight_staleness: 1

每个 batch 保留采样时的 logprob 和逐 token 的 policy 版本。actor 更新前，训练端用当前 actor 权重重算 logprob。TIS 限制后端数值差异和 policy 滞后对更新的影响。DIS 损失按概率比筛选参与训练的 token。计算概率比时，以重算 logprob 所用的 actor 权重为基准。按上述配置，如果 episode 中任一 token 的采样权重比最新发布的权重落后超过一次更新，就丢弃整条 episode，并重新采样该任务。

.. warning::

   开启这些选项后，SAO 的损失计算方式会改变，并额外执行一次 actor 前向计算。扩批前，先用小规模运行检查 TIS 权重的截断比例、有效样本量和 policy 滞后。启用这些选项后的优化器更新、权重同步和 checkpoint 恢复尚未验证。logprob 重算和 TIS 不能证明 SGLang 与 Megatron 的数值一致。

试用共享多模态预处理（实验性）
--------------------------------

共享多模态预处理（Shared Multimodal Preprocessing）复用 SGLang TokenSpace，让 rollout 和训练使用同一套图像预处理。它核对推理服务实际使用并返回的 prompt token ID，从带校验和的文件重建图像，并在训练模型的语言输入处核对 M-RoPE。此模式支持 Qwen3.5 图像输入，使用 PIL 预处理和 Megatron-Bridge，要求 PP=1、CP=1。文字和生成回复中不能包含视觉特殊标记；这条路径不支持视频和音频。

.. warning::

   这条路径仍处于实验阶段。默认的 SGLang 0.5.12.post1 环境不能运行此模式。候选依赖文件 ``requirements/agentic/sglang-token-space.txt`` 固定了 SGLang commit ``18a44326f6de5134fe921927d760d81f8ee48fc9``、Torch 2.14.1 和 Transformers 5.19.0。请准备独立环境，并重新匹配 CUDA、Megatron-Bridge 等训练内核；仍需在这个环境中联合验证 GPU 推理服务、权重同步和训练。保留现有训练环境。

   这个开发版仅用于启用共享多模态预处理的 rollout。其他 rollout 路径以及独立的奖励模型、具身推理服务会在启动前拒绝该版本。这些任务应使用受支持的 SGLang 发布版，不能因为安装成功就认定兼容。

受支持的 SGLang 发布版统一使用 ``rollout.cuda_graph_max_bs`` 配置项。SGLang 0.5.19 更改了对应服务参数的名称，RLinf 会完成映射，并将实际取值限制在 ``rollout.max_running_requests`` 以内。关闭共享多模态预处理时也适用这些规则。Search-R1 和具身任务的 GPU 回归尚未完成。

新环境通过验证后，在部署配置中启用以下设置。从 rollout 环境中移除 ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE``，同时检查启动 shell 是否继承了这个变量。rollout、actor 和 critic 节点须能通过相同路径访问 ``output_dir``。

.. code-block:: yaml

   rollout:
     sglang:
       enable_shared_multi_modal_processor: true
   tools:
     rpent:
       store_images_on_disk: true
       keep_episodes: all

公开开关为 ``rollout.sglang.enable_shared_multi_modal_processor``，默认关闭。旧配置项 ``rollout.sglang.enable_token_space_processor`` 仍作为兼容别名接受，行为保持一致。新旧字段同时存在会报错，即使取值一致也不例外；迁移时直接替换旧配置项。使用当前训练模板的命令改为 ``rollout.sglang.enable_shared_multi_modal_processor=true``。评测模板没有预设这个字段，命令中使用 ``+rollout.sglang.enable_shared_multi_modal_processor=true``。

这个开关仍映射到 SGLang 原生的 ``enable_token_space_processor`` 参数。下文的 SGLang 缓存字段、``sglang-token-space-v1`` 轨迹格式和 checkpoint 模式标记保持不变。仅修改配置项名称，不会改变 checkpoint 的输入约定。

缺少 SGLang TokenSpace 接口时，rollout 初始化会报错。Server 和 Engine 都必须返回实际使用的 prompt token ID。采样前先预处理一次图像，再按展开后的 prompt 长度计算输出和思考的 token 预算。这一步对吞吐的影响尚未测量。

切换模式时使用全新的 rollout 状态。异步 checkpoint 恢复会拒绝不同模式，训练端也会拒绝 Processor 指纹变化或混合图像格式。Torch 和内核升级后，不能直接假定旧 optimizer 状态仍然兼容。恢复 checkpoint 前须保留其引用的图像文件。

训练验收前，先在部署环境验证多轮图像、历史压缩和打包后的 M-RoPE。再按固定的 96 点方案比较 logprob，记录平均值、P95 和最大绝对误差，以 0.03/0.10 为诊断参考。验收还需确认 Token、图片和位置输入正确，更新数值有限，权重同步及新进程恢复成功；重算和 TIS 不能证明原生数值一致。基础模型 logprob 与经过采样惩罚、top-k、top-p 修改的概率须分开比较。CPU 输入检查不能证明 GPU 数值对齐。

减少重复图像预处理
~~~~~~~~~~~~~~~~~~

此模式可以让 rollout、actor 和 critic 共用图像预处理缓存。缓存保存处理后的像素和网格信息。rollout 和训练端仍各自运行冻结的视觉编码器，不共享视觉 embedding。

共享预处理缓存和 CPU 预取默认都关闭。启用时，在上面的部署配置中加入以下内容。将 ``<cache-dir>`` 替换为本作业专用缓存目录的绝对路径。所有作业进程都必须能用同一路径访问缓存目录。

.. code-block:: yaml

   rollout:
     sglang:
       enable_token_space_preprocess_cache: true
       token_space_preprocess_cache_dir: <cache-dir>
       token_space_preprocess_cache_trust_url: true
       token_space_preprocess_cache_check_rate: 0.0
   actor:
     multimodal:
       cache_bytes: 0
       prefetch_batches: 2
       prefetch_bytes: 268435456
   critic:
     multimodal:
       cache_bytes: 0
       prefetch_batches: 2
       prefetch_bytes: 268435456

actor 和 critic 可以分别设置预算。同一配置项的新旧字段均未填写时，取值默认为 0，对应的缓存或预取功能关闭。``tools.rpent`` 下的旧字段 ``training_image_cache_bytes``、``training_image_prefetch_batches`` 和 ``training_image_prefetch_bytes`` 仍同时作用于两个训练角色。同一配置项的新旧字段同时存在时，取值必须一致，否则配置校验报错。独立评估无需配置 actor 或 critic。 示例模板已改用训练角色下的字段。通过命令行设置旧字段时，使用 Hydra 的 ``++`` 语法，例如 ``++tools.rpent.training_image_cache_bytes=268435456``。

缓存要求图片路径指向的内容保持不变。缓存按处理器指纹分目录保存预处理结果。抽查比例的取值为 0 到 1，提高该值会重算部分命中的内容，检查缓存是否一致。各进程须使用相同设置。

预取配置限制每个训练 rank 的微批次数和总字节数，预取队列和当前在用的微批次都计入上限。训练线程执行前向计算，单个 CPU 线程准备后续批次。示例中的 256 MiB 不包含处理器临时内存，也不包含单独配置的 ``multimodal.cache_bytes`` 缓存。规划主存时，将每个 rank 的预算乘以 actor 和 critic 的 rank 总数。提高预算前，先测量实际内存峰值。单个微批次超出预算会报错。预取要求 TokenSpace 轨迹记录处理后的像素字节数。重放缺少该字段的旧轨迹时，将 ``multimodal.prefetch_batches`` 设为 0。

.. warning::

   缓存读取会反序列化 Python 对象，目录只能由可信进程写入。原生磁盘缓存没有淘汰机制或容量上限，须使用作业专属目录，并配置文件系统配额或外部容量监控。所有作业进程退出后才能清理缓存；checkpoint 恢复依赖的图片文件须保留。新进程从保存的图片引用重建 CPU 预取队列。

CPU 测试覆盖缓存复用、图片校验、预取顺序和线程清理。GPU 上的 logprob、训练梯度和吞吐，以及在新进程中恢复训练的行为，仍需在部署环境验证。每项优化须与同一后端关闭该选项的结果比较；跨后端的 96 点误差单独报告，不能用算法校正后的结果代替。

可选运行设置
------------

训练和评测默认设置 ``rollout.sglang.disable_overlap_schedule=true``，供 presence penalty 和可选 Thinking 限制读取当前 CPU token 历史。修改前需在目标 SGLang 版本验证采样 logprob 与强制 token 对齐。

``agentloop.chat_template_kwargs.enable_thinking=false`` 可关闭 thinking；``agentloop.prompt=simple_brief`` 则通过提示词引导模型将每次决策的推理控制在约 256 个 token，不设置解码器的硬性上限。切换前固定模型和任务 seed，从新 episode 对比原生成功率与成本；不能删除已有轨迹中的 thinking token 或 behavior logprob。

若要限制 chat template 已开启的首段 Thinking，设置 ``agentloop.thinking_token_budget=512`` 和 ``rollout.sglang.enable_custom_logit_processor=true``，并保持 ``enable_thinking=true`` 和 ``rollout.sglang.disable_overlap_schedule=true``。启用预算时，配置校验会拒绝重叠调度，避免 CPU token 历史落后于实际解码。预算默认是 ``null``。引擎记录实际强制插入的闭合 Token，该 Token 的训练 mask 为零；自然生成的闭合 Token 仍参与训练。此选项不限制后续重新开启的 Thinking 段。组装轨迹时必须保持返回的 Token ID、behavior logprob、强制标记和权重版本的对应关系。

设置 ``actor.model.lm_head_chunk_size=256``，按块计算参与 loss 的回复 token 的输出层结果；默认 ``0`` 不分块。反向传播时会重新计算这些结果，支持 PP1/CP1 的 MBridge Qwen3.5，包括 TP 和 sequence parallel。使用安装页记录的 Megatron 版本。Entropy 只作为 detached 诊断指标，不支持 entropy loss bonus；该模式也不支持 FP8、MTP、延迟 embedding 梯度、muP 和 autocast。这些约束界定受支持的执行方式，不代表已有吞吐或学习收益。

训练默认使用 SGLang Server，设置 ``rollout.sglang.backend_type=engine`` 可切换到进程内后端。评测默认使用 Engine。两个后端都保留图像、Thinking 预算标记和 policy 版本；Server 作为子进程运行，只绑定回环地址。

工具结果和图片保留在后续输入中。Policy loss 只作用于 policy 生成的 assistant token，user/tool/history token 不参与监督。逐 token 前缀兼容的连续 turn 合并，压缩或历史改写后分段。Actor 和 critic 使用相同的 response 位置映射；共享轨迹 packing 在任意一个计分 token 重叠时都会保留独立行，避免合并覆盖学习目标；这也适用于纯文本 SearchR1。回归测试比较 packing 前后的 loss 和梯度。response-only logits 要求 CP1。

启用 ``tools.rpent.store_images_on_disk=true`` 时，所有 rollout 和 learner 进程必须能读取 **相同绝对路径下的图像** 。``tools.rpent.output_dir`` 需要在模型节点与环境节点间共享，并在轨迹或保存的 checkpoint 仍引用它时保留；不同节点上同名的本地目录不够。没有共享图像存储时保留默认 ``false``，让轨迹携带图像字节，但内存成本更高。不要清理可恢复 checkpoint 所需的 Episode 产物。

保留多轮图像的 token 边界
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

合并 Qwen3.5 图像 turn 时，在每台模型节点启动 Ray **之前** 启用保留 token 的 processor：

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

adaptive 训练和评测模板也通过 ``rollout.env_vars`` 传入此变量，使 worker 启动不只依赖终端环境。

该配置使用 SGLang 的外部 processor 注册入口。图像请求携带已展开的 prompt IDs 和原始图片；适配器核对图片数量及各图 pad 段与处理后 grid 的对应关系，保留原始文本和动作 IDs，再由 SGLang 根据这一序列计算 offset 和 M-RoPE。对已采样 token 解码后重新编码，即使文本相同，也可能改变 token 边界。字符串请求保持上游行为；该图像适配器会拒绝携带视频、音频或预计算图像输入的 token-ID 请求。

使用包含 config、tokenizer 和 processor 文件的本地模型目录运行回归：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES='' SGLANG_USE_CUDA_IPC_TRANSPORT=0 \
     RLINF_TEST_QWEN_VL_MODEL=/path/to/Qwen3.5-4B \
     PYTHONPATH=.:tests/unit_tests \
     python -m pytest tests/unit_tests/test_sglang.py -k qwen_image_processor

这些 CPU 检查覆盖输入 IDs、图像 tensor、offset，以及 HF 与 SGLang 的 M-RoPE 一致性。部署新版本前，还需要验证真实引擎的生成和 logprob、完整 learner 更新及 checkpoint 恢复。
