在 Megatron 训练中复用 SGLang 的图像输入
========================================

Qwen3.5 连续看两张图时，第二轮要接着第一轮实际生成的 token 往下走。图像回放会保存 SGLang 每轮看到的 token 序列、预处理后的像素、图像网格和 M-RoPE 位置。Megatron 读取这些输入，再运行自己的视觉编码器。

用一张卡验证两轮看图
--------------------

准备一张空闲的 A100 80 GB 和本地 Qwen3.5-4B 权重。激活 agentic 环境后，在仓库根目录运行下面的命令。第一次使用时可先看 :ref:`环境准备 <qwen35-replay-setup>`。

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python examples/agent/qwen35/replay_images.py --model /path/to/Qwen3.5-4B

脚本会生成两张测试图，启动本地 SGLang 服务。看到 ``Turn 2 passed``，表示第二轮保留了第一轮的采样历史，训练侧读出的像素、网格和位置也通过了校验。日志里的 ``tokens`` 是到当前轮为止的输入和输出总长度，每轮最多只生成 1 个新 token。

用两张卡验证参数更新
--------------------

把卡号换成两张空闲的 A100 80 GB。同一台机器上也可以用八张卡，只需将卡号改为 ``0,1,2,3,4,5,6,7``。

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=4,5 bash examples/agent/qwen35/run_replay_update.sh /path/to/Qwen3.5-4B

脚本启动独立的 Ray 实例，采集 4 组两轮对话，再把同一批 8 行样本分别交给普通输出层 Actor、分块输出层 Actor 和标量 Critic。检查 ``dense``、``chunked``、``critic`` 三组输出：每个 rank 的 ``loss`` 应是有限值，``grad_norm`` 应为有限正数，``changed`` 应大于零。``changed`` 统计抽查的权重张量里有多少个元素发生了变化。

这个示例使用固定的优势值和回报。批量和并行参数见 ``examples/agent/qwen35/config/replay_update.yaml``，它继承 Search-R1 的 Megatron PPO 配置。两卡使用 TP=2、DP=1，八卡使用 TP=2、DP=4。

接入自己的 Agent
----------------

在 rollout worker 的环境中注册图像处理器，然后再启动 SGLang。上面的示例脚本已经完成了这一步。

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

在训练配置中开启图像回放和采样 logprob。使用 Critic 时也要设置 ``critic.expect_multi_modal_inputs=true``。训练侧要求 PP=1、CP=1；采样侧的 rollout worker 和本机 tokenizer 进程需要访问同一个临时目录。

.. code-block:: yaml

   rollout:
     return_logprobs: true
     sglang:
       enable_shared_multi_modal_processor: true
   actor:
     expect_multi_modal_inputs: true

以先看图 A、再看图 B 为例：

1. 第一轮传入图 A。新图在 prompt 里先占一个图像 token，``history`` 设为 ``None``。采样返回后，保存展开后的 prompt、实际生成的 token 和图 A 的区间。
2. 第二轮从上轮返回的 ``prefix`` 继续，追加新文本和图 B 的占位 token。按 A、B 的顺序传入两张原图，图 A 沿用上轮的区间，图 B 传入新区间。SGLang 会重新处理图 A，并与上轮记录核对。

每轮生成后各存一行训练样本。下面的 ``prefix``、``spans`` 和 ``history`` 留给下一轮使用：

.. code-block:: python

   result = await self.generate(
       prompt_ids, image_data=images, image_spans=spans, image_history=history,
       image_store_dir=shared_directory,
   )
   turn = AgentLoopOutput(
       prompt_ids=result["prompt_token_ids"],
       response_ids=result["output_ids"],
       response_logprobs=result["logprobs"],
       response_images=result["multimodal_input"],
   )
   prefix = result["prompt_token_ids"] + result["output_ids"]
   spans = result["multimodal_input"]["image_spans"]
   history = result["image_history"]

相同的 token 前缀可能对应不同图片，因此每行都要保留自己的 ``response_images``；纯文本行填 ``None``。设置 ``image_store_dir`` 后，相同的像素只保存一次。只要还有样本或 checkpoint 引用这些文件，就要让训练进程能读到它们。省略这个参数时，轨迹直接保存像素张量。

查看一行图像样本
----------------

下表列出 ``result["multimodal_input"]`` 的字段。图片按它们在 prompt 中出现的顺序排列。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 字段
     - 训练时怎么用
   * - ``format``、``prompt_token_ids``
     - 格式标识为 ``rlinf-qwen-image-inputs-v1``。整数列表保留本轮采样用的完整 prompt，其中的图片占位符已经展开。
   * - ``image_spans``、``image_grid_thw``
     - 每张图对应一个左闭右开区间和一行时间、高度、宽度网格。区间只含图像 token，两端的视觉分隔符不计入。模型在回复里生成的图片标记按文本处理。
   * - ``mrope_positions``、``mrope_position_delta``
     - 前者记录整个 prompt 的三轴位置，形状为 ``[3, prompt 长度]``；后者是形状为 ``[1]`` 的偏移量，用于计算后续生成 token 的位置。
   * - ``pixel_values``
     - 随轨迹保存的预处理像素。训练端读取时转为 BF16。
   * - ``pixel_refs``、``pixel_values_bytes``
     - 使用共享目录时，每张图用文件路径和 SHA-256 引用代替内联像素。字节数用于预估训练侧 BF16 像素张量的大小。

实测结果
--------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 检查项
     - 结果
   * - 两轮输入回放
     - 4 组对话，共 8 轮。采样历史、BF16 像素、网格和 M-RoPE 位置均通过校验。
   * - 参数更新
     - 同一批 8 行样本在 8 张 A100 80 GB 上完成了三次独立更新，两个 Actor 和标量 Critic 各一次。两卡配置也通过了这三项检查。
   * - 两端的 logprob
     - 另取 6 个请求，每个请求比较 16 个生成 token。冻结权重后，SGLang 与 Megatron 的 logprob 绝对差均值为 0.058638，最大值为 0.763876，96 个位置中有 17 个超过 0.10。

前两项验证输入能正确重放并用于更新参数。两端的打分仍有差异；策略成功率需要单独评测。

.. _qwen35-replay-setup:

环境准备
--------

已验证的组合是 Qwen3.5-4B、SGLang 0.5.12.post1 发布包和 Megatron Bridge 0.5.0，仅覆盖图片输入。A100 实测使用 PyTorch 2.11.0+cu128、Transformers 5.6.1、Megatron Core 0.18.0 和 Transformer Engine 2.17。Qwen3.5 的 Gated Delta 层需要 ``fla-core==0.6.0``。

已有上述环境可以直接跑示例。新建环境时，在仓库根目录执行：

.. code-block:: bash

   git clone --depth 1 --branch core_v0.18.0 https://github.com/NVIDIA/Megatron-LM.git Megatron-LM-0.18.0
   MEGATRON_PATH="$PWD/Megatron-LM-0.18.0" bash requirements/install.sh agentic --sglang 0.5.12.post1 --venv .venv-agentic
   source .venv-agentic/bin/activate

其他训练选项见 :doc:`../agentic_config`。
