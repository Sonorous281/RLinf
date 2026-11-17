共享 Qwen3.5 图像预处理
=======================

共享预处理让 SGLang rollout 与 Megatron 训练复用相同的图像输入。本指南适用于已经接入图像观测、并在多轮对话中保留采样 Token ID 的 Qwen3.5 智能体工作流。

安装预处理依赖
--------------

默认图像路径使用模型的 Hugging Face Processor。共享预处理需要主动启用 SGLang TokenSpace，并使用固定版本的实验依赖。机器需有兼容 CUDA 13 的驱动，以及源码编译所需的 Rust。在独立环境中安装：

.. code-block:: bash

   bash requirements/install.sh agentic --sglang token-space --venv .venv-token-space

命令安装固定提交的 SGLang 及配套 Megatron 依赖，默认的 agentic 安装方式保持不变。

配置回放与缓存预算
------------------

在已有的 Qwen3.5 Megatron 智能体配置中设置以下配置项。将流水线并行和上下文并行规模均设为 1。缓存目录需替换为 rollout 与 learner 均可访问的绝对路径：

.. code-block:: yaml

   rollout:
     sglang:
       enable_shared_multi_modal_processor: true
       enable_token_space_preprocess_cache: true
       token_space_preprocess_cache_dir: /absolute/shared/image-cache
       token_space_preprocess_cache_trust_url: true
   actor:
     expect_multi_modal_inputs: true
     model:
       load_mtp: false
       lm_head_chunk_size: 256
     multimodal:
       cache_bytes: 268435456
       prefetch_batches: 2
       prefetch_bytes: 536870912
   critic:
     expect_multi_modal_inputs: true
     multimodal:
       cache_bytes: 268435456
       prefetch_batches: 2
       prefetch_bytes: 536870912

示例为每个 learner rank 分配 256 MiB 图像预处理缓存和 512 MiB 预取预算。两个 actor rank 和两个 critic rank 的缓存与预取预算合计为 3 GiB，模型状态、原图和 Processor 临时空间另计。预算为零时关闭相应功能。

分块评分需要显式关闭 MTP；可以计算熵指标，但熵奖励系数必须为零。训练与 rollout 两端都应冻结视觉编码器和投影层的参数。

检查输入一致性
--------------

rollout 记录实际使用的 prompt ID、图像网格、Processor 配置、像素校验值和 M-RoPE 校验值。文件、像素或位置编码发生变化时，训练端回放会报错。例如，第二轮加入图片后，第一轮的采样 ID 和已展开图像区间应保持不变，仅展开新图片对应的区间。

磁盘缓存保存的是图像预处理结果，learner 仍执行视觉编码器前向计算。输入相同不能证明 SGLang 与 Megatron 的视觉 Embedding 或最终 Logprob 相同；视觉 Embedding 和最终 Logprob 需要分别验证，比较时使用相同的冻结权重和 Token。

.. warning::

   可信缓存 URL 必须指向内容不变的文件。只要待训练轨迹或 checkpoint 仍引用图片，就应确保图片和缓存可访问。共享预处理支持图片，不支持视频和音频。

其余训练选项见 :doc:`../agentic_config`。
