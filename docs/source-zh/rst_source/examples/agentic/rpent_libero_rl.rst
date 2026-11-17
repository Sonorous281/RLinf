RPent 多模态强化学习
====================

本示例用 RLinf 训练 Qwen3.5-4B 规划器，通过 RPent 在 LIBERO 中执行六种末端控制工具。规划器接收图像和文字，生成工具调用；RLinf 负责采样与 Actor/Critic 更新。训练更新语言参数，冻结视觉编码器和 projector。

.. important::

   默认图像路径保留采样 token，并校验图像网格与位置；实验性共享多模态预处理路径还校验 processor、pixel values 和 M-RoPE 重放。这些是输入一致性检查，两条路径都不能据此宣称真实 SGLang/Megatron logprob 已通过数值验收。共享多模态预处理默认关闭，依赖版本见安装页。

概述
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型

      完整的多模态 Qwen3.5-4B checkpoint

   .. grid-item-card:: 算法

      同步 PPO 或异步 SAO

   .. grid-item-card:: 工具

      末端执行器控制与 SAM3 分割

   .. grid-item-card:: 硬件

      单机 A100 验证；可选三节点部署

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 工作流
     - 范围
   * - 本页：``libero_rl``
     - 标准 LIBERO 的 40 个任务；seed 1–32 用于训练，seed 0 用于评测。划分的是同一组任务的初始状态。
   * - :doc:`Pi0.5 Planner PPO <rpent>`
     - LIBERO-PRO、Pi0.5、任务记忆和文本观测。使用该页面的配置与安装说明。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 行为
   * - 观测
     - 初始化与动作后返回实测状态；``view_env`` 返回相机图像。
   * - 动作
     - ``move_to``、``set_gripper``、``view_env``、``segment``、``back_project`` 和 ``finish``。
   * - 奖励
     - 以环境原生成功信号为准：成功为 1，失败为 0。规划器调用 ``finish`` 不代表任务成功。
   * - 任务提示
     - 任务指令与所选工具提示词，默认使用 ``simple``。

安装
----

使用 ``Sonorous281/RLinf`` 与 ``Sonorous281/RPent`` 配套的 ``dev/rpent-rlinf-integration`` 分支。先完成 :doc:`rpent_libero_rl_setup`，准备两个 Python 环境、完整模型、SAM3 权重、模拟器资源、任务数据和 Ray 服务。旧 baseline 分支 README 指定的是另一套 RPent 版本，不适用于本部署。

先使用下面的单机 A100 配置。较大的部署使用三台八卡机器，扩展前须测量当前负载的显存和吞吐。RPent 独立采集器使用另一套模型服务环境，默认基准也为 LIBERO-PRO。

准备任务数据
------------

任务生成器输出 ``train.jsonl``，并记录任务 ID、初始状态 seed、数量和哈希。每行包含任务指令 ``prompt``，以及 JSON 字符串形式的 ``answer``：

.. code-block:: json

   {"prompt": "open the bottom drawer", "answer": "{\"suite\": \"libero_goal\", \"task\": 0, \"seed\": 1}"}

``suite`` 和 ``task`` 指定模拟器任务；实际指令由生成器读取。Rollout 保留采样 token ID、behavior logprob、图像、policy 版本与原生结果。Policy loss 只覆盖模型生成的回复 token；工具观测和强制闭合 token 的 loss mask 为 0。

填写部署路径
------------

安装后，将部署路径模板复制到仓库外：

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf-token-space/bin/activate"
   cd "$STACK_ROOT/repos/RLinf"
   mkdir -p "$STACK_ROOT/config/deployment"
   cp examples/agent/rpent/config/deployment/paths.yaml "$STACK_ROOT/config/deployment/local.yaml"

编辑 ``$STACK_ROOT/config/deployment/local.yaml``，把每个 ``???`` 替换为实际绝对路径。``data.train_data_paths`` 和 ``data.val_data_paths`` 填 YAML 列表，分别指向训练与评测 JSONL；其余字段填写日志目录、RPent 解释器、LIBERO 配置目录、episode 目录和完整模型目录。训练、评测和恢复共用这个文件，命令末尾仍可覆盖单次运行的值。保留该文件和 Hydra 输出的最终配置，便于复现运行参数。

运行
----

``train_qwen3.5_4b_libero_rl_sao_a100`` 用于 Batch8/Env4 的小规模验证，已开启共享多模态预处理、Logprob 重算和 TIS。Thinking 默认不设 token budget，需主动配置才能限制。它需要安装页中的实验依赖环境，不能使用默认 SGLang 环境。源码候选版已在 4 张 A100 80 GB GPU 上通过 smoke 测试，完成两步更新，并从 Step 2 恢复至 Step 3。整理后的代码已通过 CPU 和配置检查，尚未重新验证 GPU 训练与恢复。

.. list-table::
   :header-rows: 1

   * - 资源
     - 分配
   * - GPU 0–1
     - Actor 和 Critic，TP2；依次更新，优化器卸载到 CPU
   * - GPU 2
     - SGLang Server，TP1，最多 4 个活跃请求
   * - GPU 3
     - 一个共享 SAM3 服务，以及 4 个环境的 EGL 渲染
   * - 主存
     - 本任务预留 416 GiB，其他进程至少保留 144 GiB 可用空间

磁盘至少预留 400 GB 保存三份 checkpoint，每份约 128 GB，另需留出 episode 图片和预处理缓存空间。配置不自动监测主存和磁盘用量，需由作业调度器监测上述限额。

配置将上下文限制为 16384 Token，每次输出最多 8192 Token，每个 episode 最多 25 轮。这些是验证预算，不代表推荐的效果训练设置。GPU 4–7 不属于此作业。4 个环境共用 SAM3，不能为每个环境单独启动一份。

确认安装页启动的 Ray 集群和 SAM3 服务仍在运行。A100 配置连接 GPU 3 上端口为 18147 的 SAM3。复用已有共享服务时，将 ``tools.rpent.sam3_endpoint`` 设为对应地址，服务启停仍由原所有者负责。

生成 seed 0 的评测清单并选取一局。Smoke、训练和恢复共用同一套 Ray 与 SAM3 服务。Smoke 限制为 3 轮、40 个环境步，每次请求最多生成 1024 Token：

.. code-block:: bash

   "$STACK_ROOT/envs/rpent/bin/python" examples/agent/rpent/prepare_tasks.py \
     --workflow libero_rl --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
     --seeds 0 --split eval --output-dir "$STACK_ROOT/data/libero-rl-eval"
   head -n 1 "$STACK_ROOT/data/libero-rl-eval/eval.jsonl" > "$STACK_ROOT/data/libero-rl-eval/smoke.jsonl"
   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl_smoke_a100 \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local \
     data.val_data_paths="[$STACK_ROOT/data/libero-rl-eval/smoke.jsonl]"

命令末尾可添加 ``--cfg job --resolve``，不加载模型即可检查最终配置。Smoke 输出 ``runner.output_dir/rpent-libero-rl-smoke/eval_results.json``，其中 ``summary.episodes=1``。检查原生终态及 episode 子进程退出；短预算内任务失败不等于采集链路失败。

单 episode Smoke 通过后，先跑两步更新，再用新训练进程恢复：

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf-token-space/bin/activate"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_a100 \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local \
     runner.output_dir="$STACK_ROOT/logs"

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_a100 \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local \
     runner.experiment_name=rpent-libero-rl-a100-resume \
     runner.output_dir="$STACK_ROOT/logs" \
     runner.resume_dir="$STACK_ROOT/logs/rpent-libero-rl-a100/checkpoints/global_step_2" \
     runner.max_steps=3

两条命令都将结果写入 ``$STACK_ROOT/logs``。使用新的实验目录；第二条命令启动新 driver，加载完整 checkpoint 根目录，不能指向其中的 ``actor/hf_checkpoint``。保留 episode 图片和预处理缓存，不覆盖第一次运行的目录。检查 ``async_trace.jsonl``：恢复后应到达 Step 3，Actor/Critic 更新次数为 3/6，发布版本为 3；消费的 sample ID 不得与恢复前重复，两套优化器都从保存的状态继续更新，并生成新的 ``global_step_3/data/data.pt`` 标记。进程正常退出本身不能证明恢复正确。

扩展到多节点
--------------

上面的单机命令已经覆盖训练与恢复。需要扩大负载时，再安装默认环境，并按安装页的可选三节点步骤启动服务：

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_local \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local

同步 PPO 使用 ``train_qwen3.5_4b_libero_rl_ppo``，并按实际资源覆盖 placement 与 batch。SAO 除了异步采集，还使用 DIS loss、SAO GAE，并先更新 Critic、重算 value 再更新 Actor；它不只是同步 PPO 的调度变体。训练前先验证有限 loss/梯度、实际参数更新、权重同步与恢复。短 smoke 不能证明最长 262144-token 工作负载能放入显存。

结果与验证
----------

四卡 A100 候选版本的归档记录包含以下基础设施结果。它们不代表整理后的代码已重新完成 GPU 验证，也不用于判断学习收益。

.. list-table::
   :header-rows: 1

   * - 阶段
     - 实测结果
   * - Batch8/Env4 训练两步
     - Actor/Critic 更新次数为 2/4，发布版本为 2，消费 sample ID 0–15
   * - 新进程从 Step 2 恢复至 Step 3
     - Actor/Critic 更新次数为 3/6，发布版本为 3，消费 sample ID 16–23，无重复消费
   * - Step 3 checkpoint
     - 包含两套优化器状态、模型导出和 rollout 数据状态，共 20 个文件，约 128 GB


.. list-table::
   :header-rows: 1

   * - 证据
     - 检查内容
   * - 输出
     - ``runner.output_dir`` 下的实验目录保存训练日志与 Actor/Critic checkpoint；``tools.rpent.output_dir`` 保存 episode 请求、图像和结果。
   * - 运行正确性
     - 在部署版本上确认 loss/梯度有限、参数确实更新、权重同步正常，以及 checkpoint 可以恢复。
   * - 任务结果
     - ``rpent/success_rate`` 反映训练 rollout 结果。分别报告原生成功数／完成数与基础设施失败；学习收益需用同条件独立评测衡量。
   * - CPU 组件回归
     - 覆盖前缀、mask、轨迹合并、Gloo 数据交换、进程清理和配置组合；未安装 Megatron/SGLang 的测试会跳过，不能计为通过。
   * - 真实 GPU 数值
     - 未验收。默认与共享预处理路径须分别验证固定 96 点（报告 mean/P95/max，0.03/0.10 为诊断参考）、多轮图像及 M-RoPE。重算或 TIS 不等于消除原生 logprob 误差。
   * - Checkpoint 与恢复
     - 源码候选版已完成两步更新及新进程 Step 3 恢复，整理后的代码尚未重新验证 GPU 训练与恢复。须验证两套优化器、更新后权重同步及新进程恢复；SAO 还须核对 buffered/pending 状态与 policy 版本。
   * - 独立效果评测
     - 尚无证明学习收益的结果。比较相同任务、初始状态与预算下训练前后的原生成功率。

同时查看 ``rpent/turns`` 和 ``rpent/infra_retry_rate``。模板设置 ``agentloop.max_infra_retry_rate=1.0``，允许已恢复的重试继续训练；单局重试耗尽仍报错。此前针对强制闭合 token 的数值检查超出诊断容差，不能作为本部署的验证依据。

恢复与评测
----------

恢复训练
~~~~~~~~

使用 ``runner.resume_dir=/path/to/checkpoints/global_step_N`` 从完整 SAO checkpoint 恢复两套模型/优化器与 buffered/pending rollout 状态。只选择包含 actor/critic 状态及最后写入的 ``data/data.pt`` 标记的完整 checkpoint。未完成任务的 prompt 会重新入队，模拟器不会从中断位置继续运行；恢复时保持数据集、算法、world size 和引用图像兼容。同步 PPO checkpoint 不包含 SAO 所需的队列状态。

独立评测
~~~~~~~~

在另行分配的资源上运行下方命令。评测默认使用一台八卡模型节点和一台独立环境节点，须按实际资源调整 placement。

评测模板使用 Planner 采样参数：temperature 为 1、top-k 为 20、top-p 为 0.95、presence penalty 为 1.5；训练使用全词表采样。初始模型与训练后模型的评测参数必须一致，并显式对齐任务、episode 预算和历史窗口。A100 Smoke 只检查基础设施，不用于判断学习收益。RPent 独立采集器使用另一套模型服务流程。

导出时 ``--vlm`` 必须指向训练初始化所用的完整模型，以保留冻结的视觉权重和 processor。从 SFT 开始训练时相应替换路径。导出目录必须是新目录；HF 导出只用于推理，不能替代包含优化器和队列的恢复 checkpoint。

.. code-block:: bash

   # Export a completed checkpoint into a new full-model directory.
   python examples/agent/rpent/checkpoint.py export \
     --checkpoint /path/to/checkpoints/global_step_5 \
     --vlm "$STACK_ROOT/models/Qwen3.5-4B" \
     --output "$STACK_ROOT/models/policy-step5"

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local \
     runner.output_dir="$STACK_ROOT/logs/eval-step5" \
     rollout.model.model_path="$STACK_ROOT/models/policy-step5" \
     tools.rpent.output_dir="$STACK_ROOT/data/eval-episodes-step5"

评测初始模型时，只改变模型与输出路径，复用同一份 manifest、任务/初态 ID、prompt、采样参数、轮数/动作预算及压缩设置。分别报告原生成功数/总数、轮数耗尽和基础设施失败。训练完成队列的成功率不是独立评测；固定 40 条 case 也不足以确认很小的学习增益。

选择训练配置
------------

.. list-table::
   :header-rows: 1

   * - 配置名（不含 ``.yaml``）
     - 用途
   * - ``train_qwen3.5_4b_libero_rl_ppo``
     - 同步 PPO
   * - ``train_qwen3.5_4b_libero_rl_sao``
     - 按 episode 完成顺序调度的 SAO
   * - ``train_qwen3.5_4b_libero_rl_sao_3node``
     - 三节点 SAO placement
   * - ``train_qwen3.5_4b_libero_rl_sao_local``
     - 需填写本地路径的三节点 SAO
   * - ``eval_qwen3.5_4b_libero_rl``
     - 独立多模态评测

启动器默认仍为原版 ``libero``，必须显式传入 ``libero_rl`` 配置名。PPO → SAO → 三节点 → local 配置依次继承公共设置。

SAO 使用 DIS actor loss（clip 边界 0.3/5.0）、``sao_gae``\ （alpha 为 1.5）、SAO critic loss 和采样时记录的 behavior logprob。:doc:`完全异步 Search-R1 <searchr1/async>` 使用独立推理卡，更新调度不同。

更新耗时较长时，可设置 ``rollout.drain_before_training=true``，先完成在途 episode 再暂停生成。Buffer 容量必须覆盖在途上限；该选项不改变任务超时预算。

重算与 TIS、Thinking 预算、共享多模态预处理、缓存和预取的设置见 :doc:`../../guides/agent-workflows/rpent-multimodal`。基础 SAO 配置默认 ``recompute_logprobs=false``、``importance_sampling_fix=false``，直接使用采样 logprob，不校正后端差异与 policy 滞后。扩批前需在目标工作负载上验证这一选择。 本页 A100 验证配置已开启重算与 TIS。

保留复现记录
------------

.. list-table::
   :header-rows: 1

   * - 记录
     - 内容
   * - 代码
     - RLinf、RPent 的完整 commit SHA；测试未提交代码时另存源码差异。
   * - 环境
     - Python、Torch、CUDA、SGLang、Megatron Core/Bridge 与内核版本，模型目录和版本。
   * - 工作负载
     - 任务 manifest 及哈希、任务 ID、seed、split 和最终解析的 Hydra 配置。
   * - 资源
     - GPU 型号、数量和 placement，显存／主机内存峰值与运行时间。
   * - 证据
     - 分别记录 CPU、GPU 输入／logprob、更新、同步、新进程恢复和独立评测结果，保留日志及失败记录。

.. toctree::
   :hidden:

   rpent_libero_rl_setup
