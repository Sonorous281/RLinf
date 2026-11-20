RPent Episode PPO 与 GRPO
=========================

训练 Qwen3.5-4B 工具调用规划器，由远端 LIBERO-PRO 环境执行工具调用。rollout、训练和评测使用同一份完整多模态 SFT checkpoint，并保持 SFT 时的 RPent 工作流。本例采用 ``episode``：显式感知，每次最多顺序执行三个动作。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型

      Qwen3.5-4B；冻结视觉编码器和 projector

   .. grid-item-card:: 算法

      同步 GRPO，或带独立 critic 的 PPO

   .. grid-item-card:: 任务

      LIBERO-PRO Goal、Spatial、Object

   .. grid-item-card:: 硬件

      四台各 8 张 H200 的主机；模型、SAM3 与环境进程共卡

任务
~~~~

GRPO 配置执行 150 次更新，每次按完整组选择训练数据。

.. list-table::
   :header-rows: 1

   * - 项目
     - 配置
   * - 训练池
     - 30 个任务的原生状态；同一任务内去重，保留原始状态编号
   * - GRPO batch
     - 32 组 × 8 条，共 256 条 episode；同组使用同一任务和原生初始状态
   * - 超发
     - 同时启动 40 组 × 8 条，选最先完整结束的 32 组；取消剩余 64 条，确认生成请求已停止，并释放全部 320 个环境后开始训练
   * - 奖励
     - 原生成功为 1，正常失败为 0；全 0、全 1 组均保留
   * - 训练池回归评测集
     - 30 个任务的 seed 0、1，共 60 条训练池 episode

观测与动作
~~~~~~~~~~

配置保持 SFT 的 prompt、工具 schema 和历史保留规则。

.. list-table::
   :header-rows: 1

   * - 项目
     - 约定
   * - 观测
     - 显式获取 1024 像素相机图像；首次观测前的 turn 可以没有图像
   * - 动作
     - ``execute_actions`` 最多顺序执行三个动作；感知工具为 ``view_env``、``segment``、``back_project``，输出工具为 ``finish``
   * - 奖励
     - 以环境成功信号为准；规划器调用 ``finish`` 不代表任务成功
   * - Prompt
     - thinking 开启，不发送 effort 档位；保留推理和图像历史；达到上下文的 70% 或 80 条消息时压缩，保留 10 条
   * - 预算
     - 上下文上限为 262144 个 token，每次最多输出 16384 个 token；最多 50 次请求、10000 环境步、规划器 7200 秒、collector 14400 秒

GRPO 如何训练
-------------

每组按奖励均值和标准差计算 advantage，保留奖励相同的组。GRPO 不使用 critic，也不再做第二次全 batch advantage 归一化。配置使用 group、agent、turn 三级 loss scaling，按有效 token 求均值，clip 下界 0.2、上界 0.28；学习率 1e-6、warmup 为 0，KL 和 entropy 系数为 0，梯度裁剪为 1，每个 batch 学习一遍。实际采样的推理和 tool-call token 参与学习；prompt、工具反馈、图像和 padding 的 mask 为 0。

按完成速度选择完整组可能造成选择偏差。用 sampling manifest 比较选入和取消的组；不同训练 batch 的成功率不能替代固定条件下的 policy 对比。

安装
----

每台 H200 主机使用 Linux x86-64 的 ``reason`` 镜像。保留 ``/opt/venv/reason`` 中的 Python 3.12.12、Torch 2.11.0/CUDA 12.8、torchvision 0.26.0、FA2 2.8.3 和 Transformer Engine 2.17。宿主机需具备 NVIDIA 驱动、CUDA 12.8 开发工具链及 EGL/OpenGL 库。

将两个 ``rpent-ppo`` checkout 放在 ``$STACK_ROOT/repos/RLinf``、``$STACK_ROOT/repos/RPent``，记录各自的 Git revision。完整离线包放在 ``$STACK_ROOT/envs/offline_package``；模型和场景资源分别放在 ``$STACK_ROOT/models``、``$STACK_ROOT/data``。每个节点执行安装，再激活同一个环境：

.. code-block:: bash

   export STACK_ROOT=/mnt/public/xzxuan
   bash "$STACK_ROOT/envs/offline_package/releases/h200-minimal-20261120-v5/install.sh"
   source "$STACK_ROOT/envs/offline_package/releases/h200-minimal-20261120-v5/activate.sh"
   python -m pip check

安装器检查镜像版本与 SHA256，备份被替换的包，再完全离线安装本地 wheel。重复安装会检查已安装包的文件，并保持版本不变。离线包结构与依赖构建说明见 ``requirements/rpent/README.md``。

最小安装包保留镜像的 SGLang 0.5.12.post1、Transformers 5.6.1、NumPy 2.2.0、commit 为 ``dc0ee41ad`` 的 Megatron-Core checkout，以及官方 Bridge 0.5.0。补装 FLA core 0.6.0、MuJoCo 3.3.0、robosuite 1.5.2、SAM3 和配套项目。OpenAI SDK 3.19.0 提供 RPent planner 所需的接口。benchmark wheel 使用已准备的 robosuite 1.5 适配，包括地面 Panda 模型的 ``NullBase`` 声明。Core 和 Bridge 均不打补丁、不重新打包；保留镜像激活脚本设置的 Core 源码路径。安装器保存 pip-check 报告，报告中的元数据冲突还需结合运行测试判断。

RLinf 使用未修改的 SGLang 发布版，保存实际推理的 prompt ID、像素、图片区间、grid 和 M-RoPE，供 learner 回放。早期兼容 wheel 包单独保留为历史版本。

RLinf、RPent planner 子进程、SGLang、SAM3 和 LIBERO 均使用 ``/opt/venv/reason/bin/python``，``tools.rpent.python`` 也设为该路径。安装后的 benchmark 包通过软链接引用共享资源。模型及资源准备见 RPent 的 ``tools/rl/INSTALL.md``。不要替换运行中作业使用的环境。

下载模型
--------

启动服务前准备以下资源。安装 Python 包不会自动下载模型权重或场景资源。

.. list-table::
   :header-rows: 1

   * - 资源
     - 使用方
     - 准备方式
   * - 完整 Qwen3.5-4B SFT 目录
     - RLinf rollout 和 learner
     - 权重、config、tokenizer、chat template、processor；各模型主机使用相同绝对路径
   * - SAM3 checkpoint
     - 环境主机的 SAM3 服务
     - 申请 ``facebook/sam3`` 访问权限，下载 ``sam3.pt``
   * - LIBERO-PRO 资源
     - 环境 worker
     - 场景资源，以及包内 BDDL 文件和原生初始状态

对比 SFT 时，将 ``MODEL_PATH`` 指向原始完整多模态 SFT 目录，目录中需包含 processor 和 tokenizer 文件：

.. code-block:: bash

   export MODEL_PATH=/absolute/path/to/full-sft-checkpoint

基础模型实验则单独下载 Qwen 完整模型：

.. code-block:: bash

   export MODEL_PATH="$STACK_ROOT/models/Qwen3.5-4B"
   hf download Qwen/Qwen3.5-4B --local-dir "$MODEL_PATH"

在环境主机下载 SAM3 checkpoint：

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
   hf download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"

直接控制的 ``episode`` 工作流不使用 Pi0.5 权重。Pi0.5 属于 :doc:`原始规划器示例 <rpent>`；切换到该示例会改变动作工具。使用 episode 工作流做过 SFT 的模型应保持 ``workflow=episode``。

运行
----

启动 SAM3 和环境服务
~~~~~~~~~~~~~~~~~~~~

四个节点各自激活环境，并启动本机服务：

.. code-block:: bash

   source "$STACK_ROOT/envs/offline_package/releases/h200-minimal-20261120-v5/activate.sh"
   python -m robots.rl.services \
     --sam3-checkpoint "$SAM3_CHECKPOINT_PATH" \
     --envs-per-gpu 3 --output-dir "$STACK_ROOT/logs/services-node0"

每次启动使用新的输出目录。该进程统一管理每卡一个 SAM3、端口 28424 的本机 pool 和端口 28490 的环境 backend，退出时关闭自己的子进程。每卡提供三个环境槽位，四个节点可同时运行全部 80 个 smoke 候选。正式的 320 候选配置需要另外验证容量。episode 结束后释放槽位；学习前释放全部环境，SAM3 继续驻留。

四个节点加入同一私有 Ray 集群。启动 Ray 前激活统一环境，并分别导出不同的 ``RLINF_NODE_RANK``，参见 :doc:`/rst_source/guides/multi_node`。worker 使用所在节点的 loopback backend；训练入口只在 head 启动一次。

准备任务和路径
~~~~~~~~~~~~~~

在已安装 LIBERO-PRO 的环境主机上生成任务文件：

.. code-block:: bash

   export TASK_DIR="$STACK_ROOT/tasks"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds all --split train \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds 0,1 --split eval \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"

将任务目录和 manifest 复制到模型主机的共享文件系统。包含 60 条 episode 的评测集是训练池回归，不是泛化评测。

复制路径模板，在仓库外保存部署配置：

.. code-block:: bash

   export CONFIG_DIR="$STACK_ROOT/config"
   mkdir -p "$CONFIG_DIR/deployment"
   cp "$RLINF_ROOT/examples/agent/rpent/config/deployment/episode.yaml" \
     "$CONFIG_DIR/deployment/local.yaml"

填完 ``local.yaml`` 中全部 ``???``。在 ``data`` 下按列表填写刚生成的训练、评测 JSONL 文件：

.. code-block:: yaml

   data:
     train_data_paths: ["/data/tasks/train.jsonl"]
     val_data_paths: ["/data/tasks/eval.jsonl"]

将以下模型路径都指向同一份完整 SFT：

.. code-block:: text

   rollout.model.model_path
   actor.tokenizer.tokenizer_model
   critic.tokenizer.tokenizer_model
   actor.megatron.ckpt_convertor.hf_model_path
   critic.megatron.ckpt_convertor.hf_model_path

``tools.rpent.python`` 指向 planner 解释器，``repo_path`` 指向匹配的 RPent 源码。episode 输出和预处理 cache 在各模型主机上必须有相同绝对路径；只要轨迹或 checkpoint 仍引用图片，就保留这些图片文件。

启动训练
~~~~~~~~

先在 Ray head 运行四节点 smoke test：

.. code-block:: bash

   cd "$RLINF_ROOT"
   source "$STACK_ROOT/envs/offline_package/releases/h200-minimal-20261120-v5/activate.sh"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_h200_smoke_grpo \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

该 smoke 从 20 个候选组中选入 16 个完整组，每组 4 条，并发 80 个环境，完成一次更新后保存 checkpoint。正式配置继续使用 ``train_qwen3.5_4b_episode_sync_grpo`` 的 32/40 组、每组 8 条，须另行配置足够的环境容量。

27B 验收使用以下配置，部署 profile 中的模型和 tokenizer 路径均指向完整 27B checkpoint：

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_27b_h200_grpo_5step \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

每批启动 40 个候选组，每组 8 条 episode，选入最先完整结束的 32 组。完成 5 次 optimizer 更新，每次更新后保存 checkpoint。每张 GPU 配置 10 个环境和 1 个 SAM3 实例。

采用 rollout TP1、actor TP4/CP1/DP8，开启序列并行（SP），关闭多 token 预测（MTP）。logprob 和 LM-head 分块均为 256 token，Core 和 Bridge 保持原样。LM-head 分块与只计算回复位置的 logits 都仅支持 CP1。评估 CP 并行前，设置 ``actor.model.lm_head_chunk_size=0`` 和 ``actor.response_only_logits=false``，再单独验证多模态输入、mask、梯度及损失归一化。

准备好的 ``Qwen3.8-27B`` 目录声明的是 dense Qwen3.5 架构。RLinf 从模型配置读取层数、维度、注意力头数和 embedding 绑定关系。验收时还须恢复保存的 checkpoint，并用恢复后的权重生成新回复。验收结束后停止运行，不自动延长到 150 次更新。

完整训练使用 ``train_qwen3.5_4b_episode_sync_grpo``：40 个候选组中选入最先完整结束的 32 组，每组 8 条 episode。启动前先验证环境服务容量。

actor 使用 TP2/CP1/DP16，rollout 为 32 个 TP1 副本。actor、critic、rollout 保持 ``NCCL_MAX_CTAS=8``，SGLang 显存比例为 0.55。该比例不是共用 GPU 的总显存上限；对比实验时保持相同的 placement 和显存设置。

PPO 可以共用相同的预处理环境，启动时选择 PPO 配置和 TokenSpace 配置：

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_oversample320 \
     +preprocessing@_global_=token_space \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

该 PPO 配置同时启动 320 个独立候选，选最先完整结束的 256 条，使用独立 critic 和按完整 episode 时间顺序计算的 GAE，不使用 GRPO 组内归一化。

新环境首次运行时，先核实完整 optimizer 更新、权重同步和下一批真实在线 episode。一批采样完成或出现单个 TensorBoard tag，均不能证明 optimizer 已完成一次更新。新环境的 GPU 数值一致性及其他驱动、内核版本的兼容性仍需验证。

恢复、导出和评测
~~~~~~~~~~~~~~~~

通过 ``runner.resume_dir`` 从相同配置的完整 checkpoint 恢复。按角色检查模型和 optimizer 分片、scheduler/RNG，以及原子发布的数据状态。数据游标属于当前 epoch，不必等于全局更新数。保留中断的原始 episode；超过最近完整 checkpoint 的内存更新不能视为可恢复。参见 :doc:`/rst_source/guides/resume`。

完成第 150 次更新后，将训练得到的语言参数导出到原始完整多模态模型：

.. code-block:: bash

   python examples/agent/rpent/checkpoint.py export \
     --checkpoint "$CHECKPOINT_150" --vlm "$MODEL_PATH" --output "$EXPORT_PATH"

使用匹配的 model 环境，在准备好的训练池回归评测集上评测导出权重，并沿用训练的 TokenSpace rollout 设置：

.. code-block:: bash

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_episode_token_space \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local \
     rollout.model.model_path="$EXPORT_PATH"

评测使用独立输出目录；保留原始 SFT 和各次 checkpoint。正常失败以及输出、上下文、轮数、时间预算耗尽均为零奖励；只有分类明确的基础设施错误使用有界重试。第二次基础设施尝试仍失败时，应停止接收该批数据并检查原因。

可视化与结果
------------

TensorBoard 记录 optimizer loss、梯度范数、学习率、采样成功率、TIS 诊断和更新耗时。sampling manifest 记录选入组、取消成员和清理回执；成功率分母使用实际选入的 episode，不使用目录数量。episode 文件保留请求、结果、图片引用和训练 turn。

episode 训练默认关闭完整 HTTP 请求、响应和压缩前后输入的抓取。排查 planner 传输问题时设置 ``tools.rpent.record_api_trace=true``；直接运行 RPent 时使用 ``--record-api-trace``。开关不改变 prompt、压缩决策、token mask 或训练记录。

分别报告任务失败、基础设施错误、选入和取消组构成、token/turn 成本，以及准确的源码和依赖版本。60 条训练池回归不能当作独立测试集上的结果。

纯图像控制
----------

pear 配置在每次动作后提供双相机 RGB 和实测机器人状态，沿用 GRPO 算法及每批 32 组、每组 8 条、超发 40 组的设置。仅提供 ``move_to`` 和 ``set_gripper`` 两个工具，``move_to`` 使用绝对位姿。环境判定任务成功或预算耗尽后，episode 自动结束。非法动作回复按正常失败处理，不再请求模型纠正。

沿用上文的环境、完整多模态 planner checkpoint、任务清单和部署配置。启动相同版本的 RPent backend；仅运行 pear 的 backend 不需要 SAM3 端点或权重。在 Ray head 启动：

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_pear_sync_grpo \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

配置默认使用 ``tools.rpent.world_cord=general``。原点在安装基座中心，x 向前、y 向左、z 向上。抓取 TCP 是两指指垫中心的实测中点。初始观测给出桌面相对基座的高度。现有 LIBERO 感知工作流也会按此设置返回几何坐标。设为 ``tools.rpent.world_cord=base`` 可保留仿真世界坐标和原控制点。

thinking 默认开启。添加 ``agentloop.chat_template_kwargs.enable_thinking=false`` 后，要求模型直接调用工具，不输出推理文本。RPent CLI 对应 ``--reasoning`` 和 ``--no-reasoning``。切换工作流或坐标系时，使用独立运行目录，并固定源码版本；原有 SFT 工作流仍保留默认行为。
