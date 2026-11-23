RPent Episode 训练
==================

使用同步 GRPO 或 PPO 训练 Qwen3.5-4B 规划器。RLinf 负责模型推理、规划和训练。各仿真器在独立环境中运行，通过 RPC 提供观察、实测动作和原生奖励。

安装
----

使用 Linux ``reason`` 镜像和配套的 RPent、RLinf checkout。镜像提供 Python 3.12、Torch、SGLang、Megatron-Core、Bridge、Transformer Engine 和 FlashAttention 2。在镜像环境中安装 RPent planner 依赖，仿真器按 `RPent 指南 <https://github.com/zhexuanxu/RPent/blob/rpent-ppo/docs/source-zh/rst_source/simulators/rl.rst>`_ 分别安装。

.. code-block:: bash

   source /opt/venv/reason/bin/activate
   export MEGATRON_PATH=/opt/venv/reason/Megatron-LM
   (cd "$RPENT_ROOT" && uv pip install --python "$VIRTUAL_ENV/bin/python" \
     -r requirements/episode-planner.txt)
   uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e "$RPENT_ROOT"

主栈使用包含 tokenizer 和图像预处理器的完整多模态权重。仿真端准备任务资产和可访问的 SAM3 服务。主栈和仿真端使用同一 RPent 版本。

准备任务清单
------------

训练 JSONL 的每行包含文本 ``prompt`` 和字符串 ``answer``，后者由仿真任务 JSON 编码得到。例如一条 RoboTwin 任务可以这样写。

.. code-block:: json

   {"prompt":"Adjust the bottle","answer":"{\"environment\":\"robotwin\",\"task_name\":\"adjust_bottle\",\"seed\":100000,\"task_config\":\"demo_randomized\"}"}

LIBERO 使用 ``suite``、``task`` 和 ``seed``。RoboCasa 和 RoboDojo 使用 ``task_name`` 和 ``seed``。RoboDojo 的 ``seed`` 选择 ``Assets/Eval_Layout`` 中已安装的组，``layout`` 指定组内索引。在 LIBERO 环境中运行 ``examples/agent/rpent/prepare_tasks.py`` 生成原生任务池，其余仿真器各提供一份任务清单。

启动训练
--------

每类仿真先启动一个 RPent 工具服务。下面沿用四环境示例中的四个地址，在八张 GPU 上分配训练和模型 rollout。仿真服务独立运行，GPU placement 和后端槽位数按实际资源配置。

如果平台尚未提供 Ray 集群，在 RLinf 主机上运行 ``ray start --head``。把它打印的 head 地址设为 ``RAY_ADDRESS``，使驱动和所有 worker 接入同一集群。配置中的 placement 范围选择训练集群里的物理 GPU。

.. code-block:: bash

   cd "$RLINF_ROOT"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_mixed_grpo \
     cluster.num_nodes=1 \
     cluster.component_placement.actor=0-7 \
     cluster.component_placement.rollout=0-7 \
     cluster.component_placement.critic=0-7 \
     cluster.component_placement.reward=0-7 \
     cluster.component_placement.agentloop=0-7 \
     rollout.model.model_path="$MODEL_PATH" \
     tools.rpent.repo_path="$RPENT_ROOT" \
     tools.rpent.output_dir="$RUN_DIR/episodes" \
     runner.output_dir="$RUN_DIR" \
     env.rpent.libero.train_data_paths="$TASK_DIR/libero.jsonl" \
     env.rpent.robotwin.train_data_paths="$TASK_DIR/robotwin.jsonl" \
     env.rpent.robocasa.train_data_paths="$TASK_DIR/robocasa.jsonl" \
     env.rpent.robodojo.train_data_paths="$TASK_DIR/robodojo.jsonl"

将各 ``env.rpent.<simulator>.endpoint`` 设为 planner 可以访问的工具服务地址。每项可选配 ``rollout_episodes`` 和 ``sampling_episodes``。``data`` 中的两个 batch size 按组填写，每环境配额按轨迹填写。全局有效 32 组、候选 40 组且每组 8 条时，四环境默认各用 64 条、候选 80 条。显式设置某项配额时，为所有启用环境填写该项，总和与全局轨迹数一致，每项都能组成完整组。

每次更新从各任务池取样，同组共享任务和初态。各仿真器按完成顺序选够自己的完整组，正常失败照常计入。环境和生成队列完成清理后再学习。

单环境使用 ``train_qwen3.5_4b_episode_sync_grpo``，通过 ``data.train_data_paths`` 和 ``tools.rpent.remote_toolkit_endpoint`` 提供任务与地址。续训将 ``runner.resume_dir`` 指向最近的完整 checkpoint，并使用相同任务池、配额和模型接口。运行目录中保存实际配置、组选择记录、optimizer checkpoint 和训练指标。
