在 LIBERO 中训练 RPent 规划器
================================

RPent 把相机图像发给 Qwen3.5 规划器，再把模型选择的工具调用转换为机械臂末端动作。RLinf 用模拟器给出的最终奖励更新规划器。只有环境判定任务完成，这条轨迹的奖励才是 1。正常结束但未完成任务的轨迹奖励为 0。模拟器或模型服务出错会中止运行。

选择训练方式
------------

* **同步 GRPO** 使用 LIBERO-Pro。同一任务和初始状态采样 8 条轨迹，每次发起 5 组，取最先完成的 4 个整组，其余取消。Actor 在更新前重新计算动作的 logprob，截断重要性采样的权重上限为 2。
* **异步 SAO** 使用标准 LIBERO。学习器更新期间，环境继续采样。Critic 拟合价值后重新估计优势，供 Actor 更新。恢复时加载学习器和已完成的组，再用原组号重新发起未完成的轨迹。

两个训练配置都使用单机 4 张 NVIDIA 卡。0、1 号卡运行张量并行的学习器，2 号卡运行一个 SGLang Server，3 号卡运行 SAM3 和 EGL 渲染，最多同时执行 4 条轨迹。Session server 和 SGLang 需要共享临时目录。恢复检查点时，保留原运行目录和其中的训练图像。

安装训练环境和模拟器
--------------------

``RPENT_MODEL`` 指向 Hugging Face 格式的本地权重目录。验证训练流程可以使用公开的 `Qwen3.5-4B <https://huggingface.co/Qwen/Qwen3.5-4B>`__。要复现历史 GRPO 的初始策略，需要相同的 RPent SFT595 权重。单机验证使用四张 80 GB A100。

在当前 RLinf 仓库安装训练环境并激活。

.. code-block:: bash

   UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic --python 3.12 --torch 2.11.0 --engine sglang --sglang 0.5.12.post1 --transformers 5.6.1 --install-rlinf
   source .venv/bin/activate

四卡配置将优化器状态放在主存中。如果使用新安装的 Megatron Core 0.18（``dc0ee41a``），请在恢复训练前应用一次优化器状态恢复补丁。若安装时指定了已有的 Core 仓库，将第二条命令中的路径换成该仓库的位置。

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/Sonorous281/RLinf/e563431a3dd234643086a4baadcdb075772672ce/requirements/agentic/megatron-core-r0.18.0-cpu-offload.patch -o /tmp/rlinf-core-offload.patch
   git -C .venv/Megatron-LM apply /tmp/rlinf-core-offload.patch

按照要运行的配置安装模拟器。GRPO 使用 SFT595 训练时的动作接口，一次调用最多依次执行三个动作。运行 GRPO 或评估该模型时，使用以下版本。

.. code-block:: bash

   git clone https://github.com/RLinf/RPent.git ../RPent
   git -C ../RPent fetch origin 762a57c2b78278aff8d5c2c8f752ee42501bd5dc
   git -C ../RPent checkout --detach FETCH_HEAD
   export RLINF_ROOT="$PWD"
   (cd ../RPent && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --prerelease=allow --torch-backend cu128 -r requirements/episode-ppo-env.txt)
   export RPENT_PYTHON="$(cd ../RPent && pwd)/.venv/bin/python"

SAO 使用 ``Sonorous281/RPent`` 的 ``053fe8bd4bb2dbbd6a0652331261ef9949c68e2f`` 提交，按该版本的 ``requirements/libero-rl.txt`` 单独安装环境。这个版本分别提供移动末端和开合夹爪的工具。将 ``RPENT_PYTHON`` 指向对应的解释器。

按 `RPent 资源准备说明 <https://github.com/Sonorous281/RPent/blob/053fe8bd4bb2dbbd6a0652331261ef9949c68e2f/docs/source-zh/rst_source/simulators/libero_rl_setup.rst>`__ 准备模拟器 XML、初始状态和 SAM3 权重。在训练终端设置下列路径，在另一个使用相同环境变量的终端运行最后一条命令，启动 SAM3。

.. code-block:: bash

   export RPENT_MODEL=/path/to/Qwen3.5-4B
   export LIBERO_CONFIG_PATH=/path/to/libero-config
   export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
   export RPENT_RUN_DIR="$PWD/logs"
   "$RPENT_PYTHON" -m rpent.robots.components.sam3_server --cuda-device 3 --port 8101

训练与评估
----------

从 goal、object、spatial 三个任务集生成训练清单。每行包含任务指令和一段编码为 JSON 字符串的任务描述。例如，``{"suite":"libero_goal","task":0,"seed":4,"init_state_id":4}`` 表示在 goal 任务集的第 0 个任务上使用第 4 个初始状态。

.. code-block:: bash

   export RPENT_TRAIN_DATA="$PWD/data/rpent/train.jsonl"
   "$RPENT_PYTHON" examples/agent/rpent/prepare_tasks.py --libero-type pro --seeds 0-31 --output-dir data/rpent
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_grpo

使用 SAO 时，以 ``--libero-type standard`` 生成任务清单，然后运行：

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_sao

评估时，将 ``RPENT_MODEL`` 指向已导出的 Hugging Face 检查点，选择训练清单以外的初始状态。评估配置默认在 0 号卡推理、1 号卡渲染。如果 SAM3 地址有变化，用 ``RPENT_SAM3_URL`` 指向已有服务。

.. code-block:: bash

   export RPENT_EVAL_DATA="$PWD/data/rpent/eval.jsonl"
   "$RPENT_PYTHON" examples/agent/rpent/prepare_tasks.py --libero-type pro --seeds 32-39 --split eval --output-dir data/rpent
   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b

在 ``RPENT_RUN_DIR`` 下查看 ``rpent-eval/eval_results.json``，其中有完成的轨迹数、总成功率及各任务集的成功率。每条轨迹的 ``request.json`` 记录任务和初始状态，``planner.json`` 记录交互，``outcome.json`` 记录环境判定。用于训练的样本包含提示词和输出的 token ID、采样时的 logprob、策略版本以及图像引用。损失只计算模型输出的 token，工具返回的观测进入下一轮提示词。

例如，在一次标准 LIBERO 采样中，任务为 ``libero_10`` 的第 0 个任务，初态为 2。规划器先调用 ``view_env`` 获取 1024 × 1024 的相机画面，再调用 ``segment``，得到分数为 0.891 的分割掩码。经过 25 轮交互、262 个环境步后，这条轨迹达到请求次数上限。环境记录为 ``terminal=true``、``success=false``，因此训练奖励为 0。分割分数只用于感知。

参考结果
--------

历史 H100 实验从 Qwen3.5-4B-SFT595 开始训练，使用 LIBERO-Pro 的 30 个任务，每个任务有 50 个初始状态。每次更新从 40 组中取最先完成的 32 组，共完成计划中 150 次更新的 96 次。前 10 次更新在 2560 条准入轨迹中成功 688 次（26.88%），末 10 次成功 1940 次（75.78%）。这里统计的是训练中被选中的轨迹。四卡示例的批量和上下文长度均小于历史实验。
