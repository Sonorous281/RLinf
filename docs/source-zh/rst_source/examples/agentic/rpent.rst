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

RPent 的模拟器进程用 tokenizer 和图像处理器计算上下文长度。如果导出后的检查点使用了模拟器环境无法识别的 tokenizer 类，可在训练或评估命令末尾加上 ``tools.rpent.tokenizer_path=/path/to/compatible-tokenizer``。可以使用训练起点的完整模型目录，里面需要有对应的 tokenizer、对话模板和图像处理配置。先确认两套环境对同一段带图对话算出的 token 数一致。模型权重仍从 ``RPENT_MODEL`` 读取。

在当前 RLinf 仓库安装训练环境并激活。

.. code-block:: bash

   UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic --python 3.12 --torch 2.11.0 --engine sglang --sglang 0.5.12.post1 --transformers 5.6.1 --install-rlinf
   source .venv/bin/activate

四卡配置将优化器状态放在主存中。如果使用新安装的 Megatron Core 0.18（``dc0ee41a``），请在恢复训练前应用一次优化器状态恢复补丁。若安装时指定了已有的 Core 仓库，将第二条命令中的路径换成该仓库的位置。

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/Sonorous281/RLinf/e563431a3dd234643086a4baadcdb075772672ce/requirements/agentic/megatron-core-r0.18.0-cpu-offload.patch -o /tmp/rlinf-core-offload.patch
   git -C .venv/Megatron-LM apply /tmp/rlinf-core-offload.patch

先选择模拟器版本。运行 GRPO 或评估 SFT595 时，一次工具调用最多依次执行三个末端动作。

.. code-block:: bash

   repo=RLinf/RPent revision=762a57c2b78278aff8d5c2c8f752ee42501bd5dc recipe=episode-ppo-env

运行 SAO 时选用下面的版本。它用两个工具分别控制末端移动和夹爪。该版本的依赖文件会从名为 ``RLinf`` 的同级目录安装 RLinf。

.. code-block:: bash

   repo=Sonorous281/RPent revision=053fe8bd4bb2dbbd6a0652331261ef9949c68e2f recipe=libero-rl

选好版本后，在 RLinf 仓库根目录执行以下命令，给模拟器创建单独的环境。

.. code-block:: bash

   export RLINF_ROOT="$PWD"
   rpdir="../RPent-$recipe"
   git clone --no-checkout "https://github.com/$repo.git" "$rpdir"
   git -C "$rpdir" fetch --depth 1 origin "$revision" && git -C "$rpdir" checkout --detach FETCH_HEAD
   (cd "$rpdir" && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --prerelease=allow --torch-backend cu128 -r "requirements/$recipe.txt")
   export RPENT_PYTHON="$(cd "$rpdir" && pwd)/.venv/bin/python"

如果 GRPO 在第一次 reset 时找不到 ``libero_floor_base_style.xml``，说明当前模拟器安装包缺少场景资源。MuJoCo 会在已导入的 LIBERO-Pro 包内查找这份 XML。将完整源码以可编辑模式安装到模拟器环境后，再启动训练。

.. code-block:: bash

   git clone --branch rpent https://github.com/RLinf/LIBERO-PRO.git ../LIBERO-PRO && git -C ../LIBERO-PRO checkout --detach d1e11fb181b8544487d27742c0caa3a4d46452ad
   uv pip install --python "$RPENT_PYTHON" --no-deps -e ../LIBERO-PRO

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

在 ``RPENT_RUN_DIR`` 下查看 ``rpent-eval/eval_results.json``，其中有完成的轨迹数、总成功率及各任务集的成功率。每条轨迹的 ``request.json`` 记录任务和初始状态，``planner.json`` 记录交互，``outcome.json`` 记录环境判定。训练样本的 ``input_ids`` 包含渲染后的提示词和模型输出。``response_mask`` 标出参与损失计算的动作 token，``rollout_logprobs`` 保存采样时的 logprob，``rollout_weight_versions`` 记录生成每个动作 token 的策略版本，``multi_modal_inputs`` 引用对应图像。工具返回的观测进入下一轮提示词。

例如，在一次标准 LIBERO 采样中，任务为 ``libero_10`` 的第 0 个任务，初态为 2。规划器先调用 ``view_env`` 获取 1024 × 1024 的相机画面，再调用 ``segment``，得到分数为 0.891 的分割掩码。经过 25 轮交互、262 个环境步后，这条轨迹达到请求次数上限。环境记录为 ``terminal=true``、``success=false``，因此训练奖励为 0。分割分数只用于感知。

在 GRPO 中复用规划历史
------------------------------

长轨迹可能拆成多行训练样本。后一行如果完整保留了前一行的 token、图像和位置编码，actor 可以把这段历史只算一次。两行必须来自同一条轨迹，在同一个数据并行副本上前后相邻。两行的动作都按各自原有的权重计入损失。

前缀复用默认关闭。在上面的 GRPO 启动命令末尾加一个开关即可启用。已验证的稠密 Qwen3.5-4B 配置使用 TP2、PP1、CP1、EP1。沿用示例的分块输出层，重算旧 logprob 和训练都使用单行微批。自定义配置时，保持 dropout、CUDA Graph 和数据并行通信重叠关闭。

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_grpo actor.reuse_nested_prefixes=true

在 actor 日志中查找 ``Nested prefix training``。如果实际送入解码器的行数（physical rows）小于训练样本行数（logical rows），说明这个副本在这批样本中复用了前缀。

在两张 A100 上回放一条历史 LIBERO-Object 轨迹，两组的词表输出都按 256 个 token 分块。重算旧 logprob 加上训练的前向和反向计算，关闭复用时耗时 11.84 秒，开启后为 8.63 秒。23,201 个动作 token 的旧 logprob 逐项相同。计时不含 rollout 和优化器更新。

参考结果
--------

历史 H100 实验从 Qwen3.5-4B-SFT595 开始训练，使用 LIBERO-Pro 的 30 个任务，每个任务有 50 个初始状态。每次更新从 40 组中取最先完成的 32 组，共完成计划中 150 次更新的 96 次。前 10 次更新在 2560 条准入轨迹中成功 688 次（26.88%），末 10 次成功 1940 次（75.78%）。这里统计的是训练中被选中的轨迹。H100 实验的上下文上限为 262,144 个 token，每轮最多输出 16,384 个。四卡 GRPO 示例分别使用 65,536 和 8,192。

另一次四卡 GRPO 短跑使用 SFT595 和固定的 LIBERO-Pro 任务，关闭 thinking，每条轨迹最多 16 轮，每轮最多输出 2,048 个 token。4 组共 32 条轨迹进入训练，第 5 组取消。32 条原生奖励都是 0，Actor 梯度范数为 0。保存检查点后，SGLang Server 的策略版本为 1，下一轮规划器已收到工具调用。

在四张 A100 上验证 SAO 时，使用 4 个环境、每批 4 条轨迹，每条最多 4 轮、每次最多生成 512 个 token。连续完成两次更新后，在新进程中恢复并完成第三次更新。每份完整检查点约 128 GiB，恢复期间整组进程的主存 PSS 峰值约 403 GiB。本次短跑的 24 条已结束轨迹均未完成任务；这个结果用于核对训练和恢复流程。
