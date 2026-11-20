RPent 安装与部署
================

本页准备 :doc:`rpent_libero_rl` 所需的配套代码、模型资产和服务。每个部署环境只需完成一次；日常训练和评测使用主示例入口。

本页将训练环境与模拟器环境分开安装。使用 H200 ``reason`` 镜像时，按 :doc:`rpent_episode_sync` 在同一环境中安装，并保留镜像原有的 Core 和 Bridge。

安装代码与环境
--------------

使用 Linux NVIDIA 主机，准备兼容驱动、CUDA 12.8 toolkit、EGL/OpenGL 库、Git 和 ``uv``。将配套的 RLinf、RPent checkout 分别放在 ``$STACK_ROOT/repos/RLinf`` 和 ``$STACK_ROOT/repos/RPent``。各节点使用相同版本和路径，安装前记录两边的 commit。

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"
   mkdir -p "$STACK_ROOT/repos" "$STACK_ROOT/envs" "$STACK_ROOT/models" "$STACK_ROOT/data"
   git clone --branch dev/rpent-rlinf-integration https://github.com/Sonorous281/RLinf.git "$STACK_ROOT/repos/RLinf"
   git clone --branch dev/rpent-rlinf-integration https://github.com/Sonorous281/RPent.git "$STACK_ROOT/repos/RPent"
   git -C "$STACK_ROOT/repos/RLinf" rev-parse HEAD
   git -C "$STACK_ROOT/repos/RPent" rev-parse HEAD
   git clone https://github.com/NVIDIA/Megatron-LM.git "$STACK_ROOT/repos/Megatron-LM"
   git -C "$STACK_ROOT/repos/Megatron-LM" checkout dc0ee41ada177d4c74ba46bb501bba5967264815
   export MEGATRON_PATH="$STACK_ROOT/repos/Megatron-LM"
   cd "$STACK_ROOT/repos/RLinf"
   UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
     --venv "$STACK_ROOT/envs/rlinf" --python 3.12.12 --torch 2.11.0 \
     --engine sglang --sglang 0.5.12.post1 --transformers 5.6.1 \
     --use-mirror --no-root --install-rlinf

安装器复用已有的 ``MEGATRON_PATH``。它在解析依赖时会临时修改依赖元数据，因此请使用空闲 checkout 和新环境。``--no-root`` 要求原生系统依赖已准备好；``UNINSTALL_FA4=1`` 选择经过验证的 FA2 路径。其他平台参数见 ``bash requirements/install.sh --help``。

Core 的 CPU optimizer 补丁只适用于 HybridDeviceOptimizer 的 checkpoint 路径。设置 ``actor.optim.optimizer_cpu_offload=false`` 时，阶段卸载不需要该补丁。H200 最小部署保留原装 Core 和 Bridge。

从配套 RPent checkout 安装独立的模拟器／SAM3 环境。它的 requirements 从同级目录安装 RLinf，保证两套环境使用相同的源码版本。启动前核对输出的模块路径：

.. code-block:: bash

   cd "$STACK_ROOT/repos/RPent"
   uv venv --python 3.12 "$STACK_ROOT/envs/rpent"
   source "$STACK_ROOT/envs/rpent/bin/activate"
   uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
   python -c 'import rlinf, robots.libero_rl; print(rlinf.__file__, robots.libero_rl.__file__)'

以下记录已验证的兼容组合，不保证未来任意一次依赖解析都会得到完全相同的环境。

.. list-table::
   :header-rows: 1

   * - 组件
     - 已验证版本
   * - RLinf
     - Python 3.12.12；Torch 2.11.0+cu128；Transformers 5.6.1；SGLang 0.5.12.post1
   * - 训练内核
     - Megatron Core 0.18.3，源码 commit 如上；Bridge 0.5.0；Transformer Engine 2.17.0；适配 Torch 2.11 的 FlashAttention 2.8.3
   * - Qwen 线性注意力
     - flash-linear-attention/fla-core 0.6.0；TileLang 0.1.8；Triton 3.6.0
   * - 环境模拟器
     - Torch 2.11.0+cu128；Transformers 4.57.6；NumPy 1.26.4；MuJoCo 3.3.0；robosuite 1.5.2；rpent-libero/rpent-liberopro 0.2.0；SAM3 0.1.0

使用 ``uv pip freeze``、``python -c 'import torch; print(torch.__version__, torch.version.cuda)'`` 和 ``git -C "$MEGATRON_PATH" rev-parse HEAD`` 核对并在验证后保留版本清单。源码编译需要匹配的 CUDA 编译器；可联网的纯 CPU 节点适合下载 wheels/源码，但不一定能编译 CUDA 扩展。离线计算节点应先在兼容 Linux/Python/CUDA 构建机准备完整依赖，再传输 wheels 和 Git 源码，通过 ``uv pip install --no-index --find-links /path/to/wheelhouse ...`` 安装。部分 wheel 缓存不等于完整离线安装包。

所有 Ray 节点必须有相同的 RLinf 环境、源码和模型文件。RPent 保留独立环境，不要为了统一版本把 RLinf 的 Transformers 升级为 RPent 版本。若使用自定义激活脚本，保留安装器设置的 Megatron、CUDA、NCCL 路径。本方案使用 FA2 并关闭 Torch 编译，在启动 Ray 前设置 ``NVTE_FLASH_ATTN=1``、``NVTE_FUSED_ATTN=0``、``TORCH_COMPILE_DISABLE=1``。环境节点还需要可用的 EGL。

精确复现某次运行时，安装前将两个仓库 checkout 到该次记录的 commit。分支名用于获取交付代码，不能代替实验的固定版本记录。

准备模型和标准 LIBERO 资产
--------------------------

将完整 Qwen3.5-4B 或兼容的全参 SFT 导出放到 ``$STACK_ROOT/models/``。其中必须包括视觉权重、tokenizer、processor 和 chat template；纯语言抽取模型不能处理 adaptive 带图回合。例如在联网机器下载基础模型：

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir "$STACK_ROOT/models/Qwen3.5-4B"

按照 `SAM3 仓库 <https://github.com/facebookresearch/sam3>`__ 的说明获取权重，并将 ``SAM3_CHECKPOINT_PATH`` 指向本地 ``sam3.pt``。Python 包不包含模拟器资产。在 RPent 环境下载标准资产并生成独立 benchmark 配置：

.. code-block:: bash

   source "$STACK_ROOT/envs/rpent/bin/activate"
   export LIBERO_ASSETS="$STACK_ROOT/data/libero/assets"
   export LIBERO_CONFIG_PATH="$STACK_ROOT/data/libero-standard-config"
   env -u LIBERO_ASSET_PATH libero-download-assets \
     --assets-dir "$LIBERO_ASSETS" --skip-existing --no-cache
   mkdir -p "$LIBERO_CONFIG_PATH"
   python - <<'PY'
   import os
   from pathlib import Path
   import yaml
   import libero.libero as libero
   assets = Path(os.environ['LIBERO_ASSETS']).resolve()
   package_assets = Path(libero.__file__).resolve().parent / 'assets'
   # LIBERO also constructs scene/mesh paths relative to its installed package.
   if not package_assets.exists():
       if package_assets.is_symlink():
           raise RuntimeError(f'Broken LIBERO assets link: {package_assets}')
       package_assets.symlink_to(assets, target_is_directory=True)
   for scene in ['libero_tabletop_base_style.xml', 'libero_floor_base_style.xml',
                 'libero_kitchen_tabletop_base_style.xml',
                 'libero_living_room_tabletop_base_style.xml']:
       assert (package_assets / 'scenes' / scene).is_file(), scene
   config = libero.get_default_path_dict()
   config['assets'] = str(assets)
   Path(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml').write_text(yaml.safe_dump(config))
   PY

LIBERO 的部分 XML/mesh 路径直接相对安装包构造，独立于 config.yaml，所以还必须补齐包内 assets 链接。已有完整资产目录时保留；遇到断链应修复路径，不替换场景几何。保留 ``get_default_path_dict()`` 返回的包内 benchmark、BDDL 和初始状态路径。``train_qwen3.5_4b_libero_rl_sao_3node`` 配置把此路径显式传给 episode 子进程，避免父环境中的 LIBERO-PRO 配置悄悄替换标准 benchmark。

继续使用 RPent 解释器生成标准 40-task 训练清单：

.. code-block:: bash

   cd "$STACK_ROOT/repos/RLinf"
   python examples/agent/rpent/prepare_tasks.py --workflow libero_rl \
     --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
     --seeds 1-32 --output-dir "$STACK_ROOT/data/libero-rl-tasks"

这里按同一组 40 个任务的初始状态划分，不是未见过的任务族；称为独立 holdout 前还需检查与 SFT 训练清单是否重叠。生成 1,280 行及记录 task/初始状态数量、seeds、哈希的 manifest。seed 0 保留评测。生成器拒绝越过各任务初始状态范围的 seed，避免取模后重复同一初始状态。可使用 ``--seeds 33-39 --split dev`` 或 ``--seeds 40-49 --split test``，为各 split 指定独立输出目录。原来的 memory 清单生成器仍为默认 ``--workflow libero``，仍要求 ``--memory-dir``。

启动三节点 Ray 和 SAM3
----------------------

**每个节点** 启动 Ray 前都先 source RLinf 环境，导出 ``MEGATRON_PATH``、``NVTE_FLASH_ATTN=1``、``NVTE_FUSED_ATTN=0``、``TORCH_COMPILE_DISABLE=1``，并设置足够的 ``RLINF_TIMEOUT``，例如 ``10800``。按实际网络选择网卡，``RLINF_COMM_NET_DEVICES=eth0`` 仅为示例。Ray worker 在启动时继承这些环境变量。

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"  # same path used during installation
   export MEGATRON_PATH="$STACK_ROOT/repos/Megatron-LM"
   source "$STACK_ROOT/envs/rlinf/bin/activate"
   export PYTHONPATH="$STACK_ROOT/repos/RPent:$STACK_ROOT/repos/RLinf:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal
   export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 TORCH_COMPILE_DISABLE=1
   export RLINF_TIMEOUT=10800
   export HEAD_IP=10.0.0.10  # 各节点可达的 head 地址

然后在模型节点 0 启动 head：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head --port 6387 --node-ip-address="$HEAD_IP" --num-gpus 8

模型节点 1 和环境节点 2 使用同样的源码/环境，各自填写 rank 和本地地址：

.. code-block:: bash

   export RLINF_NODE_RANK=1  # 环境节点使用 2
   export NODE_IP=10.0.0.11 # 环境节点填写自己的地址
   ray start --address="$HEAD_IP:6387" --node-ip-address="$NODE_IP" --num-gpus 8

通过 ``ray status --address="$HEAD_IP:6387"`` 确认三节点、24 GPU。不要在已占用节点上启动冲突的第二个 Ray 集群。长训练前验证多节点 NCCL 通信；防火墙、网卡和 RDMA 设置取决于集群。

在节点 2 的独立终端使用 RPent 环境启动 SAM3：

.. code-block:: bash

   source "$STACK_ROOT/envs/rpent/bin/activate"
   cd "$STACK_ROOT/repos/RPent"
   export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8101

Agent worker 和 episode 子进程都位于节点 2，因此 ``http://127.0.0.1:8101`` 连接本地 SAM3。Policy 的本地 HTTP endpoint 通过 RLinf channel 将模型请求发往两台模型节点，无须跨节点暴露 SAM3。


资源与服务就绪后，返回 :doc:`rpent_libero_rl` 启动训练或评测。

PPO 和 SAO 训练默认使用全词表采样：temperature 为 1、top-k 为 -1、top-p 为 1、presence penalty 为 0、repetition penalty 为 1。评测保留独立 Planner 的采样配置；对比 policy 时，应显式使用相同的采样参数。默认启用 Thinking，且不强制限制思考 token 数（\ ``agentloop.thinking_token_budget: null``\ ）。需要限制时再设置该选项。

训练默认使用 SGLang Server 后端。设置 ``rollout.sglang.backend_type=engine`` 可切换到进程内 Engine 后端，两者复用相同的 rollout 和权重同步接口。SAO 示例通过 ``runner.synchronous: false`` 异步采集 episode，actor 和 critic 仍依次更新。rollout 与训练使用独立 GPU 时可以重叠执行；共卡时，rollout 必须先暂停并卸载，再开始训练。此示例不使用无 critic 的 ``runner.async_mode``\ 路径。

进阶选项见 :doc:`../../guides/agent-workflows/rpent-multimodal`。
