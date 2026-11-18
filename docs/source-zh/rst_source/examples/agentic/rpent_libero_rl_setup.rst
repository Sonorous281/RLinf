RPent 安装与部署
================

本页准备 :doc:`rpent_libero_rl` 所需的配套代码、模型资产和服务。每个部署环境只需完成一次；日常训练和评测使用主示例入口。

安装代码与环境
--------------

在配备 NVIDIA GPU 的 Linux 主机上准备 EGL/OpenGL 库、Git 和 ``uv``。从本分支的 RLinf 仓库根目录执行以下命令，创建独立工作树，并将 RPent 固定到 ``build/libero-rl-pins`` 分支的 ``5a98ff3`` 提交。RPent 依赖文件中的 Git 包也已固定版本。

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"
   mkdir -p "$STACK_ROOT/repos" "$STACK_ROOT/envs" "$STACK_ROOT/models" "$STACK_ROOT/data"
   git worktree add --detach "$STACK_ROOT/repos/RLinf" HEAD
   git clone --branch build/libero-rl-pins https://github.com/Sonorous281/RPent.git "$STACK_ROOT/repos/RPent"
   git -C "$STACK_ROOT/repos/RPent" checkout --detach 5a98ff3
   git -C "$STACK_ROOT/repos/RLinf" rev-parse HEAD
   git -C "$STACK_ROOT/repos/RPent" rev-parse HEAD

单机 A100 按接下来四节依次安装和启动，只需一套 RLinf 环境和一套 RPent 环境。本页只描述单机部署；其他 placement 模板需另行验证。

安装 TokenSpace 的 A100 环境
-----------------------------

单机 A100 配置需要独立环境：Python 3.12.13、Rust 工具链、支持 C++20 的 GCC，以及 CUDA 13.0.3。SGLang 固定为 commit ``18a44326f6de5134fe921927d760d81f8ee48fc9``；Torch 为 2.14.1，Transformers 为 5.19.0，Bridge 为 0.5.0，Transformer Engine 为 2.17.0，FlashAttention 为 2.8.3。这些依赖沿用原集成版本的组合，本分支尚未完成干净环境 GPU 验证。

机器已具备兼容的 NVIDIA 驱动、Rust 和 GCC 时，执行以下命令。安装器会将固定版本的 CUDA 工具链装入新环境：

.. code-block:: bash

   cd "$STACK_ROOT/repos/RLinf"
   unset PYTHONPATH MEGATRON_PATH SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE
   export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
   export TORCH_CUDA_ARCH_LIST=8.0 FLASH_ATTN_CUDA_ARCHS=80
   UV_TORCH_BACKEND=cu130 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
     --venv "$STACK_ROOT/envs/rlinf-token-space" --python 3.12.13 \
     --engine sglang --sglang token-space --torch 2.14.1 --transformers 5.19.0 \
     --no-root --no-apex --install-rlinf

安装器会应用 ``requirements/agentic`` 中与版本配套的三份补丁，分别修复 Megatron 的 CPU 优化器恢复、Bridge 的 ASR 重复注册，以及 FlashAttention 的 C++20 编译选项。补丁随安装生效，不需要额外的 overlay 目录或 ``sitecustomize.py``。此配置使用 Transformer Engine 的优化器，因此传入 ``--no-apex``；其他安装方式不变。

使用 570 系列驱动的 A100 机器需要兼容的 CUDA 前向兼容库才能运行这套 CUDA 13 环境。在受支持的 Linux 部署中，将 NVIDIA 的 ``cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb`` 解压到本任务目录，并在本任务的 ``LD_LIBRARY_PATH`` 中加入其中的 ``usr/local/cuda-13.0/compat``。校验 SHA256 为 ``b0ee4fb1f439e919ca5c5fd7c5cff3e8b8afdbdb0b5ee38f7c09c047ea41a035``。这不会替换系统内核驱动，不要为测试修改共享机器的驱动。启动 Ray 前，须验证设备分配和内核执行，仅能导入 CUDA 模块还不够。

RPent 按下一节安装到独立环境。其依赖文件固定了 SAM3 和两套 LIBERO 的源码 commit，不应混入旧环境的 Python 包。模型权重和模拟器资源可以只读共享。保留两套环境的 ``uv pip freeze`` 输出、最终配置和源码版本。

安装 RPent 环境
-------------------

SAM3 和 LIBERO 使用 Transformers 4.57.6，A100 训练环境使用 5.19.0，因此保留两套 Python 环境；两边导入同一份 RLinf 代码。

从配套 RPent checkout 安装独立的模拟器／SAM3 环境。它的 requirements 从同级目录安装 RLinf，保证两套环境使用相同的源码版本。启动前核对输出的模块路径：

.. code-block:: bash

   cd "$STACK_ROOT/repos/RPent"
   uv venv --python 3.12 "$STACK_ROOT/envs/rpent"
   source "$STACK_ROOT/envs/rpent/bin/activate"
   uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
   python -c 'import rlinf, robots.libero_rl; print(rlinf.__file__, robots.libero_rl.__file__)'

实测的 A100 主机为训练和模拟器使用了不同版本的 CUDA 运行库。A100 部署配置通过以下脚本启动模拟器，让它使用自己的 CUDA 运行库。如果模拟器需要驱动兼容库，启动 Ray 前将 ``RPENT_EPISODE_LIBRARY_PATH`` 设为该目录。使用系统驱动库时可留空。

.. code-block:: bash

   cat > "$STACK_ROOT/envs/episode-python" <<'SH'
   #!/usr/bin/env bash
   set -eu
   stack_root=$(cd "$(dirname "$0")/.." && pwd)
   export PYTHONPATH="$stack_root/repos/RPent:$stack_root/repos/RLinf"
   export LD_LIBRARY_PATH="${RPENT_EPISODE_LIBRARY_PATH:-}"
   exec "$stack_root/envs/rpent/bin/python" "$@"
   SH
   chmod +x "$STACK_ROOT/envs/episode-python"


准备模型和标准 LIBERO 资产
--------------------------

将完整 Qwen3.5-4B 模型或兼容的完整 SFT 模型导出放到 ``$STACK_ROOT/models/``。其中必须包括视觉权重、tokenizer、processor 和 chat template。仅包含语言参数的模型无法处理图像输入。例如在联网机器下载基础模型：

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

启动单机 A100 服务
------------------

四卡配置使用 GPU 0–3，八卡配置使用 GPU 0–7。确认所选 GPU 均可用，再为作业启动独立的 Ray 集群；临时目录放在容量足够的数据盘，避免填满系统盘。以下路径示例要求 ``$STACK_ROOT`` 位于数据盘。Ray 的 Unix socket 路径有长度限制，请为 ``RPENT_RAY_TMP`` 选择简短路径。

四卡验证直接使用下面的默认值。跑八卡配置时，在 Ray 终端和 SAM3 终端都先执行以下命令，再分别启动服务。

.. code-block:: bash

   export RPENT_GPUS=0,1,2,3,4,5,6,7 RPENT_GPU_COUNT=8 RPENT_SAM3_GPU=6 RPENT_SAM3_PORT=18261

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf-token-space/bin/activate"
   cd "$STACK_ROOT/repos/RLinf"
   export MEGATRON_PATH="$VIRTUAL_ENV/Megatron-LM"
   export PYTHONPATH="$STACK_ROOT/repos/RPent:$STACK_ROOT/repos/RLinf:$MEGATRON_PATH"
   export CUDA_VISIBLE_DEVICES="${RPENT_GPUS:-0,1,2,3}" RLINF_NODE_RANK=0
   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=""
   export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 TORCH_COMPILE_DISABLE=1
   export RLINF_TIMEOUT=10800
   export RPENT_RAY_TMP="$STACK_ROOT/ray"
   mkdir -p "$RPENT_RAY_TMP"
   ray start --head --port 6387 --node-ip-address=127.0.0.1 --num-gpus "${RPENT_GPU_COUNT:-4}" \
     --temp-dir "$RPENT_RAY_TMP" --include-dashboard=false
   export RAY_ADDRESS=127.0.0.1:6387
   ray status --address="$RAY_ADDRESS"

在第二个终端启动共享 SAM3。保留该终端，训练与恢复都连接同一服务：

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"  # same path used during installation
   source "$STACK_ROOT/envs/rpent/bin/activate"
   cd "$STACK_ROOT/repos/RPent"
   export CUDA_VISIBLE_DEVICES="${RPENT_GPUS:-0,1,2,3}"
   export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
   python -m rpent.robots.components.sam3_server \
     --cuda-device "${RPENT_SAM3_GPU:-3}" --host 127.0.0.1 --port "${RPENT_SAM3_PORT:-18147}"

端口已被占用时，先核对服务归属；不要停止别人的 Ray 或 SAM3。可为本任务改用空闲端口，并同步修改连接地址。已有 SAM3 可共享时，直接配置其 endpoint，不重复加载模型。

服务就绪后，返回 :doc:`rpent_libero_rl`。四卡按顺序验证单个 episode、两步训练及新进程恢复；八卡可直接使用“八卡 SAO 训练与性能采集”的命令。
