RPent Installation And Deployment
=================================

Prepare the paired sources, model assets and services used by :doc:`rpent_libero_rl`. Run these steps once per deployment; use the main example for training and evaluation.

Install Code And Environments
-----------------------------

Use a Linux NVIDIA host with EGL/OpenGL libraries, Git and ``uv``; the A100 dependency requirements are listed below. Run the following commands from the root of the RLinf checkout containing this example. Set ``RPENT_SOURCE`` to the paired RPent checkout. Both checkouts must have the required changes committed, and ``requirements/libero-rl.txt`` in RPent must pin Git dependencies to commits. The commands create isolated worktrees from each checkout's HEAD. Use those same revisions and paths on every node.

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"
   export RPENT_SOURCE=/path/to/RPent
   mkdir -p "$STACK_ROOT/repos" "$STACK_ROOT/envs" "$STACK_ROOT/models" "$STACK_ROOT/data"
   git worktree add --detach "$STACK_ROOT/repos/RLinf" HEAD
   git -C "$RPENT_SOURCE" worktree add --detach "$STACK_ROOT/repos/RPent" HEAD
   git -C "$STACK_ROOT/repos/RLinf" rev-parse HEAD
   git -C "$STACK_ROOT/repos/RPent" rev-parse HEAD

For the single-node A100 run, follow the next four sections in order. Install one RLinf environment and one RPent environment. The default RLinf profile and multi-node services at the end are optional alternatives.

Install the TokenSpace A100 Environment
---------------------------------------

The single-node A100 profile needs a separate environment. Use Python 3.12.13, a Rust toolchain, GCC with C++20 support, and CUDA 13.0.3. The pinned SGLang commit is ``18a44326f6de5134fe921927d760d81f8ee48fc9``; Torch is 2.14.1, Transformers 5.19.0, Bridge 0.5.0, Transformer Engine 2.17.0 and FlashAttention 2.8.3. The source candidate completed a clean-environment GPU run through step 3 recovery. Reproduce that check after changing source or dependencies.

With a compatible NVIDIA driver, Rust and GCC available, the installer supplies the pinned CUDA toolkit in the new environment:

.. code-block:: bash

   cd "$STACK_ROOT/repos/RLinf"
   unset PYTHONPATH MEGATRON_PATH SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE
   export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
   export TORCH_CUDA_ARCH_LIST=8.0 FLASH_ATTN_CUDA_ARCHS=80
   UV_TORCH_BACKEND=cu130 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
     --venv "$STACK_ROOT/envs/rlinf-token-space" --python 3.12.13 \
     --engine sglang --sglang token-space --torch 2.14.1 --transformers 5.19.0 \
     --no-root --no-apex --install-rlinf

The installer applies three version-specific patches from ``requirements/agentic``: Megatron CPU optimizer recovery, Bridge's duplicate ASR registration and FlashAttention's C++20 build flags. The patches apply during installation; no overlay directory or ``sitecustomize.py`` is required. ``--no-apex`` omits Apex; this profile uses Transformer Engine and CPU optimizer offload.

A100 hosts with a 570-series driver need a compatible CUDA forward-compatibility library for this CUDA 13 stack. On supported Linux deployments, extract NVIDIA's ``cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb`` into a job-owned directory and set its ``usr/local/cuda-13.0/compat`` directory in ``LD_LIBRARY_PATH`` for the job. Verify SHA256 ``b0ee4fb1f439e919ca5c5fd7c5cff3e8b8afdbdb0b5ee38f7c09c047ea41a035``. This does not replace the host kernel driver; do not change a shared host's driver for this test. A CUDA import alone is insufficient: check device allocation and kernel execution before starting Ray.

Install the RPent environment in the next section. That requirements file pins SAM3 and both LIBERO sources to commits; do not mix in an existing environment's Python packages. Model weights and simulator assets may be shared read-only. Save ``uv pip freeze`` from both environments with the resolved configuration and source revisions.

Install the RPent Environment
-----------------------------

RPent uses Transformers 4.57.6 for SAM3 and LIBERO, while the A100 learner uses 5.19.0. Keep these dependencies in separate environments; both import the same RLinf checkout.

Install the separate simulator/SAM3 environment from the paired RPent checkout. Its requirements install RLinf from the sibling checkout, so both environments use the same source revision. Verify the printed module paths before launching:

.. code-block:: bash

   cd "$STACK_ROOT/repos/RPent"
   uv venv --python 3.12 "$STACK_ROOT/envs/rpent"
   source "$STACK_ROOT/envs/rpent/bin/activate"
   uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
   python -c 'import rlinf, robots.libero_rl; print(rlinf.__file__, robots.libero_rl.__file__)'


Prepare Models And Standard LIBERO Assets
-----------------------------------------

Place the full Qwen3.5-4B model, or a compatible SFT export containing the full model, under ``$STACK_ROOT/models/``. It must include the vision weights, tokenizer, processor and chat template. A language-only model cannot process image inputs. Download the base model on a networked machine, for example:

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir "$STACK_ROOT/models/Qwen3.5-4B"

Obtain the SAM3 checkpoint following the `SAM3 repository <https://github.com/facebookresearch/sam3>`__, and set ``SAM3_CHECKPOINT_PATH`` to its local ``sam3.pt``. Assets are not bundled in the Python packages. In the RPent environment, download the standard assets and write a dedicated benchmark config:

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

The installed package's assets link is necessary because LIBERO constructs some XML/mesh paths relative to its source, independently of config.yaml. Preserve an existing complete assets directory; repair a broken link instead of replacing scene geometry. Keep the package's benchmark/BDDL/initial-state paths from ``get_default_path_dict()``. The ``train_qwen3.5_4b_libero_rl_sao_3node`` recipe passes this config explicitly to episode subprocesses, so a parent process configured for LIBERO-PRO cannot silently change the standard benchmark.

Create the standard 40-task training list with the RPent interpreter:

.. code-block:: bash

   cd "$STACK_ROOT/repos/RLinf"
   python examples/agent/rpent/prepare_tasks.py --workflow libero_rl \
     --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
     --seeds 1-32 --output-dir "$STACK_ROOT/data/libero-rl-tasks"

These are initial-state splits over the same 40 tasks, not held-out task families. Check the SFT training manifest for overlap before calling them independent holdouts. This writes 1,280 rows and a manifest containing task/state counts, seeds and hashes. Seed 0 is reserved for evaluation. The generator rejects seeds outside each task's initial-state range instead of silently reusing an initial state through modulo indexing. For separate dev/test manifests, use ``--seeds 33-39 --split dev`` or ``--seeds 40-49 --split test`` with a separate output directory. The old memory-based generator remains the default ``--workflow libero`` and still requires ``--memory-dir``.

Start Single-Node A100 Services
-------------------------------

The TokenSpace profile uses GPUs 0–3. Check that all four are idle, then start a separate Ray cluster for this job. Put ``$STACK_ROOT`` on a data disk with enough space. Choose a short ``RPENT_RAY_TMP`` path to stay within the Unix socket path limit.

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf-token-space/bin/activate"
   cd "$STACK_ROOT/repos/RLinf"
   export MEGATRON_PATH="$VIRTUAL_ENV/Megatron-LM"
   export PYTHONPATH="$STACK_ROOT/repos/RPent:$STACK_ROOT/repos/RLinf:$MEGATRON_PATH"
   export CUDA_VISIBLE_DEVICES=0,1,2,3 RLINF_NODE_RANK=0
   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=""
   export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 TORCH_COMPILE_DISABLE=1
   export RLINF_TIMEOUT=10800
   export RPENT_RAY_TMP="$STACK_ROOT/ray"
   mkdir -p "$RPENT_RAY_TMP"
   ray start --head --port 6387 --node-ip-address=127.0.0.1 --num-gpus 4 \
     --temp-dir "$RPENT_RAY_TMP" --include-dashboard=false
   export RAY_ADDRESS=127.0.0.1:6387
   ray status --address="$RAY_ADDRESS"

Start shared SAM3 in a second terminal. Keep it running for training and recovery:

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"  # same path used during installation
   source "$STACK_ROOT/envs/rpent/bin/activate"
   cd "$STACK_ROOT/repos/RPent"
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
   python -m rpent.robots.components.sam3_server \
     --cuda-device 3 --host 127.0.0.1 --port 18147

If a port is occupied, identify its owner; do not stop another job. Choose a free port for this job and update its connection address. If an existing SAM3 service is available to share, configure its endpoint instead of loading another model.

Optional: Install the Default RLinf Environment
-----------------------------------------------

For the default SGLang profile, install Megatron and RLinf:

.. code-block:: bash

   git clone https://github.com/NVIDIA/Megatron-LM.git "$STACK_ROOT/repos/Megatron-LM"
   git -C "$STACK_ROOT/repos/Megatron-LM" checkout dc0ee41ada177d4c74ba46bb501bba5967264815
   export MEGATRON_PATH="$STACK_ROOT/repos/Megatron-LM"
   cd "$STACK_ROOT/repos/RLinf"
   git -C "$MEGATRON_PATH" apply \
     "$STACK_ROOT/repos/RLinf/requirements/agentic/megatron-core-r0.18.0-cpu-offload.patch"
   UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
     --venv "$STACK_ROOT/envs/rlinf" --python 3.12.12 --torch 2.11.0 \
     --engine sglang --sglang 0.5.12.post1 --transformers 5.6.1 \
     --use-mirror --no-root --install-rlinf

The installer honors an existing ``MEGATRON_PATH``. It temporarily edits dependency metadata while resolving, so use an idle checkout and a new environment. ``--no-root`` requires native system dependencies to be installed already. ``UNINSTALL_FA4=1`` selects the validated FA2 route. See ``bash requirements/install.sh --help`` for other platforms.

The ``git apply`` command above applies the patch once before installation. It lets Megatron restore CPU-offloaded master weights and Adam step counters from a Torch checkpoint.


The tested stack uses these versions. This table records compatibility, rather than promising that every future dependency resolution reproduces the same environment.

.. list-table::
   :header-rows: 1

   * - Component
     - Tested versions
   * - RLinf
     - Python 3.12.12; Torch 2.11.0+cu128; Transformers 5.6.1; SGLang 0.5.12.post1
   * - Training kernels
     - Megatron Core 0.18.3 at the commit above; Bridge 0.5.0; Transformer Engine 2.17.0; FlashAttention 2.8.3 built for Torch 2.11
   * - Qwen linear attention
     - flash-linear-attention/fla-core 0.6.0; TileLang 0.1.8; Triton 3.6.0
   * - Simulator environment
     - Torch 2.11.0+cu128; Transformers 4.57.6; NumPy 1.26.4; MuJoCo 3.3.0; robosuite 1.5.2; rpent-libero/rpent-liberopro 0.2.0; SAM3 0.1.0

Check the active environment with ``uv pip freeze``, ``python -c 'import torch; print(torch.__version__, torch.version.cuda)'`` and ``git -C "$MEGATRON_PATH" rev-parse HEAD``. Preserve a package inventory after validating it. A matching CUDA compiler is needed for source builds; a networked CPU-only machine can download wheels/sources, but does not necessarily build the CUDA extensions. For offline nodes, prepare the complete dependency set on a compatible Linux/Python/CUDA build host, transfer it and the Git sources, then install with ``uv pip install --no-index --find-links /path/to/wheelhouse ...``. A partial wheel cache is not a complete offline installation.

All Ray nodes need the same RLinf environment, source revision and model files. Keep the Transformers versions separate; do not change either to match the other. When supplying custom activation profiles, preserve the installer's Megatron, CUDA and NCCL paths. This recipe uses FA2 and disables Torch compilation; set ``NVTE_FLASH_ATTN=1``, ``NVTE_FUSED_ATTN=0`` and ``TORCH_COMPILE_DISABLE=1`` before starting Ray. EGL must work on the environment node.

For an exact replay, check out the two commit IDs recorded with that run before installing. A branch name identifies the delivery channel, not an immutable experiment version.

Start Three Ray Nodes And SAM3
------------------------------

Before ``ray start`` on **each node**, source the RLinf environment and export ``MEGATRON_PATH``, ``NVTE_FLASH_ATTN=1``, ``NVTE_FUSED_ATTN=0``, ``TORCH_COMPILE_DISABLE=1``, and a suitable ``RLINF_TIMEOUT`` such as ``10800``. Choose the actual network interface; ``RLINF_COMM_NET_DEVICES=eth0`` is an example, not a universal interface name. Ray workers inherit this environment at startup.

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"  # same path used during installation
   export MEGATRON_PATH="$STACK_ROOT/repos/Megatron-LM"
   source "$STACK_ROOT/envs/rlinf/bin/activate"
   export PYTHONPATH="$STACK_ROOT/repos/RPent:$STACK_ROOT/repos/RLinf:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal
   export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 TORCH_COMPILE_DISABLE=1
   export RLINF_TIMEOUT=10800
   export HEAD_IP=10.0.0.10  # reachable head-node address

Then start the head on model node 0:

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head --port 6387 --node-ip-address="$HEAD_IP" --num-gpus 8

On model node 1 and environment node 2, respectively, use the same source/environment and set the corresponding rank and local address:

.. code-block:: bash

   export RLINF_NODE_RANK=1  # use 2 on the environment node
   export NODE_IP=10.0.0.11 # use the environment node's address there
   ray start --address="$HEAD_IP:6387" --node-ip-address="$NODE_IP" --num-gpus 8

Run ``ray status --address="$HEAD_IP:6387"`` and verify three nodes and 24 GPUs. Do not start a second conflicting Ray cluster on an occupied node. Verify multi-node NCCL communication before a long job; network/firewall/RDMA settings depend on the cluster.

In a separate terminal on node 2, start SAM3 using the RPent environment:

.. code-block:: bash

   export STACK_ROOT="$HOME/rpent-stack"  # same path used during installation
   source "$STACK_ROOT/envs/rpent/bin/activate"
   cd "$STACK_ROOT/repos/RPent"
   export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8101

The agent workers and episode processes all run on node 2, so ``http://127.0.0.1:8101`` reaches their local SAM3 service. The local policy HTTP endpoint forwards model requests through RLinf channels to the two model nodes. There is no requirement to expose SAM3 across nodes.


Return to :doc:`rpent_libero_rl` to launch training or evaluation.

PPO and SAO training use full-vocabulary sampling by default, with temperature 1, top-k -1, top-p 1, presence penalty 0 and repetition penalty 1. Evaluation retains the standalone planner sampling settings; explicitly use the same sampling settings when comparing policies. Thinking is enabled, with no separate limit on thinking tokens (``agentloop.thinking_token_budget: null``). Set this option explicitly to impose a limit.

Training uses the SGLang Server backend by default. Set ``rollout.sglang.backend_type=engine`` to use the in-process Engine backend. Both use the same rollout and weight synchronization interfaces. The SAO recipe collects episodes asynchronously with ``runner.synchronous: false``; actor and critic updates remain sequential. Rollout can overlap training when they use separate GPUs. Shared GPUs require rollout to pause and offload before training. This recipe does not use the critic-free ``runner.async_mode`` path.

Configure advanced options in :doc:`../../guides/agent-workflows/rpent-multimodal`.
