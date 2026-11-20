RPent Episode PPO and GRPO
==========================

Train a Qwen3.5-4B planner while remote LIBERO-PRO environments execute its tool calls.
Use the same full multimodal SFT checkpoint and the same RPent workflow for rollout,
training and evaluation. This recipe uses the ``episode`` workflow with explicit
perception and ordered batches of at most three actions.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models

      Qwen3.5-4B planner; frozen vision encoder and projector

   .. grid-item-card:: Algorithms

      Synchronous GRPO, or PPO with an independent critic

   .. grid-item-card:: Tasks

      LIBERO-PRO Goal, Spatial and Object

   .. grid-item-card:: Hardware

      Four 8-H200 hosts; model, SAM3 and environment processes share GPUs

Tasks
~~~~~

The supplied GRPO configuration runs 150 updates. Each update selects whole groups.

.. list-table::
   :header-rows: 1

   * - Item
     - Setting
   * - Training pool
     - Native states of 30 tasks; duplicate states within a task are removed, preserving original state IDs
   * - GRPO batch
     - 32 groups × 8 episodes = 256 episodes; all eight members share a task and native initial state
   * - Oversampling
     - Start 40 groups × 8 episodes; select the first 32 complete groups, cancel the other 64 episodes, confirm generation has stopped and release all 320 environments before learning
   * - Reward
     - Native success 1, normal failure 0; retain all-zero and all-one groups
   * - Regression panel
     - 30 tasks × seeds 0 and 1 = 60 training-pool episodes

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The workflow preserves the SFT planner's prompts, tool schemas and history policy.

.. list-table::
   :header-rows: 1

   * - Item
     - Contract
   * - Observation
     - Explicit camera observations at 1024 pixels; a turn before the first observation can contain no image
   * - Action
     - ``execute_actions`` with at most three ordered actions; ``view_env``, ``segment``, ``back_project`` and the ``finish`` output tool
   * - Reward
     - Environment success; a planner's ``finish`` response does not establish success
   * - Prompt
     - Thinking enabled without an effort tier; preserve reasoning and images, compact at 70% of context or 80 messages and retain 10 messages
   * - Limits
     - 262144 context tokens, up to 16384 output tokens per request, 50 requests, 10000 environment steps, 7200 seconds per planner and 14400 seconds per collector

How GRPO Works
--------------

The learner computes advantages from each group's mean and standard deviation.
It keeps every selected group, including groups whose rewards are identical.
There is no critic or second batch-wide advantage normalization.
The configuration uses group-, agent- and turn-level loss scaling, token-mean
aggregation, clip bounds 0.2/0.28, learning rate 1e-6 with zero warmup, zero KL
and entropy coefficients, gradient clipping at 1 and one pass per batch.
Sampled reasoning and tool-call tokens contribute to learning; prompt, tool
feedback, image and padding tokens have mask zero.

Selecting faster complete groups can bias which tasks reach training. Compare
selected and cancelled groups using the sampling manifests. Different training
batches are not a controlled evaluation of policy improvement.

Installation
------------

Use the Linux x86-64 ``reason`` image on each H200 node. Keep the image’s Python 3.12.12,
Torch 2.11.0/CUDA 12.8, torchvision 0.26.0, FA2 2.8.3 and Transformer Engine
2.17 in ``/opt/venv/reason``. The host needs its NVIDIA driver,
CUDA 12.8 development toolkit and EGL/OpenGL libraries.

Put the two ``rpent-ppo`` checkouts under ``$STACK_ROOT/repos/RLinf`` and
``$STACK_ROOT/repos/RPent``. Copy the complete offline package to
``$STACK_ROOT/envs/offline_package`` and the separately downloaded models and
assets to ``$STACK_ROOT/models`` and ``$STACK_ROOT/data``. Record both Git
revisions. Install on every node, then activate the same environment:

.. code-block:: bash

   export STACK_ROOT=/mnt/public/xzxuan
   bash "$STACK_ROOT/envs/offline_package/releases/h200-minimal-20261120-v5/install.sh"
   source "$STACK_ROOT/envs/offline_package/releases/h200-minimal-20261120-v5/activate.sh"
   python -m pip check

The installer checks image versions and SHA256 hashes, backs up replaced
packages and installs local wheels without network access. Repeating it keeps
the same package versions. ``requirements/rpent/README.md`` describes the
package layout and dependency builds.

The minimal release keeps the image's SGLang 0.5.12.post1, Transformers 5.6.1,
NumPy 2.2.0, Megatron-Core checkout at ``dc0ee41ad`` and official Bridge 0.5.0.
It installs FLA core 0.6.0, MuJoCo 3.3.0, robosuite 1.5.2, SAM3 and the paired
projects. OpenAI SDK 3.19.0 supplies the interfaces required by RPent's planner.
The benchmark wheels use the prepared robosuite 1.5 adapter, including the
``NullBase`` declaration for ground-mounted Panda models.
Core and Bridge are neither patched nor repackaged; preserve the Core source
path set by the image activation. The installer retains its pip-check report;
optional metadata conflicts are distinct from runtime failures.

RLinf captures serving prompt IDs, pixels, image spans, grid and M-RoPE for
learner replay with the unmodified SGLang release. The earlier compatibility
wheel bundle remains a separate historical release.

RLinf, its RPent planner subprocesses, SGLang, SAM3 and LIBERO all use
``/opt/venv/reason/bin/python``. Use this path for ``tools.rpent.python``.
The installed benchmark packages link to the shared assets. Follow RPent's
``tools/rl/INSTALL.md`` for model and asset preparation. Do not replace an
environment while its jobs are running.

Download the Model
------------------

Prepare the following assets before launching services. Python installation
does not download model weights or scene assets.

.. list-table::
   :header-rows: 1

   * - Asset
     - Used by
     - Preparation
   * - Full Qwen3.5-4B SFT directory
     - RLinf rollout and learner
     - Weights, config, tokenizer, chat template and processor at the same absolute path on all model hosts
   * - SAM3 checkpoint
     - Environment host's SAM3 services
     - Request access to ``facebook/sam3``, then download ``sam3.pt``
   * - LIBERO-PRO assets
     - Environment workers
     - Scene assets plus the package's BDDL files and native initial states

For an SFT comparison, point ``MODEL_PATH`` to the original full multimodal
SFT directory, including its processor and tokenizer files:

.. code-block:: bash

   export MODEL_PATH=/absolute/path/to/full-sft-checkpoint

For a base-model run, prepare a separate model directory instead:

.. code-block:: bash

   export MODEL_PATH="$STACK_ROOT/models/Qwen3.5-4B"
   hf download Qwen/Qwen3.5-4B --local-dir "$MODEL_PATH"

Download the SAM3 checkpoint on the environment host:

.. code-block:: bash

   export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
   hf download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"

The direct-control ``episode`` workflow needs no Pi0.5 weights. Pi0.5 belongs
to the separate :doc:`original planner recipe <rpent>`; changing to that recipe
changes the model's action tools. For a checkpoint trained with the episode workflow, keep ``workflow=episode``.

Run It
------

Start SAM3 and the Environment Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On each of the four nodes, activate the environment and start its local services:

.. code-block:: bash

   source "$STACK_ROOT/envs/offline_package/releases/h200-minimal-20261120-v5/activate.sh"
   python -m robots.rl.services \
     --sam3-checkpoint "$SAM3_CHECKPOINT_PATH" \
     --envs-per-gpu 3 --output-dir "$STACK_ROOT/logs/services-node0"

Use a fresh output directory on each start. The service manager launches one SAM3 per
GPU, a local pool on port 28424 and an environment backend on port 28490.
It stops its child processes on exit. Each GPU has three environment slots;
the four nodes can host all 80 smoke candidates concurrently. Validate environment
capacity separately before running the 320-candidate configuration.
Completed episodes release their slots; all slots are released before learning.
SAM3 stays resident during the learner phase.

Join the four nodes to one private Ray cluster. Export a distinct
``RLINF_NODE_RANK`` and activate the environment before starting Ray, as in
:doc:`/rst_source/guides/multi_node`. Each worker uses its node's loopback
backend. Run the training entry script only on the head.

Prepare Tasks and Local Paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate task files on the environment host, where LIBERO-PRO is installed:

.. code-block:: bash

   export TASK_DIR="$STACK_ROOT/tasks"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds all --split train \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"
   python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
     --workflow episode --libero-type pro --seeds 0,1 --split eval \
     --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$TASK_DIR"

Copy the task directory and manifests to the model hosts' shared filesystem.
The 60-episode panel measures training-pool regression, not generalization.

Create an external deployment file from the supplied template:

.. code-block:: bash

   export CONFIG_DIR="$STACK_ROOT/config"
   mkdir -p "$CONFIG_DIR/deployment"
   cp "$RLINF_ROOT/examples/agent/rpent/config/deployment/episode.yaml" \
     "$CONFIG_DIR/deployment/local.yaml"

Fill every ``???`` in ``local.yaml``. Use the generated training and evaluation
JSONL files for ``data.train_data_paths`` and ``data.val_data_paths``.
Write dataset paths as lists under ``data``:

.. code-block:: yaml

   data:
     train_data_paths: ["/data/tasks/train.jsonl"]
     val_data_paths: ["/data/tasks/eval.jsonl"]

Point each of these model paths to the same full SFT directory:

.. code-block:: text

   rollout.model.model_path
   actor.tokenizer.tokenizer_model
   critic.tokenizer.tokenizer_model
   actor.megatron.ckpt_convertor.hf_model_path
   critic.megatron.ckpt_convertor.hf_model_path

Set ``tools.rpent.python`` to ``/opt/venv/reason/bin/python`` and ``repo_path``
to the matching RPent checkout. The episode output and preprocessing cache
directories must have the same absolute paths on all model hosts. Keep image
files while trajectories or checkpoints reference them.

Launch Training
~~~~~~~~~~~~~~~

First run the four-node smoke from the Ray head. It selects 16 complete
groups of four from 20 candidate groups, runs one update and saves a checkpoint:

.. code-block:: bash

   cd "$RLINF_ROOT"
   source "$STACK_ROOT/envs/offline_package/releases/h200-minimal-20261120-v5/activate.sh"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_h200_smoke_grpo \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

For 27B acceptance, use the following profile with all deployment model and
tokenizer paths set to the full 27B checkpoint:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_27b_h200_grpo_5step \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

The profile launches 40 groups of eight episodes and trains on the first
32 complete groups. It runs five optimizer updates and saves a checkpoint after
each. Provision ten environments and one SAM3 per GPU.

Use rollout TP1 and actor TP4/CP1/DP8 with sequence parallelism (SP).
Disable multi-token prediction (MTP) and use 256-token logprob and LM-head chunks.
Keep Core and Bridge unchanged. LM-head chunking and response-only projection
support CP1. Before evaluating CP, set ``actor.model.lm_head_chunk_size=0`` and
``actor.response_only_logits=false``, then validate multimodal inputs, masks,
gradients and loss normalization separately.
The prepared ``Qwen3.8-27B`` directory declares the dense Qwen3.5 architecture.
RLinf reads its layer count, dimensions, attention heads and embedding ties
from the model config.
Restore a saved checkpoint and generate fresh replies from the restored weights
before accepting the run. Do not extend it to 150 updates automatically.

For a full run, use ``train_qwen3.5_4b_episode_sync_grpo``. It selects 32
complete groups from 40 candidate groups, with eight episodes per group.
Validate the environment service capacity first.
Actor placement is TP2/CP1/DP16; rollout uses 32 TP1 replicas.
The configuration keeps ``NCCL_MAX_CTAS=8`` for actor/critic/rollout and
SGLang's memory fraction at 0.55. This fraction is not a total memory cap for
a GPU shared with the learner. Keep placement and memory settings fixed when comparing runs.

To use PPO with the same preprocessing environment, select the PPO recipe
and the shared TokenSpace configuration:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_oversample320 \
     +preprocessing@_global_=token_space \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

PPO uses 320 independent candidates and the first 256 complete episodes,
an independent critic and GAE computed in temporal order over each complete
episode. GRPO's group normalization
does not apply to that PPO recipe.

Inspect the first complete optimizer update, weight synchronization and the next
online episode before leaving a new installation unattended. Completing a batch
of sampled episodes or recording a single TensorBoard tag does not prove that
an optimizer update has completed. GPU numerical parity and compatibility with other driver or kernel versions
remain unverified for a new installation.

Resume, Export and Evaluate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``runner.resume_dir`` to resume a complete checkpoint from the same
configuration. Check actor/critic model and optimizer shards as applicable,
scheduler/RNG state, and the atomically published data state. The dataset cursor
is local to its epoch; it need not equal the global update number. Preserve
partial raw episodes, and do not treat updates beyond the last complete
checkpoint as recoverable. See :doc:`/rst_source/guides/resume`.

After update 150, export the trained language policy into the original full
multimodal model:

.. code-block:: bash

   python examples/agent/rpent/checkpoint.py export \
     --checkpoint "$CHECKPOINT_150" --vlm "$MODEL_PATH" --output "$EXPORT_PATH"

Evaluate the export on the prepared panel with the matching model environment.
Use the TokenSpace rollout settings from the training profile:

.. code-block:: bash

   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_episode_token_space \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local \
     rollout.model.model_path="$EXPORT_PATH"

Use an output directory separate from training. Keep the original SFT directory
and every checkpoint. Normal failures and output/context/turn/time limits have
zero reward; only classified infrastructure failures have the configured bounded
retry. If the second infrastructure attempt also fails, stop accepting data from
that batch and investigate.

Visualization and Results
-------------------------

TensorBoard reports optimizer loss, gradient norm, learning rate, sampled
success, TIS diagnostics and update timings. The sampling manifests identify
the selected groups, cancelled members and cleanup receipts; compute success
rates from selected episodes rather than directory counts. The episode files
contain request/outcome data, image references and sampled training turns.

Full HTTP request/response and compaction traces are disabled for episode
training by default. Set ``tools.rpent.record_api_trace=true`` only when
investigating a planner transport problem; direct RPent launches use
``--record-api-trace``. This does not change the prompts, compaction decisions,
token masks or saved training records.

Report task failures, infrastructure failures, selected/cancelled group
composition, token/turn costs and the exact source/dependency revisions
separately. A 60-episode training-pool regression is not a held-out benchmark.

Use RGB-Only Control
--------------------

Select the pear recipe to train with paired RGB images and measured robot state after every action. It inherits the GRPO algorithm and the 32-group, eight-member batch with 40 candidate groups. The only tools are ``move_to`` for absolute poses and ``set_gripper``. Episodes end automatically when the environment reports success or a budget is exhausted. Invalid action replies are normal failures, with no correction request.

Use the same installation, full multimodal planner checkpoint, task manifests and deployment file above. Start a matching RPent backend; a pear-only backend needs no SAM3 endpoint or SAM3 weights. Launch from the Ray head:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_pear_sync_grpo \
     --config-dir "$CONFIG_DIR" +deployment@_global_=local

The recipe sets ``tools.rpent.world_cord=general``: positions are measured from the mounting base, with x forward, y left and z up, and the TCP is the measured finger-pad midpoint. The initial observation gives table height above the base. All LIBERO perception coordinates follow this choice when using an existing perception workflow. Set ``tools.rpent.world_cord=base`` to retain simulator-world coordinates and the original control point.

Thinking is enabled by default. Add ``agentloop.chat_template_kwargs.enable_thinking=false`` for direct tool calls without reasoning text. The RPent CLI uses ``--reasoning`` and ``--no-reasoning``. Use separate run directories and pin source revisions when changing workflows or coordinates; an existing SFT workflow remains available with its original defaults.
