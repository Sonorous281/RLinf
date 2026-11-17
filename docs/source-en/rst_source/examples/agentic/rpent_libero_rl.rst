RPent Multimodal Reinforcement Learning
=======================================

Train a Qwen3.5-4B planner on LIBERO with RPent and RLinf. RPent executes six end-effector tools and collects observations; RLinf samples model responses and updates the actor and critic. This workflow trains the planner with images and text, while the vision encoder and projector stay frozen.

.. important::

   The default image path preserves sampled tokens and checks image grids and positions. Experimental shared multimodal preprocessing also checks processor, pixel-value and M-RoPE replay. These checks establish input consistency, not measured SGLang/Megatron logprob parity for either path. Shared multimodal preprocessing is off by default and requires the dependency versions in the installation guide.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model

      Full multimodal Qwen3.5-4B checkpoint

   .. grid-item-card:: Algorithm

      Synchronous PPO or asynchronous SAO

   .. grid-item-card:: Tools

      Direct end-effector control and SAM3 segmentation

   .. grid-item-card:: Hardware

      Single-node A100 validation; optional three-node deployment

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Workflow
     - Scope
   * - This page: ``libero_rl``
     - Standard LIBERO, 40 tasks; seeds 1–32 for training, seed 0 for evaluation. These are initial-state splits of the same tasks.
   * - :doc:`Original Planner PPO <rpent>`
     - LIBERO-PRO with Pi0.5, task memory and text observations. Use its own configurations and installation instructions.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Contract
   * - Observation
     - Measured state after initialization/actions; ``view_env`` returns camera images.
   * - Action
     - ``move_to``, ``set_gripper``, ``view_env``, ``segment``, ``back_project`` and ``finish``.
   * - Reward
     - Native environment success: 1 for success, 0 for failure. A planner ``finish`` call alone does not establish success.
   * - Task prompt
     - Task language with the selected tool prompt; default ``simple``.

Installation
------------

Use the paired ``dev/rpent-rlinf-integration`` branches of ``Sonorous281/RLinf`` and ``Sonorous281/RPent``. Complete :doc:`rpent_libero_rl_setup` for the two Python environments, full model and SAM3 weights, simulator assets, task data and Ray services. The baseline branch's README specifies a different RPent revision and must not be used for this deployment.

Start with the single-node profile below. The larger deployment uses three nodes with eight GPUs each; measure memory and throughput before scaling. The standalone RPent collector uses a different serving environment and defaults to LIBERO-PRO.

Prepare the Task Data
---------------------

The task generator writes ``train.jsonl`` and a manifest with task IDs, initial-state seeds, counts and hashes. Each input line has a task instruction and a JSON-encoded ``answer``:

.. code-block:: json

   {"prompt": "open the bottom drawer", "answer": "{\"suite\": \"libero_goal\", \"task\": 0, \"seed\": 1}"}

The suite/task pair selects the simulator task; use the generator to obtain its actual instruction. Rollouts retain sampled token IDs, behavior logprobs, image inputs, policy versions and the native outcome. Only model-generated response tokens contribute to the policy loss. Tool observations and forced closing tokens have zero loss mask.

Configure Deployment Paths
--------------------------

After installation, copy the deployment path template outside the repository:

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf-token-space/bin/activate"
   cd "$STACK_ROOT/repos/RLinf"
   mkdir -p "$STACK_ROOT/config/deployment"
   cp examples/agent/rpent/config/deployment/paths.yaml "$STACK_ROOT/config/deployment/local.yaml"

Edit ``$STACK_ROOT/config/deployment/local.yaml`` and replace every ``???`` with an absolute path. Set ``data.train_data_paths`` and ``data.val_data_paths`` to YAML lists of training and evaluation JSONL files. Fill the remaining fields with the log directory, RPent interpreter, LIBERO config directory, episode directory and full-model directory. Training, evaluation and resume share this file; command-line overrides still take precedence. Retain it with Hydra's resolved configuration to reproduce the run.

Run It
------

Use ``train_qwen3.5_4b_libero_rl_sao_a100`` for a bounded Batch8/Env4 run. It enables shared multimodal preprocessing, logprob recalculation and TIS. Thinking has no token budget unless explicitly configured. It uses the experimental installation described on the setup page; the default SGLang environment cannot run this profile. The source candidate passed a smoke test, completed two updates, and resumed from step 2 to step 3 on four A100 80 GB GPUs. The revised code has passed CPU and configuration checks; the GPU run has not been repeated.

.. list-table::
   :header-rows: 1

   * - Resource
     - Assignment
   * - GPU 0–1
     - Actor and Critic, TP2; sequential updates with CPU optimizer offload
   * - GPU 2
     - SGLang Server, TP1, at most 4 active requests
   * - GPU 3
     - One shared SAM3 service and EGL rendering for 4 environments
   * - Host RAM
     - Allow 416 GiB for the job and leave at least 144 GiB available for other processes

Reserve at least 400 GB of free disk for three checkpoints (about 128 GB each), plus episode images and preprocessing cache. The YAML sets no RAM or disk watchdog; monitor those limits in the job scheduler.

The profile caps context at 16384 tokens, output at 8192 tokens and an episode at 25 turns. These are validation settings, not recommended limits for a learning experiment. GPU 4–7 are outside this job. Four environments share the same SAM3 endpoint; do not start one SAM3 copy per environment.

Confirm that the Ray cluster and SAM3 service started on the setup page are still running. The A100 profile connects to SAM3 on GPU 3 at port 18147. If using an existing shared service, set ``tools.rpent.sam3_endpoint`` to its address; its owner controls its lifecycle.

Generate the seed-0 evaluation list and select one episode. Use the same Ray and SAM3 services for smoke, training and recovery. The smoke caps each episode at 3 turns and 40 environment steps, with 1024 generated tokens per request:

.. code-block:: bash

   "$STACK_ROOT/envs/rpent/bin/python" examples/agent/rpent/prepare_tasks.py \
     --workflow libero_rl --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
     --seeds 0 --split eval --output-dir "$STACK_ROOT/data/libero-rl-eval"
   head -n 1 "$STACK_ROOT/data/libero-rl-eval/eval.jsonl" > "$STACK_ROOT/data/libero-rl-eval/smoke.jsonl"
   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl_smoke_a100 \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local \
     data.val_data_paths="[$STACK_ROOT/data/libero-rl-eval/smoke.jsonl]"

Append ``--cfg job --resolve`` to inspect the final configuration without loading a model. The smoke writes ``runner.output_dir/rpent-libero-rl-smoke/eval_results.json`` with ``summary.episodes=1``. Check the native outcome and episode subprocess cleanup; a failed task within this short budget can still validate collection.

After the one-episode smoke passes, run two updates, then restore in a new training process:

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

Both commands write under ``$STACK_ROOT/logs``. Use fresh experiment directories; the second command starts a new driver and loads the complete checkpoint root, not its ``actor/hf_checkpoint`` subdirectory. Keep the episode images and preprocessing cache available; do not overwrite the first run. Check ``async_trace.jsonl`` for step 3, actor/critic update counts 3/6 and published version 3. Confirm sample IDs do not repeat those consumed before the checkpoint, both optimizers resume, and the new ``global_step_3/data/data.pt`` marker exists. A zero process exit code alone does not establish recovery.

Scale to Multiple Nodes
-----------------------

The single-node commands above complete training and recovery. For larger workloads, install the default profile on three nodes and use the optional multi-node setup instructions before launching SAO:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_local \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local

For synchronous PPO, use ``train_qwen3.5_4b_libero_rl_ppo`` and adjust placement and batch size for the allocated resources. SAO also changes the algorithm: it uses DIS loss and SAO GAE, updates the critic, recomputes values, then updates the actor. It is more than asynchronous scheduling of PPO. Check finite losses and gradients, parameter updates, weight synchronization and recovery before scaling. A short smoke run does not establish capacity for the full 262144-token workload.

Visualization and Results
-------------------------

The archived four-A100 candidate run recorded the following infrastructure results. These are not fresh results for the revised source checkout, and they do not measure learning improvement.

.. list-table::
   :header-rows: 1

   * - Stage
     - Observed result
   * - Two training steps, Batch8/Env4
     - Actor/Critic updates: 2/4; published version: 2; consumed sample IDs: 0–15
   * - New-process recovery, step 2 to step 3
     - Actor/Critic updates: 3/6; published version: 3; consumed sample IDs: 16–23, with no reuse
   * - Checkpoint at step 3
     - Both optimizer states, model exports and rollout data state; 20 files, about 128 GB


.. list-table::
   :header-rows: 1

   * - Evidence
     - What to check
   * - Run outputs
     - Training logs and actor/critic checkpoints under ``runner.output_dir`` / experiment name; episode requests, images and results under ``tools.rpent.output_dir``.
   * - Runtime checks
     - Finite losses/gradients, nonzero parameter updates, successful weight synchronization and checkpoint recovery on the deployed revision.
   * - Task results
     - ``rpent/success_rate`` reports rollout outcomes. Report native successes / completed cases and infrastructure failures separately; use matched independent evaluation to measure learning.
   * - CPU component regression
     - Covers prefixes, masks, trajectory merging, Gloo data exchange, process cleanup and config composition. Tests skipped for missing Megatron/SGLang dependencies do not count as passes.
   * - Real GPU numerical checks
     - Not yet validated. Validate the default and shared preprocessing paths separately on the fixed 96 points (mean/P95/max; 0.03/0.10 are diagnostic references), multi-turn images and M-RoPE. Recomputing logprobs or using TIS does not eliminate native logprob error.
   * - Checkpoint and recovery
     - The source candidate completed two updates and fresh-process recovery to step 3. The revised code has not repeated the GPU run. Verify both optimizers, post-update weight synchronization and fresh-process resume; SAO also needs buffer/pending-state and policy-version checks.
   * - Independent evaluation
     - No result establishes a learning gain. Compare native success before and after training with matched tasks, initial states and budgets.

Also inspect ``rpent/turns`` and ``rpent/infra_retry_rate``. The template sets ``agentloop.max_infra_retry_rate=1.0`` to accept recovered retries; exhausted per-episode retries still fail. Earlier numerical checks with forced closing tokens exceeded the diagnostic tolerance and do not validate this deployment.

Resume and Evaluate
-------------------

Resume Training
~~~~~~~~~~~~~~~

Use ``runner.resume_dir=/path/to/checkpoints/global_step_N`` to resume a complete SAO checkpoint, retaining both model/optimizer states and its buffered/pending rollout state. Use only a completed checkpoint with actor/critic state and the final ``data/data.pt`` marker. Pending prompts are requeued; this is not a persisted simulator state for halfway-finished episodes. Keep the dataset, algorithm, world size and referenced images compatible. A synchronous PPO checkpoint cannot provide the SAO queue state.

Independent Evaluation
~~~~~~~~~~~~~~~~~~~~~~

Run the commands below on separately allocated resources. The default placement uses one eight-GPU model node and a separate environment node; adjust it for your evaluation deployment.

The evaluation template uses Planner sampling: temperature 1, top-k 20, top-p 0.95 and presence penalty 1.5. Training uses full-support sampling. Keep evaluation sampling fixed between the starting and trained models, and explicitly align task selection, episode budgets and the history window. The A100 smoke is an infrastructure check, not a learning evaluation. RPent’s standalone collector uses a separate serving workflow.

The export needs the same full VLM used to initialize training, because it retains the frozen vision weights and processor. Replace ``--vlm`` accordingly when training from SFT. Use a new output directory; an HF export is for inference and cannot replace the optimizer/queue checkpoint for resume.

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

Run the same evaluation again for the starting model by changing only the model and output paths. Keep the manifest, task/initial-state IDs, prompt, sampling, turn/action budgets and compaction fixed. Report native success with numerator/denominator, turn-budget exhaustion and infrastructure failures separately. Training completion-queue success is not independent evaluation, and a fixed 40-case panel alone cannot establish a small learning gain.

Choose a Training Configuration
-------------------------------

.. list-table::
   :header-rows: 1

   * - Config (without ``.yaml``)
     - Use
   * - ``train_qwen3.5_4b_libero_rl_ppo``
     - Synchronous PPO
   * - ``train_qwen3.5_4b_libero_rl_sao``
     - SAO with completed-episode scheduling
   * - ``train_qwen3.5_4b_libero_rl_sao_3node``
     - SAO placement across three nodes
   * - ``train_qwen3.5_4b_libero_rl_sao_local``
     - Three-node SAO with required local paths
   * - ``eval_qwen3.5_4b_libero_rl``
     - Independent multimodal evaluation

The launchers default to the original ``libero`` configurations. Always pass the ``libero_rl`` config name explicitly. The PPO → SAO → three-node → local profiles inherit common settings.

SAO uses DIS actor loss (clip bounds 0.3/5.0), ``sao_gae`` (alpha 1.5), the SAO critic loss and recorded behavior logprobs. The fully asynchronous :doc:`Search-R1 <searchr1/async>` recipe has separate rollout GPUs and a different update schedule.

For long updates, ``rollout.drain_before_training=true`` finishes active episodes before pausing generation. Its buffer capacity must cover the in-flight limit. This changes admission timing, not task timeouts.

For recalculation/TIS, thinking budgets, shared multimodal preprocessing, caching and prefetching, use :doc:`../../guides/agent-workflows/rpent-multimodal`. The base SAO configuration defaults to ``recompute_logprobs=false`` and ``importance_sampling_fix=false``: it trains from sampling logprobs without correcting backend differences or policy lag. Validate that choice on the target workload before scaling. The A100 validation profile enables both recalculation and TIS.

Keep a Reproduction Record
--------------------------

.. list-table::
   :header-rows: 1

   * - Record
     - Contents
   * - Code
     - Full RLinf and RPent commit SHAs; include a source diff when testing an uncommitted checkout.
   * - Environment
     - Python, Torch, CUDA, SGLang, Megatron Core/Bridge and kernel versions; model directory and revision.
   * - Workload
     - Task manifest and hash, task IDs, seeds, split, and resolved Hydra config.
   * - Resources
     - GPU model, count and placement; peak GPU/host memory and job duration.
   * - Evidence
     - Separate CPU, GPU inputs/logprobs, updates, synchronization, new-process recovery and evaluation results; retain logs and failed attempts.

.. toctree::
   :hidden:

   rpent_libero_rl_setup
