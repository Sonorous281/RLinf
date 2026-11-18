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

Use RLinf branch ``feat/rpent-planner-ppo-sao`` and the pinned RPent revision on the setup page. Complete :doc:`rpent_libero_rl_setup` for the two Python environments, full model and SAM3 weights, simulator assets, task data and Ray services. The baseline branch's README specifies a different RPent revision and must not be used for this deployment.

Start with the single-node profile below. The larger deployment uses three nodes with eight GPUs each; measure memory and throughput before scaling. The standalone RPent collector uses a different serving environment and defaults to LIBERO-PRO.

Prepare the Task Data
---------------------

The task generator writes ``train.jsonl`` and a manifest with task IDs, initial-state seeds, counts and hashes. Each input line has a task instruction and a JSON-encoded ``answer``:

.. code-block:: json

   {"prompt": "open the bottom drawer", "answer": "{\"suite\": \"libero_goal\", \"task\": 0, \"seed\": 1}"}

The suite/task pair selects the simulator task; use the generator to obtain its actual instruction. Rollouts retain sampled token IDs, behavior logprobs, image inputs, policy versions and the native outcome. Only model-generated response tokens contribute to the policy loss. Tool observations and forced closing tokens have zero loss mask.

Configure Single-Node Paths
---------------------------

Set ``STACK_ROOT`` as shown in the setup guide. The commands below use ``+deployment@_global_=a100`` to locate the model, task lists and simulator interpreter. Run them in the terminal where you started Ray:

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf-token-space/bin/activate"
   cd "$STACK_ROOT/repos/RLinf"

For a different layout, copy ``examples/agent/rpent/config/deployment/paths.yaml`` to ``deployment/local.yaml`` under your own config directory and fill in the absolute paths. Replace the deployment option below with ``--config-dir /path/to/config +deployment@_global_=local``.

Run It
------

The four-GPU A100 profile uses 8 episodes per update and 4 environments. It enables shared multimodal preprocessing, logprob recomputation and TIS. The thinking budget defaults to unlimited.

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
     --workflow libero_rl --libero-type standard --libero-config-path "$STACK_ROOT/data/libero-standard-config" \
     --seeds 0 --split eval --output-dir "$STACK_ROOT/data/libero-rl-eval"
   head -n 1 "$STACK_ROOT/data/libero-rl-eval/eval.jsonl" > "$STACK_ROOT/data/libero-rl-eval/smoke.jsonl"
   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl_smoke_a100 \
     +deployment@_global_=a100 \
     data.val_data_paths="[$STACK_ROOT/data/libero-rl-eval/smoke.jsonl]"

Append ``--cfg job --resolve`` to inspect the final configuration without loading a model. The smoke writes ``runner.output_dir/rpent-libero-rl-smoke/eval_results.json`` with ``summary.episodes=1``. Check the native outcome and episode subprocess cleanup; a failed task within this short budget can still validate collection.

After the one-episode smoke passes, run two updates, then restore in a new training process:

.. code-block:: bash

   source "$STACK_ROOT/envs/rlinf-token-space/bin/activate"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_a100 \
     +deployment@_global_=a100

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_a100 \
     +deployment@_global_=a100 \
     runner.experiment_name=rpent-libero-rl-a100-resume \
     runner.resume_dir="$STACK_ROOT/logs/rpent-libero-rl-a100/checkpoints/global_step_2" \
     runner.max_steps=3

Both commands write under ``$STACK_ROOT/logs``. The second starts a fresh driver and loads the complete checkpoint root, not ``actor/hf_checkpoint``. Keep the original episode images and preprocessing cache. Check the step, two critic updates, actor update and weight synchronization in the logs, and verify that both optimizers continue their saved counters. The new directory should contain ``global_step_3/data/data.pt``. A normal exit alone does not prove recovery.

For synchronous PPO, use ``train_qwen3.5_4b_libero_rl_ppo`` and adjust placement and batch size. SAO updates the critic, recomputes values, then updates the actor. It also checks the infrastructure retry rate. Validate small updates, synchronization and recovery before trying more nodes or longer contexts.

Eight-GPU SAO Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reserve eight A100 80 GB GPUs. Actor and Critic use GPUs 0–3 (TP2, DP2), two TP1 rollout servers use GPUs 4–5, and GPUs 6–7 render 16 environments each. The environments share one SAM3 service on GPU 6, port 18261. Use the eight-GPU Ray and SAM3 settings on the setup page and the 1,280-row standard LIBERO training file (seeds 1–32). The original job reserved 576 GiB of host RAM plus 144 GiB of headroom.

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_a100_8gpu \
     +deployment@_global_=a100 \
     rollout.collect_meta_stats=true

This profile admits 32 episodes per update, allows 100 turns per episode and saves every five updates. Set ``runner.max_steps=N`` for a bounded run. Long visual sequences exceeded the four-GPU profile’s CPU prefetch budget, so this profile loads learner images on demand. Request profiling is optional. The override above records per-rank request metadata, including end-to-end latency when SGLang supplies it.

Visualization and Results
-------------------------

The GPU observations below are from RLinf ``a352ea10`` with the runtime fixes included in this recipe and RPent ``5a98ff3``. The eight-GPU run also enabled request profiling. These runs used an existing installation of the pinned dependencies.

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
     - On four A100s, SAO saved a checkpoint at step 2. A new training process restored it, updated the actor and critic, and saved a checkpoint at step 3. The rollout server loaded weight version 3. On eight A100s, the first Batch32/Env32 update completed and both rollout servers loaded version 1. Eight-GPU recovery still needs verification.
   * - Independent evaluation
     - No result establishes a learning gain. Compare native success before and after training with matched tasks, initial states and budgets.

Also inspect ``rpent/turns`` and ``rpent/infra_retry_rate``. The template sets ``agentloop.max_infra_retry_rate=1.0`` to accept recovered retries; exhausted per-episode retries still fail. Earlier numerical checks with forced closing tokens exceeded the diagnostic tolerance and do not validate this deployment.

Resume and Evaluate
-------------------

Resume Training
~~~~~~~~~~~~~~~

Use ``runner.resume_dir=/path/to/checkpoints/global_step_N`` to resume a complete SAO checkpoint, retaining both model/optimizer states and its buffered/pending rollout state. Use only a completed checkpoint with actor/critic state and the final ``data/data.pt`` marker. Pending prompts are requeued; this is not a persisted simulator state for halfway-finished episodes. Keep the dataset, algorithm, world size and referenced images compatible. This branch only reads ``rpent-episode-buffer-v1``; old integration-branch SAO queue checkpoints are not supported. Consumed groups are not replayed, completed unconsumed groups are retained, and unfinished episodes restart under the same group IDs. A synchronous PPO checkpoint cannot provide the SAO queue state.

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
     +deployment@_global_=a100 \
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

For shared multimodal preprocessing, caching and prefetching, use :doc:`../../guides/agent-workflows/qwen35-multimodal`. The base SAO configuration defaults to ``recompute_logprobs=false`` and ``importance_sampling_fix=false``: it trains from sampling logprobs without correcting backend differences or policy lag. Validate that choice on the target workload before scaling. The A100 validation profile enables both recalculation and TIS.

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

Configuration Migration
-----------------------

Synchronous advantages use upstream ``gae_dynamic`` instead of ``actor_gae``. Critic loss moves to ``algorithm.critic_loss_type``: ``critic`` for PPO and ``mse`` for SAO. Async mode sets ``runner.async_mode=true``. Admission and buffer capacities use ``rollout_buffer.max_inflight_groups`` and ``max_buffered_groups``; policy lag is bounded by ``agentloop.max_actor_accept_staleness``. The old ``drain_before_training`` option is no longer used.

The A100 profile enables ``algorithm.recompute_logprobs`` and ``importance_sampling_fix`` with a TIS cap of 2. PPO ratios use the pre-update actor's recomputed probabilities; importance weights compare these with sampling probabilities. Tool observations, padding and forced tokens are excluded. Monitor ``tis/ess_fraction``, ``tis/clipped_fraction`` and logprob differences; TIS does not establish backend numerical equivalence. ``agentloop.thinking_token_budget`` defaults to ``null``; set it only to impose a thinking limit.

.. toctree::
   :hidden:

   rpent_libero_rl_setup
