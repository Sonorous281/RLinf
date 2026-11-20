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

      The deployment example uses two 8-GPU model nodes and one 8-GPU environment node

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

The setup page uses three nodes with eight GPUs each. Adjust placement and concurrency to your allocated resources before launch; Measure GPU memory requirements for your workload. The standalone RPent collector uses a different serving environment and defaults to LIBERO-PRO.

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

   source "$STACK_ROOT/envs/rlinf/bin/activate"
   cd "$STACK_ROOT/repos/RLinf"
   mkdir -p "$STACK_ROOT/config/deployment"
   cp examples/agent/rpent/config/deployment/paths.yaml "$STACK_ROOT/config/deployment/local.yaml"

Edit ``$STACK_ROOT/config/deployment/local.yaml`` and replace every ``???`` with an absolute path. Set ``data.train_data_paths`` and ``data.val_data_paths`` to YAML lists of training and evaluation JSONL files. Fill the remaining fields with the log directory, RPent interpreter, LIBERO config directory, episode directory and full-model directory. Training, evaluation and resume share this file; command-line overrides still take precedence. Retain it with Hydra's resolved configuration to reproduce the run.

Run One Episode First
---------------------

This configuration uses two GPUs on one node: GPU 0 serves the model, and GPU 1 handles simulator rendering and SAM3. Complete the installation and asset steps first, but start a single-node Ray cluster with ``RLINF_NODE_RANK=0``. Start SAM3 separately on GPU 1 and set ``tools.rpent.sam3_endpoint`` to its address. This two-GPU configuration has not passed GPU validation; measure its memory requirements before increasing the workload.

Select one fixed evaluation case and cap it at 3 turns, 40 environment steps and 1024 generated tokens per request:

.. code-block:: bash

   "$STACK_ROOT/envs/rpent/bin/python" examples/agent/rpent/prepare_tasks.py \
     --workflow libero_rl --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
     --seeds 0 --split eval --output-dir "$STACK_ROOT/data/libero-rl-eval"
   head -n 1 "$STACK_ROOT/data/libero-rl-eval/eval.jsonl" > "$STACK_ROOT/data/libero-rl-eval/smoke.jsonl"
   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl_smoke \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local \
     data.val_data_paths="[$STACK_ROOT/data/libero-rl-eval/smoke.jsonl]"

Append ``--cfg job --resolve`` to the ``run_eval.sh`` command to inspect the resolved configuration without loading a model. A completed run writes ``runner.output_dir/rpent-libero-rl-smoke/eval_results.json`` with ``summary.episodes`` equal to 1, and retains requests, outcomes and logs in the episode directory. Inspect tool calls and the native outcome, and confirm the episode subprocess has exited. The task can fail within this budget: the check covers collection and cleanup, not task success, numerical parity or learning.

Run It
------

After the single-episode check, allocate the three-node placement and start Ray as described in the installation guide, then launch SAO:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_local \
     --config-dir "$STACK_ROOT/config" +deployment@_global_=local

For synchronous PPO, use ``train_qwen3.5_4b_libero_rl_ppo`` and adjust placement and batch size for the allocated resources. SAO also changes the algorithm: it uses DIS loss and SAO GAE, updates the critic, recomputes values, then updates the actor. It is more than asynchronous scheduling of PPO. Check finite losses and gradients, parameter updates, weight synchronization and recovery before scaling. A short smoke run does not establish capacity for the full 262144-token workload.

For grouped GRPO with the original episode workflow, use ``train_qwen3.5_4b_episode_sync_grpo`` and an RPent checkout that provides ``robots.rl.rl_episode``. The recipe trains on 32 groups of 8 episodes and launches 40 candidate groups. All eight members share the task and initial state, but sample independently. The first 32 complete groups enter training; the other groups are cancelled and cleaned up before learning. Groups with identical rewards remain in the batch. This recipe uses no critic and sets the upper clipping bound to 0.28. It applies ``group_level``, ``agent_level`` and ``turn_level`` loss scales to the existing trajectory, agent and valid-token fields.

The GRPO recipe captures the exact serving prompt tokens and multimodal inputs for learner replay. With the minimal H200 environment, RLinf supports the unmodified SGLang 0.5.12.post1 release; no patched serving wheel or preprocessing-cache path is required. Collect fresh rollouts with capture enabled, since legacy image records lack these inputs. The planner keeps the original episode workflow, prompts, tools and action schema. See :doc:`rpent_episode_sync` for installation and the five-step 27B acceptance configuration.

Visualization and Results
-------------------------

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
     - Full training acceptance is pending for this branch. Verify both optimizers, post-update weight synchronization and fresh-process resume; SAO also needs buffer/pending-state and policy-version checks.
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

The template matches the training task, sampling, episode budgets and history window. Apply any training overrides to evaluation too. RPent's standalone batch collector is for serving and data collection; ``eval_qwen3.5_4b`` belongs to the original LIBERO workflow.

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

For recalculation/TIS, thinking budgets, shared multimodal preprocessing, caching and prefetching, use :doc:`../../guides/agent-workflows/rpent-multimodal`. SAO defaults to ``recompute_logprobs=false`` and ``importance_sampling_fix=false``: it trains from sampling logprobs without correcting backend differences or policy lag. Validate that choice on the target workload before scaling.

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
