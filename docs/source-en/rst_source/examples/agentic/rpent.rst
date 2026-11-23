Train an RPent planner on LIBERO
================================

RPent sends camera images to a Qwen3.5 planner and executes its tool calls as end-effector actions in LIBERO. RLinf updates the planner using the simulator's final success flag. A successful episode earns 1 and an unsuccessful terminal episode earns 0. Simulator and model-service errors fail the run.

Choose a recipe
---------------

* **Synchronous GRPO on LIBERO-Pro** samples eight episodes per task and initial state. Each update starts five groups, admits the first four complete groups and cancels the rest. The actor recomputes action logprobs for truncated importance sampling (clip 2).
* **Asynchronous SAO on standard LIBERO** collects episodes during learner updates. The critic fits values and recomputes the advantages used by the actor. On restart, the runner restores the learner and completed groups and reissues unfinished episodes with their original group IDs.

Both recipes use four NVIDIA GPUs on one machine. GPUs 0 and 1 run the tensor-parallel learner, GPU 2 runs one SGLang Server, and GPU 3 runs SAM3 and up to four EGL environments. The session server and SGLang share a temporary directory for image capture. Keep the run directory and its training images accessible when resuming.

Install the learner and simulator
---------------------------------

Set ``RPENT_MODEL`` to a local Hugging Face checkpoint. The public `Qwen3.5-4B <https://huggingface.co/Qwen/Qwen3.5-4B>`__ weights can run the integration. Reproducing the historical GRPO initialization requires the RPent SFT595 checkpoint. The single-node validation uses four A100 80 GB GPUs.

The RPent simulator counts context tokens with a tokenizer and an image processor. If its Transformers version cannot load the tokenizer class in an exported checkpoint, append ``tools.rpent.tokenizer_path=/path/to/compatible-tokenizer`` to the training or evaluation command. The original model directory can supply the tokenizer, chat template and image processor. Check that both environments count the same number of tokens for an identical prompt containing an image. Model weights still come from ``RPENT_MODEL``.

From the RLinf checkout, install the learner and activate its environment.

.. code-block:: bash

   UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic --python 3.12 --torch 2.11.0 --engine sglang --sglang 0.5.12.post1 --transformers 5.6.1 --install-rlinf
   source .venv/bin/activate

The four-GPU configuration offloads optimizer state to host memory. On a fresh Megatron Core 0.18 checkout at ``dc0ee41a``, apply the tested optimizer restore fix once before using checkpoints. If you installed from a separately managed Core checkout, use its directory in the second command.

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/Sonorous281/RLinf/e563431a3dd234643086a4baadcdb075772672ce/requirements/agentic/megatron-core-r0.18.0-cpu-offload.patch -o /tmp/rlinf-core-offload.patch
   git -C .venv/Megatron-LM apply /tmp/rlinf-core-offload.patch

Choose the simulator for your recipe. For GRPO and evaluation, the SFT595 policy uses up to three ordered end-effector actions per tool call.

.. code-block:: bash

   repo=RLinf/RPent revision=762a57c2b78278aff8d5c2c8f752ee42501bd5dc recipe=episode-ppo-env

For SAO, choose the revision with separate tools for motion and gripper commands. Its requirements load RLinf from a sibling directory named ``RLinf``.

.. code-block:: bash

   repo=Sonorous281/RPent revision=053fe8bd4bb2dbbd6a0652331261ef9949c68e2f recipe=libero-rl

From the RLinf checkout, install the selected revision in its own environment.

.. code-block:: bash

   export RLINF_ROOT="$PWD"
   rpdir="../RPent-$recipe"
   git clone --no-checkout "https://github.com/$repo.git" "$rpdir"
   git -C "$rpdir" fetch --depth 1 origin "$revision" && git -C "$rpdir" checkout --detach FETCH_HEAD
   (cd "$rpdir" && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --prerelease=allow --torch-backend cu128 -r "requirements/$recipe.txt")
   export RPENT_PYTHON="$(cd "$rpdir" && pwd)/.venv/bin/python"

If the first GRPO reset fails on ``libero_floor_base_style.xml``, the installed simulator package is missing its scene assets. MuJoCo resolves this XML inside the imported LIBERO-Pro package. Install the complete source tree in editable mode in the simulator environment.

.. code-block:: bash

   git clone --branch rpent https://github.com/RLinf/LIBERO-PRO.git ../LIBERO-PRO && git -C ../LIBERO-PRO checkout --detach d1e11fb181b8544487d27742c0caa3a4d46452ad
   uv pip install --python "$RPENT_PYTHON" --no-deps -e ../LIBERO-PRO

Prepare the XML assets, initial states and SAM3 weights with the `RPent resource instructions <https://github.com/Sonorous281/RPent/blob/053fe8bd4bb2dbbd6a0652331261ef9949c68e2f/docs/source-en/rst_source/simulators/libero_rl_setup.rst>`__. Set the paths below in your training shell. Run the last command in a separate terminal with the same environment.

.. code-block:: bash

   export RPENT_MODEL=/path/to/Qwen3.5-4B
   export LIBERO_CONFIG_PATH=/path/to/libero-config
   export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
   export RPENT_RUN_DIR="$PWD/logs"
   "$RPENT_PYTHON" -m rpent.robots.components.sam3_server --cuda-device 3 --port 8101

Train and evaluate
------------------

Generate training rows from the goal, object and spatial suites. Each JSONL row contains a language instruction and a JSON-encoded task with its suite, task index, seed and initial-state index.

.. code-block:: bash

   export RPENT_TRAIN_DATA="$PWD/data/rpent/train.jsonl"
   "$RPENT_PYTHON" examples/agent/rpent/prepare_tasks.py --libero-type pro --seeds 0-31 --output-dir data/rpent
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_grpo

To reuse long planner histories, set ``reuse_nested_prefixes: true`` under ``actor`` in the GRPO training YAML. This option is off by default. Two consecutive rows from the same episode can share a decoder pass if the shorter row is an exact prefix of the longer one, including its tokens, images and positions.

The actor computes the old log probabilities for both rows in one pass. During an update, it applies the output projection in each row's original microbatch, preserving its action mask, loss weight and importance-ratio check. Use the recipe's dense Qwen3.5 model, chunked output projection and single-row microbatches. Keep dropout, CUDA graphs and overlapping DP communication disabled.

For SAO, generate the task file with ``--libero-type standard`` and run:

.. code-block:: bash

   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_sao

To evaluate, set ``RPENT_MODEL`` to the exported Hugging Face checkpoint and reserve initial states outside the training split. Evaluation uses GPU 0 for inference and GPU 1 for rendering. Set ``RPENT_SAM3_URL`` if the SAM3 service uses a different address.

.. code-block:: bash

   export RPENT_EVAL_DATA="$PWD/data/rpent/eval.jsonl"
   "$RPENT_PYTHON" examples/agent/rpent/prepare_tasks.py --libero-type pro --seeds 32-39 --split eval --output-dir data/rpent
   bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b

Read ``rpent-eval/eval_results.json`` under ``RPENT_RUN_DIR`` for the episode count and the success rates overall and by suite. Each episode's ``request.json`` identifies the task, ``planner.json`` records its interaction and ``outcome.json`` holds the simulator verdict. In a training row, ``input_ids`` contains the rendered prompt and sampled response. The learner uses ``response_mask`` to score only policy actions. ``rollout_logprobs`` holds their behavior logprobs, ``rollout_weight_versions`` tracks the policy that produced each action token, and ``multi_modal_inputs`` references the image tensors. Tool observations become part of the next prompt.

In one standard LIBERO episode (suite ``libero_10``, task 0, seed 2), the planner called ``view_env``. The tool returned a 1024 × 1024 camera view. It then called ``segment``, which returned a mask with score 0.891. After 25 planner turns and 262 environment steps the episode reached its request limit. The recorded values were ``terminal=true``, ``success=false`` and reward 0. The segmentation score never enters the reward.

Reference run
-------------

A historical H100 run started from Qwen3.5-4B-SFT595 and used all 50 initial states for each of 30 LIBERO-Pro tasks. It admitted 32 of 40 groups per update and completed 96 of 150 planned updates. The first 10 updates recorded 688 successes in 2,560 admitted episodes (26.88%). The last 10 recorded 1,940 in 2,560 (75.78%). These are training rollouts. The H100 run used a 262,144-token context and up to 16,384 output tokens per turn. The four-GPU GRPO recipe uses 65,536 and 8,192, respectively.

A separate four-A100 GRPO smoke test used SFT595 on a fixed LIBERO-Pro task with thinking disabled, at most 16 turns and 2,048 output tokens per turn. It admitted four complete groups of eight and cancelled the fifth group. All 32 admitted rewards were zero and the actor gradient norm was zero. After saving the checkpoint, the SGLang Server reported policy version 1 and served the next planner requests.

The four-A100 SAO acceptance run used four environments, four episodes per batch, up to four planner turns per episode and 512 tokens per response. It completed two updates and a third after restarting from the full checkpoint. Each checkpoint occupied about 128 GiB. Peak host proportional set size during recovery was 403 GiB. None of the 24 terminal episodes in this short run solved its task.
