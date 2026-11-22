Replay SGLang Image Inputs in Megatron Training
===============================================

In a Qwen3.5 conversation, the second image turn builds on the tokens the model
sampled in the first. RLinf records each turn's token IDs, processed pixels,
image grid and M-RoPE positions. Megatron reads those inputs and runs its own
vision encoder.

Check two image turns on one GPU
--------------------------------

Activate the agentic environment and run this command from the repository root.
Select a free A100 80 GB and supply the path to your Qwen3.5-4B weights.
For a new environment, see :ref:`Environment setup <qwen35-replay-setup>`.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python examples/agent/qwen35/replay_images.py --model /path/to/Qwen3.5-4B

The script creates two test images and starts a local SGLang server.
``Turn 2 passed`` means the first turn's sampled history was preserved and
the learner's pixels, grids and positions passed the replay checks. The
``tokens`` count includes the input and output through the current turn. Each
turn generates at most one new token.

Check an optimizer update on two GPUs
-------------------------------------

Select two free A100 80 GB GPUs. For eight GPUs on the same node, change
the device list to ``0,1,2,3,4,5,6,7``.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=4,5 bash examples/agent/qwen35/run_replay_update.sh /path/to/Qwen3.5-4B

The script starts a private Ray instance, collects four two-turn conversations
and sends the same eight rows to a dense actor, a chunked actor and a scalar
critic. Check all three outputs (``dense``, ``chunked``, ``critic``). Each rank
should report a finite ``loss``, a finite positive ``grad_norm`` and a positive
``changed`` count. The count measures how many elements of a sampled weight
tensor changed.

The example uses fixed advantages and returns. Batch and parallelism settings
are in ``examples/agent/qwen35/config/replay_update.yaml``, which inherits the
Search-R1 Megatron PPO configuration. Two GPUs use TP=2, DP=1. Eight GPUs use
TP=2, DP=4.

Use replay in your agent
------------------------

Register the image processor in the rollout worker's environment before
starting SGLang. The example scripts above do this for you.

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

Enable image replay and rollout logprobs in your training configuration. If
you train a critic, also set ``critic.expect_multi_modal_inputs=true``. The
learner requires PP=1 and CP=1. Each rollout worker and its local tokenizer
process must share a temporary directory.

.. code-block:: yaml

   rollout:
     return_logprobs: true
     sglang:
       enable_shared_multi_modal_processor: true
   actor:
     expect_multi_modal_inputs: true

For a conversation that receives image A and then image B:

1. On the first turn, pass image A with a single image placeholder in the
   prompt and set ``history`` to ``None``. Keep the expanded prompt, the sampled
   output IDs and the returned span for A.
2. On the second turn, append the new text and a placeholder for B to the
   returned ``prefix``. Send both original images in order, with the recorded
   span for A and the new span for B. SGLang processes A again and checks it
   against the previous turn.

Save one training row per turn. Carry ``prefix``, ``spans`` and ``history`` into
the next request:

.. code-block:: python

   result = await self.generate(
       prompt_ids, image_data=images, image_spans=spans, image_history=history,
       image_store_dir=shared_directory,
   )
   turn = AgentLoopOutput(
       prompt_ids=result["prompt_token_ids"],
       response_ids=result["output_ids"],
       response_logprobs=result["logprobs"],
       response_images=result["multimodal_input"],
   )
   prefix = result["prompt_token_ids"] + result["output_ids"]
   spans = result["multimodal_input"]["image_spans"]
   history = result["image_history"]

Keep ``response_images`` with its row: identical token prefixes can refer to
different images. Use ``None`` on text-only rows. With ``image_store_dir``, each
distinct pixel tensor is stored once. Keep the files readable by the learner
while any sample or checkpoint refers to them. Omit the argument to store
pixels inline.

Read a captured image row
-------------------------

These fields are in ``result["multimodal_input"]``. Images follow prompt order.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field
     - How the learner uses it
   * - ``format``, ``prompt_token_ids``
     - The format is ``rlinf-qwen-image-inputs-v1``. The integer list contains the full prompt used to sample this turn, with image placeholders expanded.
   * - ``image_spans``, ``image_grid_thw``
     - One half-open token span and one temporal, height and width grid per image. Each span excludes the vision delimiters. Image markers in sampled text retain their text interpretation.
   * - ``mrope_positions``, ``mrope_position_delta``
     - Positions for all three axes of the prompt, shaped ``[3, prompt length]``, and an offset of shape ``[1]`` for positions of generated tokens.
   * - ``pixel_values``
     - Inline pixels from the processor. The learner casts them to BF16.
   * - ``pixel_refs``, ``pixel_values_bytes``
     - With shared storage, a path and SHA-256 digest per image replace inline pixels. The byte count estimates the size of the learner's BF16 pixel tensor.

Validation results
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Check
     - Result
   * - Two-turn input replay
     - All eight turns across four conversations passed the checks for sampled history, BF16 pixels, grids and M-RoPE positions.
   * - Optimizer updates
     - On eight A100 80 GB GPUs, both actors and the scalar critic each completed one optimizer update on the same eight rows. All three checks also passed on two GPUs.
   * - Backend logprobs
     - A separate frozen-weight comparison used 16 generated tokens from each of six requests. The mean absolute SGLang/Megatron logprob difference was 0.058638, the maximum was 0.763876, and 17 of 96 positions exceeded 0.10.

The first two checks cover input replay and parameter updates. Backend scores
still differ. Policy success requires a separate evaluation.

.. _qwen35-replay-setup:

Environment setup
-----------------

The tested combination covers images with Qwen3.5-4B, the SGLang 0.5.12.post1
release wheel and Megatron Bridge 0.5.0. The A100 runs used PyTorch
2.11.0+cu128, Transformers 5.6.1, Megatron Core 0.18.0 and Transformer
Engine 2.17. Qwen3.5 gated delta layers require ``fla-core==0.6.0``.

If you already have this environment, run the examples directly. For a new
environment, run these commands from the repository root:

.. code-block:: bash

   git clone --depth 1 --branch core_v0.18.0 https://github.com/NVIDIA/Megatron-LM.git Megatron-LM-0.18.0
   MEGATRON_PATH="$PWD/Megatron-LM-0.18.0" bash requirements/install.sh agentic --sglang 0.5.12.post1 --venv .venv-agentic
   source .venv-agentic/bin/activate

See :doc:`../agentic_config` for other agent training options.
