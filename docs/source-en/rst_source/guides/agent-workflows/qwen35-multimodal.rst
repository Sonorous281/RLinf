Share Qwen3.5 Image Preprocessing
=================================

Use shared preprocessing to replay the same image inputs in SGLang rollout and
Megatron training. This guide assumes a Qwen3.5 agent workflow that sends image
observations and preserves the sampled token IDs across turns.

Install the Processor Stack
---------------------------

The model's Hugging Face processor handles images by default. To share image
preprocessing between rollout and training, install the experimental SGLang
TokenSpace dependencies.
It requires a CUDA 13 compatible driver and Rust for source builds. Install it in
a separate environment:

.. code-block:: bash

   bash requirements/install.sh agentic --sglang token-space --venv .venv-token-space

This installs the pinned SGLang source and the matching Megatron dependencies.
The default agentic installation remains unchanged.

Configure Replay and Cache Budgets
----------------------------------

Apply these overrides to an existing Qwen3.5 Megatron agent configuration with
pipeline and context parallel sizes set to one. Replace the cache directory with
an absolute path visible to rollout and learner processes:

.. code-block:: yaml

   rollout:
     sglang:
       enable_shared_multi_modal_processor: true
       enable_token_space_preprocess_cache: true
       token_space_preprocess_cache_dir: /absolute/shared/image-cache
       token_space_preprocess_cache_trust_url: true
   actor:
     expect_multi_modal_inputs: true
     model:
       load_mtp: false
       lm_head_chunk_size: 256
     multimodal:
       cache_bytes: 268435456
       prefetch_batches: 2
       prefetch_bytes: 536870912
   critic:
     expect_multi_modal_inputs: true
     multimodal:
       cache_bytes: 268435456
       prefetch_batches: 2
       prefetch_bytes: 536870912

The example gives each learner rank 256 MiB of processed-image cache and a
512 MiB prefetch budget. With two actor and two critic ranks, these budgets sum
to 3 GiB of host memory; model state, raw images and processor scratch space are
additional. Cache and prefetch are disabled when their budgets are zero.

Chunked scoring requires MTP to be disabled explicitly. It supports entropy
metrics with zero entropy bonus. Keep the vision encoder and projection frozen
in both training and rollout.

Check the Input Contract
------------------------

Rollout records the served prompt IDs, image grids, processor settings, pixel
checksums and M-RoPE checksums. Replay rejects changed files, pixels or positions.
For example, adding a second image must leave the first turn's sampled IDs and
expanded image span intact; only the new image span is expanded.

The disk cache stores image preprocessing results. The learner still runs
a forward pass through the vision encoder. Matching these inputs does not establish equal
vision embeddings or final logprobs between SGLang and Megatron; measure those
separately on the same frozen weights and tokens.

.. warning::

   Trusted cache URLs must refer to immutable files. Keep image files and the
   cache accessible for as long as pending trajectories or checkpoints use them.
   Shared preprocessing accepts images; video and audio are unsupported.

See :doc:`../agentic_config` for the other agent training options.
