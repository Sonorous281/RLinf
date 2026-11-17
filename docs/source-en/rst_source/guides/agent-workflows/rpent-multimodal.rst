Configure RPent Multimodal Training
===================================

Use these options after completing :doc:`the single-node smoke <../../examples/agentic/rpent_libero_rl>`. Keep the default and experimental shared preprocessing environments separate.

Enable Logprob Recalculation and TIS
---------------------------------------------

To test SAO with training-side logprobs and truncated importance sampling (TIS), add these settings to the SAO config. The default SAO recipe leaves both options disabled.

.. code-block:: yaml

   algorithm:
     recompute_logprobs: true
     importance_sampling_fix: true
     importance_sampling_clip: 2.0
   rollout:
     return_logprobs: true
     max_weight_staleness: 1

Each batch retains its sampling logprobs and per-token policy versions. Before the actor update, the learner recalculates logprobs with the current actor weights. TIS limits how much backend differences and policy lag can affect an update. The DIS loss uses the actor weights used for recalculation as the reference when filtering tokens by probability ratio. With these settings, an episode is discarded if any token was sampled with weights more than one update behind the latest published version. Its task is then retried.

.. warning::

   Enabling these options changes how SAO computes its loss and adds an actor forward pass. Check TIS weight clipping, effective sample size and policy lag during a small run before scaling. Optimizer updates, weight synchronization and checkpoint recovery with these options have not been validated. Logprob recalculation and TIS do not establish numerical agreement between SGLang and Megatron.

Try Shared Multimodal Preprocessing (Experimental)
--------------------------------------------------

Shared Multimodal Preprocessing uses SGLang TokenSpace to process images for rollout and training. It checks the prompt IDs returned by inference, replays images from checksum-protected files, and checks the M-RoPE positions at the training model's language input. This mode supports Qwen3.5 image inputs with PIL preprocessing and Megatron-Bridge. It requires PP=1 and CP=1. Literal vision markers in text and generated responses are unsupported; video and audio are outside this path.

.. warning::

   This path is experimental. The default SGLang 0.5.12.post1 environment cannot run it. The candidate dependency file, ``requirements/agentic/sglang-token-space.txt``, pins SGLang commit ``18a44326f6de5134fe921927d760d81f8ee48fc9``, Torch 2.14.1 and Transformers 5.19.0. Prepare a separate environment with matching CUDA and Megatron-Bridge kernels; GPU serving, weight synchronization and training still need to be validated together in this environment. Keep the existing training environment unchanged.

   RLinf permits this development build only for rollout with shared multimodal preprocessing enabled. Other rollout paths and standalone reward or embodied servers reject it before launch. Use a supported SGLang release for those workloads; installing this build does not establish their compatibility.

Keep using ``rollout.cuda_graph_max_bs`` with supported SGLang releases, including 0.5.19, which renamed the corresponding server option. RLinf caps the value at ``rollout.max_running_requests``. These rules also apply when shared multimodal preprocessing is disabled. GPU regression checks for Search-R1 and embodied workloads remain pending.

After validating that environment, enable the following settings in the deployment profile. Remove ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`` from the rollout environment, including inherited shell settings. Keep ``output_dir`` accessible at the same path on rollout, actor and critic nodes.

.. code-block:: yaml

   rollout:
     sglang:
       enable_shared_multi_modal_processor: true
   tools:
     rpent:
       store_images_on_disk: true
       keep_episodes: all

The public switch is ``rollout.sglang.enable_shared_multi_modal_processor`` and defaults to false. RLinf still accepts ``rollout.sglang.enable_token_space_processor`` as a legacy config alias with the same behavior. A config containing both names is rejected, even if the values agree; replace the old key when migrating. In commands that use the current training templates, set ``rollout.sglang.enable_shared_multi_modal_processor=true``. The evaluation template has no default key, so use ``+rollout.sglang.enable_shared_multi_modal_processor=true``.

RLinf maps this switch to SGLang's native ``enable_token_space_processor`` argument. The SGLang cache option names below, the ``sglang-token-space-v1`` trajectory format and checkpoint mode markers are unchanged. Renaming the config key alone does not change the checkpoint input contract.

Rollout initialization fails if the SGLang TokenSpace APIs are unavailable. Both Server and Engine must return the prompt token IDs they actually use. An image preprocessing pass runs before sampling to calculate output and thinking budgets from the expanded prompt; its throughput cost has not been measured.

Start with fresh rollout state when switching modes. Async checkpoint recovery rejects a different mode, and the learner rejects changed processor fingerprints or mixed media formats. Do not assume optimizer state is portable across the Torch/kernel upgrade. Retain image files for checkpoint recovery.

To validate the stack for training, first verify multi-turn images, compaction and packed M-RoPE in the deployment environment. Then run the fixed 96-point logprob comparison. Report mean, P95 and maximum absolute error, with 0.03/0.10 as diagnostic references. Acceptance also requires correct token/image/position inputs, finite updates, weight synchronization and recovery in a new process; recalculation and TIS do not prove native numerical parity. Compare base model logprobs separately from probabilities modified by sampling penalties, top-k and top-p. CPU input checks do not establish GPU numerical parity.

Reduce Repeated Image Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This mode can share preprocessed images between rollout, actor and critic. The cache contains processed pixels and grid metadata. Each backend still runs its own frozen vision encoder; visual embeddings are not shared.

Shared preprocessing caches and CPU prefetching are disabled by default. To enable them, add the following to the deployment profile above. Replace ``<cache-dir>`` with the absolute path to a directory reserved for this job. Every process in the job must be able to access the directory at that path.

.. code-block:: yaml

   rollout:
     sglang:
       enable_token_space_preprocess_cache: true
       token_space_preprocess_cache_dir: <cache-dir>
       token_space_preprocess_cache_trust_url: true
       token_space_preprocess_cache_check_rate: 0.0
   actor:
     multimodal:
       cache_bytes: 0
       prefetch_batches: 2
       prefetch_bytes: 268435456
   critic:
     multimodal:
       cache_bytes: 0
       prefetch_batches: 2
       prefetch_bytes: 268435456

Actor and critic budgets are independent. If neither a learner field nor its legacy setting is present, the value defaults to zero, disabling the corresponding cache or prefetch feature. The legacy fields under ``tools.rpent`` (``training_image_cache_bytes``, ``training_image_prefetch_batches`` and ``training_image_prefetch_bytes``) still apply to both learners. If a legacy setting and its learner setting are both present, their values must match; conflicting values fail validation. Standalone evaluation needs no actor or critic configuration. The example template uses the learner fields. To override a legacy field from the command line, use Hydra’s ``++`` syntax, for example ``++tools.rpent.training_image_cache_bytes=268435456``.

Cached image paths must always refer to the same image contents. The cache stores preprocessed outputs in a separate directory for each processor fingerprint. Set the check rate between 0 and 1 to recompute a sample of cache hits and verify their contents. Keep this setting identical across processes.

The prefetch settings limit the number and total bytes of prepared microbatches per learner rank, including the current microbatch. One CPU thread prepares the following batch while the training thread runs the forward pass for the current microbatch. The 256 MiB example excludes processor scratch space and the separate ``multimodal.cache_bytes`` cache. Multiply the per-rank budgets by the number of actor and critic ranks, and measure peak host memory before increasing them. A microbatch larger than the budget fails with an error. Prefetching requires token-space trajectories that record the size of the preprocessed pixel data in bytes. Set ``multimodal.prefetch_batches: 0`` when replaying older trajectories.

.. warning::

   Only trusted processes may write the cache directory: cache reads deserialize Python objects. The native disk cache has no eviction or disk-size limit. Use a job-owned directory with a filesystem quota or external capacity monitoring. Remove it only after all job processes have exited; image files must remain available for checkpoint recovery. A new process rebuilds its CPU prefetch queue from the saved image references.

CPU tests cover cache reuse, image integrity, prefetch ordering and thread cleanup. Validate GPU logprobs, training gradients, throughput and training recovery in a new process in the deployment environment. Compare each optimization against the same backend with the option disabled; report the separate 96-point cross-backend diagnostic alongside those results.

Optional Runtime Settings
-------------------------

Training and evaluation set ``rollout.sglang.disable_overlap_schedule=true`` so presence penalties and optional thinking limits read the current CPU token history. Validate sampling logprobs and forced-token alignment on the deployed SGLang version before changing it.

Set ``agentloop.chat_template_kwargs.enable_thinking=false`` to disable thinking, or ``agentloop.prompt=simple_brief`` for prompt guidance toward about 256 reasoning tokens per decision. The prompt does not enforce a decoder limit. Compare native success and cost from fresh episodes with fixed models and seeds; do not remove thinking tokens or behavior logprobs from recorded trajectories.

To cap the first thinking span already opened by the chat template, set ``agentloop.thinking_token_budget=512`` and ``rollout.sglang.enable_custom_logit_processor=true``, keeping ``enable_thinking=true`` and ``rollout.sglang.disable_overlap_schedule=true``. Configuration validation rejects a budget with overlap enabled because its CPU token history can lag decoding. The default budget is ``null``. The engine records which closing token it forces; that token has zero training mask, while a naturally generated closing token remains trainable. A later reopened thinking span is not capped by this option. Keep the returned IDs, behavior logprobs, forced-token flags and weight versions together through trajectory assembly.

Set ``actor.model.lm_head_chunk_size=256`` to calculate the output layer in blocks for response tokens that contribute to the loss. The default ``0`` disables chunking. Results are recomputed during the backward pass. It supports MBridge Qwen3.5 with PP1/CP1, including TP and sequence parallel. Use the Megatron version listed on the setup page. Entropy is a detached diagnostic only, so an entropy loss bonus is unsupported. FP8, MTP, deferred embedding gradients, muP and autocast are also rejected for this mode. These are support constraints, not evidence of higher throughput or learning gains.

Training uses the SGLang Server backend by default; set ``rollout.sglang.backend_type=engine`` for the in-process backend. Evaluation uses Engine by default. Both backends preserve images, thinking-budget metadata and policy versions. The server binds to loopback and runs as a child process.

Tool results and images remain in subsequent prompts. Only policy-generated assistant tokens receive policy loss; user/tool/history tokens do not. Prefix-compatible turns merge, while context compression/history rewriting starts a new row. The critic and actor use the same response-position mapping. Shared trajectory packing keeps rows separate if even one scored token overlaps: merging would overwrite a learning target. This applies to text-only SearchR1 as well as RPent; packing regressions compare both losses and gradients. Response-only logits require CP1.

When enabling ``tools.rpent.store_images_on_disk=true``, every rollout and learner process must read the **same absolute image paths**. ``tools.rpent.output_dir`` must be shared across model and environment nodes and retained while trajectories or saved checkpoints reference it. A same-named directory on separate local disks is insufficient. Leave the default ``false`` for installations without shared image storage; it retains image bytes in trajectories at a higher memory cost. Do not prune episode artifacts needed by a recoverable checkpoint.

Preserve Image-Turn Token Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For merged Qwen3.5 image turns, enable the token-preserving processor on each model node **before starting Ray**:

.. code-block:: bash

   export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rlinf.hybrid_engines.sglang.multimodal

The adaptive train/eval templates also pass this variable through ``rollout.env_vars``, so worker launch does not depend solely on the terminal environment. This uses SGLang's external processor registry. Image requests carry already expanded prompt IDs and raw images. The adapter validates the number of images and each image's pad span against its processed grid, retains the exact text/action IDs, and lets SGLang derive offsets and M-RoPE from that sequence. Decoding and re-encoding sampled tokens can otherwise change their boundaries, even when the decoded text is identical. String requests retain the upstream behavior. Token-ID requests with video, audio or precomputed image inputs are rejected by this image-only adapter.

Run the processor regressions with a local model directory containing its config, tokenizer and processor files:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES='' SGLANG_USE_CUDA_IPC_TRANSPORT=0 \
     RLINF_TEST_QWEN_VL_MODEL=/path/to/Qwen3.5-4B \
     PYTHONPATH=.:tests/unit_tests \
     python -m pytest tests/unit_tests/test_sglang.py -k qwen_image_processor

These CPU checks cover input IDs, image tensors, offsets and HF/SGLang M-RoPE agreement. They do not replace real-engine generation/logprob checks or a complete learner update and checkpoint recovery before deploying a new processor version.
