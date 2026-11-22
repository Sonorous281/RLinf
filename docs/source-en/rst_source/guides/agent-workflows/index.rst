Agent Workflows
===============

Use this guide when you operate agentic or reasoning workloads instead of
embodied RL jobs.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`Agentic Guides <../agentic>`
     - Operational workflows for agentic tasks.
   * - :doc:`Session Server <../session_server>`
     - Collect and assemble agent trajectories on the rollout nodes. Control
       and data travel separately, requests go through an SGLang router, and
       agents can use the OpenAI chat completions API.

   * - :doc:`Qwen3.5 Image Replay <qwen35-multimodal>`
     - Replay the image inputs from SGLang in Megatron and verify a training update.

.. toctree::
   :hidden:

   Agentic Guides <../agentic>
   Qwen3.5 Image Replay <qwen35-multimodal>
   Session Server <../session_server>
