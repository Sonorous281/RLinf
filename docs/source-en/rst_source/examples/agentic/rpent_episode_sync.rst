RPent episode training
======================

Train a Qwen3.5-4B planner with synchronous GRPO or PPO. RLinf runs inference,
the planner and the learner. Each simulator runs in its own environment and
returns observations, measured actions and native rewards over RPC.

Installation
------------

Use the Linux ``reason`` image and paired RPent and RLinf checkouts. The image
provides Python 3.12, Torch, SGLang, Megatron-Core, Bridge, Transformer Engine
and FlashAttention 2. Install the RPent planner dependencies in the image
environment. Install each simulator on its own host using the
`RPent guide <https://github.com/zhexuanxu/RPent/blob/rpent-ppo/docs/source-en/rst_source/simulators/rl.rst>`_.

.. code-block:: bash

   source /opt/venv/reason/bin/activate
   export MEGATRON_PATH=/opt/venv/reason/Megatron-LM
   (cd "$RPENT_ROOT" && uv pip install --python "$VIRTUAL_ENV/bin/python" \
     -r requirements/episode-planner.txt)
   uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e "$RPENT_ROOT"

Provide the original multimodal model directory with its tokenizer and image
processor. Simulator hosts need the task assets and access to SAM3. Use the same
RPent revision on the planner and simulation hosts.

Prepare tasks
-------------

Each row in a training JSONL file has a text ``prompt`` and an ``answer`` string
containing the simulator task as JSON. For example, a RoboTwin row is:

.. code-block:: json

   {"prompt":"Adjust the bottle","answer":"{\"environment\":\"robotwin\",\"task_name\":\"adjust_bottle\",\"seed\":100000,\"task_config\":\"demo_randomized\"}"}

LIBERO tasks use ``suite``, ``task`` and ``seed``. RoboCasa and RoboDojo use
``task_name`` and ``seed``. In RoboDojo, select a seed group installed in
``Assets/Eval_Layout`` and set ``layout`` to an index within that group. Run
``examples/agent/rpent/prepare_tasks.py`` inside the LIBERO environment to
build its native task pool, and supply a separate task list for each other
simulator.

Run training
------------

Start one RPent toolkit server per simulator. The following recipe uses the
four endpoints defined in the mixed GRPO example. Set the GPU placement and
backend capacities for your cluster. This example assigns training and model
rollout to eight GPUs. Four simulator backends run independently.

Start a Ray head on the RLinf host if the training platform has not provided
one. Use the address printed by ``ray start --head`` as ``RAY_ADDRESS``.
The driver and its workers connect to that head. The placement ranges select
physical GPUs on the training cluster.

.. code-block:: bash

   cd "$RLINF_ROOT"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_mixed_grpo \
     cluster.num_nodes=1 \
     cluster.component_placement.actor=0-7 \
     cluster.component_placement.rollout=0-7 \
     cluster.component_placement.critic=0-7 \
     cluster.component_placement.reward=0-7 \
     cluster.component_placement.agentloop=0-7 \
     rollout.model.model_path="$MODEL_PATH" \
     tools.rpent.repo_path="$RPENT_ROOT" \
     tools.rpent.output_dir="$RUN_DIR/episodes" \
     runner.output_dir="$RUN_DIR" \
     env.rpent.libero.train_data_paths="$TASK_DIR/libero.jsonl" \
     env.rpent.robotwin.train_data_paths="$TASK_DIR/robotwin.jsonl" \
     env.rpent.robocasa.train_data_paths="$TASK_DIR/robocasa.jsonl" \
     env.rpent.robodojo.train_data_paths="$TASK_DIR/robodojo.jsonl"

Set each ``env.rpent.<simulator>.endpoint`` to the toolkit address reachable by
the planner. An entry also accepts ``rollout_episodes`` and ``sampling_episodes``.
The two sizes in ``data`` count groups. The two simulator quotas count episodes.
With 32 effective groups, 40 candidate groups and eight episodes per group,
the four simulators default to 64 effective and 80 candidate episodes each.
To set a quota explicitly, supply that field for all enabled simulators and
keep their sum equal to the corresponding global episode count. Quotas must
contain whole groups.

A sampler draws from all task pools each update. Members of a group share one
task and initial state. Each simulator contributes its first complete groups
up to its own quota, including normal task failures. The learner starts after
candidate environments and generation queues are clean.

For a single simulator, use ``train_qwen3.5_4b_episode_sync_grpo`` with
``data.train_data_paths`` and ``tools.rpent.remote_toolkit_endpoint``.
Set ``runner.resume_dir`` to the latest complete checkpoint to resume. Keep
the same task pools, simulator quotas and model interface during recovery.
The run directory stores the resolved configuration, selected groups,
optimizer checkpoints and training metrics.
