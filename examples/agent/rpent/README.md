# RPent Planner Training

[中文](README_zh.md)

Start with the [episode PPO/GRPO guide](../../../docs/source-en/rst_source/examples/agentic/rpent_episode_sync.rst)
([中文](../../../docs/source-zh/rst_source/examples/agentic/rpent_episode_sync.rst)).
It covers the shared reason environment, model assets, services,
dataset preparation, training, export and evaluation.

- `train.py` launches synchronous PPO/GRPO or asynchronous SAO.
- `eval.py` runs the selected planner on a fixed task panel.
- `prepare_tasks.py` exports native task states and their manifests.
- `checkpoint.py` exports a trained language policy with its frozen vision model.

For explicit saved-cohort recovery, pass a verified `replay` configuration to
`train.py`. It verifies the original policy and complete groups, consumes that
batch once, then resumes online sampling. Ordinary training samples online.

The original Pi0.5 planner and the direct-control SAO examples have separate
configurations and documentation. Select the workflow used to produce your SFT checkpoint;
switching workflows changes the model's prompts and tools.

## Code Ownership

A synchronous GRPO update follows this path:

```text
dataset groups → RPent planner → policy endpoint → SGLang
                       ↓
                 remote environment / SAM3
                       ↓
complete groups → trajectory assembly → learner → weight sync → next batch
```

| Module | Responsibility |
| --- | --- |
| `rlinf/agents/rpent/config_validation.py` | Validate resources, multimodal inputs, planner budgets and algorithm contracts |
| `rlinf/agents/rpent/launcher.py` | Own subprocess launch, timeout, cancellation and reaping |
| `rlinf/agents/rpent/history.py` | Preserve sampled token prefixes across tool feedback and compaction |
| `rlinf/agents/rpent/rpent_agent_loop.py` | Serve planner requests, assemble episodes and finalize candidate selection |
| `rlinf/agents/rpent/episode.py` | Decode tasks, merge compatible turns and attach the native outcome |
| `rlinf/agents/rpent/records.py` | Persist sampled tokens and multimodal contracts for recovery |
| `rlinf/agents/rpent/replay.py` | Read and verify an explicitly selected saved batch |

RPent provides prompts and tool schemas, executes actions, and reports the
simulator's native success signal. RLinf
owns policy inputs, grouped selection, advantages, optimizer updates and
checkpoints. The `preprocessing/token_space` configuration is shared by GRPO,
PPO and evaluation; algorithm settings stay in their training recipes.
