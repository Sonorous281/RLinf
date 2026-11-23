# RPent environment

Use the reason image for RLinf, the model and the planner. Install the paired
RPent checkout with the API planner dependencies.

```bash
source /opt/venv/reason/bin/activate
export MEGATRON_PATH=/opt/venv/reason/Megatron-LM
uv pip install --python "$VIRTUAL_ENV/bin/python" -r "$RPENT_ROOT/requirements/episode-planner.txt"
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e "$RPENT_ROOT"
```

Each simulator has its own Python environment and toolkit service. Follow the
[training guide](../../docs/source-en/rst_source/examples/agentic/rpent_episode_sync.rst)
for simulator installation, task manifests, mixed quotas and recovery.

The versioned H200 offline release and its scripts belong to the frozen
historical deployment. Resume its checkpoints with their matching source,
software and action contract.
