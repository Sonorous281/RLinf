# Install RPent in the Reason Image

Install the minimal H200 release into `/opt/venv/reason`. RLinf, SGLang, RPent,
SAM3 and LIBERO use that interpreter. Core and Bridge retain their original
files and import paths. The LIBERO wheels are built from the prepared benchmark
sources: ground-mounted Panda models declare the robosuite 1.5 `NullBase`
instead of the obsolete `None` base. Their other Python files are unchanged. The earlier compatibility-wheel package is historical;
use the versioned minimal release for fresh reason images.

```bash
export STACK_ROOT=/mnt/public/xzxuan
export PACKAGE_ROOT="$STACK_ROOT/envs/offline_package/releases/h200-minimal-20261120-v5"
bash "$PACKAGE_ROOT/install.sh"
source "$PACKAGE_ROOT/activate.sh"
```

Copy the complete release, including its manifest, wheels and resources, to
shared storage before installing on each node. Model weights and LIBERO assets
remain separate under `models/` and `data/`. The package manifest lists their
asset paths. The two source checkouts belong in `repos/RLinf` and `repos/RPent`.
To install from the RLinf checkout, run
`bash requirements/rpent/install_minimal.sh /path/to/release`.

| Component | Image or required addition |
| --- | --- |
| Python | Retain 3.12.12 |
| Torch / torchvision | Retain 2.11.0+cu128 / 0.26.0+cu128 |
| SGLang / FlashInfer | Retain 0.5.12.post1 / 0.6.11.post1 |
| Core / Bridge | Retain Core checkout `dc0ee41ad` / official Bridge 0.5.0 |
| FA2 / Transformer Engine / Triton | Retain 2.8.3 / 2.17.0 / 3.6.0 |
| Transformers / NumPy | Retain 5.6.1 / 2.2.0 unless runtime validation requires a change |
| Qwen linear attention | Add `fla-core==0.6.0` |
| Simulation | Add MuJoCo 3.3.0, robosuite 1.5.2, LIBERO/PRO and their runtime dependencies |
| Perception | Add SAM3, its tokenizer resource and image dependencies |
| Planner | Add the paired RPent/RLinf wheels and planner dependencies; OpenAI SDK 3.19.0 provides the required planner interfaces |

Installation checks wheel/resource hashes and the original Core commit,
records before/after inventories, and refuses replacements without a recorded
runtime incompatibility. Protected GPU packages and Core/Bridge must remain
unchanged. Replacement packages are backed up. Rerunning the installer verifies installed package files and keeps their
versions unchanged. The installer saves `pip check` output; validate reported
metadata conflicts with runtime tests. Do not run the historical `build_wheels.py` to prepare this release.

## Start Services and Ray

Set a fresh `GRPO_RUN_ROOT` and a unique node rank, 0 through 3, on each node.
Activate the package before starting services or Ray:

```bash
export GRPO_RUN_ROOT="$STACK_ROOT/logs/my-27b-grpo"
export RLINF_NODE_RANK=0  # 1, 2 or 3 on the other nodes
python -m robots.rl.services --gpus 0,1,2,3,4,5,6,7 \
  --sam3-per-gpu 1 --envs-per-gpu 10 \
  --sam3-checkpoint "$SAM3_CHECKPOINT_PATH" \
  --output-dir "$GRPO_RUN_ROOT/services-node$RLINF_NODE_RANK"
```

Keep that service process running. Each node provides eight SAM3 instances,
a pool at loopback port 28424 and an environment backend at port 28490. Across
four nodes, 320 environment slots are available. Verify simultaneous capacity,
real segmentation/actions and cleanup before accepting a deployment.

In a second shell on each node, activate the same environment and export the
same node rank before `ray start`. Set reachable head and local IP addresses:

```bash
export HEAD_IP=<head-ip>
export NODE_IP=<this-node-ip>
# Head only:
ray start --head --port 26379 --node-ip-address="$HEAD_IP" --num-gpus 8
# Other nodes:
ray start --address="$HEAD_IP:26379" --node-ip-address="$NODE_IP" --num-gpus 8
```

Confirm four nodes and 32 GPUs with `ray status`. Run the driver only on the
head. Keep the image activation's Core path; the minimal activation does not
replace it with a wheel or source overlay.

## Run the 27B Five-Step Acceptance

Use `train_qwen3.5_27b_h200_grpo_5step` with a deployment profile containing
the complete local 27B checkpoint, task manifest, RPent checkout, benchmark
config and fresh output paths. `Qwen3.8-27B` is the prepared directory name;
its config declares dense `Qwen3_5ForConditionalGeneration`. RLinf reads the
layer count, dimensions, attention heads and embedding ties from that config.

```bash
cd "$RLINF_ROOT"
bash examples/agent/rpent/run_train.sh train_qwen3.5_27b_h200_grpo_5step \
  --config-dir "$CONFIG_DIR" +deployment@_global_=local
```

Use the deployment-profile format in the episode recipe. Set all model/tokenizer
paths to the 27B checkpoint and `tools.rpent.python` to
`/opt/venv/reason/bin/python`. RLinf launches RPent planner subprocesses;
they use the node-local environment backend and SAM3 pool.

The profile selects 32 complete groups of eight from 40 candidate groups,
uses the `episode` workflow in simulator coordinates (`world_cord=base`),
and runs five optimizer updates with a checkpoint after each.
Use rollout TP1 and actor TP4/CP1/PP1/DP8 with sequence
parallelism (SP). Multi-token prediction (MTP) and the
MCore HybridDeviceOptimizer are disabled; stage offload remains enabled.
Logprob and LM-head chunks are 256. The remaining GRPO settings inherit the
4B recipe, including the three loss scales and clip upper bound 0.28.

Verify that optimizer steps complete, losses and gradients are finite, language
parameters change, and vision parameters stay frozen. Check that learner inputs
match the captured serving inputs. Restore a checkpoint, verify its optimizer,
RNG and data state, then generate fresh replies from the restored weights.
All-zero advantages do not prove effective learning. Stop the processes started
for acceptance after verification.
