# RPent 规划器训练

从 [episode PPO/GRPO 指南](../../../docs/source-zh/rst_source/examples/agentic/rpent_episode_sync.rst) 开始
（[English](README.md)）。指南覆盖共享 reason 环境、模型资源、服务、任务清单、训练、导出和评测。

- `train.py` 启动同步 PPO/GRPO 或异步 SAO。
- `eval.py` 在固定任务集上运行规划器。
- `prepare_tasks.py` 导出原生任务状态和 manifest。
- `checkpoint.py` 将训练后的语言参数与冻结的视觉模型合并导出。

需要从已保存 cohort 恢复时，给 `train.py` 传入经过校验的 `replay` 配置。它校验原始 policy 和完整组，只使用该批一次，再恢复在线采样。常规训练始终在线采样。

原始 Pi0.5 规划器与直接控制的 SAO 示例使用各自的配置和文档。选择生成 SFT checkpoint 时使用的工作流；切换工作流会改变 prompt 和工具。

## 代码职责

一次同步 GRPO 更新经过以下步骤：

```text
任务组 → RPent planner → policy 接口 → SGLang
              ↓
        远端环境 / SAM3
              ↓
完整组 → 轨迹组装 → learner → 权重同步 → 下一批
```

| 模块 | 职责 |
| --- | --- |
| `rlinf/agents/rpent/config_validation.py` | 校验资源、多模态输入、planner 预算和算法约定 |
| `rlinf/agents/rpent/launcher.py` | 管理子进程的启动、超时和取消，并回收已退出的子进程 |
| `rlinf/agents/rpent/history.py` | 在工具反馈与压缩过程中保持采样 token 前缀 |
| `rlinf/agents/rpent/rpent_agent_loop.py` | 处理 planner 请求、组装 episode、完成候选选择 |
| `rlinf/agents/rpent/episode.py` | 解码任务、合并兼容 turn、关联原生结果 |
| `rlinf/agents/rpent/records.py` | 保存采样 token 和恢复所需的多模态约定 |
| `rlinf/agents/rpent/replay.py` | 读取并校验指定的已保存批次 |

RPent 提供 prompt 和工具 schema，执行动作，并报告模拟器的原生成功信号。RLinf 负责 policy 输入、分组选择、advantage 计算、optimizer 更新和 checkpoint 保存。GRPO、PPO、评测共用 `preprocessing/token_space` 配置；算法参数保留在各训练配置中。
