# Slack Serving

日常从 `final/README.md` 开始。`final/cooperative_4gpu/` 是当前最后采用的完整合作版本；`final/attention_2gpu/` 是双卡 attention 与多轮执行基础版，两者用途不同。

GitHub 私有仓库：[Eyan-L/Slack-Serving](https://github.com/Eyan-L/Slack-Serving)。仓库跟踪 `final/`、本说明和项目操作约定，包含代码、配置、冻结依赖、提交脚本及来源校验信息。`final/data/corpus.json` 是复现所需的输入语料，随最终版提交。下列历史实验、结果、原始 trace、整理记录及运行缓存仅保留在本地 / NSCC，不纳入 Git。

```text
final/                          最终代码、配置、冻结依赖和提交入口
experiments/
  01_single_gpu/                单卡带宽、KV 增长和工作区间
  02_peer_kv/                   远端 KV 原型、正确性和早期吞吐
  03_dynamic_kv/                动态分配、迁移、prefill 优化探索
  04_flashinfer/                FlashInfer decode、prefill、多轮与四卡
  05_trace_serving/             真实 subagent + 背景请求回放
results/<阶段>/<项目>/<作业ID>/   对应结果、图表和作业证据
results/06_presentations/       历史汇报素材
traces/                         NSCC 原始五份 trace（本地不额外复制原始压缩包）
archive/organization_20261101/   本次移动映射、兼容入口恢复脚本和整理记录
```

## 最后采用的版本及证据

主版本采用 FlashInfer attention、本地优先分配、动态迁移 **1+2+4 / 提前 2048 槽位**；四张物理卡运行两个模型，以三路 NV4 为 offpath 提供远端 KV。它是三路供给的硬件代理，不能视为已经实测真正双卡 NV12。

| 配置 | 作业 | Element 组完成 / s | Vuls 组完成 / s | 完成输出 / tok/s |
|---|---|---:|---:|---:|
| 独立双卡，100% / 100% KV | 26078142 | 387.854 | 368.513 | 315.473 |
| **合作，80% / 117.35% 有效 KV** | **26079252** | **328.319** | **293.322** | **294.295** |
| 匹配容量的原生全本地对照 | 26079802 | 333.637 | 298.972 | 298.198 |

结果入口：

- `results/05_trace_serving/cooperative_trace_background/26079252/analysis/review.json`
- `results/05_trace_serving/dual_trace_background/26078142/analysis/review.json`
- `results/05_trace_serving/dual_trace_capacity_control/26079802/analysis/review.json`

这些是单次实测；吞吐按共同观察窗口内已完成调用的输出 token 计算，未完成背景请求不计入完成输出。独立组与合作组还改变了分配和容量划分，不能将差值全部归因于 NVLink。容量匹配对照用于判断完整双卡实现的额外成本。

## 使用与追溯

`final/` 包含整理后的源码副本、原始配置和依赖来源。历史 `experiments/` 与结果的常规文件原样保留，没有删除中间结果或 raw trace；NSCC 的历史结果索引可链接到对应实验原有 `runs/`。

历史冻结脚本中保留了原始绝对路径。如果需要直接运行旧入口，按 `archive/organization_20261101/README.md` 恢复兼容符号链接。正常使用最终版无需恢复这些旧入口。

本次是目录与依赖路径整理，没有提交 GPU 作业，也没有改动 Windows 组会 PPT。最终副本的新路径仅做静态校验；性能与正确性的实测证据仍是上述原始作业。
