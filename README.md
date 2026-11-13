# Slack Serving

快速上手见下节，版本与来源说明见 [final/README.md](final/README.md)。`final/cooperative_4gpu/` 是当前最后采用的完整合作版本；`final/attention_2gpu/` 是双卡 attention 与多轮执行基础版，两者用途不同。

GitHub 私有仓库：[Eyan-L/Slack-Serving](https://github.com/Eyan-L/Slack-Serving)。仓库跟踪 `final/`、本说明和项目操作约定，包含代码、配置、冻结依赖、提交脚本及来源校验信息。`final/data/corpus.json` 是复现所需的输入语料，随最终版提交。下列历史实验、结果、原始 trace、整理记录及运行缓存仅保留在本地 / NSCC，不纳入 Git。

## 快速上手

当前入口面向 NSCC / PBS / Singularity 环境。模型权重、容器镜像和历史结果需要另外准备；仓库内已包含最终运行代码、冻结源码依赖和输入数据。整理后的入口尚未重新执行 GPU，已有实测来自原始冻结作业。

### 1. 获取代码并选择入口

使用具有私有仓库访问权限的账号获取代码，将项目放在计算节点和容器均可访问的共享目录：

```bash
git clone https://github.com/Eyan-L/Slack-Serving.git
cd Slack-Serving
```

| 入口 | GPU 数量 | 用途 |
|---|---:|---|
| `attention_2gpu` | 2 | 一张计算卡 + 一张 KV 供给卡，受控多轮 prefill/decode；适合先验证基础链路 |
| `cooperative_4gpu` | 4 | 两个模型服务进程 + 三路 KV 供给，五个 subagent 与持续背景请求 |
| `controls/independent_100` | 2 | 原生双卡独立运行，100% / 100% 本地 KV |
| `controls/native_matched` | 2 | 原生全本地容量对照，80% / 117.35% KV |

### 2. 准备环境并调整配置

历史环境为 A100-SXM4-40GB、Qwen3-8B BF16、TP=1，使用 PyTorch `2.13.0+cu130`、SGLang `0.5.21`、FlashInfer `0.6.18`。源码依赖这些版本的接口；所需 Python/CUDA 构建环境由原 SIF 提供，仓库未提供通用 pip 安装或镜像构建流程。合作入口要求四张卡间具备相应的 NVLink / CUDA P2P 可达性，双卡基础版要求两卡 P2P。

按自己的环境检查以下文件：

- `final/<入口>/config.json`：将 `model_path` 改为已下载的完整 Qwen3-8B 模型目录，包含 tokenizer。运行时启用离线模式。
- `final/job.sh`：调整 Singularity 模块、SIF 路径、容器内 Python 路径和 `--bind`。当前默认镜像为 `/scratch/users/ntu/n2602695/sif/sglang_v0.5.21_cuda13.sif`，Python 为 `/opt/sglang/bin/python3`；绑定目录必须覆盖代码、模型和结果目录。
- `final/<入口>/run.pbs`：调整项目账号、队列和资源申请。当前默认 `normal` / `02:00:01`；双卡任务按原 NSCC 规则路由到 g2。

`job.sh` 在运行前校验 `final/SHA256SUMS`。修改配置或部署脚本后，须更新清单中**对应文件**的哈希；保留 `provenance/` 中的历史原件。以合作入口为例：

```bash
# 将输出替换到 final/SHA256SUMS 中的同名条目。
(cd final && sha256sum job.sh cooperative_4gpu/config.json cooperative_4gpu/run.pbs)
# 更新后，从 final 目录验证清单。
(cd final && sha256sum -c SHA256SUMS)
```

### 3. 提交一次运行

以下命令在准备好环境的 NSCC 登录节点、仓库根目录执行；每条都会实际提交一个 GPU 作业，选择需要的入口即可：

```bash
# 基础双卡链路
bash final/submit.sh attention_2gpu
# 或完整合作回放
bash final/submit.sh cooperative_4gpu
```

提交后记录返回的 Job ID，使用 `qstat <Job ID>` 查看状态。运行会先执行预热和必要的正确性检查，再采集正式结果；当前配置每个性能 case 正式执行一次。

### 4. 查看输出

结果写入当前仓库下的 `results/<阶段>/<实验>/<作业数字ID>/`：

- 双卡基础版：`results/04_flashinfer/nvlink_flashinfer_multiturn/`。
- 合作版：`results/05_trace_serving/cooperative_trace_background/`。
- 两个对照：`results/05_trace_serving/dual_trace_background/`、`dual_trace_capacity_control/`。

先检查 `pbs_exit.json` 的 `exit_code` 和 `driver.log`。合作版的 `measurement/review.json`、`measurement/gpu0/result.json`、`measurement/gpu1/result.json` 保存运行状态与任务结果；逐轮 KV、请求及迁移记录在对应 GPU 子目录。双卡基础版完成后生成 `measurement/collection.json` 和各 case 的 `plain_0.json`、`validation.json`、`profile.json`。

分析脚本需要相应 Python 依赖；双卡基础版可执行：

```bash
python3 final/attention_2gpu/code/analyze.py \
  results/04_flashinfer/nvlink_flashinfer_multiturn/<作业数字ID>/measurement
```

合作版与容量对照的 `analyze.py` 仍引用固定历史作业的基线结果。这些结果未上传 GitHub，首次 clone 后需要另行同步相应基线到脚本指定的 `results/` 路径，或先调整基线读取路径，才能完成对比绘图。提交脚本不会自动执行分析或更新 PPT。

## 本地与 NSCC 工作目录

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

主版本采用 FlashInfer attention、本地优先分配、动态迁移 **1+2+4 / 每批迁移 2048 个 token 槽位**；当前触发点仍接近本地耗尽，2048 不代表提前启动的剩余空间阈值。四张物理卡运行两个模型，以三路 NV4 为 offpath 提供远端 KV。它是三路供给的硬件代理，不能视为已经实测真正双卡 NV12。

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

## 已发现的问题与下一步优化（2026-11-13）

以下是根据本地原始记录确认的问题及待验证方向，尚未修改最终运行实现，也未将推测当作性能结论。原始结果保留，不事后扣除耗时改写测量值。

### 1. 补齐预热与端到端配对复测

等容量全本地作业 `26079802` 的 onpath 存在两段合计 **4.51 s** 的模型调用外停顿；合作作业 `26079252` 的 offpath 存在合计 **5.88 s** 的异常页表准备耗时。后者对应此前预热未覆盖的 batch / 长度组合，疑似首次编译，但缺少编译日志直接确认。单次约 1%–2% 的吞吐或 onpath 时延差异不足以证明稳定收益或退化。

下一步：覆盖实际编译形状后做三组全本地 / 合作配对复测，交替执行顺序；匹配负载、容量和停止规则，分别记录 prefill、decode、页管理和调用间隙。同时报告完成调用输出吞吐与包含未完成请求部分输出的生成吞吐，保留每次结果和异常停顿。

### 2. Prefill 的远端片上多轮预取

当前完整合作版的 prefill 采用逐层本地 HBM 暂存。后续无 HBM 暂存的直接读取尝试中，提前发 K 和 K/V 双缓冲尚未消除远端开销；这些局部尝试不能证明 prefill 无法直接读取远端 KV。

下一步：让远端搬运进度与计算 tile 解耦，使用独立 shared memory 缓冲，提前多个 tile 读取，消费时才等待，并在消费后循环补充。测量预取深度、等待、shared memory / 寄存器占用及 CTA 并发。须保证每批数据在消费前到达；是否 compute bound 取决于具体 attention 形状，尤其是短追加输入。目标是隐藏远端等待并省去本地 HBM 暂存；不能预设一定快于已由计算限制的全本地 kernel。

### 3. 迁移提前启动，跨多轮延迟提交

当前异步迁移仍在下一轮开始提交 pending 迁移，主计算流会等待 copy-done；触发接近本地耗尽，覆盖窗口较短。CPU 负责发起，实际搬运使用 GPU Triton kernel，仍会竞争 SM、HBM 和 NVLink。

下一步：将迁移批量与提前启动阈值独立配置。复制期间固定本地源槽位、预留远端目标槽位，页表仍指向本地，后续若干轮继续计算；非阻塞检查复制完成后，在安全 step 边界切换页表，并在旧读取结束后归还本地槽位。剩余本地空间必须覆盖期间新增 KV；追加 prefill、请求释放和比例限制也需纳入 pending 状态管理。该方向尚未实现，不能视为免费搬运。

### 4. 约束单请求及执行窗口的远端集中度

当前全局 FIFO 迁移只约束全池 `remote/local`，未约束单个请求。合作作业 `26079252` 中，曾有请求释放前本地 **541**、远端 **11,746** 个 token（远端占约 **95.6%**）；同一循环全池本地 **86,815**、远端 **11,746**，远端/本地约 **13.53%**，仍满足全局上限。

混排按请求内部组织，本地 KV 耗尽后会产生远端密集区段；其他请求的本地读取能否同时覆盖，取决于 block / split 的执行分布。等待工具的请求也可能占有本地 KV，却不参与当前计算。因此全池存储比例不能保证当前 attention 读取比例或时延。

下一步：记录并分析每请求、活动 batch 与 split / tile 的远端分布及尾部耗时；比较按请求均衡迁移与按当前工作量选择迁移对象等策略，再决定是否需要单请求上限或跨请求调度。双卡全局上限约为 `R/L < 5.99%`，三路供给约为 `R/L < 17.97%`，均不是每个请求的上限。当前跨请求 prefix 复用关闭；未来开启后还需考虑共享 prefix 的访问频率和缓存复用。

## 使用与追溯

`final/` 包含整理后的源码副本、原始配置和依赖来源。历史 `experiments/` 与结果的常规文件原样保留，没有删除中间结果或 raw trace；NSCC 的历史结果索引可链接到对应实验原有 `runs/`。

历史冻结脚本中保留了原始绝对路径。如果需要直接运行旧入口，按 `archive/organization_20261101/README.md` 恢复兼容符号链接。正常使用最终版无需恢复这些旧入口。

本次是目录与依赖路径整理，没有提交 GPU 作业，也没有改动 Windows 组会 PPT。最终副本的新路径仅做静态校验；性能与正确性的实测证据仍是上述原始作业。
