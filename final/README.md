# 最终采用的合作版本

主入口是 `cooperative_4gpu/`，不是历史实验目录中名称相似的中间版本。

| 目录 | 用途 | 成功实测作业 |
|---|---|---|
| cooperative_4gpu | 两个模型服务进程、三路远端 KV，真实五个 subagent + 持续背景请求 | 26079252 |
| attention_2gpu | 一个计算进程 + 一个 KV 供给进程，多轮 prefill/decode/追加输入 | 26075197；参考 26075145、26075216 |
| controls/independent_100 | 双卡各自运行，100% / 100% 本地 KV | 26078142 |
| controls/native_matched | 80% / 117.35% 全本地原生容量对照 | 26079802 |

核心方案保留 FlashInfer 的 attention 数学与规划，在页地址解析中加入本地/远端地址；本地优先分配，动态迁移采用 **1+2+4、提前 2048 token 槽位**。`vendor/` 是提取的冻结历史依赖，保留来源层级以处理同名模块，不需要加载 `experiments/` 中的运行代码。

四卡主版：GPU0 跑两个 onpath 长尾并持有一份远端 KV；GPU3 跑三个 offpath；GPU1、2 仅供给 KV。参考池 C=98,304，onpath 自用上限 78,643，offpath 物理池 97,792，远端有效上限 17,573，总可用 115,365；remote/local <17.97%，远端全部计入逻辑 onpath 剩余空间。512 槽位 staging 成本已扣除。tile32 中混入 5 份远端读取。三路 NV4 是硬件代理，不能等同于真正双卡 NV12。

双卡基础版：remote/local <5.99%，decode 每 128 个 KV token 混入 8 个远端，仍用 1+2+4 / 2048。只有一张卡执行模型；它不是双模型合作回放。默认配置重新生成输入、执行所有 reference/native/local/peer case；历史 r4 曾复用 26075145 的完成 case，原配置在 provenance 中原样保存。

## 运行

在 NSCC 项目根目录按需执行：

```bash
bash final/submit.sh cooperative_4gpu
bash final/submit.sh attention_2gpu
bash final/submit.sh controls/independent_100
bash final/submit.sh controls/native_matched
```

每个命令实际提交一次作业，normal / 02:00:01，不自动提交全部。原有模型路径、SIF 和 CUDA/FlashInfer 环境为外部依赖，见 config.json 和 job.sh。两卡基础版的 corpus 原文已打包到 data/corpus.json，SHA256 已核对。当前输入是按 trace token 长度构造的测试材料，保留输入输出长度、工具等待和组等待，不代表任务答案质量评测。

新结果写入 `results/<阶段>/<项目>/<作业数字ID>/`。PBS 结束后可执行对应目录的 `analyze.py <结果目录>`；attention_2gpu 使用 `code/analyze.py <结果目录>/measurement`。分析需原有 numpy/matplotlib 环境。历史汇报生成器只供追溯，仍涉及特定 Windows 临时目录，不自动执行、不修改组会 PPT。

## 打包验证范围

`source_manifest.json` 记录 101 个 Python 源文件的原始/整理后 SHA256 及路径替换数量。`provenance/` 保存原配置和依赖清单。此次仅改新副本中的路径与根目录发现逻辑，原始实验文件未修改；验证了语法、封装路径和清单。**整理后的入口没有重新跑 GPU**，实测证据属于上表历史作业。

本地的合作组汇报生成器比 NSCC 同名脚本新，最终包采用本地版本；原件快照在 cooperative_4gpu/provenance/original_make_presentation.py，两地来源哈希分别记录在 source_manifest.json。运行核函数及其冻结依赖均与 NSCC 一致。
