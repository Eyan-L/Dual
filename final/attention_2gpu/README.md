# 双卡基础版：一张计算卡、一张 KV 供给卡

历史成功作业：`26075197.pbs101`。源码来源见 `../release.json`（controls 为 `../../release.json`）；原始配置和源文件校验清单保留在 `provenance/`。

在 NSCC 执行 `bash final/submit.sh attention_2gpu` 可提交本版本。此命令会实际提交新作业；本次整理没有执行。

历史结果：`results/04_flashinfer/nvlink_flashinfer_multiturn/26075197/`。
提交配置为本目录 `config.json`。测量实现及核函数只做依赖路径迁移，没有改动算法。
