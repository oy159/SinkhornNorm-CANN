# SinkhornNorm 优化基线

`main` 保存按向量宽度选择执行路径的基线，算子实现与
`9762d0948ff1acc291d14a3a4af371ba0cb1ff2d` 一致。CANNJudge
[提交 #929691](https://cannjudge.cn/public/op_challenge_xinan_final/sinkhorn_norm/submission/6adb4afc1dac5d235bee58bb)
为 **Pass**：5 个可见点通过，10 个点显示 Hidden，接口未返回总分。
5 个可见点耗时为 **1196.56、2.72、16.38、118.64、129.86 μs**。
相对上一基线 `ba8a6b3` 的单次历史观测，四点耗时下降 **3.49%～20.56%**；
第三点增加 **0.08 μs（0.49%）**，保留观察，不由单次数据推断波动范围。
本候选仅提交一次，Ubuntu 正式模板编译与 Host 资源检查通过。
当前没有云端 NPU，未做本版本的设备泛化测试；以上结果仅代表可见判题点。
后续优化从 `main` 创建实验分支。代码提交、提交额度、结果归档和集成要求见
[AGENTS.md](AGENTS.md)。逐版本结果见 [TEST_RESULTS.md](TEST_RESULTS.md)，
逐次原始成绩索引见 [history/submissions.jsonl](history/submissions.jsonl)。

## 源码与模板

| 路径 | 用途 |
| --- | --- |
| `src/host/tiling.cpp` | 形状与属性校验、UB 预算和分核 |
| `src/kernel/sinkhorn_norm.asc` | 单 Kernel 完成门控、softmax 和 Sinkhorn 迭代 |
| `src/host/launch.cpp` | 平台资源查询与 Kernel 启动 |
| `src/host/template_adapter.inc` | 官方 `run_kernel` 接口适配 |
| `include/sinkhorn_norm/` | 共享接口与 TilingData |
| `scripts/export-template.py` | 将共享源码导出为自包含判题源码 |
| `scripts/judge-experiment.py` | 判题额度、版本、基线及重复保护，自动归档并登记历史 |
| `testbench/` | 独立云端完整输出核查与官方 1+5 计时 |
| `template/kernel.asc` | 自动生成的上传文件 |
| `tests/`、`tools/reference/` | Host 测试与独立离线参考工具 |

只编辑 `src/`、`include/`，再显式导出；增加参与实现的文件时维护导出脚本的
`SOURCES` 清单。不要直接修改生成的 `template/kernel.asc`；其他官方模板文件保持原样。

```bash
python3 scripts/export-template.py
python3 scripts/export-template.py --check
```

当前实现使用 FP32、动态核数和 UB 预算，矩阵保留在 UB，一个 Kernel 完成全部计算。
DMA 直接搬入逐行对齐布局，WholeReduce/Brcb 与 repeat 运算完成矩阵归一化。
两行 gate 能放入一个向量时合并计算；宽矩阵采用列树归约，并在安全 eps 范围复用分母倒数。
Host 按形状和 eps 元数据选择编译期模式，每次只启动一个 Kernel。Host 仅处理元数据和启动配置，
离线参考结果不会传入算子。

## 本机验证

macOS 无需 CANN 即可运行 Host 与离线参考测试：

```bash
cmake -S . -B build/host -DCMAKE_BUILD_TYPE=Debug
cmake --build build/host --parallel 2
ctest --test-dir build/host --output-on-failure
python3 -m unittest discover -s tests -p test_cannjudge.py
python3 scripts/export-template.py --check
```

默认不构建设备库。离线 FP32 文件生成和输出比较见 [tests/README.md](tests/README.md)。
这些本机测试不执行 NPU，不能代替设备正确性或性能验证。

## CANN 9.0 编译

导出后编译官方模板工程：

```bash
python3 scripts/export-template.py
./scripts/remote-build.sh template
```

不带参数的 `./scripts/remote-build.sh` 构建独立共享库并运行 Host/离线测试。
固定镜像、远端目录及环境参数见 [远程编译说明](docs/remote-compile-cann9.md)。
编译脚本不运行 Device Kernel，也不消耗判题额度。

## CANNJudge

默认赛事和题目为 `op_challenge_xinan_final / sinkhorn_norm`，脚本按名称查询当前题目 ID。
上传仅包含可编辑的平级 `.asc/.h` 源码，保持原文；账号会话按 slot 隔离。

```bash
python3 scripts/cannjudge.py info
python3 scripts/cannjudge.py prepare --out /tmp/sinkhorn-upload-preview
python3 scripts/cannjudge.py submit --dry-run
```

`prepare` 完全离线，输出目录必须不存在且位于源码目录之外。
`info` 和 dry-run 读取线上题目及模板，不发出判题提交。
320 KiB JSON 请求和 20 文件是工具的本地预算。

账号文件可用忽略跟踪的 `scripts/accounts.env` / `scripts/acounts.env`，
或通过 `--env` 指定已有文件；也支持 `CANNJUDGE_ACCOUNT` / `CANNJUDGE_PASSWORD`。
登录后会话缓存于忽略跟踪的 `scripts/.cache/slotN/`：

```bash
python3 scripts/cannjudge.py login --env ../AddRmsNormBias/scripts/acounts.env --slot 1
```

真实提交使用实验包装器。先保存代码提交，并填写候选、已有成功基线和改动摘要：

```bash
python3 scripts/judge-experiment.py prepare \
  --experiment-id YOUR_NEW_EXPERIMENT \
  --baseline-experiment-id sinkhorn-vector-repeat-20261023 \
  --change-summary '说明具体优化机制'
```

`prepare` 完全离线。确认候选已完成编译和对应设备检查后，将 `prepare` 改为 `submit`。
包装器核对线上可编辑文件、所选账号当天全部题目的已用额度和提交间隔，
禁止基线源码复投；同一实验成功后的再次提交必须明确加 `--repeat`。
工具不自动登录、重试或补测基线。`--code-version FULL_SHA` 可在仅增加文档后
继续引用真正的代码提交，但上传文本必须与该提交精确一致。

```bash
python3 scripts/judge-experiment.py result SUBMISSION_ID
```

`result` 只查询本地历史里已有的提交。原始响应保存在仓库外的归档中，
`history/submissions.jsonl` 会立即登记 pending 并补全结果；失败、超时及 Hidden 同样保留。
多工作区可用 `--cache-root` 复用已有会话，用 `--archive-root` 指向长期归档目录。
不使用判题器复测历史基线；收益不足以判断时保留待观察结论。

云端核查与官方计时见 [testbench/README.md](testbench/README.md)：

```bash
scripts/remote-testbench.sh build
scripts/remote-testbench.sh run --cases P01,P02,C01,C02,C03,C04
```
