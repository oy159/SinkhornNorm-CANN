# SinkhornNorm-CANN

基于 Ascend C 的 FP32 SinkhornNorm 算子实现，包含门控、行 softmax 和 Sinkhorn 行列归一化。
全部张量计算在 Device 侧完成，每次调用只启动一个 Kernel；Host 负责形状校验、资源预算和启动配置。

## 比赛提交

GitCode 参赛作品提交：[PR #2723](https://gitcode.com/cann/cann-ops-competitions/pull/2723)，
可查看对应的[提交记录](https://gitcode.com/cann/cann-ops-competitions/pull/2723/commit)。

## 性能

相对最初的 Ascend C 实现，前 5 个对应判题点的加速比几何平均为 **18.36×**，最高 **67.23×**。

| 对应判题点 | 加速比 |
| --- | ---: |
| 1 | 67.23× |
| 2 | 3.09× |
| 3 | 11.20× |
| 4 | 28.34× |
| 5 | 31.73× |

加速比 = 初始版本耗时 ÷ 最终版本耗时。采用历史判题中逐点耗时的中位数：
初始版本为 2026-10-23 的 2 次通过记录，最终版本为 2026-11-07 的 3 次通过记录。
按前后大致对应的用例直接比较，包含期间用例修订的影响，用于展示本项目的整体优化幅度。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/`、`include/` | 算子源码和共享接口 |
| `submission/` | 官方模板工程，`kernel.asc` 由源码导出 |
| `scripts/` | 模板导出、判题提交、远程构建与测试 |
| `tests/`、`tools/reference/` | Host 测试与独立 CPU 参考工具 |
| `testbench/` | 设备正确性检查与性能采集 |

## 本地检查

需要 CMake、C++17 编译器和 Python 3.10+。以下检查不执行 NPU Kernel：

```bash
python3 scripts/export-template.py --check
cmake -S . -B build/host -DCMAKE_BUILD_TYPE=Debug
cmake --build build/host --parallel 2
ctest --test-dir build/host --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
```

只维护 `src/` 和 `include/`，修改后运行 `python3 scripts/export-template.py`。
源码和重新导出的 `submission/kernel.asc` 一起提交。其他官方模板文件保持原样。

## 远程构建与测试

构建环境使用 CANN 9.0，默认架构为 `dav-2201`。远端构建机需要 SSH、Docker 和构建镜像；
设备测试机需要 Ascend NPU、CANN、`msopprof`、NumPy 和 CPU PyTorch。
远程脚本使用当前 Git 提交标识测试版本，运行前先提交源码。

```bash
export ASCEND_BUILD_HOST=your-build-host
./scripts/remote-build.sh submission

export ASCEND_TEST_HOST=your-npu-host
./scripts/remote-testbench.sh build
./scripts/remote-testbench.sh run --suite smoke
```

`ASCEND_BUILD_WORKSPACE_NAME`、`ASCEND_TEST_ROOT` 可覆盖远端目录，`NPU_ARCH`、`BUILD_JOBS` 可覆盖架构和编译并行数。
设备测试默认逐个 case 预热 1 次、测试 5 次，使用官方 profiler 的 Task Duration(us)，并检查完整输出。
可用 `python3 testbench/run.py --list` 查看通用测试集，使用 `--cases P01,P02` 选择用例。

## 判题提交

只提交 `submission/` 下可编辑的算子源码，正式提交前需要与 Git HEAD 一致。
账号文件和登录缓存仅保存在本地。账号文件 `scripts/accounts.env` 示例：

```ini
slot=1
email=your-account
code=your-password
```

```bash
chmod 600 scripts/accounts.env
python3 scripts/cannjudge.py login --slot 1
python3 scripts/cannjudge.py prepare --out /tmp/sinkhorn-upload-preview
python3 scripts/cannjudge.py submit --slot 1 --dry-run
python3 scripts/cannjudge.py submit --slot 1 --watch
```

`prepare` 离线执行，目标目录必须不存在且位于源码目录之外；`--dry-run` 联网核对提交内容但不提交。
正式提交前自行核对比赛开放状态和账号剩余额度。提交结果可用 `python3 scripts/cannjudge.py result SUBMISSION_ID --slot 1` 查询。
