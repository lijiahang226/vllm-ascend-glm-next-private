# KPool master c1c35176e7 A5 x86_64 构建产物

构建日期：2026-08-30

## 构建环境

- 构建主机：`sz-blue-950pr-13-241`
- 主机架构：`x86_64`
- CANN：`/usr/local/Ascend/cann-9.1.0`
- 源码仓：`/home/npu_user1/ops-transformer_kpool_master_c1c35176e7`
- 源码分支：`master`
- 源码提交：`c1c35176e7990abac1bb07d601ebfe8ee742d59d`
- 构建算子：`key_pool,pool_key_indexer`

## Run 包

A3 和 A5 使用同一源码提交分别构建，目标 SoC 不同。由于构建主机为
x86_64，两个包的文件名均为 `cann-ops-transformer-custom_linux-x86_64.run`，
通过目录区分目标平台。

| 目标 SoC | 归档路径 | 构建命令 | 文件大小 | SHA256 |
| --- | --- | --- | ---: | --- |
| A3 / `ascend910_93` | `a3/cann-ops-transformer-custom_linux-x86_64.run` | `bash build.sh --pkg --soc=ascend910_93 --ops=key_pool,pool_key_indexer -j16` | 4,257,900 bytes | `8b4957eed52476e15932ebdcc090cf144952f373408734788b1831688f8e5c3f` |
| A5 / `ascend950` | `a5/cann-ops-transformer-custom_linux-x86_64.run` | `bash build.sh --pkg --soc=ascend950 --ops=key_pool,pool_key_indexer -j16` | 3,423,185 bytes | `5e6122e26be3b153bbd14d5f94887af5ac4d8bffadf0b153e493d31dea2f75ad` |

两个 Run 包均已通过 Makeself 自解压包生成流程，包内容包含
`KeyPool`、`PoolKeyIndexer` 的 ACLNN 头文件、算子实现、Tiling 配置和
对应目标 SoC 的二进制文件。

## Python wheel

| 产物 | 归档路径 | 构建命令 | 文件大小 | SHA256 |
| --- | --- | --- | ---: | --- |
| `cann_ops_transformer-1.0.0-py3-none-any.whl` | `wheel/cann_ops_transformer-1.0.0-py3-none-any.whl` | `cd torch_extension && python3 -m build --wheel --no-isolation` | 420,753 bytes | `ff04fd4203092dde59d768d2c25f2f2626466a0e4bb0b99b10fcb254399b0da1` |

该 wheel 为纯 Python wheel，包含：

- `key_pool`
- `pool_key_indexer`
- 两个算子的 C++ Torch 扩展封装
- 两个算子的 Python 接口和文档

wheel 未修改文件名，可直接使用标准 pip 命令安装。

## 构建日志

- `validation/20260830_master_c1c35176e7_a5build/build_logs/a3_x86_build_final.log`
- `validation/20260830_master_c1c35176e7_a5build/build_logs/a5_x86_build.log`
- `validation/20260830_master_c1c35176e7_a5build/build_logs/wheel_build.log`
