# 学习支持画像

面向老年学习服务的内部资料分析工具。当前支持版本资料检查、学习支持画像的描述性汇总、跨资料结构分析（`shared_barriers`）以及分析历史。它只描述已有观察，不自动生成干预结论。`learning/domain.py` 包含业务校验及汇总逻辑，`learning/shared_barriers.py` 负责跨资料结构；资料源和应用是同一机器的两个本地 HTTP 进程，分别保留来源契约和分析消费契约。

## 跨资料结构

兴趣、出行与听课障碍、到课与志愿活动三类资料量纲不同，且有人整类未交。`shared_barriers` 不对缺失单元补值：相关用成对完整记录估计，学员定位只用已观测单元。它区分跨资料成立的共同维度（`shared_needs`）、只在单一资料成立的结构（`view_specific`）以及无法识别的维度；维度数由置换零分布加单位根判断，单类资料只有一个强特征时不当作需求。结果同时给出重拟合对半、留出整位学员、控制校区和原始量纲支配等诊断。新学员可只交一类资料，通过 `structure_from` 沿用已保存结构给出适用范围内的解释，支撑不足时保持未分配。细节见 docs/api.md 和 docs/data-contract.md。

## 安装和验证

使用 Python 3.11.2，数值依赖固定在 requirements.txt；部署时先安装，不在服务启动或测试过程中下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q learning scripts tests
python3 -m unittest discover -s tests -v
python3 scripts/demo.py
```

最后一条命令启动两个真实服务，等待就绪后读取目录、执行当前汇总与跨资料结构分析、沿用已保存结构对另一版本定位、保存并回读结果，然后退出。它使用临时数据库，不会覆盖已有分析；服务就绪即继续，最长就绪期限为 65 秒。正常执行只需数秒。使用容器时先构建再运行，构建不计入业务演示时间：

```bash
docker build -t multiview-factors:local .
docker run --rm multiview-factors:local
```

## 持续运行

```bash
python3 -m learning.provider
python3 -m learning.service
```

以上命令在不同终端执行，应用默认把结果写入 var/analyses.sqlite3。数据字段、缺失含义和版本差异见 docs/data-contract.md；应用接口与幂等约定见 docs/api.md。扩展业务分析时应保持原有 overview 和已保存分析可查询。
