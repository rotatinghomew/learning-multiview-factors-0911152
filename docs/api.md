# 分析服务接口

资料服务在 8765 端口提供 `/releases`、`/releases/term-a` 和版本内的表分页。表页携带 revision；客户端必须读取同一清单版本并验证摘要，不能合并不同版本的页。

应用服务在 8766 端口提供 `GET /catalog`、`POST /analyses`、`GET /analyses`、`GET /analyses/{id}`。当前目录有 `overview`（资料汇总）和 `shared_barriers`（学习支持画像的共同维度与定位）。分析实现登记在 `learning.domain.ANALYSES`，函数接收完整 tables 和 parameters，返回可序列化对象。

创建请求字段为 request_key、release、analysis 和可选的 parameters 对象。request_key 是调用方的幂等编号：相同完整请求返回同一分析，相同编号不同内容返回 409。分析返回 id、request、revision 和 result。不同 request_key 是不同留存记录。缺少字段、未知分析类型或无效资料返回 422；查询不存在的记录返回 404。JSON 不允许 NaN 或 Infinity，无法估计的量使用 null 并在结果中说明原因。

保存后的分析内容不因资料更新而重算。新分析可在 result 内保存统计对象、参数、诊断及所需的版本化结构；不可覆盖其他分析或跳过资料完整性校验。数据库只保存本地分析，不是资料服务的业务真源。

## shared_barriers

`parameters` 支持 `permutations`、`alpha`、`max_dimensions`、`min_pairs`、`min_feature_loading`、`support_threshold`、`partial_support`、`seed`、`loo_limit` 和 `people`（学员编号列表，省略表示全体）。未知参数返回 422，不静默忽略。

结果分为三部分：

- `structure`：版本化结构（`structure_version` 为 `shared-barriers/1`），含特征中心/尺度、命名维度、载荷以及 `estimation` 诊断（维度数、特征值、置换零分布分位、被移除特征）。
- `shared_needs` / `view_specific`：跨两类以上资料成立的共同维度，以及只在单一资料中成立的结构；无法识别的维度不写入 `structure`，只在 `estimation.unidentified_dimensions` 中列出原因。
- `positions` / `position_summary`：每位学员在已保存结构上的定位。`support` 是该维度载荷中被实际观测到的平方载荷占比；`shared` 维度需至少在两类资料上同时有观测才标 `supported`，否则为限定范围的 `partial`；观测不足者为 `unassigned`，整类资料都没有者为 `no_observation`。任何情况下都不补值、不强制分配类型。

`diagnostics` 记录判断依据：`split_half`（按学员编号哈希对半，重拟合后维度对应程度）、`leave_one_out`（留出整位学员重拟合后维度对应程度）、`site_adjustment`（控制校区后维度是否仍成立）、`site_influence` / `age_influence`（定位分数与分组变量的关联）、`scale`（原始量纲第一主成分是否由单一资料支配）。

## 沿用已保存结构

`parameters.structure_from` 填先前 `shared_barriers` 分析的 id，服务读取该分析已保存的 `structure` 并对当前 release 的学员定位，不重新估计，也不改动来源记录。结果 `mode` 为 `apply`，`structure_source` 指向来源分析 id、revision 与结构版本。来源不是 `fit`、结构版本不兼容或结构不完整时返回 422。这样新学员只交了其中一类资料时，仍能沿用同一结构给出适用范围内的解释。
