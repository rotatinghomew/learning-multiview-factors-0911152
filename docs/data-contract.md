# 学习支持画像资料约定

资料全部是本地合成业务记录，不含真实个人信息。`term-a` 为完整版本，`term-b` 保持业务值不变而调整表行次序，用于检验分析是否不恰当地依赖到达顺序；`limited` 保留登记资料并移除一类观测，表示证据不足的版本。版本内容不可就地修改。

- `people`：业务唯一键为 `person`。
- `observations`：业务唯一键为 `person, view, feature`。
- `features`：业务唯一键为 `view, feature`。

observations 是稀疏长表，未出现的 feature 或整个 view 均为缺失而不是 0。features 给出业务合法范围，不是归一化建议；interests、access、participation 分别是偏好强度、障碍强度和参与比例，方向含义不同。people 包含零观测的人。新增学员也沿用同一 feature 身份，未知特征不可默认为已训练列。因子的符号、排列、尺度本身不具业务身份。

字段之外不能从示例的编号顺序或随机生成规律推导业务关系。涉及权重、相关范围、模型支撑的判断必须使用明确登记的字段和业务假设。
