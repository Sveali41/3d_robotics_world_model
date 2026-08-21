# RoboCasa × RAPS × PointWorld 后续执行计划

更新时间：2026-07-28

本文档从当前仓库状态出发，描述 RoboCasa RAPS 主线从“可收集 bootstrap
样本”推进到“可复现的 world-model-guided active exploration”的全部后续工作。
原始研究设计仍保留在
`docs/ROBOCASA_RAPS_POINTWORLD_PLAN.md`；本文档只承担可执行任务拆分和验收跟踪。

## 0. 当前基线和统一约定

### 0.1 已确认完成

- [x] RoboCasa kitchen 可由 `configs/kitchen_custom.yaml` 配置。
- [x] 当前场景可生成离屏 RGB render。
- [x] Fetch base/body 固定，公开 low-level action 为
  `[dx, dy, dz, gripper]`。
- [x] 已定义 10 个 agent-centric primitive。
- [x] 已定义 official-RAPS-compatible 20D high-level action：
  `10D primitive one-hot + 10D normalized parameter bank`。
- [x] `RAPSEnvWrapper` 可接收一个 20D action，并执行多个 4D low-level
  simulator step。
- [x] `PrimitiveRecord` 可保存 observations、low actions、qpos、EE pose 和
  object pose。
- [x] primitive trajectory 可重采样为 PointWorld 所需的 T=11 phase view。
- [x] bootstrap collector 可输出 RoboCasa RAPS WebDataset。
- [x] WDS 中已有 canonical `scene_flows`、`robot_flows`、RAPS action 和 mask。
- [x] PointWorld 模型代码已有可选的显式 `raps_action` embedding 接口。
- [x] `lift` / `drop` 已统一使用 schema 参数名 `z_dist`。

### 0.2 Contract 冻结

后续实现必须遵守以下 contract，除非先更新本文档并迁移所有消费者。

- [ ] 高层 action shape 固定为 `(20,)`。
- [ ] 高层 action 前 10 维是 primitive selector。
- [ ] 高层 action 后 10 维是归一化到 `[-1, 1]` 的共享参数 bank。
- [ ] low-level action shape 固定为 `(4,)`：
  `[dx, dy, dz, gripper]`。
- [ ] PointWorld temporal horizon 固定为 `T=11`。
- [ ] 每个 primitive record 满足
  `len(observations) == len(low_actions) + 1`。
- [ ] 所有 exploration budget 使用真实 low-level `env.step()` 次数计费。
- [ ] primitive 控制不读取 object ID、object pose 或未来 simulator state。
- [ ] object state 只允许用于 outcome label、GT scene flow 和离线 metrics。
- [ ] PointWorld ensemble 的所有 member 对同一 candidate 使用完全相同的
  predicted robot flow。

### 0.3 建议新增的测试结构

- [ ] 新建 `tests/robocasa/test_raps_schema.py`。
- [ ] 新建 `tests/robocasa/test_primitives.py`。
- [ ] 新建 `tests/robocasa/test_raps_wrapper.py`。
- [ ] 新建 `tests/robocasa/test_primitive_record.py`。
- [ ] 新建 `tests/robocasa/test_fetch_robot_flow.py`。
- [ ] 新建 `tests/robocasa/test_scene_correspondence.py`。
- [ ] 新建 `tests/robocasa/test_raps_wds_roundtrip.py`。
- [ ] 新建 `tests/exploration/test_candidate_selection.py`。
- [ ] 为 simulator-heavy 测试添加 `slow` marker，允许快速单元测试与完整
  GPU/CPU smoke test 分开运行。

---

## 1. Phase A：Primitive 与高层 wrapper 正确性

目标：10 个 primitive 都能可靠执行，wrapper 的 episode/budget 语义明确且可测试。

状态：已完成。自动测试 57 项通过；真实 RoboCasa smoke validation
完成 10 类 primitive × 5 trials，共 50 次，无错误。

### A1. Primitive schema 单元测试

涉及文件：

- `domain/robocasa/primitives.py`
- `tests/robocasa/test_raps_schema.py`

任务：

- [x] 检查 primitive 名称顺序固定且无重复。
- [x] 检查 `num_primitives == 10`。
- [x] 检查 `num_parameters == 10`。
- [x] 检查 `action_dim == 20`。
- [x] 对每个 primitive 执行 encode/decode round-trip。
- [x] 检查 normalized 参数边界 `-1`、`0`、`1` 的物理缩放。
- [x] 检查 action 中的 NaN/Inf 被拒绝。
- [x] 检查 unknown primitive 被拒绝。
- [x] 检查缺失参数产生明确错误。
- [x] 检查 `open_gripper` / `close_gripper` 不读取参数 bank。
- [x] 检查 `lift` / `drop` 使用 `z_dist`。
- [x] 检查 directional primitive 使用 `x_dist` / `y_dist`。

验收：

- [x] schema 测试在不启动 simulator 的情况下全部通过。
- [x] schema contract 在测试中明确断言为 20D，而不是旧计划中的 28D。

### A2. Primitive 状态机测试

涉及文件：

- `domain/robocasa/primitives.py`
- `tests/robocasa/test_primitives.py`

任务：

- [x] 为 fake context 构造 EE position 和 gripper state。
- [x] 测试 `move_delta_ee_pose` 的目标点。
- [x] 测试 `top_grasp` 的 descend/close 状态切换。
- [x] 测试 `lift` 只产生正 Z 目标。
- [x] 测试 `drop` 只产生负 Z 目标。
- [x] 测试 `move_left` / `move_right` 的 Y 方向符号。
- [x] 测试 `move_forward` / `move_backward` 的 X 方向符号。
- [x] 测试 gripper primitive 的 command sign。
- [x] 测试达到目标后 status 为 `success`。
- [x] 测试超过 `max_steps` 后 status 为 `timeout`。
- [x] 测试每个 primitive 输出的 low action 都是有限 `(4,)` 数组。
- [x] 明确负 distance 参数的语义：
  保留 `abs()` 并将参数视为 magnitude，或收紧 schema 到非负范围。
- [x] 将最终 distance 语义写入注释和测试。

验收：

- [x] 10 个 primitive 均有独立成功路径测试。
- [x] 至少一个 primitive 有 timeout 路径测试。
- [x] primitive controller call graph 不读取 object pose。

### A3. Simulator primitive smoke test

建议新增：

- `bin/validate_robocasa_primitives.py`
- `logs/validation/primitives/summary.json`

任务：

- [x] 固定 seed 初始化当前 kitchen。
- [x] 每个 primitive 至少执行 5 次。
- [x] 保存 primitive 名称、参数、duration、status、termination reason。
- [x] 保存每次执行前后 EE pose。
- [x] 保存 base/body 最大漂移。
- [x] 保存 NaN/Inf 检查结果。
- [ ] 可选保存每类 primitive 的 GIF。
- [x] 对无法自然 success 的场景记录 timeout，而不是无限执行。

验收：

- [x] 10 类 primitive 都实际进入 simulator。
- [x] 所有 low action 有限且 shape 为 `(4,)`。
- [x] `lift` / `drop` 不再产生参数 `KeyError`。
- [x] 输出机器可读 `summary.json`。

### A4. TimeLimit 与 budget 修正

涉及文件：

- `bin/run_robocasa_kitchen.py`
- `domain/robocasa/primitive_wrapper.py`
- `bin/collect_robocasa_raps_wds.py`

任务：

- [x] 确认 `gym.make()` 返回对象上是否挂载 low-level
  `TimeLimit(max_episode_steps=200)`。
- [x] 决定并实现一种策略：
  移除 low-level TimeLimit，或将其上限设置为高层 episode 总 low-level budget。
- [x] `max_primitives_per_episode` 只按 primitive 数计费。
- [x] `max_low_level_steps_per_episode` 只按真实 simulator step 数计费。
- [x] 单个 primitive 内遇到 base env `terminated/truncated` 立即停止。
- [x] budget 在 primitive 中途耗尽时，record 的 status/reason 明确。
- [x] primitive 正常完成不能自动终止整个高层 episode。
- [x] reset 后 wrapper 的 primitive ID、low-step counter 和 record list 清零。

验收：

- [x] 高层 episode 可超过 200 个累计 low-level step，直到配置预算耗尽。
- [x] duration 精确等于该 primitive 内部 `env.step()` 调用数。
- [x] episode 总 low-level steps 不超过配置上限。
- [x] terminated/truncated 语义有自动测试。

---

## 2. Phase B：Dense trajectory 与 WDS contract

目标：每个 primitive 的 dense record 完整、可回放，并能无损转换为 T=11 WDS。

状态：已完成。Phase A+B 自动测试共 75 项通过；真实 RoboCasa record/WDS
round-trip 和 PointWorld 完整 batch 均已验证。

### B1. 扩展 PrimitiveRecord

涉及文件：

- `exploration/trajectory_recorder.py`
- `domain/robocasa/primitive_wrapper.py`

新增或补齐字段：

- [x] `parameters_physical`。
- [x] `qvel [H+1, Dq]`。
- [x] `gripper_open [H+1]`。
- [x] `full_controller_actions [H, controller_dim]`，若环境可提供。
- [x] `contact_events [H, ...]`。
- [x] `termination_reason`。
- [x] `base_pose [H+1, 7]`。
- [x] `body_qpos [H+1, Db]`。
- [x] robot type。
- [x] URDF path 或稳定资源标识。
- [x] active joint ordering。
- [x] camera names。
- [x] scene/layout/style IDs。
- [x] collector/schema version。

实现细节：

- [x] observation 和 simulator state 使用深拷贝，避免后续 step 覆盖历史值。
- [x] 每次 append 后检查各序列长度。
- [x] finalize 时执行完整性验证。
- [x] 中断或异常 record 标记为 invalid，不写入正式训练 shard。
- [x] outcome summary 与 dense contact history 分开保存。

验收：

- [x] `observations/qpos/qvel/ee_pose/object_poses` 长度均为 `H+1`。
- [x] `low_actions/contact_events` 长度均为 `H`。
- [x] record 可序列化并重新载入。

### B2. T=11 phase sampling

涉及文件：

- `exploration/trajectory_recorder.py`

任务：

- [x] 为 `H=1`、`H<10`、`H=10`、`H>10` 编写测试。
- [x] 保证 sample indices 单调不下降。
- [x] 首帧 index 必须是 0。
- [x] 尾帧 index 必须是 H。
- [x] 短 primitive 的重复尾帧必须由 temporal mask 标为 padding。
- [x] 明确 low action 在重采样后的含义：
  nearest action、区间聚合或只保留 dense sidecar。
- [x] 长 primitive 第一版保持全局 phase resample；如改为滑动窗口，
  每个 window 必须保存 window index 和 primitive ID。
- [x] `primitive_phase` 与 sample indices 一致，而不只是固定线性占位。

验收：

- [x] 所有 horizon 测试通过。
- [x] T=11 输出的时间索引和 mask 可解释。
- [x] 不存在 observation/action off-by-one。

### B3. WDS round-trip

涉及文件：

- `exploration/robocasa_collector.py`
- `common/utils.py`
- `PointWorld/dataset_components/decoders.py`
- `PointWorld/dataset_components/dataloader.py`

任务：

- [x] 定义并保存 `sample_schema_version`。
- [x] 保存 primitive key 到 episode/primitive ID 的映射。
- [x] 保存 physical parameters。
- [x] 保存 dense record sidecar，或保存可追溯的原始 record 路径。
- [x] 检查 train/test split 对 seed 可复现。
- [x] 检查 WDS decode 后 action、duration、mask 与写入前完全一致。
- [x] 检查 canonical scene/robot arrays dtype 和 shape。
- [x] 检查每个 sample 都有 status 和 outcome metrics。
- [x] 检查不使用 reward-dependent filtering。
- [x] 为损坏 tar member 提供清晰报错。

验收：

- [x] `raps_action.shape == (20,)`。
- [x] `scene_flows.shape[0] == 11`。
- [x] `robot_flows.shape[0] == 11`。
- [x] temporal mask shape 为 `(11,)`。
- [x] round-trip 数值字段逐项相等。
- [x] PointWorld dataloader 可读取一个完整 batch。

---

## 3. Phase C：Fetch robot-flow 几何验证

目标：确认真实未来 qpos 经 Fetch URDF/FK 生成的 robot point trajectory 是可信 GT。

### C1. 依赖和资源检查

涉及文件：

- `PointWorld/integration/pointworld_wrapper.py`
- `PointWorld/robot_sampler.py`
- Fetch URDF/mesh assets

任务：

- [x] 确认 `urdfpy`、`addict`、`pytorch_kinematics` 在 `rlPractice` 环境可导入。
- [x] 确认 Fetch URDF 和所有 mesh 路径存在。
- [x] 记录 URDF joint names。
- [x] 记录 ManiSkill Fetch active joint names。
- [x] 验证 joint-name mapping 无缺失、重复或静默 fallback。
- [x] 运行 `RobotSampler.presample()`。
- [x] 固定 seed 后 sampled robot points 可复现。

验收：

- [x] Fetch sampler 可独立运行并输出有限 `[Nr,3]`。
- [x] 不允许使用 Panda URDF 解释 Fetch qpos。

### C2. FK 与坐标系验证

建议新增：

- `bin/validate_fetch_robot_flow.py`
- `logs/validation/fetch_robot_flow/`

任务：

- [x] 保存同一帧的 simulator link pose。
- [x] 保存 URDF/FK link pose。
- [x] 对每个关键 link 计算 translation/rotation error。
- [x] 验证 robot points 位于固定 robot-base frame。
- [x] 检查 base transform 只应用一次。
- [x] 检查 fixed base/body 在全部 timestep 的漂移。
- [x] 检查 EE point motion 与 simulator EE pose 方向一致。
- [x] 检查 open/close gripper 时 finger points 相向/背向移动。

建议指标：

- [x] base translation drift。
- [x] body joint max drift。
- [x] link translation MAE/max。
- [x] link rotation error。
- [x] EE trajectory EPE。
- [x] robot-point trajectory EPE。

验收：

- [x] 指标阈值写入配置，而不是只做肉眼判断。
- [x] 生成 JSON 指标。
- [x] 关键失败时 validation command 返回非零状态。

### C3. Robot overlay 可视化

任务：

- [x] 将 URDF/FK robot points 投影到 `fetch_head`。
- [x] 将 URDF/FK robot points 投影到 `fetch_hand`。
- [x] 与 segmentation robot mask 叠加显示。
- [x] 保存至少 open-gripper 和 close-gripper 两个状态。
- [x] 保存 overlay 图片或 GIF。
- [x] 计算投影点落在 robot mask 内的比例。

验收：

- [x] 两个相机均有可检查的 overlay artifact。
- [x] robot mask overlap 达到预设阈值。

完成证据（2026-07-28）：

- `bin/validate_fetch_robot_flow.py` 在真实 RoboCasa Fetch 环境中通过，失败时返回非零状态。
- `configs/fetch_robot_flow_validation.yaml` 固化所有数值和 overlay 阈值。
- 10,000 个 robot points 在固定 seed 下逐点可复现，shape 为 `[10000,3]` 且全部有限。
- 14 个关键 link 的 translation MAE/max 为 `3.88e-7 / 7.01e-7 m`，rotation mean/max 为 `2.21e-4 / 5.20e-4 rad`。
- base/body drift 都为 `0`；EE relative trajectory EPE 为 `2.32e-7 m`；robot-point EPE mean/max 为 `2.00e-7 / 7.17e-7 m`。
- finger sampled-point separation：initial/open/closed 为 `0.0410 / 0.1091 / 0.0151 m`，方向检查通过。
- `fetch_head` / `fetch_hand` 最低 robot-mask overlap 为 `1.000 / 0.910`。
- simulator/URDF link matrices、JSON summary 和四张 open/closed overlay 保存在 `logs/validation/fetch_robot_flow/`。
- Phase A-C 回归：`78 passed`。

---

## 4. Phase D：Scene point correspondence 与 GT flow

目标：同一批 t=0 scene points 在未来帧中保持物理身份一致。

### D1. Actor-local correspondence

涉及文件：

- `PointWorld/integration/pointworld_wrapper.py`
- `common/utils.py`

任务：

- [x] t=0 scene point 按 segmentation actor/link ID 分类。
- [x] 动态 actor point 转换到 actor-local frame。
- [x] 每个未来 timestep 使用 actor pose 恢复同一物理点。
- [x] fixture/static points 始终保持初始坐标。
- [x] robot points 从 scene points 中可靠移除。
- [x] actor 消失时保留 physical existence，并单独更新 visibility。
- [x] 不允许按每帧重新采样后的点索引直接相减。
- [x] 支持 object translation 和 rotation。
- [x] 支持 object-object collision 后继续追踪原 actor。

验收：

- [x] 静态厨房点最大 flow 接近零。
- [x] 刚体点间距离在物体运动中保持不变。
- [x] 物体 translation 与 actor pose delta 一致。
- [x] 物体 rotation 与 actor rotation matrix 一致。

### D2. Visibility / existence / temporal mask

任务：

- [x] `scene_exists` 表示物理 point 是否存在。
- [x] `scene_visibility` 表示 point 是否被对应相机观测。
- [x] `scene_depth_valid_mask` 表示投影/depth 是否有效。
- [x] `temporal_valid_mask` 只描述 primitive phase padding。
- [x] 四类 mask 不互相替代。
- [x] 多相机 merge 时保留每个相机自己的 visibility。

验收：

- [x] 遮挡物体的 point 可存在但不可见。
- [x] padding timestep 不参与 loss。
- [x] invalid depth 不进入 scene supervision。

### D3. GT scene-flow 可视化和测试

建议新增：

- `bin/validate_robocasa_scene_flow.py`
- `logs/validation/scene_flow/`

任务：

- [x] 保存静态 kitchen 测试。
- [x] 保存单物体平移测试。
- [x] 保存单物体旋转测试。
- [x] 保存 grasp/lift 测试。
- [x] 保存 object-object contact 测试。
- [x] 用颜色或箭头显示 t=0→t=10 flow。
- [x] 输出 static false-motion、dynamic EPE、rigidity error。

验收：

- [x] 每种测试都有 artifact 和 JSON。
- [x] GT flow 与 simulator actor pose 的误差低于配置阈值。

完成证据（2026-07-28）：

- `PointWorld/integration/scene_correspondence.py` 使用稳定 object ID、segmentation ID 和 actor-local 坐标构造固定点身份 trajectory；future RGB-D 不参与点重采样。
- 修复 world-frame actor pose delta 直接作用于 base-frame points 的坐标系错误。
- `voxel_downsample()` 改为 deterministic segmentation-aware 截断，小物体不会再被全局随机采样完全删除。
- WDS 分别保存 `scene_exists`、`scene_visibility`、`scene_depth_valid_mask`、`scene_pose_valid` 和 `temporal_valid_mask`；per-camera mask 保持独立。
- PointWorld supervision 在 loss boundary 才组合 visibility、depth、pose 和 temporal mask；padding timestep 与 invalid depth 不参与 loss。
- 真实 RoboCasa/WDS 验证包含 7,500 scene points，robot scene leakage 为 `0`，mask roundtrip 通过。
- physical-but-occluded / physical-but-invalid-depth point 数量为 `35,859 / 29,919`。
- 五类测试最大 actor-pose EPE 为 `3.73e-7 m`，最大 rigidity error 为 `1.08e-7 m`，静态厨房 false motion 为 `0`。
- `base_camera` / `fetch_head` / `fetch_hand` visibility 分别为 `0.099 / 0.925 / 0.671`，每个相机均保留独立 mask。
- JSON 和五张 flow artifact 保存在 `logs/validation/scene_flow/`。
- Phase A-D 回归：`83 passed`。

---

## 5. Phase E：Uniform-RAPS 固定预算 baseline

目标：在训练 WM 前，建立可复现且公平的 exploration baseline。

### E1. Baseline runner

建议新增：

- `bin/run_robocasa_exploration_baseline.py`
- `configs/robocasa_baseline.yaml`

任务：

- [ ] 支持 `uniform_raps`。
- [ ] 支持 `random_cartesian`。
- [ ] 两个 baseline 使用相同 scene seeds。
- [ ] 两个 baseline 使用相同 low-level transition budget。
- [ ] 两个 baseline 使用相同 reset policy。
- [ ] 两个 baseline 使用相同 camera/data contract。
- [ ] primitive distribution 从配置读取并保存到 metadata。
- [ ] 每个 run 保存完整 config snapshot。
- [ ] 支持多 seed。
- [ ] 支持断点续跑，且不会重复 sample key。

### E2. Exploration metrics

建议新增：

- `exploration/metrics.py`

任务：

- [ ] contact rate。
- [ ] bilateral finger contact rate。
- [ ] grasp rate。
- [ ] moved-object rate。
- [ ] moved distance distribution。
- [ ] no-contact rate。
- [ ] primitive success/failure/timeout rate。
- [ ] 每类 primitive sample count。
- [ ] 每类 primitive duration distribution。
- [ ] 有效 depth/scene point coverage。
- [ ] 每千 low-level transitions 的有效 primitive 数。
- [ ] wall-clock collection throughput。
- [ ] 按 seed、primitive、参数 bucket 输出统计。

验收：

- [ ] 输出 `metrics.json`。
- [ ] 输出便于画图的 `metrics.csv`。
- [ ] budget 统计来自真实 low-level step，不来自 primitive count。
- [ ] Uniform-RAPS 与 random Cartesian 可直接比较。

### E3. Baseline 数据集

任务：

- [ ] 决定正式 bootstrap low-level transition budget。
- [ ] 收集至少 3 个 seeds。
- [ ] 固定 train/test split。
- [ ] 保存 dataset manifest 和 sample count。
- [ ] 检查每类 primitive 最低样本量。
- [ ] 检查 success、failure、no-contact 均有覆盖。
- [ ] 检查 WDS 可完整迭代。
- [ ] 检查无 NaN/Inf。

验收：

- [ ] 形成可引用的 Uniform-RAPS baseline dataset。
- [ ] 形成等预算 random Cartesian dataset。
- [ ] baseline metrics 进入实验报告。

---

## 6. Phase F：PointWorld 单模型 RoboCasa 训练

目标：先证明单个 PointWorld 能学习 primitive-conditioned scene dynamics，再进入 ensemble。

### F1. 配置接入

涉及文件：

- `configs/main.yaml` 或新增 RoboCasa 专用训练配置
- `PointWorld/arguments.py`
- `PointWorld/pointworld/base.py`
- `PointWorld/dataset_components/`

任务：

- [ ] RoboCasa RAPS 训练设置 `raps_action_dim: 20`。
- [ ] `data_dirs` 指向正式 RoboCasa WDS。
- [ ] 使用正确的 domain/canonical direct-flow decoder。
- [ ] 检查 batch 中包含 `raps_action`。
- [ ] 检查 batch 中包含 temporal/scene valid masks。
- [ ] 加入 primitive duration conditioning，或记录第一版暂不使用的决定。
- [ ] 加入 primitive phase conditioning，或记录第一版暂不使用的决定。
- [ ] normalization stats 只从 train split 生成。
- [ ] checkpoint 保存 action contract 和 schema version。

### F2. 小数据 overfit gate

任务：

- [ ] 选择 32 个 primitive samples。
- [ ] 选择 64 个 primitive samples。
- [ ] 选择 128 个 primitive samples。
- [ ] 训练到明显 overfit。
- [ ] 报告 total flow loss。
- [ ] 报告 moved-point EPE。
- [ ] 报告 static false-motion。
- [ ] 报告每类 primitive error。
- [ ] 可视化 prediction 与 GT。

必须完成的消融：

- [ ] robot flow only。
- [ ] RAPS action only。
- [ ] robot flow + RAPS action。
- [ ] shuffled RAPS action。
- [ ] zero robot flow。

验收：

- [ ] 小数据 train error 明显下降。
- [ ] moved-object flow 方向正确。
- [ ] shuffled action 相比正确 action 明显恶化，或有合理解释。
- [ ] zero robot flow 与真实 robot flow 的预测不同。
- [ ] static false-motion 低于预设阈值。

### F3. 单模型 held-out baseline

任务：

- [ ] 在正式 bootstrap train split 训练。
- [ ] 使用固定 held-out test split。
- [ ] 保存 best/last checkpoint。
- [ ] 保存 normalization stats。
- [ ] 保存训练 config 和 dataset manifest。
- [ ] 按 primitive 和 motion bucket 报告误差。
- [ ] 同时报告 GT robot-flow 条件下的 scene error。

验收：

- [ ] 单模型达到进入 ensemble 前预设的 held-out baseline。
- [ ] 可从 checkpoint 重现同一 evaluation。

---

## 7. Phase G：Candidate RAPS action → predicted robot flow

目标：在执行 candidate 之前，仅用当前 robot state 和 RAPS action 生成 T=11
robot trajectory。

### G1. Nominal bridge 校准

涉及文件：

- `exploration/robot_flow_bridge.py`

任务：

- [ ] 输入当前 qpos、EE pose、gripper state 和 20D candidate。
- [ ] 为 10 个 primitive 定义 nominal phase trajectory。
- [ ] gripper primitive 生成 finger motion，而不是全零 robot flow。
- [ ] waypoint 通过 Fetch IK/qpos/FK 转换成完整 robot points。
- [ ] 不允许只对全部 robot points施加同一个 EE translation。
- [ ] duration 预测或 nominal duration 与训练 phase 一致。
- [ ] 所有 candidates 使用当前 observation，不读取未来 qpos。
- [ ] 所有 candidates 不读取 object state。

### G2. Validity filter

建议新增：

- `exploration/candidate_filter.py`

任务：

- [ ] normalized parameter range 检查。
- [ ] predicted qpos joint-limit 检查。
- [ ] predicted EE workspace 检查。
- [ ] NaN/Inf 检查。
- [ ] duration/budget 检查。
- [ ] 可选 self-collision approximation。
- [ ] 不允许使用 object pose 做 candidate feasibility 过滤。
- [ ] 所有过滤原因写入 candidate audit。

### G3. Bridge evaluation

建议新增：

- `bin/evaluate_raps_robot_flow_bridge.py`

指标：

- [ ] duration MAE。
- [ ] qpos MAE。
- [ ] EE trajectory EPE。
- [ ] robot-point trajectory EPE。
- [ ] gripper point EPE。
- [ ] candidate invalid precision/recall，如果有 label。
- [ ] 按 primitive 和参数 bucket 统计。

验收：

- [ ] candidate 执行前可输出 `[K,11,Nr,3]`。
- [ ] 不依赖未来 observation/qpos。
- [ ] 不依赖 object state。
- [ ] held-out robot-flow EPE 低于预设阈值。
- [ ] GT-flow 与 predicted-flow 条件下的 PointWorld error 差距可接受。

### G4. Learned bridge（仅在 nominal bridge 不达标时）

建议新增：

- `exploration/raps_robot_trajectory_model.py`
- `bin/train_raps_robot_trajectory_model.py`

任务：

- [ ] 输入 initial qpos、EE pose、gripper state、20D action。
- [ ] 输出 T=11 qpos/gripper trajectory。
- [ ] 使用 qpos、EE FK、robot-point auxiliary loss。
- [ ] 保存 normalization 和 joint ordering。
- [ ] 与 nominal bridge 使用相同 evaluation。
- [ ] 冻结 bridge 后再用于 ensemble acquisition。

验收：

- [ ] learned bridge 明显优于 nominal bridge，才替换默认实现。

---

## 8. Phase H：PointWorld ensemble 与 uncertainty

目标：训练可校准的 epistemic ensemble，并能批量评价同一状态下的 K 个 candidate。

### H1. Ensemble 训练

建议新增：

- `bin/train_pointworld_ensemble.py`
- `configs/robocasa_ensemble.yaml`

任务：

- [ ] 默认 ensemble size `E=5`。
- [ ] 每个 member 使用不同 initialization seed。
- [ ] 每个 member 使用不同 dataloader order。
- [ ] 每个 member 使用稳定 bootstrap membership。
- [ ] 所有 member 使用相同 validation set。
- [ ] 所有 member 使用相同 normalization contract。
- [ ] checkpoint 保存 member seed。
- [ ] checkpoint 保存 bootstrap manifest。
- [ ] checkpoint 保存 dataset/schema version。
- [ ] 支持 interrupted training resume。

验收：

- [ ] 每个 member 单独达到单模型 held-out baseline。
- [ ] 不同 member 输出不是完全相同。
- [ ] ensemble 可加载并批量推理。

### H2. Candidate batch inference

涉及文件：

- `exploration/pointworld_raps.py`
- PointWorld batch adapter

任务：

- [ ] 同一 observation 的 scene encoder 每个 member 只运行一次。
- [ ] 缓存 `encoded_scene_feat0`。
- [ ] K candidates 沿 batch 维并行。
- [ ] 所有 member 使用相同 candidate action。
- [ ] 所有 member 使用相同 predicted robot flow。
- [ ] 输出 prediction shape `[E,K,T,Ns,3]`。
- [ ] 输出推理耗时和显存统计。

验收：

- [ ] K=32 和 K=128 均可运行。
- [ ] batched inference 与逐 candidate inference 数值一致。

### H3. Disagreement calibration

涉及文件：

- `exploration/pointworld_raps.py`

任务：

- [ ] point-level epistemic variance。
- [ ] robot-path neighborhood weighting。
- [ ] temporal/visibility mask。
- [ ] moved/static-aware score 诊断。
- [ ] top-quantile score 诊断。
- [ ] 分离 epistemic disagreement 与模型 aleatoric `log_var`。
- [ ] 测试 uncertainty 与真实 prediction error 的相关性。
- [ ] 检查 uncertainty 是否只集中在 invalid depth/boundary。
- [ ] 重复加入相同训练数据后检查 variance 是否下降。

验收：

- [ ] uncertainty 与 held-out error 正相关。
- [ ] score 对 mask 和 invalid candidate 稳定。
- [ ] 保存 calibration plot 和 JSON。

---

## 9. Phase I：WM-guided active collection 闭环

目标：形成一个可直接运行、可中断恢复、可审计的 active exploration command。

### I1. Active pipeline 入口

建议新增：

- `bin/run_robocasa_active_pipeline.py`
- `configs/robocasa_active.yaml`
- 在 `main.py` 增加明确 subcommand。

每个 decision：

- [ ] 读取当前 observation。
- [ ] 采样 K 个 20D RAPS candidates。
- [ ] 运行 validity filter。
- [ ] 为 valid candidates 预测 T=11 robot flow。
- [ ] 构建 PointWorld candidate batch。
- [ ] 运行 ensemble inference。
- [ ] 计算 disagreement score。
- [ ] epsilon/temperature 策略选择一个 candidate。
- [ ] 只在真实环境执行被选中的 candidate。
- [ ] 保存 primitive record。
- [ ] 写入 WDS/replay manifest。
- [ ] 更新真实 low-level transition budget。
- [ ] terminated/truncated 后安全 reset。

### I2. Candidate audit

涉及文件：

- `exploration/round_loop.py`

每个 decision 保存：

- [ ] round/episode/decision ID。
- [ ] 全部 candidate actions。
- [ ] decoded primitive 和 physical parameters。
- [ ] candidate validity mask。
- [ ] 每个 invalid reason。
- [ ] candidate scores。
- [ ] selection probabilities。
- [ ] selected index。
- [ ] 是否 epsilon-random。
- [ ] predicted robot-flow summary。
- [ ] selected primitive duration。
- [ ] selected primitive outcome。
- [ ] inference/selection/collection wall time。

验收：

- [ ] JSONL audit 可完整重建每次选择逻辑。
- [ ] audit 不包含未执行 candidate 的未来 simulator outcome。

### I3. Round-based retraining

任务：

- [ ] bootstrap round 使用 Uniform-RAPS。
- [ ] bootstrap 后训练/freeze ensemble。
- [ ] active round 按固定 low-level budget 收集。
- [ ] round 结束后更新 replay manifest。
- [ ] 每个 member warm-start 固定 optimizer steps。
- [ ] 训练期间 collector 不改变 ensemble。
- [ ] 下一 round 原子切换到新 checkpoint。
- [ ] 保存 round state 以支持 resume。

验收：

- [ ] 中断后不会重复消费 budget。
- [ ] 中断后不会覆盖已有 shard/checkpoint。
- [ ] 每个 round 可复现 candidate audit 和 dataset manifest。

### I4. Replay 与数据预算

任务：

- [ ] 统一 bootstrap、active 和 baseline 的 low-level transition budget。
- [ ] 明确 replay capacity 单位是 primitive sample 还是 low-level transition。
- [ ] replay sampling 不读取 test split。
- [ ] 保持固定 validation set。
- [ ] 保存每轮新增/保留/淘汰 sample key。
- [ ] 防止高 duration primitive 隐式获得更多预算。

验收：

- [ ] 所有 explorer 的真实 transition budget完全一致。
- [ ] 数据集统计和 audit 中的 budget 可以互相校验。

---

## 10. Phase J：最终实验、可视化与报告

### J1. 必做实验

- [ ] Random Cartesian baseline。
- [ ] Uniform-RAPS baseline。
- [ ] WM-guided RAPS。
- [ ] 每种方法至少 3 个 seeds。
- [ ] 每种方法相同 low-level transition budget。
- [ ] 每种方法相同初始 bootstrap budget。
- [ ] 每种方法相同 train/test split。

### J2. Exploration 指标

- [ ] contact rate。
- [ ] grasp rate。
- [ ] object-motion rate。
- [ ] moved distance。
- [ ] primitive success/failure/timeout。
- [ ] sample diversity。
- [ ] 每千 low-level steps 的有效 interaction。
- [ ] collection wall-clock cost。
- [ ] model inference/training cost。

### J3. World-model 指标

- [ ] overall scene-flow EPE。
- [ ] moved-object EPE。
- [ ] static false-motion。
- [ ] per-primitive EPE。
- [ ] GT robot-flow conditioned EPE。
- [ ] predicted robot-flow conditioned EPE。
- [ ] uncertainty/error correlation。
- [ ] calibration curve。

### J4. 消融实验

- [ ] robot flow only。
- [ ] RAPS action only。
- [ ] robot flow + RAPS action。
- [ ] shuffled RAPS action。
- [ ] nominal bridge vs learned bridge。
- [ ] unweighted disagreement vs robot-neighborhood weighted disagreement。
- [ ] greedy vs epsilon/temperature selection。

### J5. 可视化产物

- [ ] 当前 kitchen render。
- [ ] 每类 primitive rollout GIF。
- [ ] robot FK/segmentation overlay。
- [ ] GT scene-flow visualization。
- [ ] PointWorld prediction vs GT。
- [ ] candidate robot-flow comparison。
- [ ] uncertainty heatmap。
- [ ] active round metric curves。

### J6. 最终复现包

- [ ] 固定环境和依赖版本。
- [ ] 保存全部配置 snapshot。
- [ ] 保存 dataset manifests。
- [ ] 保存 normalization stats。
- [ ] 保存 checkpoints 和 member seeds。
- [ ] 保存运行命令。
- [ ] 保存所有 metrics JSON/CSV。
- [ ] 从空输出目录执行一次 end-to-end smoke run。
- [ ] 编写 README：bootstrap、train、ensemble、active、eval 五个入口。

---

## 11. 推荐实施顺序与阶段出口

严格依赖顺序：

```text
Phase A  primitive/wrapper correctness
   ↓
Phase B  dense record + WDS contract
   ↓
Phase C  Fetch robot-flow GT
   ↓
Phase D  scene-flow GT
   ↓
Phase E  fixed-budget Uniform-RAPS baseline
   ↓
Phase F  PointWorld single-model gate
   ↓
Phase G  candidate robot-flow bridge
   ↓
Phase H  ensemble + calibrated disagreement
   ↓
Phase I  active collection loop
   ↓
Phase J  final experiments
```

阶段出口：

- [x] A 完成后：10 个 primitive 和 wrapper budget 语义可信。
- [x] B 完成后：每个 primitive 都能生成无损、可追溯的 T=11 WDS sample。
- [ ] C 完成后：Fetch robot flow 可作为训练 GT。
- [ ] D 完成后：scene flow 可作为训练 GT。
- [ ] E 完成后：有公平的 exploration baseline。
- [ ] F 完成后：单模型证明能够学习 RoboCasa dynamics。
- [ ] G 完成后：candidate 执行前可获得可信 robot flow。
- [ ] H 完成后：uncertainty 可用于 candidate ranking。
- [ ] I 完成后：WM-guided active collection 真正闭环。
- [ ] J 完成后：结果可复现、可比较、可报告。

## 12. 下一批建议立即执行的任务

这是当前最小且顺序正确的一批工作：

- [x] 新建 schema 和 primitive 单元测试。
- [x] 跑 10 类 primitive simulator smoke test。
- [x] 检查并修正外层 `TimeLimit(200)`。
- [x] 为 wrapper 增加 duration/termination/budget 测试。
- [x] 扩展 `PrimitiveRecord` 的 qvel、gripper、termination reason。
- [x] 增加 WDS round-trip 测试。
- [ ] 运行 Fetch URDF/FK 独立验证。
- [ ] 生成第一张 robot-points/segmentation overlay。

完成以上八项后，再开始正式收集 Uniform-RAPS baseline，避免用尚未验证的
geometry 和 recorder 生成大规模数据。
