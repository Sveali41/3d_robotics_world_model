# RoboCasa Primitive Exploration × PointWorld 实施计划

## 1. 当前研究范围

当前阶段只解决一个问题：

> 在固定底盘、连续控制的 RoboCasa 抓取环境中，如何稳定地产生足够多且足够多样的物体交互轨迹，用于训练 PointWorld？

现阶段暂不研究：

- mini-environment 与 target environment 的对比；
- 根据 WM 自动生成厨房或物体配置；
- environment curriculum；
- 跨环境难度递进；
- 完整长任务 policy。

实验期间使用固定的 RoboCasa scene 和 object reset distribution。物体仍由 YAML 配置，但环境不会根据 WM uncertainty 自动改变。

本阶段只比较 exploration method：

```text
IID random low-level action
vs. correlated random Cartesian action
vs. uniform parameterized primitives
vs. contact/motion-guided primitives
vs. world-model-guided primitives
```

相关工作：

- [RAPS](https://mihdalal.github.io/raps/)
- [PointWorld](https://point-world.github.io/)
- [MAPLE](https://ut-austin-rpl.github.io/maple/)
- [CEE-US](https://cee-us.github.io/)

---

## 2. 核心假设

### 2.1 原始随机动作很难产生交互

独立随机采样关节或末端动作会产生大量机械臂空中抖动、错误时机的夹爪开合和完全静止的物体轨迹。即使 transition 数量很大，真正包含 object dynamics 的比例也可能很低。

### 2.2 Primitive 提高动作可行性

RAPS 风格高层 action 定义为：

\[
z=(k,\theta)
\]

其中：

- \(k\)：`MOVE_DELTA`、`PUSH_XY`、`TOP_GRASP`、`LIFT` 等 primitive；
- \(\theta\)：相对 EE 的偏移、方向、距离和夹爪幅度等连续参数。

正式 action 不包含 object ID，也不读取 simulator object pose。点云候选生成器可以
产生空间参数，但 primitive controller 只依赖机器人状态和 \(\theta\)。object GT 只用于
事后计算 contact、grasp 和 motion metrics。

primitive 内部仍然执行并记录真实的低层 Cartesian action，因此 action abstraction 不会丢失 PointWorld 所需的稠密轨迹。

### 2.3 WM 可以选择信息量更高的 primitive

均匀 primitive sampler 能提高接触率，但会重复已经学会的转移。训练出初始 WM 后，可以根据 uncertainty、contact、predicted motion、coverage 和 redundancy 选择下一次 primitive。

---

## 3. 系统结构

```text
固定 RoboCasa scene + 固定 object distribution
                    ↓
Exploration Policy / Primitive Sampler
选择 primitive k、参数 θ
                    ↓
Primitive Controller
展开为 H 个 [dx, dy, dz, gripper]
                    ↓
RoboCasa / ManiSkill
逐 timestep 执行并模拟接触
                    ↓
Dense Trajectory Recorder
保存视觉、qpos、EE、物体、接触和低层 action
                    ↓
Robot Adapter + URDF/FK
生成 robot point trajectory
                    ↓
Scene Flow Builder
生成带 correspondence 的 scene point trajectory
                    ↓
PointWorld Training
                    ↓
Uncertainty / Error / Coverage
                    ↓
选择下一批 primitive
```

本阶段没有 environment generator。

---

## 4. 当前代码状态

### 已完成

- [x] RoboCasa scene 和物体通过 `configs/kitchen_custom.yaml` 配置；
- [x] 支持物体 `group`、fixture 和局部 `pos`；
- [x] Fetch 底盘固定；
- [x] base action 已从公开 action space 移除；
- [x] 控制方式切换为 `pd_ee_delta_pos`；
- [x] action 从 11 维降到 4 维 `[dx,dy,dz,gripper]`；
- [x] PointWorld wrapper 保留 Panda/DROID adapter；
- [x] 新增 Fetch adapter；
- [x] 数据收集器按 active-joint name 映射 qpos；
- [x] Panda 9 维和 Fetch 15 维 qpos mapping 已验证；
- [x] 配置支持 `robot_type: droid_panda / fetch`。
- [x] 实现 20D official-RAPS-compatible high-level action schema；
- [x] 实现 10 个无 object ID 的 agent-centric primitives；
- [x] Uniform sampler 只随机 primitive 和连续参数；
- [x] TOP_GRASP / PUSH_XY 参数化物理验证通过；

### 仍需解决

- [x] 将 7 维 action 降成 4 维 `[dx,dy,dz,gripper]`；
- [x] 建立第一版 object/contact outcome context；
- [x] 实现 primitive 状态机；
- [ ] 保存 primitive 内全部低层 timestep；
- [ ] 实现接受 20D action 的高层 `RAPSEnvWrapper`；
- [ ] 实现 RAPS action 到候选 robot-flow 的预测桥梁；
- [ ] 完成 Fetch URDF/FK 几何验证；
- [ ] 完成 scene point correspondence；
- [ ] 建立 exploration metrics；
- [x] 完成可视化 Uniform-RAPS sampler；
- [ ] 完成固定数据预算的 Uniform-RAPS dataset baseline；
- [ ] 最后加入 WM-guided primitive selection。

`configs/kitchen_custom.yaml` 中的 `mini_env` 只是当前代码遗留名称，本计划不把它当成研究中的 mini-environment。

---

## 5. 计划代码结构

```text
robocasa_exploration/
├── __init__.py
├── primitives.py                 # 已实现：schema/context/primitives/uniform sampler
├── primitive_wrapper.py
├── trajectory_recorder.py
├── flow_builder.py
├── robot_trajectory_model.py
├── candidate_generator.py
├── pointworld_ensemble.py
├── exploration_metrics.py
├── active_selector.py
└── replay.py

collect_robocasa_exploration.py
train_pointworld_ensemble.py
run_pointworld_raps_exploration.py
evaluate_robocasa_exploration.py
configs/robocasa_exploration.yaml
```

职责：

- `run_robocasa_kitchen.py`：环境定义和人工可视化；
- `primitives.py`：20D schema、参数范围、primitive 状态机和 Uniform sampler；
- `primitive_wrapper.py`：高层 step 和多步低层执行；
- `trajectory_recorder.py`：保存每个 timestep；
- `flow_builder.py`：生成 PointWorld robot/scene trajectories；
- `robot_trajectory_model.py`：执行前预测候选 primitive 的 robot trajectory；
- `candidate_generator.py`：uniform / point-cloud proposal；
- `pointworld_ensemble.py`：模型加载、共享 scene encoding 和 batched disagreement；
- `active_selector.py`：WM-guided candidate ranking；
- `exploration_metrics.py`：接触率、物体运动率、覆盖率和重复度。

primitive 不写入 `domain/ManiSkill` vendor controller。

---

## 6. Phase 0：固定环境与 baseline

### 固定实验条件

- kitchen layout 和 style；
- robot spawn、camera 和 control frequency；
- object groups 和 placement distribution；
- physics 参数、episode length 和 seeds。

可以先用一个物体调试 primitive，再使用多个物体测试 explorer，但这不是 mini/target curriculum。

### Baseline A：IID random

使用当前 `env.action_space.sample()`，记录：

- 总 transition 数；
- robot-object contact transitions；
- object moved transitions；
- 最大物体位移；
- effective-interaction episode 比例。

### Baseline B：Correlated random Cartesian

避免每个 timestep 完全独立：

\[
a_t=\rho a_{t-1}+(1-\rho)\epsilon_t
\]

也可以让一个随机 action 持续 3–10 step。这是比 IID random 更强、更公平的低层 baseline。

### 验收

- [ ] 同 seed reset 可复现；
- [ ] headless 环境运行至少 500 step；
- [ ] baseline metrics 保存为 JSON/CSV；
- [ ] contact 和 motion 检测经过可视化验证；
- [ ] 所有 explorer 使用相同真实 transition budget。

---

## 7. Phase 1：4D Cartesian action

当前 Fetch mapping：

```text
arm:     [0:3] -> dx, dy, dz
gripper: [3:4]
body:    [4:7]
base:    [7:9]
```

fixed-base wrapper 已隐藏 base，下一步同时隐藏 body。

### 实现

- [x] 动态读取 `arm/gripper/body/base` action slices；
- [x] public action 只保留 arm 和 gripper；
- [x] body/base 自动填零；
- [x] 保存并维持 body/base 初始状态；
- [x] 越界 action clip；
- [x] 暴露 public-to-full mapping 供调试。

### Contract

```text
action[0] = normalized dx
action[1] = normalized dy
action[2] = normalized dz
action[3] = gripper
```

### 验收

- [x] `env.action_space.shape == (4,)`；
- [x] dx/dy/dz 分别产生正确 controller target；
- [x] gripper 独立开合；
- [x] body/base 运行 500 step 后保持固定；
- [x] 4D random baseline 可稳定运行。

---

## 8. Phase 2：PrimitiveContext

Primitive 不直接散落访问 RoboCasa 内部字段。建立：

```python
PrimitiveContext:
    ee_position
    ee_orientation
    gripper_open
    object_ids
    object_groups
    object_poses
    object_sizes
    robot_object_contacts
    object_object_contacts
    fixture_contacts
```

### Object registry

每次 reset 建立：

```text
object_index
object_name
object_group
actor handle
segmentation ID
initial pose
```

### Contact detector

至少区分：

```text
left finger ↔ object
right finger ↔ object
gripper/arm ↔ object
object ↔ fixture
object ↔ object
```

### Motion detector

\[
d_t^o=\lVert p_t^o-p_{t-1}^o\rVert
\]

同时记录 rotation change；阈值必须高于静止 jitter。

### 验收

- [ ] 打印所有 object keys、groups 和 poses；
- [x] PUSH 时 finger-object contact 正确触发；
- [ ] 静止物体不被频繁误判为 moved；
- [ ] object-object contact 可独立统计；
- [x] context 不依赖 apple/banana 等具体名称。

### 当前实现状态

`robocasa_exploration/primitives.py` 已提供第一版 `PrimitiveContext`，统一读取：

- EE position（arm controller root frame）；
- gripper opening fraction；
- 全部 object ID、actor handle、root/world position；
- left/right finger-object contact force；
- `is_grasping(object_id)`。

object group/size、orientation、object-object/fixture contact 和通用逐帧 motion
detector 仍属于 recorder 集成阶段，尚未宣称完成。

---

## 9. Phase 3：Primitive 接口

统一高层 action：

```python
PrimitiveAction(
    primitive_type: str,
    parameters: dict,
)
```

当前固定高层向量遵循原始 RAPS 的 hybrid action contract：

```text
10D primitive one-hot + 10D normalized argument bank = 20D
```

每个 primitive 在 20D bank 中拥有固定参数 slice；未被选择 primitive 的参数槽为零。

每个 primitive 实现：

```python
class Primitive:
    def reset(self, context, parameters): ...
    def next_action(self, context) -> np.ndarray: ...
    def is_done(self, context) -> bool: ...
    def status(self) -> str: ...
```

状态包括：

```text
running
success
timeout
unreachable
collision_abort
lost_object
```

每个 primitive 必须有最大 step 和明确 termination。

### 当前实现状态

- [x] `PrimitiveAction` 与统一 `Primitive` 基类；
- [x] 20D action encode/decode round-trip；
- [x] action 和执行逻辑不包含 object ID / simulator object pose；
- [x] 每个 primitive 每次只输出一个 4D simulator action；
- [x] temporal state、最大步数、success/failed/timeout；
- [x] 每次结束记录 duration、stage、物体位移、最大双指接触力和抓持事件；
- [x] `UniformPrimitiveExplorer` 按 YAML 权重采样并维护 primitive state；
- [ ] 将 primitive metadata 接入正式 PointWorld WDS recorder。

---

## 10. Phase 4：Agent-centric primitive library

当前库刻意保留一定冗余，使随机高层探索比 raw Cartesian 更容易形成有意义的运动：

```text
MOVE_DELTA_EE_POSE(dx, dy, dz)
TOP_GRASP(z_down)
LIFT(z_dist)
DROP(z_dist)
MOVE_LEFT(y_dist)
MOVE_RIGHT(y_dist)
MOVE_FORWARD(x_dist)
MOVE_BACKWARD(x_dist)
OPEN_GRIPPER()
CLOSE_GRIPPER()
```

所有位移均相对 primitive 启动时的 EE pose，并位于 arm-controller root frame。
`TOP_GRASP` 的下降 waypoint 只由这些参数计算；执行期间不追踪任何物体。
`MOVE_DELTA_EE_POSE` 是保留表达能力的 raw primitive。

### 当前实现状态

- [x] 10 个 official-RAPS-compatible primitives；
- [x] 固定参数范围与 normalized encoding；
- [x] TOP_GRASP 参数化测试产生 contact 和 grasp event；
- [ ] directional move 参数化测试产生可控的平移；
- [x] object pose/contact 只进入 outcome logger；
- [ ] 正式数据集上的 primitive 成功率和 moved/contact coverage benchmark。

### 验收

- [ ] primitive 可单独启动、成功、失败或 timeout；
- [ ] directional-motion rate 高于 random action；
- [ ] GRASP 同时产生成功与失败；
- [ ] 参数和 termination reason 完整记录；
- [ ] primitive 不依赖完整任务 reward。

---

## 11. Phase 5：Primitive wrapper

一次高层 step 可以执行多个低层 step：

```python
obs, reward, terminated, truncated, info = wrapped_env.step(
    PrimitiveAction("top_grasp", parameters={"z_down": 0.20})
)
```

内部流程：

```text
初始化 primitive
→ 读取 context
→ 生成 4D action
→ env.step
→ recorder 保存 transition
→ 更新 context
→ 检查 termination
→ 重复
```

高层 `info` 保存：

```text
primitive_type
parameters
duration_steps
status
contact_count
object_translation
object_rotation
low_level_return
```

### 验收

- [ ] duration 等于内部 `env.step` 数；
- [ ] terminated/truncated 立即停止 primitive；
- [ ] primitive 之间无残留状态；
- [ ] wrapper 可选择性挂载 recorder；
- [ ] 相同 seed 和参数可复现近似轨迹。

---

## 12. Phase 6：Dense trajectory recorder

```text
PrimitiveRecord
├── episode_id / primitive_id
├── primitive_type / parameters / status
├── low_level_actions          [H, 4]
├── full_controller_actions    [H, controller_dim]
├── observations               [H+1, ...]
├── qpos / qvel                [H+1, Dq]
├── ee_pose                    [H+1, 7]
├── object_poses               [H+1, No, 7]
├── contacts                   [H, ...]
├── robot_point_trajectory     [H+1, Nr, 3]
└── scene_point_trajectory     [H+1, Ns, 3]
```

PointWorld 当前需要 T=11。长 primitive 使用滑动窗口，例如 `0..10, 5..15, 10..20`。短轨迹 padding 时必须保存 temporal valid mask。

### 验收

- [ ] `len(observations) == len(actions) + 1`；
- [ ] 每个 window 可追溯到 primitive；
- [ ] padding 有 mask；
- [ ] 中断 episode 不保存损坏 sample；
- [ ] WebDataset pipeline 可读取。

---

## 13. Phase 7：Robot point flow

### 已完成

- [x] Panda 与 Fetch adapters；
- [x] qpos joint-name mapping；
- [x] gripper normalization；
- [x] collection 透传 `robot_type` 和 joint names。

### 待完成

- [ ] 补齐 `urdfpy/addict/pytorch_kinematics`；
- [ ] Fetch URDF 运行 `RobotSampler.presample()`；
- [ ] 生成 `[T,Nr,3]` robot trajectory；
- [ ] 验证 base/body 静止；
- [ ] 验证 finger closing flow；
- [ ] robot points overlay 到相机 segmentation；
- [ ] 保存 robot type、URDF 和 joint ordering metadata。

### 几何验收

- [ ] 静止机器人 flow 接近零；
- [ ] EE point motion 与 EE pose 一致；
- [ ] mesh overlay 与 robot mask 重合；
- [ ] 不允许 Panda URDF 解释 Fetch qpos；
- [ ] fixed-base transform 不重复应用。

---

## 14. Phase 8：Scene flow ground truth

不能按索引相减每帧重新采样的点云。应当：

1. 在 t=0 获取 scene points；
2. 用 segmentation 确定所属 actor/link；
3. 转换到 actor local frame；
4. 后续通过 actor pose 变换同一个 local point；
5. 静态 fixture points 保持不动；
6. 使用 exists/visibility mask；
7. robot 与 scene points 分离；
8. 坐标统一到固定 robot-base frame。

### 验收

- [ ] 静态台面 flow 接近零；
- [ ] 物体平移 flow 一致；
- [ ] 物体旋转符合刚体变换；
- [ ] object-object collision 后 correspondence 正确；
- [ ] GT flow 可视化与仿真一致。

---

## 15. Phase 9：Uniform primitive exploration

WM 未训练时使用固定 primitive 分布：

```text
PUSH_XY        30%
TOP_GRASP      25%
MOVE_DELTA     15%
LIFT / DROP    20%
OPEN / CLOSE    9%
RETREAT          5%
```

参数范围草案：

```yaml
primitive_sampler:
  top_grasp:
    z_down: [0.0, 0.25]
  lift:
    z_dist: [0.0, 0.16]
```

目标数据构成：

```text
50–60% 接触且物体运动
20–30% 接触但失败
10–20% 无接触
```

主动参数扰动产生抓偏、擦过、提前/延迟闭合、短距离 push、斜向接触和滑落等边界数据。

### 验收

- [ ] contact rate 高于 4D random；
- [ ] object-motion rate 高于 4D random；
- [ ] 每个 primitive 有最低样本量；
- [ ] 同时包含 success、failure 和 no-contact；
- [ ] PointWorld 可在小数据上 overfit。

---

## 16. Exploration metrics

### 效率

- contact transitions / total transitions；
- moved-object transitions / total transitions；
- effective-interaction episodes / total episodes；
- 首次有效交互所需 step；
- 最大/平均物体位移；
- grasp/lift success rate。

### 覆盖率

- object × primitive coverage（仅离线评估，不参与候选动作生成或打分）；
- push direction/distance bins；
- grasp-offset bins；
- contact-point coverage；
- object state-change coverage。

### 重复度

轨迹 summary 包含 object group、primitive、parameter bins、初始 EE-object pose、contact point、object displacement 和 status。使用 bucket count、kNN distance 或 embedding distance 衡量重复。

### WM usefulness

- full-scene EPE；
- moved-object EPE；
- contact-neighborhood EPE；
- static false-motion；
- multi-step rollout error；
- 达到固定误差所需 transition 数。

---

## 17. Phase 10：Heuristic-guided RAPS

在 WM selector 前建立强 heuristic baseline：

\[
S_h=w_cC_{contact}+w_mM_{object}+w_rR_{rarity}-w_dD_{redundancy}-w_iC_{invalid}
\]

流程：采样 K 个 candidates，过滤不可达参数，优先低计数 bucket，对长期无接触参数降权，并保留 20% uniform primitive。

这一步用于区分提升来自简单 coverage heuristic，还是确实需要 WM uncertainty。

---

## 18. Phase 11：PointWorld uncertainty

先用 Uniform-RAPS 数据训练初始 PointWorld。推荐训练 3–5 个不同 seed/bootstrap data 的 ensemble。PointWorld 不使用 environment reward 或 intrinsic reward，只使用 scene-flow prediction loss：

\[
\mathcal L_{PW}=\ell(\hat{\Delta P}_{1:T},\Delta P^{GT}_{1:T})
\]

ensemble disagreement 是 explorer 的 acquisition score，不是 PointWorld 的训练目标：

\[
U(P,z)=\operatorname{Var}_m[\hat{\Delta P}^{(m)}(P,z)]
\]

uncertainty 重点计算在预测 robot trajectory 邻域、predicted moving points 和 contact neighborhood，避免静态背景稀释结果。正式选择过程不使用 object ID 或 GT object mask。

在固定 validation trajectories 上比较 predicted uncertainty 与 actual flow error。

### 验收

- [ ] disagreement 与真实 error 正相关；
- [ ] uncertainty 不只集中在无效深度和边界；
- [ ] 可按 primitive、参数 bucket 和无语义空间区域输出 uncertainty；
- [ ] 所有 ensemble model 接受同一 candidate；
- [ ] 记录 uncertainty 计算成本。

---

## 19. Phase 12：WM-guided primitive selection

固定 observation，采样：

\[
z_i=(k_i,\theta_i)
\]

每个 candidate 经历：

```text
primitive parameters
→ RAPSRobotTrajectoryModel / nominal planner
→ predicted qpos trajectory
→ Fetch FK robot point trajectory
→ PointWorld ensemble prediction
→ information score
```

主方法第一版只使用 epistemic disagreement：

\[
S(z)=U_{ensemble}(P,z)-\lambda_iC_{invalid}(z)
\]

contact、motion、rarity 和 coverage 分数只作为单独 heuristic baseline / ablation，避免把人工交互目标混入“纯 PointWorld uncertainty”主方法。

第一版不训练 RL policy，采用 candidate sampling + ranking：

```text
采样 K 个候选
→ 过滤
→ WM 评分
→ top-1 或 softmax 采样
→ 执行并记录真实结果
→ 更新 WM 和 coverage statistics
```

保留 `80% WM-guided + 20% uniform`，避免过早锁定。

---

## 20. Phase 13：闭环探索

```text
Uniform-RAPS bootstrap D0
→ train robot-trajectory model G0
→ train PointWorld ensemble M0 with prediction loss
→ generate primitive candidates
→ G0 predicts candidate robot flows
→ rank by PointWorld disagreement
→ collect B1
→ D1 = replay(D0) ∪ B1
→ update G1 and M1
→ repeat
```

每轮保存 model、dataset manifest、全部 candidates、scores、选择结果、真实 contact/motion、prediction error、calibration 和 exploration metrics。

Replay 按 moved-object、contact-only、no-contact、rare primitive、rare parameter 和 rare outcome 分层。第一版使用 50% 当前轮 + 50% historical replay。

整个闭环中 `environment_reward=0`；collector 不按 success 筛选或拒绝轨迹。PointWorld 用监督 flow loss，selector 用 disagreement score，二者严格分离。

---

## 21. Baselines 与消融

| 方法 | Action space | 选择方式 |
|---|---|---|
| IID Random Joint | joint delta | 独立随机 |
| Correlated Random Cartesian | 4D EE | temporal noise |
| Uniform RAPS | primitives | 固定分布 |
| Heuristic RAPS | primitives | contact/motion/coverage |
| PW-RAPS | primitives | PointWorld ensemble disagreement |
| PW-RAPS + coverage | primitives | disagreement + coverage ablation |

消融：

- 删除 uncertainty；分别加入 contact/motion/rarity/redundancy heuristic；
- 删除 uniform exploration floor；
- 只保留成功或删除 no-contact；
- primitive metadata 是否输入 WM；
- robot point flow vs 4D Cartesian action；
- ensemble uncertainty vs single model；
- 一个长 primitive vs 多个短 primitive。

公平性：相同 scene distribution、真实 transition budget、WM architecture、optimizer steps、validation set、replay capacity、camera 和 flow pipeline。额外 planning 计算成本单独报告。

---

## 22. 配置草案

```yaml
environment:
  env_id: RoboCasaKitchenCustom-v1
  kitchen_config: configs/kitchen_custom.yaml
  robot_type: fetch
  control_mode: pd_ee_delta_pos
  fixed_base: true
  fixed_body: true

pointworld:
  domain: maniskill_robocasa
  robot_type: fetch
  robot_urdf_path: domain/ManiSkill/mani_skill/assets/robots/fetch/fetch.urdf
  sequence_length: 11
  sequence_stride: 5
  ensemble_size: 3

exploration:
  method: uniform_primitives
  transition_budget: 10000
  random_floor: 0.20
  candidates_per_step: 32

primitives:
  enabled: [move_delta_ee_pose, top_grasp, lift, drop, move_left, move_right, move_forward, move_backward, open_gripper, close_gripper]
  probabilities:
    move_delta_ee_pose: 0.15
    top_grasp: 0.10
    lift: 0.12
    drop: 0.08
    move_left: 0.10
    move_right: 0.10
    move_forward: 0.10
    move_backward: 0.10
    open_gripper: 0.05
    close_gripper: 0.10

active_score:
  uncertainty: 1.0
  # 以下启发式项默认关闭，只在单独的 ablation 中启用
  contact: 0.0
  predicted_motion: 0.0
  rarity: 0.0
  redundancy: 0.0
  invalid: 2.0
```

---

## 23. MVP

MVP 只实现：

```text
固定 RoboCasa scene
固定 Fetch
4D action
object/contact outcome context
agent-centric PUSH_XY + TOP_GRASP
均匀 RAPS 参数
dense recorder
Fetch robot flow
scene GT flow
PointWorld T=11 windows
Random Cartesian vs Uniform parameterized RAPS
```

成功标准：

- [ ] RAPS contact rate 高于 random Cartesian；
- [ ] RAPS object-motion rate 高于 random Cartesian；
- [ ] primitive 内完整轨迹可回放；
- [ ] robot/scene flow 对齐；
- [ ] PointWorld 可在小数据 overfit；
- [ ] 等量数据下 Uniform RAPS 的 moved-object validation error 更低。

只有 Uniform-RAPS 数据、robot-flow bridge 和 PointWorld overfit 都成功后，才加入 ensemble selector。

---

## 24. 严格实施顺序

1. 固定 baseline 与 metrics；
2. action 降为 4 维；
3. 实现 PrimitiveContext；
4. 实现固定 hybrid RAPS action schema；
5. 实现 agent-centric primitive library；
6. 实现 wrapper 和 recorder；
7. 验证 Fetch robot flow；
8. 验证 scene GT flow；
9. 比较 Random Cartesian 与 Uniform RAPS；
10. 加入 point-cloud candidate parameter generator；
11. 建立 Uniform-RAPS；
12. 训练 PointWorld ensemble；
13. 验证 uncertainty calibration；
14. 实现 Heuristic RAPS；
15. 实现 WM-RAPS；
16. 运行完整消融。

停止条件：

- 4D action 不稳定时不做 primitive；
- contact detector 未验证时不报告探索效率；
- recorder 丢 timestep 时不生成 PointWorld 数据；
- flow 未对齐时不训练正式 WM；
- Uniform-RAPS 未优于 random 时不加入 WM selector；
- uncertainty 与真实 error 不相关时不用于主动选择。

---

## 25. 最近三个开发任务

### Task 1：4D action

隐藏 `body`，只暴露 `[dx,dy,dz,gripper]`。

### Task 2：PrimitiveContext

统一获取 EE pose、object poses/IDs、finger-object contacts 和 object motion。

状态：第一版 runtime context 已完成；通用 orientation/motion/contact registry 留给
recorder 集成。

### Task 3：Agent-centric RAPS MVP

先证明 parameterized RAPS 在相同 transition budget 下，比随机 Cartesian exploration 产生更多接触、更多物体运动，以及更有用的 PointWorld 数据。

状态：20D hybrid action、10 个无 object ID primitive 与可视化 explorer 已实现；
TOP_GRASP 已验证 contact/grasp，PUSH_XY 已验证接触和约 3.9 cm 物体位移。
与 random Cartesian 的等预算数据集实验尚未进行。

---

## 26. 当前阶段预期贡献

当前不声称 environment generation 或 curriculum 的贡献。研究问题是：

> Can parameterized robot action primitives and world-model uncertainty produce more interaction-rich and informative data than low-level random exploration for learning 3D manipulation dynamics?

两个逐步贡献：

1. **RAPS-style feasible exploration for 3D world-model data collection**；
2. **PointWorld-guided primitive acquisition**。

先完整解决探索问题，再决定是否重新加入环境配置生成。

---

## 27. 下一阶段总目标：Reward-free PointWorld–RAPS exploration

> 第 27–39 节是从当前代码状态继续开发的权威规格。前面的章节保留设计演化、
> 已完成工作和早期实验计划；若表述冲突，以第 27–39 节为准。

下一阶段只做 reward-free dynamics exploration：

```text
RAPS                   解决“机器人怎样执行可行的结构化动作”
PointWorld             解决“场景在该机器人动作下怎样变化”
Ensemble disagreement  解决“当前最值得收集哪个动作的数据”
```

严格区分三种信号：

| 信号 | 使用者 | 来源 | 是否训练 PointWorld |
|---|---|---|---|
| World-model loss | PointWorld | predicted flow vs GT flow | 是 |
| Acquisition score | Explorer / selector | PointWorld ensemble disagreement | 否 |
| Task reward | 下游 task policy | 任务成功条件 | 探索阶段不存在 |

探索阶段统一设定：

```text
environment task reward = 0
success filtering       = false
PointWorld reward input = none
selector signal         = epistemic disagreement
```

环境中的 `success`、object ID、contact、grasp 和 object motion 只用于离线评价与分层统计，不进入正式 action generation，也不作为 PointWorld label 或主方法 reward。

### 27.1 第一版不是完整 actor-based Plan2Explore

PointWorld 当前是固定 T=11 的 scene-flow predictor，不是 Dreamer RSSM；它没有 recurrent latent imagination、actor 和 value function。因此第一版采用：

```text
candidate sampling
→ PointWorld ensemble one-primitive prediction
→ disagreement ranking
→ receding-horizon execution
```

该方法准确命名为：

> PointWorld ensemble disagreement-guided RAPS exploration（PW-RAPS）

它实现 Plan2Explore 的核心 acquisition principle，但不声称逐行复现原版 Plan2Explore。多步 imagined actor 是第二阶段扩展，见第 36 节。

---

## 28. 权威系统数据流

```text
Current RoboCasa observation P_t + robot state q_t
                         ↓
CandidateGenerator.sample(K)
生成 K 个 20D [one-hot primitive, normalized parameters]
                         ↓
ValidityFilter
只检查 robot workspace / parameter / self-collision feasibility
不读取 object ID 或 object pose
                         ↓
RAPSRobotTrajectoryModel G(q_t, a_i)
预测每个 candidate 的 T=11 qpos / EE trajectory
                         ↓
FetchFlowAdapter + URDF/FK
生成 candidate robot_flows [K,T,Nr,3]
                         ↓
PointWorld ensemble {M_1...M_E}
预测 scene relative flow [E,K,T,Ns,3]
                         ↓
DisagreementScorer
在 robot-path neighborhood 计算 epistemic uncertainty
                         ↓
Selector
epsilon-uniform / softmax / top-1 选择 a*
                         ↓
RAPSEnvWrapper.step(a*)
展开并真实执行 H 个 [dx,dy,dz,gripper]
                         ↓
DenseTrajectoryRecorder
保存 H+1 全部真实 timestep
                         ↓
PrimitiveResampler
重采样为 PointWorld T=11 sample
                         ↓
Replay + GT scene flow + robot flow
                         ↓
更新 G 和 PointWorld ensemble
                         └──────────────→ 下一轮
```

任何 candidate 在被选中前都不能通过 simulator rollout 偷看 scene outcome；否则相当于使用 simulator planning oracle。

---

## 29. 实施 A：RAPSEnvWrapper

### 29.1 对外 API

新增 `robocasa_exploration/primitive_wrapper.py`：

```python
class RAPSEnvWrapper(gym.Wrapper):
    action_space.shape == (28,)

    def step(self, raps_action):
        ...
        return observation, 0.0, terminated, truncated, info
```

20D contract：

```text
action[:8]   = primitive selector / one-hot logits
action[8:28] = normalized shared parameter bank
```

wrapper 使用 `RAPS_ACTION_SCHEMA.decode()`，不接受 object ID。

### 29.2 一次高层 step 的内部流程

```text
1. 保存 primitive-start observation / qpos / EE pose
2. decode 20D action
3. 初始化对应 primitive state machine
4. primitive.next_action(context) → 4D low action
5. base_env.step(low_action)
6. recorder 保存 transition
7. 更新 robot-only context 和 outcome diagnostics
8. primitive success / timeout / env done 时停止
9. 返回最终 observation 和 PrimitiveRecord
```

wrapper 的 `reward` 永远返回 `0.0`。如果底层 RoboCasa task 意外返回非零 reward，默认丢弃；只允许在 debug 模式记录 `base_env_reward_debug`，不得进入训练数据和 selector。

### 29.3 termination 语义

必须区分：

```text
primitive_done       当前 primitive 正常完成
primitive_timeout    达到 primitive max_steps
terminated           安全/物理终止条件
truncated            高层 episode 或总低层预算耗尽
```

primitive 完成不等于 episode terminated。

### 29.4 TimeLimit 修正

当前注册环境的 `max_episode_steps=200` 是低层 step 预算，而单个 primitive 可运行约 10–180 step。高层 wrapper 前必须：

- 移除外层 200-step `TimeLimit`，或将其调整为整个高层 episode 的低层总预算；
- 单独设置 `max_primitives_per_episode`；
- 单独设置 `max_low_level_steps_per_episode`；
- 任何 baseline 都按真实 low-level simulator steps 计费。

### 29.5 验收

- [ ] `env.action_space.shape == (28,)`；
- [ ] `step()` 只接收 20D action；
- [ ] 内部所有 low action 都是有限的 4D；
- [ ] reward 始终为零；
- [ ] 一个 primitive 完成后 episode 可继续；
- [ ] low-level duration 与 recorder 完全一致；
- [ ] object ID 不出现在 action decode / primitive control call graph；
- [ ] fixed base/body 在整个高层 episode 中保持零漂移。

---

## 30. 实施 B：双层轨迹与 WDS contract

新增 `robocasa_exploration/trajectory_recorder.py`，同时保存 dense low-level record 与 PointWorld high-level sample。

### 30.1 Dense PrimitiveRecord

```text
PrimitiveRecord
├── episode_id
├── primitive_id
├── primitive_type
├── raps_action_normalized        [28]
├── parameters_physical           dict
├── status / final_stage
├── duration_low_steps            scalar H
├── low_actions                   [H,4]
├── observations                  [H+1,...]
├── qpos / qvel                   [H+1,D]
├── ee_pose                       [H+1,7]
├── gripper_open                  [H+1]
├── contacts                      [H,...]
├── object_poses_eval_only        [H+1,No,7]
├── terminated / truncated
└── outcome_metrics
    ├── contacted_object_ids
    ├── grasped_object_ids
    └── object_translation
```

不得按 contact、grasp 或 success 拒绝样本；空气运动、失败接触和滑落都是合法 dynamics data。

### 30.2 PointWorld T=11 sample

当前 PointWorld 固定使用 T=11。一个 primitive 的真实 duration 为 H，使用相位重采样：

\[
i_j=\operatorname{round}\left(\frac{jH}{10}\right),\quad j=0...10
\]

生成：

```text
sample_indices               [11]
phase                        [11] = linspace(0,1,11)
low_step_time                [11]
temporal_valid_mask          [11]
joint_positions              [11,D]
robot_flows                  [11,Nr,3]
scene_flows_gt               [11,Ns,3]
scene_visibility             [11,Ns]
scene_depth_valid_mask       [11,Ns]
raps_action_normalized       [28]
primitive_duration           scalar
primitive_type               scalar/string metadata
```

如果 H<10，允许重复最近 frame，但必须保存 `temporal_valid_mask` 和原始 indices。禁止无标记 padding。

### 30.3 两种 dataset view

```text
primitive_view:
    一个 sample = 一个完整 primitive 的 T=11 相位轨迹
    用于 high-level candidate outcome prediction

low_level_window_view:
    dense trajectory 继续滑窗成连续 T=11 low-step windows
    用于保留原 PointWorld 的短时动力学监督
```

MVP 优先训练 `primitive_view`，同时保留 dense archive，避免以后重新采集。

### 30.4 WDS 新增字段

现有 WDS collector 需要新增并贯穿 decoder、pipeline、gather、collate：

```text
raps_action.npy              float32 [28]
primitive_id.npy             int64 scalar
primitive_duration.npy       int32 scalar
primitive_phase.npy          float32 [11]
temporal_valid_mask.npy      bool [11]
low_actions.npy              float32 [H,4] 或独立 dense shard
primitive_status.txt
```

不能只把字段写进 tar；必须加入 PointWorld `final_fields`、collate 和 batch device transfer，否则训练时会静默丢失。

### 30.5 Scene correspondence

RoboCasa 不能沿用当前 collector 中 peg-specific scene-flow 逻辑。通用实现：

```text
t=0 scene point
→ segmentation actor ID
→ world point 转 actor-local point
→ 每帧 actor pose × local point
→ 得到同一物理点的 GT trajectory
```

fixture/static points保持不动；动态 object points随 actor pose；visibility 与 physical existence 分开。

### 30.6 验收

- [ ] `len(observations) == H+1`；
- [ ] `len(low_actions) == H`；
- [ ] 每个 primitive 恰好产生一个 primitive-view sample；
- [ ] T=11 index 单调且首尾对应真实 primitive 首尾；
- [ ] WDS round-trip 后 20D action、duration、mask 完全相同；
- [ ] GT scene flow 可视化与 actor pose 一致；
- [ ] 数据中不存在 reward-dependent filtering。

---

## 31. 实施 C：RAPS action → robot-flow bridge

### 31.1 为什么这是硬前置

当前 PointWorld 的 action conditioning 不是 4D/20D action，而是：

```text
robot_coord_seq = data_dict["robot_flows"]  # [B,T,Nr,3]
```

训练时 robot flow 可从真实未来 qpos 经 Fetch URDF/FK 得到；但候选选择发生在执行前，真实未来 qpos 尚不存在。因此必须预测候选 robot trajectory。

### 31.2 第一版模型

新增 `RAPSRobotTrajectoryModel`：

\[
G(q_0,e_0,a^{RAPS},d)\rightarrow \hat q_{0:10}
\]

输入：

```text
initial arm/gripper qpos
initial EE pose
20D normalized RAPS action
primitive duration / nominal horizon
```

输出：

```text
predicted qpos trajectory [11,Dq]
predicted gripper_open     [11]
optional duration logits/scalar
```

随后通过已经实现的 Fetch adapter + robot sampler：

```text
predicted qpos
→ joint-name mapping
→ URDF/FK
→ predicted robot_flows [11,Nr,3]
```

### 31.3 训练目标

\[
\mathcal L_G=
\lambda_q\lVert\hat q-q\rVert_1+
\lambda_{ee}\lVert FK(\hat q)-e\rVert_1+
\lambda_r\lVert\hat R-R\rVert_1
\]

其中 \(R\) 是 robot point trajectory。第一版可先预测 qpos，再用 FK 计算 EE/robot-flow auxiliary loss。

### 31.4 可行性过滤

`ValidityFilter` 只能使用 robot-side information：

- 参数是否在 schema range；
- predicted joint limit violation；
- predicted EE workspace violation；
- primitive duration 是否超过预算；
- self-collision / fixture-independent safety approximation。

不能使用 object pose 判断“有没有物体”或提前查询真实 scene collision outcome。

### 31.5 防止 bridge error 污染 uncertainty

所有 PointWorld ensemble member 必须使用同一个冻结的 `G` 输出，这样 disagreement 主要来自 scene dynamics，而不是每个成员不同的 robot plan。

分别报告：

```text
robot qpos MAE
EE trajectory EPE
robot-point trajectory EPE
duration error
scene-flow error conditioned on GT robot flow
scene-flow error conditioned on predicted robot flow
```

只有 predicted-flow 与 GT-flow 条件下性能差距可接受后，才能用于在线 candidate ranking。

### 31.6 替代方案与顺序

可先尝试 deterministic nominal waypoint + IK rollout；如果其 robot-flow EPE 已足够低，可不训练 `G`。禁止使用复制 simulator 并真实 rollout candidate scene 作为 planner，因为这会泄露环境动力学答案。

### 31.7 验收

- [ ] candidate 执行前可生成完整 T=11 robot flow；
- [ ] 不依赖未来 observation/qpos；
- [ ] 不依赖 object state；
- [ ] held-out robot-flow EPE 达到预设阈值；
- [ ] invalid candidate 可在 PointWorld 前过滤；
- [ ] GT robot-flow 与 predicted robot-flow 两套 PointWorld 结果均被记录。

---

## 32. 实施 D：PointWorld primitive-view conditioning

### 32.1 保留现有核心输入

PointWorld 继续接收：

```text
initial scene point cloud
scene visual/geometric features
T=11 robot point trajectory
gripper state / robot velocity / acceleration
```

新增全局 action conditioning：

```text
raps_action [B,28]
primitive_duration [B,1]
primitive_phase [B,11]
```

虽然 robot flow 已隐式编码 action，显式 action embedding 仍用于区分相似 robot path 下的控制意图和 gripper primitive，并用于后续 actor 接口。必须通过消融比较：

```text
robot flow only
RAPS action only
robot flow + RAPS action
```

### 32.2 模型接入位置

新增：

```python
action_embedding = MLP(raps_action, primitive_duration)
```

将 action embedding 通过 FiLM 或 token 注入 `DynamicsPredictor`；不得把 20D action 简单复制到几千个点后造成尺度失衡而不做消融。

### 32.3 Scene encoder cache

同一 observation 下 K 个 candidate 的 scene input 相同。当前 PointWorld 已支持 `encoded_scene_feat0`，候选评分应：

```text
每个 ensemble member 对当前 scene encode 一次
→ cache encoded_scene_feat0
→ candidate 只替换 robot flow / action embedding
```

并将 K candidates 沿 batch 维并行，避免 K 次重复视觉编码。

### 32.4 PointWorld loss

主损失保持 scene relative-flow supervision：

\[
\mathcal L_{flow}=
\sum_{t,j}m_{t,j}w_j
\rho(\hat{\Delta p}_{t,j}-\Delta p_{t,j}^{GT})
\]

其中：

- \(m\)：temporal / visibility / existence valid mask；
- \(w\)：moved/static balancing；
- \(\rho\)：L1/Huber/NLL，与现有训练保持一致。

现有 heteroscedastic `log_var_head` 表示单模型 aleatoric uncertainty；主 acquisition 使用不同模型之间的 epistemic disagreement，二者不得混为一谈。

### 32.5 小数据门槛

在 ensemble 前先训练单模型：

- [ ] 32–128 个 primitive sample 可明显 overfit；
- [ ] static point false motion 接近零；
- [ ] moved-object flow 方向正确；
- [ ] action permutation 会显著恶化预测；
- [ ] zero robot-flow 与真实 robot-flow 的预测不同。

如果 action permutation 不影响 loss，说明模型忽略了 RAPS/robot-flow conditioning，不能进入 ensemble 阶段。

---

## 33. 实施 E：PointWorld ensemble

### 33.1 组成

推荐 E=5：

```text
相同 architecture / normalization / feature contract
不同 random initialization
不同 dataloader order
不同 bootstrap sample membership
同一个固定 validation set
```

bootstrap membership 必须由 sample key + model seed 稳定决定，确保恢复训练后不改变。

### 33.2 预测张量

对 K 个 candidates：

```text
pred_relative_flow [E,K,T,Ns,3]
pred_log_var       [E,K,T,Ns,1]
```

所有模型使用相同的 initial scene points、candidate action、predicted robot flow 和 masks。

### 33.3 训练调度

第一版采用 round-based 更新而非每个 primitive 后立刻训练：

```text
bootstrap collection
→ train ensemble to fixed optimizer-step budget
→ collect one active batch
→ warm-start each member for fixed steps
→ repeat
```

这样更容易复现，也能将 collection time 与 training time分开统计。

### 33.4 验收

- [ ] 每个 member 单独达到 held-out prediction baseline；
- [ ] 不同 member 不是完全相同输出；
- [ ] ensemble variance 在训练数据重复加入后下降；
- [ ] checkpoint 包含 member seed、bootstrap manifest、normalization stats；
- [ ] ensemble 可 batched score K candidates。

---

## 34. 实施 F：Disagreement acquisition

### 34.1 点级 epistemic disagreement

令模型 m 对 candidate k 的预测为 \(\hat\Delta p_{mktj}\)：

\[
\bar\Delta p_{ktj}=\frac1E\sum_m\hat\Delta p_{mktj}
\]

\[
u_{ktj}=\frac1E\sum_m
\left\lVert\hat\Delta p_{mktj}-\bar\Delta p_{ktj}\right\rVert_2^2
\]

### 34.2 不让静态厨房稀释 score

不能对所有墙、台面和柜体点简单平均。使用 candidate predicted robot path 计算：

\[
d_{kj}=\min_{t,r}\lVert p_j-R_{ktr}\rVert
\]

\[
w_{kj}=\exp(-d_{kj}^2/(2\sigma^2))
\]

主分数：

\[
U_k=\frac{\sum_{t,j}m_{tj}w_{kj}u_{ktj}}
{\sum_{t,j}m_{tj}w_{kj}+\epsilon}
\]

再报告 top-q uncertainty 作为鲁棒性指标。这里的 neighborhood 完全由 observed scene points 和 predicted robot path 计算，不需要 object segmentation。

### 34.3 Primary 与 ablation 分开

主方法：

```text
PW-RAPS score = epistemic disagreement - invalid penalty
```

单独消融：

```text
+ predicted motion
+ parameter rarity
+ interaction coverage
+ learning progress
+ aleatoric penalty
```

不要在主结果中默认加入 GT contact/object motion reward。

### 34.4 Calibration

在冻结 validation candidates 上，对执行后的真实 error 计算：

- Spearman correlation：disagreement vs actual EPE；
- uncertainty quantile reliability；
- top-10% uncertainty 的平均 error；
- high-error detection AUROC/AUPRC；
- primitive-wise calibration；
- parameter-bin calibration。

如果 disagreement 与真实误差无正相关，停止 active selection，先修复 ensemble diversity、robot-flow bridge 或数据噪声。

---

## 35. 实施 G：CandidateGenerator 与 Selector

### 35.1 Uniform candidate generator

第一版复用当前 schema ranges 和 YAML weights：

```python
candidates = [sample_raps_action(rng) for _ in range(K)]
```

它只能读取：

```text
primitive weights
parameter bounds
current robot state
remaining low-level budget
```

不能读取 simulator object pose。

### 35.2 Point-cloud proposal baseline

作为单独 heuristic baseline，可从 observation 点云：

```text
估计台面平面
→ 选择高出台面的无语义点/cluster
→ 转到 robot base frame
→ 生成 dx/dy/descend candidates
```

它不使用类别名或 GT segmentation，但属于视觉 affordance prior，必须与 pure Uniform candidate generation 分开报告。

### 35.3 Selector

默认：

```text
probability epsilon：从 valid candidates 均匀随机
otherwise：从 uncertainty softmax 或 top-1 选择
```

推荐先使用：

```yaml
candidates_per_decision: 32
uniform_floor: 0.20
selection: softmax
temperature: 0.10
```

所有候选都记录，不只记录被选中的 action：

```text
candidate actions [K,28]
valid mask [K]
predicted robot-flow summary
ensemble score [K]
selected index
selection probability
```

这使得 selector 可离线重放和审计。

---

## 36. 实施 H：闭环与完整 Plan2Explore 扩展

### 36.1 MVP round-based 闭环

```text
Round 0:
    Uniform RAPS 收集 D0
    训练 robot trajectory model G0
    训练 PointWorld ensemble M0

Round r:
    固定 G_r, M_r
    对每个 observation 生成 K candidates
    disagreement 选择并执行 B_r
    D_{r+1} = replay(D_r) ∪ B_r
    更新 G_{r+1}, M_{r+1}
    在冻结 validation set 上评估
```

每轮之间才更新模型；collection 中模型保持冻结，避免不同 episode 使用不可追踪的参数版本。

### 36.2 Replay

训练 PointWorld 时必须保留：

```text
current active batch
historical uniform bootstrap
historical high-uncertainty samples
contact / moved / no-contact strata
all primitive types
```

第一版建议固定总 replay capacity，按真实 primitive sample 数做 reservoir；训练 batch 可采用 50% current + 50% replay。分层标签只影响采样平衡，不改变 loss target。

### 36.3 Budget fairness

主环境成本单位：

```text
真实 low-level simulator steps
```

同时报告：

```text
high-level primitive decisions
accepted/invalid candidates
PointWorld forward passes
GPU planning time
training optimizer steps
wall-clock time
```

不同 primitive duration 不得只按“一个 primitive = 一个 sample”比较，否则会偏向长 primitive。

### 36.4 完整 actor-based Plan2Explore（后续）

只有单步 PW-RAPS 明确优于 Uniform-RAPS 后才实现：

```text
PointWorld scene encoder
→ pooled high-level state h_t
→ hybrid exploration actor π(k,θ|h_t)
→ imagined PointWorld rollout
→ ensemble disagreement intrinsic reward
→ actor/value update
```

完整版本还需要：

- 可递归的 high-level scene state transition；
- predicted scene points重新编码为下一 latent；
- hybrid categorical + continuous actor；
- value function和 multi-step return；
- imagination horizon（以 primitive 为单位）；
- imagined rollout stability。

在此之前不得把 candidate-ranking MVP 称为“完整 Plan2Explore actor”。

---

## 37. 实验设计与评价

### 37.1 Baselines

| ID | 方法 | 动作 | 选择信号 |
|---|---|---|---|
| B0 | IID Cartesian | 4D low action | IID random |
| B1 | Correlated Cartesian | 4D low action | temporally correlated noise |
| B2 | Uniform RAPS | 20D high action | fixed primitive/parameter distribution |
| B3 | Point-cloud proposal RAPS | 20D | geometric proposal + uniform |
| B4 | Heuristic RAPS | 20D | coverage/contact/motion heuristic |
| Ours | PW-RAPS | 20D | PointWorld epistemic disagreement |

### 37.2 固定条件

- 同一 RoboCasa layout/style/object reset distribution；
- 同一 Fetch spawn 和固定 base/body；
- 同一 camera、voxel size、scene/robot point budget；
- 同一 low-level simulator-step budget；
- 同一 PointWorld architecture、optimizer-step budget、replay capacity；
- 同一冻结 validation/test split；
- 至少 3 个 exploration seeds，正式结果建议 5 seeds。

### 37.3 Exploration outcome metrics

这些指标只评价 collector，不作为主方法 reward：

```text
contact low-steps / total low-steps
moved-object primitive rate
grasp-event rate
unique contact-region coverage
primitive / parameter-bin coverage
no-contact fraction
trajectory redundancy
first interaction low-step count
```

### 37.4 World-model metrics

```text
full-scene EPE
moved-point EPE
near-robot/contact-neighborhood EPE
static false-motion EPE
T=1 / T=5 / T=10 rollout EPE
uncertainty–error correlation
达到固定 validation EPE 所需 low-level transitions
```

test set 必须独立于每种 explorer 收集的数据；推荐建立一个固定 mixed-policy evaluation suite，包含 uniform、contact boundary、successful interaction 和 failure trajectories。

### 37.5 关键消融

- GT robot flow vs predicted robot flow；
- robot flow only vs action only vs both；
- single PointWorld log variance vs ensemble disagreement；
- global uncertainty mean vs robot-neighborhood weighting；
- top-1 vs softmax selection；
- `uniform_floor=0` vs `0.2`；
- K=8/32/64 candidates；
- Uniform workspace vs point-cloud candidates；
- primitive-view T=11 vs low-level sliding windows；
- 无 action embedding vs 20D action embedding；
- current-only training vs replay。

---

## 38. 下一版 YAML contract

```yaml
robocasa_raps:
  env_id: RoboCasaKitchenCustom-v1
  kitchen_config: configs/kitchen_custom.yaml
  robot_type: fetch
  control_mode: pd_ee_delta_pos
  fixed_base: true
  fixed_body: true
  environment_reward: disabled
  max_primitives_per_episode: 10
  max_low_level_steps_per_episode: 1500

  action:
    num_primitives: 8
    num_parameter_slots: 20
    action_dim: 28

  recorder:
    save_dense_low_level: true
    pointworld_sequence_length: 11
    primitive_resampling: uniform_phase
    save_temporal_valid_mask: true

  robot_trajectory_model:
    mode: learned_qpos        # learned_qpos / deterministic_ik
    checkpoint: null
    freeze_during_collection: true

  ensemble:
    size: 5
    bootstrap: true
    shared_validation_split: true
    warm_start_each_round: true

  exploration:
    method: uniform_raps      # uniform_raps / heuristic_raps / pw_raps
    candidates_per_decision: 32
    uniform_floor: 0.20
    selection: softmax        # top1 / softmax
    temperature: 0.10
    score: epistemic_robot_neighborhood

  loop:
    bootstrap_low_level_steps: 50000
    active_low_level_steps_per_round: 10000
    num_rounds: 5
    replay_capacity_primitives: 10000
    current_batch_fraction: 0.50

  logging:
    save_all_candidates: true
    save_base_env_reward_debug: false
    save_eval_object_gt: true
```

预算数值是实验起点，正式运行前由 smoke test 的 primitive 平均 duration 和存储成本校准。

---

## 39. 从现在开始的严格开发顺序

### Milestone 1：高层环境与数据完整性

1. 实现 `RAPSEnvWrapper.step(20D)`；
2. 修正低层 TimeLimit / 高层 episode budget；
3. 实现 dense `PrimitiveRecord`；
4. 实现 primitive T=11 phase resampler；
5. 将 RAPS metadata 接入 WDS decoder/pipeline/collate；
6. 完成可视化回放和 round-trip tests。

完成条件：随机给定 20D action，可以无 reward 地执行、保存、重载并逐帧复现数据 contract。

### Milestone 2：RoboCasa PointWorld 数据

7. 完成通用 RoboCasa scene correspondence；
8. 完成 Fetch robot-flow overlay/FK 验证；
9. 收集小规模 Uniform-RAPS dataset；
10. 单 PointWorld 在 32–128 samples 上 overfit；
11. 验证模型使用了 robot flow / RAPS action。

完成条件：PointWorld 能正确预测至少一个 PUSH_XY 和 TOP_GRASP 的 moved-point direction，并保持静态点稳定。

### Milestone 3：执行前 candidate robot flow

12. 实现 deterministic IK baseline 或 `RAPSRobotTrajectoryModel`；
13. 从 20D action 预测 T=11 qpos；
14. 经 Fetch FK 生成 candidate robot flow；
15. 比较 GT-flow-conditioned 与 predicted-flow-conditioned PointWorld。

完成条件：未执行 candidate 时可以构建 PointWorld 输入，且 robot-flow bridge error 不主导 scene prediction。

### Milestone 4：Ensemble acquisition

16. 训练 E=5 PointWorld ensemble；
17. batched score K candidates，并缓存 scene encoding；
18. 实现 robot-neighborhood disagreement；
19. 在冻结 validation set 校准 disagreement；
20. 实现 epsilon-uniform selector。

完成条件：disagreement 与真实 held-out error 显著正相关，并可在一个 observation 上稳定排序 candidates。

### Milestone 5：闭环与论文实验

21. 运行 Uniform bootstrap；
22. 运行多轮 PW-RAPS collect–train loop；
23. 固定 low-level budget 比较 B0–B4 和 Ours；
24. 完成主要消融、计算成本和多 seed 统计；
25. 只有单步 PW-RAPS 成功后再决定是否实现 actor-based Plan2Explore。

### 全局停止条件

- wrapper 丢失任一 low timestep：停止；
- reward 或 success filtering 泄漏进 collector：停止；
- action/primitive 使用 object GT：停止正式实验；
- Fetch FK/scene correspondence 未对齐：不训练正式 PointWorld；
- PointWorld 忽略 action conditioning：不训练 ensemble；
- candidate robot-flow error 过大：不做 active ranking；
- disagreement 与真实 error 不相关：不进入闭环；
- Uniform-RAPS 在相同 low-step budget 下不优于 Cartesian baseline：先修 primitive 分布与数据 contract。
