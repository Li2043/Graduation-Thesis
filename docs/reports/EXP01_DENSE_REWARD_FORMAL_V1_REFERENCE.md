# exp01_ma_formal_v1（Dense Reward / FormalV1 DER）实验配置与公式整理

**用途**：作为"下一步做 dense reward"的参考基线。本文档所有数字/代码引用均已对照实际跑出来的
`config.json` / `run_summary.json` 核实，不是从记忆或文档转述——凡是直接读取源码或运行产物确认的，
标注为"已核实"；凡是仅来自 `paper/实验迭代.md` 转述、本次未重新核对原始日志的，单独标注。

**物理位置**：`C:\Users\HP\Desktop\毕业项目\thesis\rawlsian_project\`（独立仓库，不在
`final_new_experiment` 的 E01–E32 hub/junction 体系内；hub 侧索引见 `STRUCTURE.md` 的
"E00 前置探索实验"一节）。

---

## 0. 网络：Decentralized（去中心化执行）+ 参数共享

**结论：去中心化（decentralized execution），单个共享（parameter-shared）DQN，不是中心化网络。**

已核实依据（`src/thesis/training/formal_v1_ma_train.py`）：

- 全程只创建 **一个** `EgoisticDQN` 策略对象（`policy = EgoisticDQN(...)`），main 和 ramp 两辆车共用同一份网络权重——这是"参数共享"，但**不是**中心化（centralized）。
- 每一步，两辆车各自把**自己的局部观测**单独喂给这个共享网络，独立选自己的动作：
  ```python
  action_main = policy.act(obs_dict[MAIN_AGENT], epsilon)
  action_ramp = policy.act(obs_dict[RAMP_AGENT], epsilon)
  ```
  没有把两车状态拼成一个联合状态（joint state）喂给网络，也没有输出联合动作（joint action）、没有中心化 critic——这正是"去中心化执行"的定义。
- 两车的 transition `(local_obs, action, reward, next_local_obs, done)` 各自单独存进**同一个**回放缓冲区，网络更新时不区分是哪辆车产生的样本（等价于当前论文里 Original/WSC 的"parameter-shared DQN, decentralized execution"范式，只是那时用两车、现在用四车）。

**观测是纯局部/自我中心的**（已核实，`src/thesis/env/multi_agent_highway_merge_env.py`，17 维）：

```
self_x_norm, self_speed_norm, self_lane_norm, role_is_main, role_is_ramp,
distance_to_conflict_norm, distance_to_goal_norm,
rel_x_other_controlled, rel_v_other_controlled, same_lane_other_controlled,
front_gap_current_lane_norm, front_rel_v_current_lane,
rear_gap_current_lane_norm, rear_rel_v_current_lane,
front_gap_target_lane_norm, rear_gap_target_lane_norm,
own_ttc_norm
```

即：自身状态 + 与"另一辆受控车"的相对量 + 当前/目标车道的前后车间距——不包含背景车辆的完整状态，也不是全局状态。

**网络结构**（已核实，`src/thesis/training/policies/egoistic_dqn.py`）：

```
Linear(17 → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → 4)
```

- 动作空间 4 维离散：MAINTAIN / ACCELERATE / DECELERATE / LANE_CHANGE。
- **注意与当前论文的架构差异**：这里的 target 计算是 `target_network(next_state).max(...)`——用 target 网络同时做动作选择和取值，是**vanilla DQN，不是 Double DQN**（当前 Study B 论文用的是 Double DQN）。若要迁移到现在的四车环境，这一点需要决定是照搬 vanilla DQN 还是统一升级成 Double DQN。
- Loss：Smooth L1（Huber）；优化器 Adam；replay buffer 为均匀采样（非 PER）；target network 每 500 次梯度更新硬拷贝一次。

---

## 1. Dense Reward 公式（已核实，逐行对照 `config.json` 与源码）

### 1.1 每步基础项（Egoistic，两条件共用）

来源：`src/thesis/rewards/egoistic_reward.py`

$$
r^{\text{ego}}_{i,t} = 1.0\cdot\text{progress}_{i,t} \;-\; 1.0\cdot\mathbb{1}[\text{crashed}_i] \;-\; 0.1\cdot\max\!\Big(0,\ \frac{1}{ttc_i+10^{-3}}\Big) \;-\; 0.1\cdot\frac{\text{waiting\_time}_i}{100}
$$

- `progress`：沿目标方向的纵向位移变化，按 `goal_position` 归一化：
  \(\text{progress} = \dfrac{|goal-x_{\text{prev}}| - |goal-x_{\text{curr}}|}{\max(|goal|,\,10^{-6})}\)
- 风险项：TTC 越小惩罚越大（反比例，非线性）。
- 等待项：累计等待时间线性惩罚，除以 100 归一化。
- 参数（`EgoisticReward.__init__` 默认值，训练脚本未覆盖，已核实无覆盖）：`alpha=0.1`（风险系数）、`beta=0.1`（等待系数）、`epsilon=1e-3`、`progress_weight=1.0`、`collision_penalty=1.0`、`waiting_normalizer=100.0`。

### 1.2 Rawlsian 条件的额外 shaping 信号（两条件唯一差异）

来源：`src/thesis/rewards/multi_agent_rewards.py::compute_rawlsian_signal_all_vehicles`

$$
r^{\text{rawls}}_t=
\begin{cases}
+\lambda & \Delta E_{\min,t} > \varepsilon\\
-\lambda & \Delta E_{\min,t} < -\varepsilon\\
0 & \text{otherwise}
\end{cases}
\qquad
E_{\min,t}=\min_i \text{experience\_score}_i(t)
$$

- \(\lambda = 0.2\)，\(\varepsilon = 10^{-4}\)（`config.json` 实际值：`lambda_rawlsian=0.2`, `epsilon_delta=0.0001`）。
- \(E_{\min}\) 是对**场上所有车辆**（含背景车）取 min，不只是两辆受控车。
- 这个信号是**共享的**——同一个数值同时加到 `r_main` 和 `r_ramp` 上，不是每车单独算。
- Egoistic 条件下这一项恒为 0。

### 1.3 experience_score（DER 聚合，四项加权）

来源：`src/thesis/experience/der.py::aggregate_experience_score`

$$
\text{experience}_i=
\begin{cases}
0 & \text{碰撞}\\
\mathrm{clip}_{[0,1]}\big(0.20\,\text{task}_i+0.35\,\text{safety}_i+0.20\,\text{efficiency}_i+0.25\,\text{social\_impact}_i\big) & \text{否则}
\end{cases}
$$

子项定义（均已核实为 `config.json` 实际值）：

| 子项 | 公式 | 相关阈值 |
|---|---|---|
| `task` | 1 完成任务 / 0 未完成 | — |
| `safety` | \(1-\text{ttc\_risk}\)；TTC≤2.0s→风险1，TTC≥10.0s→风险0，中间线性插值 | `ttc_critical=2.0`, `ttc_safe=10.0` |
| `speed_efficiency` | \(\mathrm{clip}_{01}(v/v_{\text{target}})\)；\(v_{\text{target}}\)：ramp=10、main/background=8 m/s | `target_speed_ramp=10.0`, `target_speed_main=8.0` |
| `waiting_score` | \(1-\mathrm{clip}_{01}(\text{waiting\_time}/100)\) | `waiting_ref=100.0` |
| `efficiency` | \(0.60\cdot\text{speed\_efficiency}+0.40\cdot\text{waiting\_score}\) | — |
| `nearby_safety` | \(1-\)（40m 内最近邻车辆的最大 TTC 风险）；无邻车则为 1 | `d_near=40.0` |
| `nearby_waiting` | \(1-\)（40m 内最近邻车辆的最大归一化等待时间）；无邻车则为 1 | — |
| `social_impact` | \(0.65\cdot\text{nearby\_safety}+0.35\cdot\text{nearby\_waiting}\)；无邻车则为 1 | — |

### 1.4 终局调整（两条件共用，逐车判定）

来源：`src/thesis/rewards/merge_task_reward.py`

- 完成 merge：**+1.0**（`merge_success_bonus`）
- 回合终止（超时/截断）且未 merge 且未撞车：**−1.0**（`non_merge_failure_penalty`）
- 该车撞车：**−10.0**（`terminal_collision_penalty`）——两条件共用同一惩罚，独立于 1.1 节 egoistic 已有的每步碰撞项之外（文档原话：不是重复计数，一个是个体每步安全偏好，一个是任务层安全约束）。

### 1.5 完整逐车最终奖励

$$
R_i = r^{\text{ego}}_{i} + r^{\text{rawls}}\ (\text{仅 Rawlsian 条件，Egoistic 恒为 0}) + \text{terminal\_merge\_adj}_i + \text{terminal\_collision\_adj}_i
$$

---

## 2. 训练配置（已核实，来自实际跑出的 `config.json`）

```yaml
episodes: 25000
max_steps: 100
seed: 0              # config 里写了 seeds:[0,1,2]，但 exp01_ma_main_25000 这批实际只跑了 seed 0
controlled_vehicle_ids: [main, ramp]
multi_agent: true
tag: ma_main_25000

# DQN 超参（formal_v1_ma_train.py 的 FormalV1MARunConfig 默认值，CLI 未覆盖，已核实）
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_episodes: 150   # 按 episode 线性衰减，不是按 step
gamma: 0.99
learning_rate: 1e-3
batch_size: 64
buffer_capacity: 50000
target_update_interval: 500   # 按梯度更新次数计
hidden_dim: 64

# reward/DER 相关（同第1节）
merge_success_bonus: 1.0
non_merge_failure_penalty: 1.0
terminal_collision_penalty: 10.0
lambda_rawlsian: 0.2
epsilon_delta: 0.0001
experience_score_weights: {task: 0.20, safety: 0.35, efficiency: 0.20, social_impact: 0.25}
efficiency_weights: {speed_efficiency: 0.60, waiting_score: 0.40}
social_impact_weights: {nearby_safety: 0.65, nearby_waiting: 0.35}
thresholds: {d_near: 40.0, waiting_ref: 100.0, ttc_critical: 2.0, ttc_safe: 10.0,
             target_speed_ramp: 10.0, target_speed_main: 8.0, target_speed_background: 8.0}
```

两个条件（egoistic / rawlsian）除 reward 计算方式外，其余配置**完全相同**（同一套 `formal_config`，同一环境、同一网络结构、同一训练超参、同一 seed=0）。

---

## 3. 结果

**已核实（本会话直接解压 `runs.zip` 读取 `run_summary.json` 确认跑过 25000 episodes、seed=0、egoistic/rawlsian 各一次）：**
run_id、condition、seed、episodes 四项元数据。

**转述自 `paper/实验迭代.md`（未在本次重新解压 episode_log.csv 逐条核对，仅转述该文档已给出的数字）：**

| 指标 | Egoistic | Rawlsian |
|---|---:|---:|
| 全程 success | 0.641 | 0.791 |
| collision | 0.238 | 0.177 |
| mean(min_experience) | 0.488 | 0.600 |
| mean(mean_experience) | 0.680 | 0.749 |

n=1（单 seed），非正式统计。若要在下一步实验里引用这组数字作为"当年确实更好"的证据，建议先用 `episode_log.csv`（在 `runs.zip` 内）重新拉一遍 last-100/全程 success 曲线，确认不是训练早期偶然的高点。

---

## 4. 源码索引

```
rawlsian_project/
├─ src/thesis/config/formal_v1_config.py         # 全部权重/阈值/训练默认值（dataclass）
├─ src/thesis/experience/der.py                  # DER / experience_score 计算
├─ src/thesis/rewards/
│  ├─ base_reward.py                             # RewardFunction 抽象接口 + 共享终局项
│  ├─ egoistic_reward.py                         # 每步 dense reward（两条件共用的基础项）
│  ├─ rawlsian_reward.py                         # 单智能体版本的 Rawlsian shaping（未在exp01用，仅供对照）
│  ├─ multi_agent_rewards.py                     # exp01 实际用的双车 reward 组合逻辑
│  └─ merge_task_reward.py                       # 终局 merge/collision 调整
├─ src/thesis/env/multi_agent_highway_merge_env.py  # 环境 + 17维局部观测构造
├─ src/thesis/training/
│  ├─ formal_v1_ma_train.py                      # exp01 训练主循环（决定了centralize/decentralize结论）
│  └─ policies/egoistic_dqn.py                   # 共享 DQN 网络结构 + 训练更新（vanilla DQN）
└─ experiments_results/exp01_ma_formal_v1/
   └─ runs.zip                                   # 实际 config.json / run_summary.json / episode_log.csv
```

---

## 5. 迁移到现在四车环境时需要注意的接口差异

- 车辆数：exp01 是 2 车（main + ramp，另有不受控背景车）；现在 Study B 是 4 受控车（Ramp-Fast/Slow, Mainline-Fast/Slow）。
- 观测维度：exp01 是 17 维局部观测；现在是 18 维（Original）/22 维（WSC）。
- DQN 版本：exp01 是 vanilla DQN；现在是 Double DQN——迁移时需要决定是否统一。
- Reward 结构：exp01 是"稠密个体项 + 离散±λ的Rawlsian shaping + 终局bonus/penalty"，与现在 Study B 的"稠密task reward + 终局social-welfare PBRS(GGI/Maximin/Mean)"是两套完全不同的公式家族（`实验迭代.md` 原文即注明"exp01–13 的 reward 不等于 Chapter 3 Base Reward V1/V2，无继承关系"）——不是同一个公式的新旧版本，而是两条独立的设计线，迁移时应视为"借鉴思路"而非"直接复用系数"。
- Rawlsian shaping 是**离散**的（±0.2 或 0），现在的 PBRS 型公式（\(F=\gamma\Phi(s_{t+1})-\Phi(s_t)\)）是**连续**的势函数差分——这是两种不同的 reward-shaping 范式，值得在设计新 dense reward 时明确选一种，而不是混用。
