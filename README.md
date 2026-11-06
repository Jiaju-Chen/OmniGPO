# OmniGPO — 标准核心实现

图信用与神经 critic 相结合的多轮 LLM agent 优化代码。
本仓库首先发布我们最标准的 **固定 `q=0.25`、图辅助 `beta=0.1`** 核心算法，
用于阅读、单元测试和接入已有训练框架。

**范围说明：这是可独立运行的算法核心包，不是完整的分布式训练工程。**
不包含模型权重、数据集、环境模拟器、Ray/FSDP/vLLM 训练器、服务器路径、
实验日志、checkpoint 或清理脚本。安装后可以跑 CPU 数值示例和测试；
不能把下面的示例当成 ALFWorld/WebShop/ScienceWorld 的端到端复现。

## 1. 方法：两个不同的作用位置

### Actor：融合步级信用，不混合 value

$$
A_i^{\mathrm{actor}}
=\eta_i A_i^{\mathrm{out}}
+w\left[(1-q)\rho_i\widetilde A_i^{\mathrm{FP}}
+q\widetilde A_i^{\mathrm{N}}\right].
$$

- `A_out`、`eta`、图支持 gate `rho`：保留原生 Gated-BEPO 计算。
- `A_FP`：当前 rollout 任务组的经验图产生的步级信用。
- `A_N`：神经 critic 的跨环境 turn GAE，按同一任务组对齐尺度。
- `q=0.25`：固定分配给神经信用的系数；不是 critic loss 的权重。
- `w=1.5`：步级信用相对 outcome 的力度。

Actor 仍使用**逐 token 概率比与逐 token PPO 裁剪**，不是整条回复的概率乘积。
`q=0` 可以精确恢复原生图 Actor 信用，但通过 `compute_credit` 调用时仍然计算
critic 和辅助目标；这不等于一条不创建 critic 的完整 Gated-BEPO 训练流水线。

### Critic：标准价值学习 + 图监督

$$
\mathcal L_V
=\mathcal L_{V,\mathrm{PPO}}
+\beta\frac{\sum_i b_i c_i
  \left(V_\phi(h_i)-\operatorname{sg}[V_{\mathrm{FP}}(s_i)]\right)^2}
 {\max(\sum_i b_i,1)}.
$$

- `beta=0.1`：固定图辅助监督强度，与 Actor 的 `q` 独立。
- `b_i`：动作前边界位置 mask；不是整段回复每个 token 都预测一遍 turn value。
- `c_i`：图老师的支持/残差检查结果。
- `sg`：停止 target 梯度，Actor 也不通过 advantage 反传至 critic。
- 分母保留所有有效边界，包括没有图支持的边界，**不改成只除以支持数**。
- PPO 基础 value loss 保留实验实现的裁剪和缩放，**不额外乘 `1/2`**。

`beta=0` 仅关闭 critic 图辅助 loss，不关闭 Actor 的图信用通道。

## 2. 图老师究竟计算什么

同一任务 UID 内合并 rollout 访问到的状态，保留经验 transition 的出现次数，
对观察到的后继做**经验均值 Bellman backup**：

$$
V_{\mathrm{FP}}(s)
=\frac{1}{|\mathcal E_s|}\sum_{(r,s')\in\mathcal E_s}
\left[r+\gamma V_{\mathrm{FP}}(s')\right],\qquad V(\mathrm{terminal})=0.
$$

这不是最短路径、不是对 action 取 max、不是 oracle value，也不是跨所有历史
batch 永久累积的全局图。状态标识 `anchor_obs` 由调用方提供，不能默认为完整
Markov/oracle state。图支持不足时辅助权重为0，但 Actor outcome 和神经通道仍存在。

同一 `(UID, trajectory, step)` 的 padding copy 不重复计入图的 transition 统计；
保留训练器原本的样本布局和边界 loss 分母，不在此偷偷改变 padding 训练权重。

## 3. 安装和最小运行

```bash
git clone https://github.com/Jiaju-Chen/OmniGPO.git
cd OmniGPO
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
python -m examples.tiny_batch
python -m pytest -q
```

CPU 即可运行。已有 CUDA PyTorch 时无需更换为 CPU 版本。
本地验证环境为 Python3.12 / PyTorch2.5.1 / NumPy1.26.4；这不是完整 GPU 训练环境锁。

## 4. 输入／输出契约

```python
from omnigpo import Batch, StandardConfig, compute_credit, critic_loss

cfg = StandardConfig(q=0.25, beta=0.1)
data = Batch(batch=tensor_fields, non_tensor_batch=turn_metadata)
compute_credit(data, cfg)

# new_values 来自当前可训练 critic；不是把 rollout 的旧 values 当成新前向。
loss, metrics = critic_loss(new_values, data, cfg)
loss.backward()
```

| 字段 | 含义 |
| --- | --- |
| `response_mask [N,L]` | 每个 turn 的有效生成 token，二值 mask |
| `values [N,L]` | rollout 时旧 critic 的预测；首个有效位置对应动作前状态 |
| `token_level_rewards [N,L]` | 原训练管线的 outcome token rewards，原样保留 |
| `uid [N]` | 同任务多 rollout 的分组 ID |
| `traj_uid [N]` | 独立轨迹 ID，必须跨任务唯一 |
| `step_idx [N]` | 轨迹内从0开始的连续环境 step，可经过 batch 重排 |
| `anchor_obs [N]` | 图使用的状态键，由环境/rollout 管线提供 |
| `rewards [N]` | 每次环境 action 恰好一次即时 reward，不能重复 episode outcome |
| `terminal_kind [N]`，可选 | `success` / `failure` / `truncated` 或空串 |

输出包括 `advantages`、`returns`、`value_mask`、`graph_value_targets`、
`graph_confidence`，以及分支诊断。原 `token_level_rewards` 不会被替换。
详细可执行例子见 [examples/tiny_batch.py](examples/tiny_batch.py)。

## 5. 标准配置和代码导航

默认：`q=.25, beta=.1, gamma=.95, lambda=.8, w=1.5`；
PPO Actor token ratio clip 为`.2`（调用 loss 时传入），critic value clip 为`.5`；
Actor 图迭代20轮，FP teacher 最多200轮，teacher residual 阈值`1e-4`；
图支持至少2条不同轨迹、2个不同后继。

| 代码 | 职责 |
| --- | --- |
| `omnigpo/api.py` | 标准配置、输入检查、信用与 critic loss 的公共接口 |
| `gigpo/core_bepo.py` | Gated-BEPO 经验图、Bellman 计算、outcome／图信用 |
| `gigpo/bepo_neural_credit.py` | 固定 q 的神经信用融合，原始 turn reward 处理 |
| `gigpo/graph_value_teacher.py` | FP teacher、置信支持 mask、节点监督 loss |
| `gigpo/ppo.py` | 跨 turn GAE、逐 token PPO、裁剪 value loss |
| `tests/` | 公式、奖励计数、mask、梯度、padding、源码来源检查 |

保留的算法文件来自同一份已记录实验源码，兼容分支不作为本包默认方法。
公开入口不支持动态 q、EMA、MC-direct、edge loss、自适应 beta 或轻量 critic。
我们没有在这个核心发布中重新实现或打包那些探索实验。

## 6. 接入训练器的边界

参见 [docs/INTEGRATION.md](docs/INTEGRATION.md)。接入方仍负责模型前向、
pre-action value 对齐、old log-prob、PPO minibatch、环境生命周期、采样随机性、
checkpoint 和评测协议。不能将“CPU 核心测试通过”写成“完整7B训练已复现”。

当前 GAE 与实验保持一致：每条采集轨迹末端按0 bootstrap，包括 time-limit
截断；若需要对 time-limit 另做 bootstrap，应单独设计和验证，不能静默替换。
图目标是当前样本的经验估计，不保证对当前策略无偏，也不宣称任意数据集都提升。

## 来源和许可

[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) 记录实验源码 SHA-256 与打包变化。
图内核与 teacher 文件保持原样；神经融合只做导入适配与非默认路由的显式拒绝；
PPO/GAE 从已核验源码机械提取。保留上游 Apache-2.0 声明，见 [LICENSE](LICENSE)
和 [NOTICE](NOTICE)。不包含未核验结果表、论文主表复现承诺或第三方模型/数据资产。
