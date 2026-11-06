# 与现有训练工程集成

本包处理 rollout 已经收集完成后的算法计算，不负责启动训练。

1. **先采样。** 保留同任务多次 rollout 的 UID／trajectory／turn 信息。每个请求
   不要被强制设置成同一个采样 seed；engine 初始化 seed 不等于 request seed。
2. **保存旧策略数据。** 计算 old token log-prob 和旧 critic value。首个 response
   位置的 value 必须由“读完当前 observation、尚未生成 action”的隐藏状态产生。
   不能拿 action 末端的 value 冒充动作前状态。
3. **计算信用一次。** `compute_credit(data, config)` 产生 detached Actor
   advantage、跨 turn GAE return 和 FP target。传入 `rewards` 是每个 turn 一次的
   环境即时奖励；`token_level_rewards` 是既有 outcome 通路，二者不可混用。
4. **训练 critic。** 对相同历史重新前向得到 `new_values`，用
   `critic_loss(new_values, data, config)`。targets、old values、graph confidence
   全部保持 detached。标准版 full critic 由训练器创建；本包没有 MLP/浅层结构。
5. **训练 Actor。** 使用 `gigpo.ppo.compute_policy_loss` 或框架中数学一致的
   token PPO loss。把同一 turn 的信用广播到 action 的有效 token，但仍逐 token
   计算 ratio 和 clipping。不要把 token log-ratio 求和后变成 joint-turn ratio。
6. **保存／验证。** 由训练工程保存模型、optimizer、scheduler、RNG、dataloader
   和必要路由状态。latest 与 best 的用途要分开，best 仅使用验证集选择，不在
   正式测试集上反复选模型。此核心包不包含清理、停止进程、自动重试或服务器脚本。

## 批次布局

- `N` 是展开后的环境 turn 行数，不等于 task 数、trajectory 数或 token 数。
- `L` 是回复的 padding 长度。mask 可以是非完整长度，但每行至少有1个有效 token。
- metadata 在 batch 重排后必须与对应 tensor 行同步重排。
- 轨迹 ID 必须跨任务唯一；每条轨迹 step_idx 连续从0开始。
- padding copy 必须与被复制行的 reward/state 保持一致。teacher 对这些复制行去重，
  但不改写原训练器的 batch 布局或 loss accumulation 规则。

## 两个控制量不是一个东西

`q` 改 Actor 的信用分配；`beta` 改 Critic 的学习目标。
`q=1` 关闭 Actor 的图步级项，但仍保留 outcome 和图辅助 critic loss。
`beta=0` 关闭 Critic 的图辅助 loss，但 Actor 仍可以使用图步级项。
默认两者为固定常数，不对其最优性或自适应性做额外承诺。

## 不在本次上传范围

Ray/FSDP/vLLM worker、模型/数据下载、ALFWorld/WebShop/ScienceWorld 环境适配、
服务器物理 GPU 映射、完整训练 YAML、数据划分、W&B、checkpoint 保留工具和终评
编排都需要另外的工程。不能仅凭本包中的数值示例声称端到端 benchmark 复现。
