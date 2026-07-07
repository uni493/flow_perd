# GRM 与 Bid2X 流量预估技术洞察

资料核验日期：2026-07-07
调研对象：

| 论文 | 链接 | 与流量预估的关系 |
| --- | --- | --- |
| Constrained Auto-Bidding via Generative Response Modeling | [arXiv:2605.27811](https://arxiv.org/abs/2605.27811) / [PDF](https://arxiv.org/pdf/2605.27811) | 显式预测未来竞价机会量 `I_{t:T}`，并同时预测出价倍率下的 cost/value 响应曲线 |
| Bid2X: Revealing Dynamics of Bidding Environment in Online Advertising from A Foundation Model Lens | [arXiv:2510.23410](https://arxiv.org/abs/2510.23410) / [PDF](https://arxiv.org/pdf/2510.23410) | 给定下一时刻 bid，预测 cost/reward/count 等竞价环境结果；其中 count/PV/BuyCnt 可理解为“出价条件下的可赢得流量” |

## 0. 结论先行

这两篇论文都不把广告流量预估做成一个孤立的日级时间序列问题。它们的共同视角是：

> 流量不是只随时间自然变化，也会被 bid、预算、CPA/ROI 约束、pacing 状态和竞争环境共同决定。因此，流量预估应该是“状态 + 决策条件下的响应建模”。

具体差异如下：

| 维度 | GRM | Bid2X |
| --- | --- | --- |
| 预测对象 | 剩余 horizon 的总竞价机会量 `I_{t:T}`，以及 `C(alpha)`、`V(alpha)` 响应曲线 | 下一 tick 的 cost、reward、count；并用辅助任务预测从当前 tick 到结束的累计目标变量 |
| “流量”定义 | 未来 bid opportunities，总供给/机会量，更接近自然剩余流量 | count / PV / BuyCnt，更接近给定 bid 后可赢得的受控流量 |
| 是否显式预测流量 | 是，独立 traffic head | 否，没有单独 traffic head；count 是目标变量之一 |
| 是否考虑出价影响 | traffic head 本身不随 alpha 变，但 cost/value 总量通过 `I_hat * response_curve(alpha)` 随 alpha 变 | 强依赖 bid，模型输入包含下一时刻 bid |
| 时间粒度 | 当前 tick 到全日/全周期结束的 horizon aggregate | next-token / next-tick，同时有 cumulative future auxiliary task |
| 训练监督 | logged future ticks 的 future-sampling，traffic 用 log loss | 自监督 next-token prediction，零膨胀分类 + 回归，多目标联合训练 |
| 工程价值 | 适合 MPC / pacing：直接给剩余机会量和约束可行边界 | 适合 bidding foundation model：统一建模不同场景下 bid -> outcome 的动态关系 |

对当前流量预估系统最关键的启发：

1. 要拆分“自然剩余机会量”和“出价后可赢得流量”。前者像 GRM 的 `I_{t:T}`，后者像 Bid2X 的 `count/PV/BuyCnt`。
2. 只预测全日总量不够，下游 MPC 更需要“当前时刻之后的剩余 horizon 预测”。
3. 如果流量会被出价策略反向影响，模型应把 bid/alpha、预算剩余、CPA/ROI slack、历史 cost/value/count 一起作为条件。
4. 对稀疏任务、低流量任务，需要零膨胀处理：先判断是否非零，再预测非零幅度。
5. 评估不能只看 WAPE/MAE，还要看下游 pacing replay：预算是否花完、CPA/ROI 是否越界、是否因为高估剩余流量导致出价过保守。

## 1. GRM 是怎么做流量预估的

### 1.1 问题定义

GRM 把 auto-bidding 拆成两层：

```text
历史状态 H_t
  -> GRM 预测未来响应 bundle
  -> analytic controller 解预算/CPA 约束
  -> 输出当前 tick 的 alpha
```

其中历史状态包含：

```text
H_t = (
  s_{1:t},          # 当前 tick 之前的上下文/状态序列
  alpha_{1:t-1},   # 已执行的 pacing multiplier 序列
  I_{1:t-1},       # 已发生的竞价机会量
  Cost_{<t},
  Val_{<t}
)
```

论文里最重要的流量变量是：

```text
I_{t:T} = sum_{k=t}^{T} I_k
```

也就是从当前 tick `t` 到 horizon 结束的未来 bid opportunities 总数。这个变量更接近“剩余自然机会量”，不是展示量，也不是赢得量。

### 1.2 模型输出

GRM 不预测每个未来 tick 的流量序列，而是预测一个 horizon-aggregate response bundle：

```text
R_hat_{t:T} = (
  I_hat_{t:T},
  C_bar_hat_{t:T}(alpha),
  V_bar_hat_{t:T}(alpha)
)
```

其中：

```text
C_total_hat(alpha) = I_hat_{t:T} * C_bar_hat_{t:T}(alpha)
V_total_hat(alpha) = I_hat_{t:T} * V_bar_hat_{t:T}(alpha)
```

也就是说，GRM 把剩余总 cost/value 拆成：

```text
剩余总量 = 未来机会量 * 每机会响应曲线
```

这是一个非常值得借鉴的建模分解。对流量预估来说，它避免了把“未来有没有机会”和“给定出价能不能买到”混在一个 label 里。

### 1.3 响应曲线参数化

GRM 用 causal Transformer 编码历史，再用 MLP 输出 7 个参数：

```text
(
  I_hat_{t:T},
  theta_C = (a_C, b_C, c_C),
  theta_V = (a_V, b_V, c_V)
)
```

其中 cost/value 曲线采用 log-sigmoid 形状：

```text
C_bar_hat(alpha) = a_C * Phi_tilde(b_C, c_C, alpha)
V_bar_hat(alpha) = a_V * Phi_tilde(b_V, c_V, alpha)
```

这个设计的含义是：

- `alpha = 0` 时没有竞价胜出，cost/value 接近 0。
- `alpha` 变大后，cost/value 上升但逐渐饱和。
- 用 `log(alpha)` 捕捉竞价中的边际收益递减。
- 用 softplus 保证 scale/sensitivity 为正，从结构上保证曲线单调。

### 1.4 训练方式

论文没有要求反事实完整观测，而是使用日志数据里的已执行 multiplier：

```text
logged data per tick:
  alpha_k, I_k, Cost_k, Val_k
```

对每个 anchor tick `t`，从未来区间 `[t, T]` 里采样若干 tick：

```text
k_m ~ Uniform({t, ..., T})
```

然后用这些未来 tick 的真实记录监督当前 `t` 的 horizon aggregate 曲线。cost/value 目标被转换为 per-opportunity：

```text
C_k(alpha_k) ~= Cost_k / I_k
V_k(alpha_k) ~= Val_k / I_k
```

训练损失核心是：

```text
L = traffic_weighted_curve_loss
  + lambda_I * (log I_hat_{t:T} - log I_{t:T})^2
```

关键点：

- cost/value 曲线损失按 `I_k` 加权，让高流量 tick 对 aggregate curve 拟合影响更大。
- traffic head 用 log-scale loss，缓解广告流量重尾。
- 只预测一个 horizon aggregate curve，降低输出维度，牺牲一点 per-tick 细节换稳定性。

### 1.5 下游如何使用流量预估

GRM 的流量预测不是为了报表，而是直接进入约束求解。

在当前 tick，模型先得到：

```text
I_hat_{t:T}
C_bar_hat(alpha)
V_bar_hat(alpha)
```

然后计算：

```text
C_total_hat(alpha) = I_hat_{t:T} * C_bar_hat(alpha)
V_total_hat(alpha) = I_hat_{t:T} * V_bar_hat(alpha)
```

controller 解两个一维方程：

```text
Budget pacing:
C_total_hat(alpha_B) = remaining_budget

CPA pacing:
C_total_hat(alpha_C) - tau * V_total_hat(alpha_C) = CPA_slack

execute:
alpha_t = min(alpha_B, alpha_C)
```

所以 `I_hat_{t:T}` 如果高估，系统会误以为后面机会充足，当前可能出价偏保守；如果低估，系统会认为后面机会不足，当前可能出价过激。论文还把 constraint violation bound 写成 traffic error 和 curve error 的函数，说明流量误差会直接传导到预算/CPA 约束安全性。

### 1.6 对工程落地的洞察

GRM 最适合借鉴的不是“具体 Transformer”，而是这三个设计：

1. **预测剩余 horizon，而不是只预测下一点。**
   MPC/pacing 需要知道从现在到结束还能有多少机会。

2. **把机会量和响应曲线拆开。**
   `I_hat_remaining` 表示市场/任务还有多少机会，`response(alpha)` 表示出价倍率下能消耗多少、拿到多少价值。

3. **用当前日已发生状态反复重规划。**
   每个 tick 重新预测、重新解约束，能让 traffic error 的影响被后续反馈校正。

## 2. Bid2X 是怎么做流量预估的

### 2.1 Bid2X 的问题不是传统 traffic forecasting

Bid2X 的核心任务是 bidding environment modeling：

```text
given:
  historical complete trajectory
  today's incomplete trajectory until t
  campaign context / constraints
  next bid b_{t+1}

predict:
  y_{t+1} = (cost_{t+1}, reward_{t+1}, count_{t+1})
```

这里的 `count` 是 winning impressions count，也就是在给定 bid 和环境下能赢到多少曝光机会。论文摘要和线上实验还提到 PV、BuyCnt 等指标。因此，Bid2X 的“流量”更准确地说是：

```text
bid-conditioned achieved traffic
```

它不是自然供给量 `request_cnt`，而是受 bid、预算、竞争和系统策略影响后的结果量。

### 2.2 输入数据结构

Bid2X 把一个 campaign 的 bidding trajectory 表示为记录序列：

```text
x = (b, c, r, ct, t)
```

含义：

- `b`: bid 或 bidding parameter
- `c`: 从当前 timestamp 到下次 bid 调整之间的累计 cost
- `r`: 同一区间的累计 reward
- `ct`: 同一区间的 winning impression count
- `t`: tick time

它利用两类轨迹：

```text
X^{tau-1}: 历史完整轨迹，用来学习变量关系
X^{tau}:   当天截至当前 tick 的不完整轨迹，用来学习日内动态
```

### 2.3 表征方式

Bid2X 的结构可以概括为：

```text
heterogeneous data
  -> uniform series embeddings
  -> variable attention encoder
  -> temporal causal attention decoder
  -> variable-aware fusion
  -> zero-inflated projection
  -> cost/reward/count prediction
```

它的关键设计是把不同信息放到不同 attention 视角里：

| 模块 | 作用 |
| --- | --- |
| Historical data embedding | 把上一日完整轨迹中每个变量序列编码成 variable token |
| Today's data embedding | 把当天每个 time slot 的所有变量值编码成 temporal token |
| Variable attention | 学习 bid、cost、reward、count 等变量之间的相关关系 |
| Temporal causal attention | 学习当天序列动态，只看过去不看未来，避免信息泄漏 |
| Variable-aware fusion | 针对每个目标变量融合变量视角和时间视角 |
| Zero-inflated projection | 处理大量 0 值：先估计非零概率，再预测数值幅度 |

### 2.4 防止信息泄漏

Bid2X 在当天轨迹 embedding 时，把 control variables 和 target variables 分开处理：

```text
control variables: bid, tick
target variables: cost, reward, count
```

为了预测 `t+1`，它会：

- 让目标变量序列右移，用 zero vector 作为 start token。
- 把未来 bid 信息放到 control series 的末尾。
- 用 causal mask 保证 temporal attention 只能看过去。

这对流量预估很重要：如果我们要预测当前时刻之后的剩余流量，就必须确保任何累计量、当日全量、未来切片都没有泄漏到输入。

### 2.5 零膨胀预测头

广告竞价数据有大量 0：

- 出价没有赢到曝光。
- 小预算/低频 campaign 在某些 tick 没有流量。
- 转化或 GMV 类 reward 更稀疏。

Bid2X 对每个目标变量先预测非零概率：

```text
p_i = Pr(y_i != 0)
```

再预测非零幅度：

```text
y_tilde_i = MLP(H_i)
```

最终输出：

```text
y_hat_i = p_i * y_tilde_i
```

训练目标是 classification + regression 的联合 loss。这个思路对 `req_cnt`、`win_cnt`、`pv` 等低量级任务很实用：不要只用 MSE/MAE 硬拟合所有点，否则模型容易在“是否有量”和“有量时多少”之间互相拖累。

### 2.6 累计预测辅助任务

Bid2X 还加了一个 self-supervised auxiliary task：

```text
predict cumulative value from current time slot to campaign end
```

这对“剩余流量预估”很有直接参考价值。主任务预测 next tick，辅助任务提供全局 horizon 视角，避免模型只学短期局部波动。

如果把 target variable 换成 `count` 或 `pv`，这个辅助任务本质上就是：

```text
从当前 tick 到结束的剩余可赢得流量预测
```

### 2.7 对工程落地的洞察

Bid2X 最值得借鉴的是这四点：

1. **把 bid 作为条件输入。**
   如果预测对象是 win_cnt、pv、cost、GMV，它们不是纯自然流量，而是 action-conditioned outcome。

2. **历史完整轨迹 + 当天不完整轨迹分开建模。**
   历史用于学习变量关系，当天用于捕捉日内动态。

3. **同时预测 next tick 和 remaining cumulative。**
   next tick 服务短周期控制，remaining cumulative 服务 pacing/MPC。

4. **零膨胀头适合长尾 campaign。**
   对大量 0 流量任务，比单一回归头更稳定。

## 3. 两篇论文对比：自然流量 vs 受控流量

广告系统里“流量”至少有三层：

```text
1. natural supply / bid opportunity
   平台自然产生的竞价机会，例如 request_cnt、bid opportunity

2. reachable / eligible traffic
   满足定向、预算、审核、频控等条件后可参与的机会

3. achieved / won traffic
   在给定 bid 和竞争环境下实际赢得的曝光、PV、BuyCnt
```

GRM 更接近第 1 层：

```text
I_hat_{t:T} = future bid opportunities
```

Bid2X 更接近第 3 层：

```text
count_hat_{t+1} = f(bid_{t+1}, history, context)
```

这意味着，落地时不能直接说“Bid2X 做了自然流量预测”。它做的是竞价环境响应预测，里面的 count/PV 是出价条件下的结果量。

## 4. 对当前流量预估系统的建议架构

如果目标是服务广告 MPC / pacing，建议把模型输出拆成三类：

```text
自然剩余机会量:
  I_hat_remaining

受控响应曲线:
  win_cnt_hat(alpha)
  cost_hat(alpha)
  value_hat(alpha)

校正状态:
  intraday_correction_factor
  uncertainty / quantile
```

一个可落地的版本如下：

```text
features:
  campaign_id / task_id embedding
  date features: weekday, holiday, promotion flag
  time_slice
  historical daily total and intraday shape
  current-day cumulative req/win/cost/value
  bid / alpha / target CPA or ROI
  remaining budget
  CPA/ROI slack
  recent pacing state

heads:
  log_I_remaining_head
  next_slice_req_head
  next_slice_win_count_head
  cumulative_remaining_win_count_head
  cost/value response curve heads
  nonzero probability heads for sparse targets
```

### 4.1 MVP 路径

可以按三阶段实现：

| 阶段 | 模型 | 作用 |
| --- | --- | --- |
| V0 | 现有全日总量预测 + 当日累计校正 | 保留简单稳定的 baseline |
| V1 | GRM-like remaining traffic head | 直接预测 `I_remaining`，用 log loss，输入当前日累计状态 |
| V2 | Bid2X-like action-conditioned outcome model | 输入 bid/alpha，预测 next slice 和 remaining cumulative 的 win_cnt/cost/value |

推荐先做 V1，因为它对当前剩余流量预估最直接；再做 V2，把流量预测接入 MPC 出价决策。

### 4.2 Label 设计

自然剩余机会量：

```text
I_remaining_{i,d,k} = sum_{t=k+1}^{47} req_cnt_{i,d,t}
```

受控剩余赢量：

```text
Win_remaining_{i,d,k} = sum_{t=k+1}^{47} win_cnt_{i,d,t}
```

下一切片响应：

```text
y_{i,d,k+1} = (
  req_cnt,
  win_cnt,
  cost,
  value
)
```

如果 `req_cnt` 是平台自然请求量，不要把 bid/alpha 作为强因果输入；如果 `win_cnt/pv/cost/value` 是受控结果量，就必须输入 bid/alpha、预算和 pacing 状态。

### 4.3 Loss 设计

可以组合：

```text
L =
  w_1 * MSE(log(I_remaining_hat + 1), log(I_remaining + 1))
  + w_2 * WAPE_loss(next_slice_req)
  + w_3 * zero_inflated_loss(win_cnt)
  + w_4 * response_curve_loss(cost/value)
  + w_5 * cumulative_remaining_loss
```

注意：

- 对重尾流量用 log loss 或相对误差更稳。
- 对稀疏 win_cnt/PV 用 zero-inflated head。
- 对下游 MPC 重要的高预算、高消耗、高价值 campaign 增加样本权重。
- 对 `time_slice` 越靠近当前、越靠近预算耗尽边界的样本，可单独加权。

### 4.4 评估设计

不要只看全局 MAE/WAPE，至少分四组评估：

| 评估面 | 指标 |
| --- | --- |
| 点预测 | MAE、RMSE、WAPE、SMAPE |
| 分桶稳定性 | 按流量规模、预算规模、time_slice、行业/定向分桶 |
| 剩余流量 | remaining WAPE，尤其是中午/下午后半段 |
| 下游决策 | 预算花费率、CPA/ROI 越界率、欠投率、过早消耗率 |

GRM 的思想提醒我们：真正重要的是 prediction error 如何传导到 constraint violation。Bid2X 的思想提醒我们：如果模型预测的是受控流量，就要在不同 bid/alpha 下评估响应是否单调、是否符合边际递减。

## 5. 风险与注意事项

1. **Bid2X 的 count 不是自然供给。**
   如果业务目标是预测平台请求量或可竞价机会，不能直接把 Bid2X 的 count 当成 supply forecast。

2. **GRM 的响应曲线不是严格因果曲线。**
   它依赖日志里已有的 alpha 覆盖。如果历史策略从未探索某些 alpha 区间，曲线外推会有风险。

3. **仅用日级全量训练会掩盖日内控制问题。**
   MPC 需要的是每个 time_slice 的 remaining forecast，而不是只要当天总量准确。

4. **预算和 bid 会污染观测流量。**
   如果 observed flow 是 win_cnt/PV/cost，它已经被策略控制过。训练时要明确它是 outcome，不是 supply。

5. **冷启动需要 metadata。**
   Bid2X 依赖历史轨迹，冷启动 campaign 需要行业、预算、定向、商品、广告主等上下文 embedding 补足。

## 6. 最可迁移的技术洞察

可以把两篇论文合成一句工程原则：

> 面向广告 MPC 的流量预估，不应只预测“未来有多少量”，而应预测“在当前状态和候选出价策略下，未来还有多少可用机会、能赢多少、会花多少钱、能产生多少价值”。

对应到模型接口，建议从：

```text
predict(task_id, date) -> total_flow
```

升级为：

```text
predict(task_id, date, time_slice, state, action_context) -> {
  natural_remaining_opportunity,
  achieved_remaining_count,
  next_slice_count,
  cost_response_curve,
  value_response_curve,
  uncertainty
}
```

这里 `state` 包含当日累计观测、预算剩余、CPA/ROI slack、历史日内形状；`action_context` 包含 bid/alpha 或候选策略。这样模型输出才能真正服务 pacing、预算控制和 ROI/CPA 约束，而不是只做离线报表预测。
