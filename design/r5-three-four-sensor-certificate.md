# R5：三与四传感器最小观测配置实验计划

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-24
- Verification Status: VERIFIED — FOUR-SENSOR GLOBAL CERTIFICATE; THREE-SENSOR LOCAL CERTIFICATE
- Version Label: code_plan_v1

## Experiment Overview

- **Title**：固定总观测长度下的三与四传感器稳定重构
- **Objective**：判断 3 个或 4 个局部平均传感器能否替代已经通过的 5 传感器方案，并明确
  全局半离散证书与有限轨迹局部证书的边界。
- **Hypothesis**：3 个传感器受秩约束不能在最难参数上产生质量伴随全局证书，但一般
  \(B\) 可能得到低瞬态的有限轨迹收缩；4 个传感器是质量伴随全局证书可能成立的最小数量，
  但必须把区间放在低模态的有效观测位置。
- **Type**：deterministic matrix analysis + nonlinear simulation

## 数学问题

沿用研究计划

\[
\partial_t u=Au+F(u),\qquad y=\mathcal C u,
\]

\[
\partial_t\hat u=A\hat u+F(\hat u)+B(y-\mathcal C\hat u),
\qquad e=\hat u-u,
\]

以及由此得到的精确误差方程

\[
\partial_t e=Ae+F(u+e)-F(u)-B\mathcal C e.
\]

总观测长度固定为 0.20。若有 \(q\) 个传感器，每个区间宽度固定为 \(0.20/q\)，区间
不重叠。这样只改变传感器数量与位置，不增加总观测覆盖长度。

对于 \(B=\gamma\mathcal C^\top/h\)，Allen--Cahn 三次项给出

\[
\frac12\frac{\mathrm d}{\mathrm dt}\|e\|_h^2
\leq
\langle e,(\nu L_h+I-B\mathcal C)e\rangle_h.
\]

\(\nu=0.005\) 时 \(\nu L_h+I\) 有 4 个正方向。三个传感器产生的
\(B\mathcal C\) 秩不超过 3，因此三传感器质量伴随路线不可能使上式在所有方向严格为负。
四个传感器只满足必要秩条件，能否通过仍由区间位置决定。

## 冻结候选与选择规则

### 三传感器

每个区间宽度为 \(1/15\)，比较以下只由几何规则确定的中心：

1. cell-centered：\((1/6,1/2,5/6)\)；
2. quarter-centered：\((1/4,1/2,3/4)\)；
3. wide-interior：\((0.20,0.50,0.80)\)；
4. asymmetric-control：\((0.20,0.45,0.80)\)。

只使用 `n=31` 的线性矩阵选择位置，不查看任何 validation 或 test 轨迹。选择顺序为：

1. 三个 \(\nu\) 的全部不稳定模态时序可观测秩满秩；
2. 一般 \(B\) 的 LMI 初始化在三个 \(\nu\) 上都具有正线性收缩率；
3. 最小化 \(\nu=0.005\) 的有限时间瞬态放大，再比较 \(T\) 条件数和 \(B\) 范数。

### 四传感器

每个区间宽度为 0.05，比较：

1. cell-centered：\((0.125,0.375,0.625,0.875)\)；
2. interior-fifths：\((0.20,0.40,0.60,0.80)\)。

对 \(\gamma\in\{0.10,0.25,0.50,1,2,4,8\}\)，只用 `n=31` 的矩阵计算选择。先要求
三个 \(\nu\) 的全局半离散裕量都至少为 0.10，再选择满足条件的最小 \(\gamma\)；若多个
位置同时通过，选择最坏裕量更大的位置。之后冻结位置和增益。

## 验证设计

- 网格：`n in {31, 63, 127}`。
- 参数：`nu in {0.005, 0.010, 0.020}`。
- validation：每个组合 4 条冻结 pilot，加第 4 模态和接近最小观测方向的 2 条困难误差。
- test：在所有 validation 门通过后才解封，每个组合使用前 4 条冻结 test，并加入相同两类
  困难误差；test 不参与位置、增益或 \(B\) 的选择。
- 噪声：共同正弦测量噪声，幅值 0.01。
- 对照：原两传感器固定增益 0.10、已通过的五传感器方案。

三传感器除了在线 \(\|e\|_h\)，还使用 LMI 初始化对应的固定可逆 \(T\) 计算

\[
-\frac{\langle Te,\partial_t(Te)\rangle_h}{\|Te\|_h^2},
\]

并与研究计划中的 \(\lambda=0.1\nu\pi^2\) 比较。该结果只解释为声明轨迹上的有限样本局部
证书，不能替代三传感器已被秩结论排除的全局质量伴随证书。

## 成功与停止门

### 四传感器全局路线

1. 冻结位置和增益在 9 个网格/参数组合上的全局半离散裕量全部大于 0，且最坏值至少 0.10；
2. validation 终点误差中位数在 9 个组合上均不差于原两传感器固定增益；
3. 幅值 0.01 噪声下的终点误差中位数相对无噪声退化不超过 10%。

### 三传感器局部路线

1. 明确报告质量伴随全局证书在 \(\nu=0.005\) 上不可能，而不是把有限轨迹结果写成全局；
2. 一般 \(B\) 在 9 个组合上线性稳定，\(T\) 条件数不超过 4，瞬态放大不超过 2.5；
3. validation 所有采样点的直接收缩率均大于 \(\lambda=0.1\nu\pi^2\)；
4. validation 终点误差中位数均不差于原两传感器固定增益；
5. 噪声退化不超过 10%。

只有相应 validation 门全部通过，才运行该路线的 test。若三传感器只失败直接收缩门但在线
性能通过，才允许把受限 \(B+T_\phi\) 联合训练放到 2060；四传感器已有全局证书时不训练。

## Setup

- **Language/Framework**：Python、NumPy、SciPy、CVXPY；预期 CPU 即可。
- **Entry Command**：
  `python tool/r5_three_four_sensor_certificate.py --validation-limit-per-combination 4 --test-limit-per-combination 4 --output out/<new-run>/results.json`
- **Working Directory**：当前 experiment worktree。
- **Timeout**：30 分钟。
- **Monitoring**：进程存活、标准输出、结果 JSON 是否生成且非空。
- **Expected Output**：新的不可覆盖 JSON、结果报告、完整测试集通过、Git 提交且仓库干净。

## Analysis Plan

- **Primary metric**：四传感器最坏全局半离散裕量；三传感器最坏 validation 直接收缩率。
- **Secondary metrics**：终点/峰值 \(\|e\|_h\)、噪声退化、\(B\) 范数、\(T\) 条件数和瞬态放大。
- **Comparison**：原两传感器固定增益、五传感器全局证书、三/四传感器之间的证据等级。
