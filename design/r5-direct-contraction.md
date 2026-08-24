# R5：直接收缩路线

## Material Passport

- ID：R5-direct-contraction-2026-08-24
- 类型：Code Experiment Plan
- 状态：COMPLETED — NO POSITIVE VALIDATION MARGIN
- 前序结果：`report/r5-dynamics-defect-repair.md`

## 研究问题

研究计划定义

\[
z=T_\phi(e),
\qquad
\partial_t z=(A-\lambda I)z,
\qquad
\lambda>0.
\]

前序实验惩罚实际变换后动力学与该固定目标之间的全部向量差异，但某些差异只改变方向，
未必使 \(z\) 的能量增长。本路线直接检查

\[
-\frac{\langle T_\phi(e),\partial_tT_\phi(e)\rangle_{M_h}}
{\|T_\phi(e)\|_{M_h}^2}.
\]

该量为正，才表示当前样本上的变换后误差能量下降。为避免自创另一套目标，本路线沿用研究
计划中的 \(\lambda\)，固定 \(\lambda=0.1\nu\pi^2\)，并检查

\[
\langle T_\phi(e),\partial_tT_\phi(e)\rangle_{M_h}
\le -\lambda\|T_\phi(e)\|_{M_h}^2.
\]

## 第一阶段：现有 checkpoint 审计

先复放 `2026-08-24-r5-tk-structure-screen-gain-010-192-direct` 选中的 `n=31`、seed 501
checkpoint，不重新训练。分别审计固定增益训练轨迹、当前观测器训练轨迹、固定增益独立验证
轨迹、当前观测器独立验证轨迹，以及 12 条幅值 0.01 的共同正弦测量噪声验证轨迹。

每组报告最小值、5% 分位数、中位数、均值、95% 分位数、最大值和正值比例；独立验证还按
\(\nu\)、早中晚时间段和误差大小分组。有限样本最小值大于零记为“声明轨迹上的正裕量”，
但不写成连续状态域上的一致定理。

## 第二阶段：直接收缩训练

若现有 checkpoint 的当前观测器独立验证最小值不为正，则在原有
\(\mathcal L_{\mathrm{stable}}\)、动力学缺陷、双向范数约束和增益约束之外，加入上述能量不等式
的归一化 hinge 损失。固定 `n=31`、seed 501--503、80 epoch、每 20 epoch 刷新轨迹，比较
直接收缩损失权重 0.1、1 和 10；其余结构保持前序选中配置不变。

每个配置先通过可逆性和观测方向审计，再要求 12 条 validation rollout 的终点误差不差于
固定增益 0.10；随后最大化当前观测器独立验证轨迹上的最坏样本收缩率，并以 5% 分位数和
终点误差打破并列。test 不参与配置和 seed 选择。

## 完成判据

结果必须同时报告训练、独立验证、最坏样本、三个 \(\nu\) 分组、误差大小、测量噪声、在线
终点误差和可逆性审计。只有无噪声当前观测器独立验证的有限样本最小值为正，才能声称在
声明轨迹上得到正的局部稳定裕量；若连直接训练仍失败，则保留违反收缩条件的状态区域与
误差尺度，作为该路线的正式负结果。

本方案已执行完毕。现有 checkpoint 与三档直接收缩训练都未使无噪声当前观测器独立验证的
有限样本最小值变为正，正式结果见 `report/r5-direct-contraction.md`。
