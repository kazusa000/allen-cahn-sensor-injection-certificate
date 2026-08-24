# R5：观测注入修复与收缩证书

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-24
- Verification Status: VERIFIED — FIVE-SENSOR CERTIFICATE PASSED; TWO-SENSOR TRAINING FAILED STOP GATE
- Version Label: code_plan_v1

## 实验问题

研究计划给出

\[
\partial_t u=Au+F(u),\qquad y=\mathcal C u,
\]

\[
\partial_t\hat u=A\hat u+F(\hat u)+B(y-\mathcal C\hat u),
\qquad e=\hat u-u.
\]

因此本实验采用的精确误差方程是

\[
\partial_t e=Ae+F(u+e)-F(u)-B\mathcal C e.
\]

前四条 R5 路线改变了目标、损失或证书区域，但都把 (B) 限制为两个局部平均传感器的
正质量伴随注入。对 Allen--Cahn 线性化

\[
A=\nu L_h,\qquad F'(0)=I,
\]

该限制只能提供秩不超过 2 的对称阻尼，而 (\nu=0.005) 和 (0.01) 分别存在 4 个和 3 个
不稳定模态。本实验检验以下修复是否能在训练前恢复可稳定性：

1. 用五个均匀分布、总观测长度不变的固定局部平均传感器建立可证基准；
2. 保留原两个传感器，但把 (B) 改为作用于全部不稳定模态的斜投影输出注入；
3. 只在上述线性门通过后，才允许训练 (T_\phi) 或 (B) 的有界残差。

本方案是前序只读诊断后的证实性实验，不宣称结果盲化或预注册。有限维输出覆盖不稳定自由度
的思路参照 Azouani--Titi 的有限决定参数反馈，局部平均测量的斜投影观测器参照 Rodrigues，
低模态注入与快模态解析拆分参照 modal LMI 路线。

## 假设

### H1：五传感器可证基准

固定

\[
[0.08,0.12],\ [0.28,0.32],\ [0.48,0.52],\ [0.68,0.72],\ [0.88,0.92],
\]

并取 (B=\mathcal C^\top/h)。由于三次项满足

\[
\langle e,F(u+e)-F(u)\rangle_h\leq\|e\|_h^2,
\]

若

\[
\lambda_{\max}\!\left(\nu L_h+I-\mathcal C^\top\mathcal C/h\right)<0,
\]

则得到全局半离散误差收缩。要求 3 个网格与 3 个 (\nu) 的 9 个组合全部具有严格正裕度。

### H2：原两个传感器可由斜投影注入稳定

保留

\[
[0.20,0.30],\qquad[0.65,0.75].
\]

对每个网格和 (\nu)，先提取 (\nu L_h+I) 的全部正特征值方向。要求这些方向在当前
(\mathcal C) 下具有满时序可观测秩。随后令 (B) 的像空间覆盖这些方向，比较极点配置和
Riccati 型初始化，并检查

\[
\max\operatorname{Re}\sigma(A+I-B\mathcal C)<0.
\]

裸极点配置只证明可稳定性，不构成最终成功；还必须报告有限时间瞬态放大、(B) 的谱范数、
噪声响应以及由

\[
(A+I-B\mathcal C)^\top T_\phi^\top T_\phi
+T_\phi^\top T_\phi(A+I-B\mathcal C)
\preceq-2\lambda T_\phi^\top T_\phi
\]

得到的正收缩裕度与条件数。

### H3：训练只能改善可行初始化

如果 H2 的线性门通过但非线性在线误差或噪声响应不通过，则在 2060 上训练围绕已证 (B)
的有界残差和可逆 (T_\phi)。主要稳定损失为

\[
\mathcal L_{\mathrm{stable}}
=\left(\max\left\{0,
\langle z,\partial_tz\rangle+\lambda\|z\|^2
\right\}\right)^2,
\qquad z=T_\phi e.
\]

固定目标动力学缺陷保留为辅助诊断，不再作为唯一通过条件。

## 设置

- 模型、网格、参数、时间范围与噪声：沿用 `design/r5-formal-contract.md`。
- 五传感器总观测长度为 0.20，与原两个宽度 0.10 的传感器总长度相同。
- 线性门覆盖 `n in {31, 63, 127}` 与 `nu in {0.005, 0.010, 0.020}`。
- 非线性验证补入第 4 个误差模态，并加入接近 (\mathcal C e=0) 的困难方向；不得继续只用
  原前三模态覆盖 (\nu=0.005)。
- CPU 分析使用 NumPy/SciPy；只有 H1、H2 和本地 smoke 通过后才允许 2060 多 seed 训练。

## 执行阶段

### 阶段 A：结构反例与五传感器证书

实现可复用的线性化审计，报告不稳定模态数、当前正伴随注入的秩上限、五传感器最坏特征值
和收缩裕度。该阶段是解析结论的矩阵核验，不依赖训练数据。

### 阶段 B：两传感器斜投影 (B)

实现完整矩阵输出注入观测器与低模态设计。对极点配置和一组冻结的 Riccati 权重做 CPU
筛选；选择规则依次为：闭环严格稳定、度量收缩裕度为正、有限时间瞬态较小、增益较小。
不得按 test 在线误差反向选择配置。

### 阶段 C：非线性与噪声验证

对五传感器基准和阶段 B 选中的两传感器注入运行配对轨迹。报告终点误差、峰值误差、相对
开环/原固定增益的变化和幅值 0.01 的共同正弦测量噪声结果。线性稳定但非线性退化必须明确
报告，不能以特征值替代在线验证。

### 阶段 D：联合训练门

只有阶段 B、C 同时通过才直接进入正式多网格验证；若 B 通过而 C 失败，则实现受约束的
(B+T_\phi) 残差训练并放到 2060。若 B 本身失败，则停止训练并返回传感器重新布置。

CPU 正式门已经表明，仅 (\nu=0.005) 的在线误差退化；把目标衰减率从约 0.005 增加到
0.4 会继续放大瞬态，不能作为修复。联合训练因此冻结为 `n=31` 的 validation-only 筛选：
seed 501--503、80 epoch、每 20 epoch 刷新当前观测器轨迹，比较固定 LMI (T_\phi) 且
(B) 残差信赖域 0.25、联合 (T_\phi) 残差且信赖域 0.25、联合残差且信赖域 0.50 三项。
主要收缩损失、稳定损失、辅助动力学缺陷、可逆性和增益正则权重依次为 10、1、0.05、1、
0.1。筛选阶段不计算 test；先要求 validation 最坏收缩率为正且 (\nu=0.005) 终点误差不差于
固定增益 0.10，再按最坏收缩率、5% 分位数和终点误差选择。只有通过才扩展三网格并解封 test。

## 输出与命令

- 实现：`src/allen_cahn_certified_observer/observer_design.py` 与通用输出注入观测器。
- 单元测试：`tests/test_observer_design.py` 及现有测试集。
- CPU 入口：`tool/r5_oblique_injection_certificate.py`。
- 原始输出：新的 `out/<时间>-r5-oblique-injection-certificate/`，禁止覆盖。
- 结论：`report/r5-oblique-injection-certificate.md`。
- 测试命令：`python3 -m pytest -q`。
- 实验命令：
  `python3 tool/r5_oblique_injection_certificate.py --output out/<新目录>/results.json`。
- CPU 硬超时：30 分钟；监控进程存活、标准输出与 `results.json` 是否生成。

## 完成标准

1. 9 个五传感器组合全部具有严格正的全局半离散收缩裕度，并保存可复算矩阵指标。
2. 9 个两传感器组合全部报告不稳定模态数、时序可观测秩、闭环谱界、增益、瞬态和度量条件数。
3. 非线性与噪声验证包含第 4 不稳定模态和观测近零方向，不使用 test 选择设计。
4. 明确分类：可证基准通过；两传感器路线通过、需要联合训练，或结构上失败。
5. 正式输出不可覆盖、测试通过、报告与变更记录完成、实验仓库提交且干净。
