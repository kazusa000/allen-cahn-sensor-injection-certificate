# R5：稳定目标修正、归一化损失与当前观测器轨迹刷新

## Material Passport

- ID：R5-normalized-refresh-2026-08-22
- 类型：Experiment Result
- 状态：VERIFIED
- 代码提交：34e453896de926d758dbaa3f09910d8ab556f109
- 执行节点：局域网直连 192.168.1.220，RTX 2060 6GB
- 筛选输出：`/home/wjj/work/wt/phd-project1-codex/project/project1/experiment/allen-cahn-certified-observer/out/2026-08-22-r5-normalized-refresh-screen-192-direct/`
- 正式输出：`/home/wjj/work/wt/phd-project1-codex/project/project1/experiment/allen-cahn-certified-observer/out/2026-08-22-r5-normalized-refresh-formal-192-direct/`
- 正式运行退出码：0

## 本轮修正

研究计划把 Allen--Cahn 写成

\[
\partial_tu=Au+F(u),
\]

其中 \(A\) 是线性扩散部分，\(F\) 包含反应项。此前 R5 联合训练错误地使用

\[
A_{s,h}=\nu L_h+I-\lambda I,
\]

把反应项 \(+u\) 又放进目标生成元。在当前 \(\nu\) 范围下，该目标的第一模态可能增长，
因此不符合研究计划中的稳定目标。本轮修正为

\[
A_{s,h}=\nu L_h-\lambda I,
\qquad
\lambda/\rho_1\in\{0.1,0.5,1.0\},
\qquad
\rho_1=\nu\pi^2.
\]

本轮同时把离散稳定损失改为

\[
\widetilde{\mathcal L}_{\mathrm{stable}}
=\frac1N\sum_{j=1}^N
\frac{
\|T_{\phi,h}(u_{j+1},e_{j+1})
-S_{\Delta t}T_{\phi,h}(u_j,e_j)\|_{M_h}^2
}{
\Delta t^2(\|e_j\|_{M_h}^2+10^{-8})
},
\]

使其与按 \(\|e\|_{M_h}^2\) 归一化的动力学缺陷处于可比较尺度。每隔 50 epoch 用
当前 \(\Gamma_{\theta,h}\) 重新运行 96 条训练 observer 轨迹；seed 先通过 certificate
审计，再按 48 条独立 validation rollout 的终点误差选择。

## 预注册筛选

筛选固定为 \(n=31\)、seed 501--502、60 epoch、每 20 epoch 刷新，比较

\[
\lambda/\rho_1\in\{0.1,0.5,1.0\},
\qquad
\omega_{\mathrm{defect}}\in\{0.1,1.0\}.
\]

六组配置全部有限且通过可逆性审计。按 12 条 validation rollout 选中

\[
\lambda/\rho_1=0.1,
\qquad
\omega_{\mathrm{defect}}=0.1.
\]

该配置的 validation 终点误差为 0.05982，同批固定增益 0.10 为 0.07078；筛选 test
不参与超参数选择。

## 三网格正式结果

正式运行使用三个网格、四个 seed、150 epoch；每个 seed 在 epoch 50 和 100 各刷新一次
训练轨迹。每个网格使用全部 48 条 validation rollout 选择 seed，再独立运行 48 条 test
和 12 条共同正弦噪声 test。

| 网格 | seed | 归一化稳定损失 | 原始稳定损失 | 动力学缺陷 | validation | 固定增益 validation | test | 噪声 test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 31 | 502 | 0.4678 | \(2.30\times10^{-6}\) | 0.5382 | 0.06965 | 0.07196 | 0.07555 | 0.06873 |
| 63 | 504 | 0.5382 | \(3.23\times10^{-6}\) | 0.6095 | 0.06788 | 0.07186 | 0.07611 | 0.06610 |
| 127 | 504 | 0.5064 | \(2.56\times10^{-6}\) | 0.6834 | 0.07055 | 0.07183 | 0.08424 | 0.07243 |

所有 12 个 seed 的训练/验证数值均有限；每个 seed 完成两次轨迹刷新。三张网格的
certificate 审计为：

| 网格 | Jacobian 最小奇异值 | Jacobian 最大奇异值 | 零误差纤维残差 | 最大方向残差 |
|---:|---:|---:|---:|---:|
| 31 | 0.791 | 1.387 | 0 | 小于 \(10^{-7}\) |
| 63 | 0.677 | 1.042 | 0 | 小于 \(10^{-7}\) |
| 127 | 0.985 | 1.095 | 0 | 小于 \(10^{-7}\) |

## 与此前结果比较

| 网格 | 旧 R5-E 完整模型 | 上一轮三项联合训练 | 本轮 | 相对旧 R5-E 改善 |
|---:|---:|---:|---:|---:|
| 31 | 0.08861 | 0.10928 | 0.07555 | 14.7% |
| 63 | 0.09682 | 0.11194 | 0.07611 | 21.4% |
| 127 | 0.08984 | 0.10401 | 0.08424 | 6.2% |

旧固定增益 0.10 的同一无噪声 test 指标为 0.08649/0.08635/0.08632。本轮在三张网格上
分别为 0.07555/0.07611/0.08424，因此第一次在全部三个无噪声测试网格上同时超过该
固定增益基线。

## 结论边界

稳定目标修正、损失归一化、当前观测器轨迹刷新和 validation rollout 选模共同解决了此前
“训练损失很小但在线误差变差”的问题；有限样本上的局部可逆性也继续通过。

但动力学缺陷仍为 0.538/0.609/0.683，没有达到可用于条件稳定界的小量级，并且 n=127
噪声 test 没有形成同样清楚的优势。因此当前证据支持“无噪声在线重构跨网格改善”，不支持
\(\beta_h^{\mathrm{stab}}>0\) 或统一噪声稳定证书。下一步若继续 R5，应直接估计
\(\varepsilon_{T,h}\)、\(K_h\)、\(\alpha_h\) 和 \(m_h\)，而不是继续用训练 epoch
代替证书判据。
