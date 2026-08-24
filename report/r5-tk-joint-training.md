# R5：T--K 风格联合训练结果

## Material Passport

- ID：R5-tk-joint-2026-08-22
- 类型：Experiment Result
- 状态：SUPERSEDED
- 代码提交：89cbd4db636fc38cee66beddd3593e9e1c1b725f
- 执行节点：局域网直连 192.168.1.220，RTX 2060 6GB
- 原始输出：/home/wjj/work/wt/phd-project1-codex/project/project1/experiment/allen-cahn-certified-observer/out/2026-08-22-r5-tk-joint-192-direct/
- 运行退出码：0

> 后续审计发现本运行把 Allen--Cahn 反应项 \(+I\) 重复放入了目标生成元，导致
> \(A_{s,h}=\nu L_h+I-\lambda I\) 在当前参数范围内不保证稳定。本文件保留为历史运行记录；
> 修正后的正式结论见 `report/r5-normalized-refresh.md`。

## 联合目标

本次不再使用 hand-designed certificate target，而是让 \(T_{\phi,h}\) 与在线
\(\Gamma_{\theta,h}\) 共同进入同一个离散稳定目标：

\[
\mathcal L_{\mathrm{stable}}
=
\left\|
T_{\phi,h}(u_{j+1},e_{j+1})
-S_{\Delta t}T_{\phi,h}(u_j,e_j)
\right\|_{M_h}^{2},
\qquad
S_{\Delta t}=\exp(\Delta t A_{s,h}),
\]

其中 \(e_j=\hat u_j-u_j\)，\(e_{j+1}\) 由当前
\(\Gamma_{\theta,h}\) 的一步 observer 更新产生。一次反向传播同时更新
\(T_{\phi,h}\) 和 \(\Gamma_{\theta,h}\)。

设置为 \(\lambda/\rho_1=0.5\)，固定物理增益起点为 0.02，三个网格各使用 4 个 seed，
训练 150 epoch。

## 结果

| 网格 | 选中 seed | 稳定验证损失 | 终点误差 | 噪声终点误差 |
|---:|---:|---:|---:|---:|
| 31 | 501 | \(1.51\times10^{-6}\) | 0.1084 | 0.0972 |
| 63 | 504 | \(1.76\times10^{-6}\) | 0.1105 | 0.0963 |
| 127 | 503 | \(1.84\times10^{-6}\) | 0.1166 | 0.1020 |

所有 12 个 seed 的训练/验证损失都是有限数，3 个网格、48 个 test case 和 12 个噪声
case 均完成。

## 约束与可逆性

结构化参数化仍然满足：

\[
T_{\phi,h}(u,0)=0,
\qquad
C_hT_{\phi,h}(u,e)=C_he
\]

数值残差约为 \(10^{-9}\) 或更小。

但是 Jacobian 奇异值范围变为：

- \(n=31\)：约 \([0.508,1.753]\)；
- \(n=63\)：约 \([0.352,1.727]\)；
- \(n=127\)：约 \([0.367,1.690]\)。

因此当前联合损失虽然降低了 \(\mathcal L_{\mathrm{stable}}\)，但没有自动给出足够好的
双 Lipschitz 或局部可逆性余量。

## 判断

这次运行回答了“是否真的联合训练”这个问题：答案是是的。与旧 R5-E 不同，当前
\(T_{\phi,h}\) 和 \(\Gamma_{\theta,h}\) 共享 \(\mathcal L_{\mathrm{stable}}\) 的计算图。

但科学结果是负的/不完整的：

1. 稳定一步残差下降，不等于 \(T_{\phi,h}\) 在误差方向上保持良好可逆性；
2. 当前联合模型的在线误差高于旧 R5-E 消融中的完整模型；
3. 因此还不能报告 \(\beta_h^{\mathrm{stab}}>0\)，也不能把这次结果称为条件稳定证书。

下一步必须在保持 T--K 联合目标的同时加入冻结方案要求的双 Lipschitz/可逆性控制，并
重新做跨网格独立复核。不能只继续增加训练 epoch。
