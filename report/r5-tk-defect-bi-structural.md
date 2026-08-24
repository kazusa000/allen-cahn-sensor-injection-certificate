# R5：稳定损失、动力学缺陷与可逆性约束联合训练

## Material Passport

- ID：R5-tk-defect-bi-structural-2026-08-22
- 类型：Experiment Result
- 状态：SUPERSEDED
- 代码提交：b5f66f039888435cd5089b3ad642fc8c7d7cfb68
- 执行节点：局域网直连 192.168.1.220，RTX 2060 6GB
- 原始输出：`/home/wjj/work/wt/phd-project1-codex/project/project1/experiment/allen-cahn-certified-observer/out/2026-08-22-r5-tk-defect-bi-structural-192-direct/`
- 本地复核输出：`out/2026-08-22-r5-tk-defect-bi-structural-192-direct/results.json`
- 运行退出码：0

> 后续审计发现本运行把 Allen--Cahn 反应项 \(+I\) 重复放入了目标生成元，导致
> \(A_{s,h}=\nu L_h+I-\lambda I\) 在当前参数范围内不保证稳定。本文件保留为历史运行记录；
> 修正后的正式结论见 `report/r5-normalized-refresh.md`。

## 训练目标

当前 R5 同时训练在线增益 \(\Gamma_{\theta,h}\) 和离线 certificate
\(T_{\phi,h}\)，总损失为

\[
\mathcal L_{\mathrm{joint}}
=\mathcal L_{\mathrm{stable}}
+\omega_{\mathrm{defect}}\mathcal L_{\mathrm{defect}}
+\omega_{\mathrm{bi}}\mathcal L_{\mathrm{bi}},
\qquad
\omega_{\mathrm{defect}}=\omega_{\mathrm{bi}}=1.
\]

第一项是离散稳定目标：

\[
\mathcal L_{\mathrm{stable}}
=\left\|
T_{\phi,h}(u_{j+1},e_{j+1})
-S_{\Delta t}T_{\phi,h}(u_j,e_j)
\right\|_{M_h}^2,
\qquad S_{\Delta t}=\exp(\Delta t A_{s,h}).
\]

第二项直接计算连续动力学缺陷：

\[
r^0_{\theta,\phi,h}(u,e)
=D_uT_{\phi,h}(u,e)F_h(u)
+D_eT_{\phi,h}(u,e)\bigl(F_h(u+e)-F_h(u)+g_{\theta,h}(u,e)\bigr)
-A_{s,h}T_{\phi,h}(u,e),
\]

\[
\mathcal L_{\mathrm{defect}}
=\frac1N\sum_{j=1}^N
\frac{\|r^0_{\theta,\phi,h}(u_j,e_j)\|_{M_h}^2}
{\|e_j\|_{M_h}^2+10^{-8}}.
\]

第三项是双 Lipschitz 范数带的安全约束：

\[
\mathcal L_{\mathrm{bi}}
=\frac1N\sum_{j=1}^N\left[
\operatorname{ReLU}\!\left(m_h\|e_j\|_{M_h}-\|T_{\phi,h}(u_j,e_j)\|_{M_h}\right)^2
+\operatorname{ReLU}\!\left(\|T_{\phi,h}(u_j,e_j)\|_{M_h}-M_h^\sharp\|e_j\|_{M_h}\right)^2
\right].
\]

## 可逆性处理

第一次只使用样本范数约束的尝试虽然让 \(\mathcal L_{\mathrm{bi}}\) 很小，但 n=31 的
Jacobian 最小奇异值仍只有 0.105，因此没有把那次结果作为正式结论。

正式版本改为只依赖状态的零空间缩放。设 \(N_h\) 是观测矩阵零空间的正交基，
每个缩放因子满足

\[
0.5=m_h\le s_{\phi,h}(u)\le M_h^\sharp=2.0,
\]

并令

\[
T_{\phi,h}(u,e)
=e+N_h\bigl((s_{\phi,h}(u)-1)\odot N_h^{\mathsf T}e\bigr).
\]

所以 \(T_{\phi,h}(u,0)=0\)、\(C_hT_{\phi,h}(u,e)=C_he\) 保持成立，且固定 \(u\)
时误差方向的 Jacobian 具有正的下界。正式运行中 \(\mathcal L_{\mathrm{bi}}=0\) 是
硬约束已经满足范数带的结果，不是遗漏损失。

## 正式结果

配置：网格 \(n=31,63,127\)，每个网格 4 个 seed（501--504），150 epoch，
\(\lambda/\rho_1=0.5\)，每个网格 48 个 test case 和 12 个噪声 case。

| 网格 | 选中 seed | 稳定验证损失 | 动力学缺陷验证损失 | 可逆性验证损失 | 终点误差 | 噪声终点误差 |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 503 | \(1.53\times10^{-6}\) | 0.3279 | 0 | 0.1093 | 0.0838 |
| 63 | 504 | \(1.63\times10^{-6}\) | 0.3419 | 0 | 0.1119 | 0.0902 |
| 127 | 503 | \(2.39\times10^{-6}\) | 0.5604 | 0 | 0.1040 | 0.0825 |

12 个训练/验证记录全部为有限数；三张网格的 48 个测试案例和 12 个噪声案例全部完成。

证书审计结果：

- 最大零误差纤维残差：0；
- 最大观测方向残差：\(2.32\times10^{-9}\)；
- Jacobian 最小奇异值：\(0.540,0.699,0.999\)；
- Jacobian 最大奇异值：\(1.209,1.000,1.256\)。

## R5 结论

这次运行完成了“稳定损失 + 动力学缺陷 + 可逆性约束”的真正联合训练，并修复了只看
样本范数却不能保证局部可逆的问题。它证明的是：在当前有限样本、有限网格和声明的局部
参数化下，certificate 的 fiber、方向和局部 Jacobian 审计可以通过。

它没有证明学习增益优于固定增益，也没有证明条件稳定界成立。旧 R5 消融完整模型的无噪声
终点误差为 \(0.0886/0.0968/0.0898\)，本次为 \(0.1093/0.1119/0.1040\)，因此在线性能
仍未超过旧基线。要进入研究计划中的条件稳定结论，还需要单独估计
\(\varepsilon_{T,h}\)、\(K_h\)、\(\alpha_h\) 和 \(\beta_h^{\mathrm{stab}}>0\)；本次
训练损失和审计不能替代这些估计。
