# R5-C：离线 certificate 审计

## Scope

R5-C first separates certificate validity from certificate performance. The certificate is evaluated only
offline on declared state/error samples. It is not part of the deployed observer RHS.

The minimum constraints are

\[
T(u,0)=0,
\qquad C_sT(u,e)=Ce,
\qquad 0<m\le\sigma_{\min}(D_eT)\le\sigma_{\max}(D_eT)<\infty.
\]

The repository contains an identity baseline and a small state-conditioned nullspace scaffold. The latter
is deliberately not called an optimal or learned certificate; its purpose is to test the audit machinery
before introducing a trainable parameterization.

## Audit outputs

For every declared sample set, record:

- sample count;
- maximum zero-fiber residual;
- maximum direction-preservation residual;
- minimum and maximum singular values of the error Jacobian;
- whether the basic constraints pass.

Finite-sample pass is not a uniform theorem. State-domain coverage, inverse error, and certificate defect
must be reported separately before any local stability claim.
