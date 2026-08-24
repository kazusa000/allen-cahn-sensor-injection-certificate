# R5-B：局部增量动力学与因果基线

## Purpose

R5-B checks the nonlinear mechanism before any learned certificate or learned correction is introduced.
For

\[
F_h(u)=\nu D_{2,h}u+u-u^{\circ3},
\]

the exact expansion around a reference state \(u\) is

\[
F_h(u+e)-F_h(u)=J_h(u)e-3u\circ e^{\circ2}-e^{\circ3},
\]

where

\[
J_h(u)=\nu D_{2,h}+I-3\operatorname{diag}(u^{\circ2}).
\]

The last two terms are retained as an explicit nonlinear remainder; they are not silently absorbed into a
linear observer claim.

## Baselines

- open loop: correction gain zero;
- causal constant-gain nudging using the current local-average innovation and the physical mass adjoint
  \(M_h^{-1}C_h^\top\);
- the R2 fixed-gain construction, when its observation and grid assumptions are satisfied.

The offline rollout harness uses the true state only to generate the measurement stream. The observer RHS
itself receives only the current estimate and current measurement.

## Gates

1. The Jacobian/remainder identity agrees with direct subtraction to floating-point tolerance.
2. The local Jacobian is symmetric for the Dirichlet finite-difference discretization and its spectrum is
   reported along the reference trajectory.
3. Zero innovation produces exactly the uncontrolled Allen–Cahn RHS.
4. Open-loop and causal nudging rollouts are reproducible and have finite diagnostics.
5. No learned certificate or predictor is used in this stage.

## Interpretation boundary

Passing R5-B establishes a correct nonlinear baseline and a local linearization diagnostic. It does not
establish a uniform nonlinear stability theorem. That claim is reserved for the state-domain and certificate
gates in R5-C through R5-F.
