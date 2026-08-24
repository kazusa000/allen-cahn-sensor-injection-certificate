# R5-A：Allen–Cahn reference solver and observation baseline

## Scope

R5-A validates the deterministic reference layer before any learned correction is introduced. The model is

\[
u_t=\nu u_{xx}+u-u^3,
\qquad u(0,t)=u(1,t)=0,
\qquad \nu\in[0.005,0.02].
\]

The spatial discretization uses the interior second-difference operator on a uniform grid and the physical
mass matrix \(M_h=hI\). Fixed-width observations are exact averages of the piecewise-linear interpolant over
physical intervals. The first observer baseline is causal constant-gain nudging.

## Gates

1. The discrete energy and the semi-discrete RHS satisfy

   \[
   \frac{dE_h}{dt}=-\|F_h(u)\|_{M_h}^2\le0.
   \]

2. DOP853 integration is reproducible under the declared tolerances and its sampled energy is non-increasing
   up to a declared numerical slack.

3. Local-average rows are finite, have the declared physical interval, and preserve constants for intervals
   away from the Dirichlet boundary.

4. The baseline observer uses only current estimates and current measurements. Truth is used only by the
   offline harness to generate the measurement stream.

5. No training, certificate, future measurement, or predictor is part of R5-A.

## Next gate

After these checks pass, R5-B will add the full-state local-incremental baseline and observation/noise
rollouts. The state/error domain and formal train/validation/test split are frozen before R5-C.
