# R5 formal contract：state domain, splits, and comparison budget

## Status

This contract is the first executable R5 pilot freeze. It defines the local domain and split for the
reference/baseline/certificate stages. Any change requires an amendment before looking at held-out results.

## Model and grids

- Allen–Cahn: `u_t = nu * u_xx + u - u^3` on `[0, 1]` with homogeneous Dirichlet boundaries.
- Viscosity values: `nu in {0.005, 0.010, 0.020}`.
- Interior grids: `n in {31, 63, 127}` with `h = 1/(n+1)` and physical mass `M_h = h I`.
- Main observation: two fixed-width local averages
  `[0.20, 0.30]` and `[0.65, 0.75]`; both widths are `0.10`.
- Observation rows are exact averages of the piecewise-linear interpolant with zero Dirichlet endpoints.

## State and error domain

Initial reference states use

\[
u_0(x)=a_1\sin(\pi x)+a_2\sin(2\pi x)+a_3\sin(3\pi x),
\qquad |a_i|\le 0.5,
\]

and initial observer errors use the same first three modes with coefficient Euclidean norm at most `0.25`.
The executable domain declaration is therefore

\[
\|u_0\|_{H_h^1}\le R_u(h),
\qquad \|e_0\|_{M_h}\le R_e(h),
\]

where `R_u(h)` and `R_e(h)` are computed from the declared coefficient bounds on each grid and saved in
the run manifest. The formal claim is local to this domain and to the observed trajectory horizon.

## Time and noise

- Reference horizon: `T = 1.0`.
- Saved output: 101 uniform times on `[0, 1]`.
- Deterministic observation stress: zero, constant positive, constant negative, and common sinusoidal noise.
- Dimensionless deterministic noise amplitudes: `{0, 0.001, 0.01}`.
- Stochastic pilot noise: independent Gaussian samples at amplitudes `{0.001, 0.01}` using a declared seed stream.
- The observer receives only current/past observations; no smoothed or future values are allowed.

## Split and seeds

- Training: 8 fixed seeds, 4 initial-state coefficient draws per `(nu, n)`.
- Validation: 4 fixed seeds, disjoint coefficient draws.
- Held-out test: 4 fixed seeds, disjoint coefficient draws and one unseen mixed-mode stress family.
- The same initial state, observation stream, and noise realization are paired across methods.
- Seeds and generated case manifest are immutable run inputs.

## Baselines and gates

Baselines are open loop, causal constant-gain nudging, and the R2 fixed-gain construction where compatible.
The learned correction must use the same observation history and training budget as the baselines.

The pilot records:

- mass-norm reconstruction error over time;
- transient peak and terminal error;
- observation-noise amplification;
- energy defect of the estimated trajectory;
- cross-grid change in each diagnostic;
- certificate zero-fiber, direction, Jacobian, inverse, and defect diagnostics.

Passing the pilot requires finite, reproducible outputs and legal online information. It does not by itself
claim a uniform theorem or superiority over the baselines.

## Compute split

The reference solver, generated manifest, and baseline smoke remain local. The full training split is first
profiled locally. Only the multi-seed learned-correction/certificate sweep is eligible for 2060, and only
after the remote repository and environment pass the remote-experiment checks.
