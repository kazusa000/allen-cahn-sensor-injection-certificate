# R5-D：合法在线 learned correction

## Model

The first learned correction is a deliberately small causal residual model:

\[
\partial_t\hat u_h=F_h(\hat u_h)+g_0M_h^{-1}C_h^Tr_h+
W_0r_h+W_1s(\hat u_h)r_h,
\qquad r_h=y_h-C_h\hat u_h,
\]

with

\[
s(\hat u_h)=\|\hat u_h\|_{M_h}.
\]

The fixed physical-gain term is a safeguard; the learned term is the residual. Training uses reference states to construct the target correction
`F_h(u_h)-F_h(estimate_h)`. Deployment uses only the current estimate and current measurement. The
reference state is never an input to the correction model.

## Why this stage is small

This is an information-contract and baseline stage, not the final nonlinear neural model. It makes the
causal restriction testable without introducing a GPU dependency. R5-E may replace this parameterization
with a trainable nonlinear model only after the R5-D inputs/outputs and metrics are stable.

## Required checks

- ridge fit is reproducible;
- zero innovation produces zero learned correction;
- held-out inputs use the same current-information interface;
- training, validation, and test samples remain disjoint;
- comparison reports physical error, peak error, energy defect, and noise response.
