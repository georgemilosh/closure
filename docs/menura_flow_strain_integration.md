# Deploying the flow-strain closure in Menura

Reference implementation and checklist for running
`EoS_nBW_Ptensor_fieldframe_f2_MLP` (six invariants, no electric-field inputs)
inside Menura. Nothing here is applied automatically: the Menura source lives
only in run-directory snapshots, so each run directory must be patched and
rebuilt.

The training-side definition is `closure/field_invariants.py`; the tests in
`tests/test_field_invariants.py` pin the stencil, the analytic values and the
rotational invariance. **The kernel below must reproduce that function
exactly** — a mismatched derivative is the same class of silent train/deploy
inconsistency as the hyper-resistive term discussed below, only harder to see.

## 1. Why these features and not `E + u_e x B`

Under Menura's own Ohm's law,

```
E = -u_i x B + (J x B)/n - (div P_e)/n + eta J - eta_h lap J ,   u_e = u_i - J/n
=>  E + u_e x B  ==  -(div P_e)/n + eta J - eta_h lap J
```

so an electron-frame electric field input feeds the closure the divergence of
its own previous output, plus a hyper-resistive term that has **no counterpart
in the ECsim training data**. Measured on `EoS_nB_PparPperp_hpub_m825K11_fhlim3`
at t = 37.5 over the reconnection region: `|E_amb|` rms 0.0351, `|E_hyp|` rms
0.0069 — the hyper-resistive share is **20 %** of the invariant domain-wide and
**45 % of E_amb at the X-point**.

The four flow-strain invariants depend only on `B` and
`u_e = (J_tot - J_i)/rho`. `J_tot` comes from Ampere's law, so nothing in the
input path depends on `P_e`: the feedback loop is cut by construction.

## 2. New features

Append to `FeatureIndex` in `parameters.h` (after `BMAGN = 10`):

```cpp
  WPAR_E  = 11,   // b.W.b            : CGL parallel-pressure driver
  DIVV_E  = 12,   // div u_e          : compressional driver
  WMIX_E  = 13,   // |(I-bb).W.b|     : gyroviscous (agyrotropic) driver
  WPERP_E = 14,   // ||traceless perpendicular block of W||
  FeatureLength = 15
```

and to `feature_map` in `kernels_calls.h`:

```cpp
  {"Wpar_e",  WPAR_E},
  {"divV_e",  DIVV_E},
  {"Wmix_e",  WMIX_E},
  {"Wperp_e", WPERP_E},
```

## 3. Kernel

`extract_features_kernel` currently reads only cell `(i,j)`. These features need
a 2-cell halo, which the ghosted arrays already provide; guard the stencil so it
is only evaluated on interior cells (`i >= 2 && i < len_x_cst+2`, likewise `j`).

`u_e` must be formed from the *unscaled* density, exactly as the existing
`VX_E` block does, and **before** `features[RHO]` is divided by `4*M_PI`.

```cpp
// --- Tier-2 flow-strain invariants ------------------------------------------
// Mirrors closure/field_invariants.py::flow_gradient_invariants.  Menura's
// fourth-order central stencil, identical to the training-time operator:
//     d_x f = (8 (f[i+1] - f[i-1]) - (f[i+2] - f[i-2])) / (12 dx)
{
  auto ue = [&](int ii, int jj, int c) -> float {
    float rho = -fields->density_b[ii][jj];          // electron charge density
    return (fields->Jtot[c][ii][jj] - fields->Ji[c][ii][jj]) / rho;
  };
  // grad[d][c] = d_d u_c ; the d_z row vanishes in 2-D.
  float grad[2][3];
  for (int c = 0; c < 3; ++c) {
    grad[0][c] = (8.f*(ue(i+1,j,c) - ue(i-1,j,c))
                    - (ue(i+2,j,c) - ue(i-2,j,c))) / (12.f*dx_cst);
    grad[1][c] = (8.f*(ue(i,j+1,c) - ue(i,j-1,c))
                    - (ue(i,j+2,c) - ue(i,j-2,c))) / (12.f*dy_cst);
  }
  // Symmetric rate of strain W_ab = 0.5 (d_a u_b + d_b u_a), with d_z = 0.
  float W[3][3];
  W[0][0] = grad[0][0];
  W[1][1] = grad[1][1];
  W[2][2] = 0.f;
  W[0][1] = W[1][0] = 0.5f*(grad[0][1] + grad[1][0]);
  W[0][2] = W[2][0] = 0.5f*grad[0][2];
  W[1][2] = W[2][1] = 0.5f*grad[1][2];

  float bmag = fmaxf(features[BMAGN], 1.0e-12f);
  float bh[3] = {features[BX]/bmag, features[BY]/bmag, features[BZ]/bmag};

  float Wb[3], par = 0.f, wnorm2 = 0.f, wb2 = 0.f;
  for (int a = 0; a < 3; ++a) {
    Wb[a] = W[a][0]*bh[0] + W[a][1]*bh[1] + W[a][2]*bh[2];
    par  += bh[a]*Wb[a];
    wb2  += Wb[a]*Wb[a];
    for (int b = 0; b < 3; ++b) wnorm2 += W[a][b]*W[a][b];
  }
  float mix2 = 0.f;
  for (int a = 0; a < 3; ++a) {
    float m = Wb[a] - par*bh[a];
    mix2 += m*m;
  }
  float divergence  = grad[0][0] + grad[1][1];
  float perp_trace  = divergence - par;
  // ||P W P||^2 = ||W||^2 - 2|W.b|^2 + (b.W.b)^2 ; removing the trace of the
  // 2-D perpendicular block subtracts 0.5 tr^2 (the perp identity has norm^2 2).
  float perp2 = wnorm2 - 2.f*wb2 + par*par - 0.5f*perp_trace*perp_trace;

  features[WPAR_E]  = par;
  features[DIVV_E]  = divergence;
  features[WMIX_E]  = sqrtf(fmaxf(mix2,  0.f));
  features[WPERP_E] = sqrtf(fmaxf(perp2, 0.f));
}
```

Units: `B` and `u_e` are already in Alfven units and `dx_cst` is in `d_i`, so
these come out in `Omega_ci` — the same units the training pipeline produces
after `code2alfven` (which divides the rate channels by `b0x`).

## 4. Verification before trusting a run

1. **Stencil parity.** Dump the four features from a 200-iteration Menura run
   and recompute them from the same `B`/`Jtot`/`Ji`/`density_b` dumps with
   `closure.field_invariants.flow_gradient_invariants`. They must agree to
   float32 round-off. Any systematic offset means the kernel and the reader
   disagree, and the model is being fed out-of-distribution inputs.
2. **Magnitudes.** On ECsim `RunID_0` cycle 10000 the measured rms values are
   `Wpar_e` 0.74, `divV_e` 1.12, `Wmix_e` 1.64, `Wperp_e` 1.67 (p99 up to 8.4).
   The config's `extra_invariant_scales` are set to those rms values. A Menura
   run whose distributions sit far from these is outside the training set.
3. **Grid noise.** These are second derivatives of `B` (via `u_e ~ curl B`), so
   they carry grid-scale noise — into a closure whose failure mode has
   consistently been grid-scale. If the p99 tail in deployment greatly exceeds
   the training tail, evaluate them on smoothed moments (the `SM2` binomial
   machinery already exists) and apply the same smoothing when regenerating the
   training features.

## 5. Decode and normalisation (independent of these features)

The trained model emits **raw** Cartesian pressures, but Menura's `P_TENSOR`
path applies `4*M_PI*exp()` to the diagonal and `4*M_PI*sinh()` off-diagonal
after de-standardising, and `JoblibLoader` requires `X.pkl`/`y.pkl` that a
`scaler_*: false` config never writes. Ship the model as an *export wrapper*
that emits what Menura already expects — i.e. append

```
Cartesian P -> log(diag), asinh(offdiag) -> (. - mean)/std
```

to the TorchScript module, write `y.pkl` with those (mean, std), write `X.pkl`
as identity `(zeros(14), ones(14))` so feature standardisation is a pass-through,
and declare `prescaler_targets: [log, log, log, arcsinh, arcsinh, arcsinh]` in
the deployment `config.yaml`. `enforce_spd` guarantees a positive diagonal, so
the `log` is always defined.

Also worth fixing in `functions_i.cpp`: `check_prescaler_targets` returns early
when the list is empty, and `prescaler_targets: null` yields an empty list — so
the one configuration that most needs the guard (raw outputs against an
exp/sinh decode) passes silently. Treat an explicit `null` as `{"none" x 6}`.
