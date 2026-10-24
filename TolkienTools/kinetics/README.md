# Multiwavelength Kinetics

Fit experimental multiwavelength kinetic spectra through `kinet_python.py`,
the launcher-compatible entry point. The module also includes `kd_to_txt.py`
for converting `.KD` files.

```bash
tolkien-tools
tolkien-tools kinetics
tolkien-tools kinetics --help
```

## Interactive workflow and preprocessing

Launcher option 4, `tolkien-tools kinetics`, and
`tolkien-tools kinetics file.KD` open the kinetics main menu before loading
data. Choose an irreversible A -> B model, consecutive irreversible
A -> B -> C, A <-> B -> C, or **Special mechanisms and studies**. The latter
contains sulfide mechanisms with/without binding, transsulfuration, global
fits and thermal disulfide acid-base speciation analysis.

Input is requested when the chosen analysis starts; a filename already given
on the command line is reused. Data-dependent preprocessing follows loading.
`--no-plot`, `--skip-preprocess-dialog`, `--global-manifest` and `activation`
retain direct access for automated runs.

Each analysis writes to `results_<input-stem>` beside the input file.
The existing reaction-start question now selects the **exact reaction time zero
(t0)**, as well as the lower time cutoff. Keep spectra with `t_original >= t0`
and apply:

```text
t_corrected = t_original - t0
```

No measurement at t0 is required. For example, entering 20.41 s and retaining
the next frame at 20.9 s gives that frame a corrected time of 0.49 s. Dropping
the first eligible frame does not move the chosen origin. The optional final
cutoff still uses original acquisition times. No additional question is asked.

With no t0 supplied, the first retained frame remains the default zero. The
legacy noninteractive `--initial-spectrum-index` reference origin applies only
when no explicit t0 is set; an explicit t0 always takes priority. Selecting a
spectrum or mean in the interactive reference dialog does not change t0.

The selected model is introduced with its scheme, absorbing species,
time profiles, parameters and interpretation notes. Special mechanisms use
400-650 nm by default; individual transsulfuration fits retain 385-650 nm.
General models use 320-820 nm. Override either bound
with `--lambda-min`, `--lambda-max` or the interactive dialog.

## Special mechanisms

### `mbfe3_sulfide_autocatalytic`

Autocatalytic MbFe(III) reduction by sulfides assumes the initial observable
species is the coordinated MbFeIII-HS complex. This is appropriate when HS-
is in excess and binding is fast compared with the fitted time window.
The analytical profiles are:

```text
x = [MbFeII] / [Mb]total
dx/dt = (k_slow + k_auto*x) * (1 - x)
[MbFeIII-HS](t) = c0 * (1 - x)
[MbFeII](t) = c0 * x
```

All times and two absorbing species are fitted. `k_slow,obs` describes the
initial slow phase; its dependence on [HS-] must be measured before assigning
a mechanistic interpretation. `x` is a phenomenological proxy for unobserved
reactive sulfur species, and `k_auto` measures apparent acceleration; it must
not be interpreted directly as k_Red(HSS-).

Before time trimming, the interactive diagnostic provisionally corrects the
baseline using 750-850 nm, detects the early spectral jump associated with
HS- addition, and proposes the first spectrum of the early MbFeIII-HS plateau
as t0. The panel shows corrected spectra, 409/428/434 nm traces and spectral
distance to the plateau. Suggested t0 and baseline region can be overridden.

### `mbfe3_sulfide_binding_autocatalytic`

This model adds reversible coordination before reduction:

```text
MbFeIII + HS- <-> MbFeIII-HS -> MbFeII

d[MbFeIII]/dt    = -k_on*[MbFeIII] + k_off*[MbFeIII-HS]
x                = [MbFeII] / [Mb]total
d[MbFeIII-HS]/dt = k_on*[MbFeIII] - (k_off + k_slow + k_cat*x)*[MbFeIII-HS]
d[MbFeII]/dt     = (k_slow + k_cat*x)*[MbFeIII-HS]
```

Three absorbing spectra (MbFeIII, MbFeIII-HS, MbFeII) and four rate constants
(k_on, k_off, k_slow,obs, k_cat) are fitted. In a single experiment k_on is
an apparent pseudo-first-order constant. With known excess [HS-], estimate
the bimolecular constant separately by dividing by [HS-]. k_off is first
order; k_cat is phenomenological and is not directly k_Red(HSS-).

The interactive flow separately offers to fix the first retained spectrum
as MbFeIII and the last as MbFeII. The MbFeIII-HS spectrum remains free unless
provided as a known spectrum.

Coupled nonlinear concentration profiles are solved with
`scipy.integrate.solve_ivp`. This keeps association, dissociation, reduction
and product accumulation explicit and allows later model extensions, at
the cost of numerical integration at each optimization trial.

### `mbfe3_hss_no_binding`

MbFe(III) reduction by HSS- without binding in the fitted window uses Soret
traces to locate addition, fitting fast MbFeIII decay at 409 nm with an
exponential plus a slow drift; 428 and 434 nm provide controls. Proposed t0
is four time constants after addition, when HSS- binding is approximately
98% complete. The subsequent MbFeIII-HSS -> MbFeII conversion is irreversible
first order with one rate constant k and no autocatalysis.

### `mbfe3_sulfide_hss_transsulfuration`

For experiments where excess HS- first forms MbFeIII-HS and a known amount
of HSS- is then added:

```text
R_HSS = [HSS-]added / [Mb]total
A = MbFeIII-HS
B = MbFeIII-HSS
P = MbFeII
S = effective free added HSS-
x = P / [Mb]total

dA/dt = -(k_slow + k_auto*x)*A - k_ts*A*S
dB/dt =  k_ts*A*S - k_fast*B
dP/dt =  (k_slow + k_auto*x)*A + k_fast*B
dS/dt = -k_ts*A*S

A(0) = c0
B(0) = 0
P(0) = 0
S(0) = R_HSS*c0
```

MbFeIII-HS and MbFeIII-HSS are kinetically distinct but share a spectrum.
The two visible species are `[MbFeIII-Sx](t) = A(t) + B(t)` and
`[MbFeII](t) = P(t)`. Four constants are fitted: k_slow,obs, k_auto, k_ts and
k_fast. Endogenous autocatalysis is represented by k_auto*x; added HSS- is a
consumable external species initialized with `--hss-ratio R`.

### `mbfe3_sulfide_hss_transsulfuration_no_auto`

This variant retains intrinsic slow reduction of initial MbFeIII-HS but
omits endogenous polysulfide autocatalysis:

```text
dA/dt = -k_slow*A - k_ts*A*S
dB/dt =  k_ts*A*S - k_fast*B
dP/dt =  k_slow*A + k_fast*B
dS/dt = -k_ts*A*S
```

Initial conditions and visible species match the autocatalytic variant.
At t=0 the MbFeII formation rate is k_slow*c0 because no MbFeIII-HSS exists
yet. Subsequent buildup and reduction of B can produce sigmoidal traces.
The fitted constants are k_slow,obs, k_ts and k_fast.

## Noninteractive fitting and known spectra

For any supported mechanism, select its identifier with `--model`:

```bash
tolkien-tools 4 experiment.dat \
  --model mbfe3_sulfide_autocatalytic \
  --baseline-mode none --no-plot

tolkien-tools 4 experiment.dat \
  --model mbfe3_sulfide_binding_autocatalytic \
  --baseline-mode none --no-plot

tolkien-tools 4 experiment.dat \
  --model mbfe3_sulfide_hss_transsulfuration \
  --hss-ratio 20 --baseline-mode none --no-plot

tolkien-tools 4 experiment.dat \
  --model mbfe3_sulfide_hss_transsulfuration_no_auto \
  --hss-ratio 20 --baseline-mode none --no-plot
```

Fix the reactant spectrum to the first retained experimental spectrum with
`--fix-initial-spectrum`; fix the product to the last with
`--fix-final-spectrum`. Either or both may be used:

```bash
tolkien-tools 4 experiment.dat --model a_to_b_to_c \
  --fix-initial-spectrum --fix-final-spectrum --fit-method nnls
```

The initial-spectrum option is available for all mechanisms and cannot be
combined with `--initial-spectrum-weight`, which is only a soft constraint.
Interactive endpoint questions follow spectrum/time trimming, so the
endpoints are the ones retained for fitting.

### Experimental MCR-ALS

`a_to_b_to_c` and `a_rev_b_to_c` also accept `--fit-method mcr_als`.
Starting from NNLS concentrations, this alternates nonnegative spectral and
concentration estimation with concentration closure. Keep concentrations
near the kinetic profile using:

```bash
--mcr-kinetic-weight 1 --mcr-max-iter 200
```

This is exploratory: reported rate constants are the projection of the MCR
result onto the chosen mechanism. With a known induction period, use a
spectrum from the pre-injection plateau as the fixed A reference rather
than fixing the first post-injection spectrum.

## Global transsulfuration fits across R values

Both transsulfuration variants can share rate constants and pure spectra
across experiments. Manifest columns are `filename`, `R` and `c0_after`:

```csv
filename,R,c0_after
experiment_R5.TXT,5,7.868852459e-6
experiment_R10.TXT,10,7.741935484e-6
```

```bash
tolkien-tools 4 --global-manifest series.csv \
  --global-root /path/to/data \
  --model mbfe3_sulfide_hss_transsulfuration_no_auto \
  --fix-initial-spectrum --fix-final-spectrum \
  --global-output-dir global_results
```

Defaults: HSS- addition at 47 s, baseline region 750-820 nm and fit window
410-650 nm. Global corrected time remains `t_original - 47 s` even if the
first retained spectrum is later, consistent with an explicit single-file t0.
Each experiment has equal total objective weight, preventing longer series
from dominating.

k_slow, k_ts, k_fast and (when included) k_auto are global; R, c0_after and
times remain file-specific. Literature/COPASI values provide numerical
starting guesses only; they are neither fixed nor penalized.
An independently measured k_fast can be fixed explicitly:

```bash
--global-fix-k-fast 0.001
```

The summary marks it as fixed; other constants remain globally fitted.

## Activation and acid-base equilibrium analysis

Fit rate constants measured at several temperatures and pH values using two
forms in rapid equilibrium. Select **Special mechanisms and studies**, then
**Thermal analysis of disulfide acid-base speciation**, or run:

```bash
tolkien-tools kinetics activation measurements.csv --output results_activation
tolkien-tools kinetics activation --interactive
```

CSV/TSV headers: `T_K,pH,k`, with absolute temperature in kelvin,
dimensionless pH, positive rate constants in s^-1 and decimal points.
Optional columns: `id`, `sigma_log_k`. See the
[technical manual](activacion_acido_base.md) (Spanish) for equations,
manual entry, checks, outputs and limitations. The separate
[kinetic model manual](modelos_cineticos_tolkien_tools.md) is also retained
in its original language.

### Reoptimized uncertainty and compensation maps

After the fit, the default analysis generates
`profile_compensation_matrix_six_parameters.png/pdf/svg`. The diagonal shows
one-parameter profiles with the other five parameters refitted. Each off-diagonal
panel fixes two parameters and refits the other four, showing valleys of
compensation; mirrored panels share the same numerical grid. Red crosses mark
the original fit, and `r` is the local linearized parameter correlation.
With `--fixed-pka`, pKa remains fixed during all refits and the output is a
five-parameter matrix (`profile_compensation_matrix_five_parameters.*`).

The fitted variables are ln(k_HA,ref), ln(k_A,ref), the two intrinsic activation
enthalpies, acidity enthalpy, and (when free) pKa at the reference temperature.
Entropies are derived quantities, not additional independent parameters.

Defaults are 33 points per axis over +/-3 local SE, clipped to the numerical
bounds. Use 21 points for a faster preview:

```bash
tolkien-tools kinetics activation measurements.csv --profile-points 21
```

`--profile-extent` adjusts the half-width, and `--profile-max-nfev` adjusts
the optimization limit per refit. These calculations can take several minutes;
progress is printed as each profile/map completes. `--no-profiles` skips this
step while retaining the fit and local SE analysis. `--no-plot` omits figures
but still saves numerical profiles unless combined with `--no-profiles`.

The objective uses the original log-rate residuals and their supplied weights.
For a nonzero fitted SSE, the color scale shows SSE/fitted SSE (weighted SSE
when sigma_log_k is supplied), with descriptive contours at 1.05, 1.2, 1.5,
2 and 3. For effectively zero SSE, objective increases replace the undefined
ratio. If local SE is unavailable or effectively zero, explicitly documented
exploration scales replace SE-based ranges. Failed refits are masked and
counted; a lower minimum found during profiling is flagged for a new fit.

These contours are **not confidence regions**, and local parameter correlation
is not a correlation of measurements. Keep interpreting the existing local SE
as conditional, linearized uncertainty; the profile maps additionally reveal
nonlinear compensation over the explored range.

Numerical outputs accompany the figures:

- `profiles.json`: one-dimensional profiles and reoptimized parameters.
- `matrix_pair_grids/pair_i_j.npz`: axes, objective values, full reoptimized
  parameters and convergence masks for each pair.
- `parameter_correlation.json`: local parameter correlation matrix; unavailable
  entries are null, including the fixed pKa row/column.
- `profile_matrix_manifest.json`: settings, axis scales/ranges, weighting,
  failed-node counts and interpretation notes.
- `result.json` and `summary.txt`: the fit plus a profile-analysis summary.

## Internal organization

- `kinet_python.py`: compatible entry point.
- `kinet_cli.py`: arguments, interactive dialogs and orchestration.
- `kinet_common.py`: shared dataclasses and labels.
- `kinet_io.py`: table loading and `.KD` conversion.
- `kinet_preprocessing.py`: baseline, time/wavelength trimming and exclusions.
- `kinet_models.py`: kinetic concentration profiles.
- `kinet_fitting.py`: NNLS and rate optimization.
- `kinet_linalg.py`: linear algebra/SVD helpers.
- `kinet_plotting.py`: exploratory plots, diagnostics and final panel.
- `kinet_t0.py`: time-zero estimator for reduction without binding.
- `kinet_export.py`: concentrations, pure spectra, summaries and PNG output.

- `kinet_activation.py`: two-form activation/acid-base fits and local uncertainty.
- `kinet_activation_profiles.py`: reoptimized profiles and compensation matrices.

## Irreversible binding with autocatalytic reduction

Choose **Special mechanisms and studies → MbFe(III)+HS- irreversible binding**
(special option 3).
CLI model: `--model mbfe3_sulfide_irreversible_binding_autocatalytic`.
The reversible binding integrator is reused with `k_off = 0` fixed exactly.
Only `k_on`, `k_slow` (reduction) and `k_cat` are optimized; spectral reference
and endpoint options remain available. Reports identify dissociation as fixed,
not an estimated parameter. This differs from the general A → B → C model,
which has no autocatalytic reduction term.

## Unified per-species reference dialog

Reactant, each intermediate, and product have one reference question:

- `yes` / `si` / `sí`: fix the first retained spectrum for reactant, or the
  last retained spectrum for product, after time trimming.
- `no`: fit that species spectrum freely.
- An original index (`23`): fix that spectrum.
- An inclusive range (`23-25`): fix their arithmetic mean.
- A filename (`reference.dat`): use that species spectrum from the file.

Indices are original, 1-based, before time trimming. Selected measured spectra
receive the same baseline correction and wavelength crop as the fitted data,
then are averaged and fixed in shape and amplitude. Explicitly dropped spectra
cannot be selected. An intermediate has no automatic first/last endpoint:
choose an index, range, filename, or `no`.

**Every reference file must contain molar absorptivity epsilon in M^-1 cm^-1,
already baseline-corrected, for the module's 1-cm optical path convention.**
This requirement is displayed immediately after a filename is entered. Raw
absorbance files must be converted to epsilon before loading. The model
multiplies epsilon by the entered concentration; a small concentration such as
1e-5 M is not the amplitude-correction multiplier.

Only for file references, one follow-up asks whether to adjust amplitude:
`no` / Enter keeps the multiplier fixed at 1; `yes` fits a positive relative
multiplier constrained to **0.7–1.3**. There are no questions about units or
custom amplitude limits. Spectral shape is preserved and no additional baseline
is applied to files. Each species has its own amplitude choice, remembered when
revising preprocessing. Optimization uses log space, not a multiplier grid.

Files may be plain whitespace-separated wavelength/epsilon pairs, or labeled
multi-spectrum files with `# wavelength A B C` (positional model labels).
Single-spectrum files can be used for any species. Paths may contain spaces;
case is preserved. Interpolation is restricted to the common measured wavelength
coverage, without extrapolation. Reports record selections, file paths/hashes,
epsilon units, amplitude mode, bounds and the fitted multipliers. Reference files
are protected against output overwrites.

Interactive references do not redefine the entered t0. The legacy
noninteractive `--initial-spectrum-index` time-origin convention applies only
when no explicit t0 is supplied. Existing `--known-spectrum`
CLI imports retain their historical adjustable-scale behavior and broad bounds;
the simplified interactive file workflow uses the fixed 0.7–1.3 interval.

## FeMb + disulfides with reversible binding

Special mechanisms option 7 (`--model femb_disulfide_reversible`) implements
A <-> B -> C without autocatalysis, with three species and no extra selection
questions. Defaults: fit 400–650 nm, baseline suggestion 750–850 nm; references
and the exact t0 use the existing dialogs.

The backend uses four deterministic starting points, followed by sampling
reversible families with the same two kinetic eigenvalues. Spectra are profiled
by NNLS. Candidates must lie within 0.1% of the lowest spectral RMSE. The
system-specific chemical filters require:

- Soret maxima A 405–414, B 423–430, C 432–438 nm.
- Intermediate/product Soret peak epsilon ratio between 0.50 and 0.95.

Among eligible candidates, selection minimizes the mean of the three squared
RMS discrepancies between kinetic fractions and per-frame nonnegative,
sum-one spectral estimates. Profiles are evaluated from t >= 5 s, or over all
frames if none reach 5 s. These estimates use the same species spectra and
measure internal consistency rather than independent validation.

The peak windows and amplitude interval are assumptions of this preset, not
universal spectral constraints. The B409/max ratio is diagnostic only; shoulder
shape is not constrained. No derivative criterion or interactive candidate
selector is currently implemented. If no absorption-comparable candidate passes
the chemical filters, a candidate is returned with an explicit failure flag for
inspection; chemical acceptance must not be inferred from its return alone.

Preset rate bounds (s^-1): k_on 1e-5–5, k_off 1e-7–5, k_slow/k_red 1e-6–1.
These bounds replace the generic --k-min/--k-max settings and are not
automatically expanded. Known-file amplitude bounds remain respected.

The preset exports `_selection.json` with all candidates, criteria and selection
status, plus `_spectral_concentrations.dat`, and overlays spectral estimates on
the kinetic profiles. Reversible models with unknown species spectra can admit
substantially different rate/spectrum combinations with similar residuals.
Inspect species spectra and local concentration discrepancies; a selected
solution is not evidence that its individual rates are uniquely identified.


### Numeric exports (all kinetic mechanisms)

Final exports contain no text headers; units and species order are recorded
separately in `*_export_layout.json`. Spectral CSV matrices (`*_experimental.csv`,
`*_simulated.csv`, `*_residuals_mAU.csv`) have time in row 1, wavelength in column 1,
and an empty top-left cell. Experimental values include all accepted baseline
corrections and cuts; simulation uses the identical axes. Residuals are experimental
minus simulated, in mAU, without clipping. The final panel uses a fixed +/-20 mAU
color scale and indicates exceedances.

`*_pure_spectra.dat` contains wavelength followed by the species epsilon columns.
`*_concentrations.dat` has time in row 1 and numeric row identifiers in column 1:
rows 1..N after the time row hold all model concentrations; rows N+1..2N hold
sparse spectral projections in the same species order. Empty projection cells
mean no display point, not zero concentration. `*_spectral_concentrations.dat`
also provides these sparse points directly (time row then species rows).
Concentrations are molar. Projections are nonnegative and sum to c0; they use the
fitted species spectra and are not independent concentration measurements.
Display points are selected by time targets, never residual quality. No fitted
observations are removed by this display sampling. Older concentration tables
with time in column 1 are superseded by this layout.


### Special mechanisms menu

1. Thermal analysis of disulfide acid-base speciation
2. MbFe(III)+HS- postbinding
3. MbFe(III)+HS- irreversible binding
4. MbFe(III)+HS- reversible binding
5. MbFe(III)+HSS- postbinding
6. MbFe(III)+HSS- irreversible binding
7. MbFe(III)+HSS- reversible binding
8. Transsulfuration reactions

The transsulfuration submenu contains individual fits with and without
autocatalysis, followed by the corresponding two global fits (options 1–4).
Selecting Back returns to the parent menu without requesting an input file.


### Unified species-reference dialog

Individual kinetic fits, including the generic A -> B, A -> B -> C and
A <-> B -> C models, ask **Use a known spectrum for species X?** once per
species. Options appear on separate lines: `yes` (first/last retained frame,
endpoints only), `no` (fit freely), an original 1-based index, an inclusive
range to average, or a reference file. Only a file selection adds an amplitude
question (fixed multiplier 1 or fitted multiplier 0.7–1.3). Imported spectra
must contain baseline-corrected epsilon in M^-1 cm^-1 for a 1 cm optical path.
Headerless numeric reference tables are accepted: wavelength followed by the
species columns in positional A/B/C order; a single spectrum column can be
used for any species. Legacy labeled files remain supported. This also allows
reusing the numeric pure-spectra exports directly. Supplied command-line
references are reported without asking for the same species again.


### Interactive equivalent-fit explorer

Interactive **MbFe(III)+HSS- reversible binding** fits open a candidate explorer
after optimization. The slider orders complete RMSE-equivalent solutions by
increasing intermediate peak absorptivity. The window shows the B family,
current A/B/C spectra, model concentrations and sparse spectral projections,
and residuals on a fixed +/-20 mAU scale. Rates, RMSE, concentration discrepancy,
peak ratio and the 409 nm shoulder diagnostic update together. No optimization
is performed while moving the slider. Gray background spectra are sampled to
at most 80 curves for readability; every available candidate is browsable.

Initially only candidates passing the configured chemical bounds are shown.
**Include chemically flagged fits** makes other RMSE-equivalent solutions
available, marked explicitly; if none passes, flagged solutions are shown from
the start. Accepting one records that its chemical bounds were not satisfied.
**Recommended solution** restores the automatic candidate. **Accept displayed
solution** uses the entire current fit for reporting and numeric export; the
selection JSON records manual selection, its chemical status and the original
recommendation. Closing without accepting retains the automatic result.

This explorer is limited to the solver's sampled candidate families; it does
not establish exhaustive coverage or uniqueness. It requires an interactive
Matplotlib backend. Noninteractive/no-plot runs keep automatic selection.


### HSS- irreversible binding

Select **Special mechanisms and studies -> MbFe(III)+HSS- irreversible binding**
(option 6). CLI model: `femb_disulfide_irreversible`. This reuses the reversible
FeMb/disulfide fitter with `k_off=0` fixed exactly and excluded from optimization;
only `k_on`, `k_slow` (k_red), and optional file-reference amplitude multipliers
are fitted. The mechanism is A -> B -> C without autocatalysis, equivalent to
the generic consecutive irreversible mechanism. The analytical equal-rate limit
is handled by the existing consecutive-reaction profile implementation.

The preset retains shared NNLS fitting, species reference controls, multistart,
RMSE comparison and concentration diagnostics. Its intermediate peak window is
427-433 nm (A: 405-414, C: 432-438), with B/C peak ratio 0.50-0.95. The default
fit range is 400-650 nm, baseline suggestion 750-850 nm. Reports explicitly
identify k_off as fixed. Continuous reversible eigenvalue families and their
interactive explorer are not used for this two-rate mechanism.
