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
After removing spectra and applying time cutoffs (`t >= t_start` and optional
`t <= t_end`), corrected time normally becomes:

```text
t_corrected = t_original - original_time_of_first_retained_spectrum
```

The first retained spectrum is therefore at `t = 0`, even if the requested
start cutoff precedes it. For general A-B-C models, an earlier spectrum can
instead be selected by its original index as the fixed reactant reference,
including outside the fitted time window. In that case, corrected time is
referenced to the selected A spectrum.

The selected model is introduced with its scheme, absorbing species,
time profiles, parameters and interpretation notes. Special mechanisms use
385-650 nm by default; general models use 320-820 nm. Override either bound
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
410-650 nm. Unlike the ordinary single-file clock, global corrected time
remains `t_original - 47 s` even if the first retained spectrum is later.
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
