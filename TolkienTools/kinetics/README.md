# Multiwavelength Kinetics

Fit a time series of absorption spectra to a kinetic mechanism and recover
rate constants, species spectra, and concentration profiles. The fit uses all
selected wavelengths and times together, rather than fitting a single trace.

The easiest way to enter the kinetics module is:

```bash
tolkien-tools 4 filename
```

Replace `filename` with your experimental file, for example
`tolkien-tools 4 experiment.KD`. This opens the kinetics menu; the file is
loaded when you choose an analysis. You can also use `tolkien-tools kinetics`
and select a file when prompted. For all command-line options, run
`tolkien-tools kinetics --help`.

## Chemometrics and multiwavelength fitting

### Treat the experiment as a spectral matrix

The data matrix `D` has one row per wavelength and one column per acquisition
time. Under the additive Beer–Lambert model:

```text
D(lambda, t) = l * sum_i epsilon_i(lambda)*c_i(t) + residual(lambda, t)
D = l * E * C + residual
```

`E` contains the species spectra, `C` contains their concentration profiles,
and the module uses `l = 1 cm`. Every wavelength shares the same kinetic
constants and concentration profiles; each species has its own spectrum.
This combines information from overlapping bands and different spectral
regions, including wavelengths where an intermediate contributes even when
it is difficult to distinguish in a single trace.

For each trial set of rate constants, the chosen mechanism generates `C`.
The standard NNLS fit then estimates the unknown nonnegative spectra `E`,
respecting any supplied references. The optimizer adjusts the rates to
minimize the sum of squared absorbance residuals across all retained
wavelengths and times. The global spectral RMSE summarizes this full matrix;
it is not an average of independently fitted single-wavelength rate constants.
Here “global” refers to one spectral time series fitted jointly across
wavelengths, not automatically to multiple experiments sharing parameters.

### Factor analysis: how many species are spectrally distinguishable?

The module performs factor analysis by singular value decomposition (SVD)
of the absorbance matrix, without mean-centering:

```text
D = U * S * V_transpose
D approximately equals Q * W when only the leading factors are retained
```

The singular values on the diagonal of `S` indicate the strength of successive
independent components. The interface plots them for the initial data and
the preprocessed fit window, and includes them in the final diagnostic panel.
Look for dominant values separated from a smaller tail consistent with noise.
For example, three clearly significant factors support considering three
spectrally distinguishable absorbing components, such as A, B and C.
This is a visual diagnostic; the module does not automatically select the
number of species or the reaction mechanism from a significance threshold.

The factors are mathematical spectral/temporal combinations, not pure species
spectra or chemical concentration profiles. The number of significant factors
estimates the effective spectral rank, which can be smaller than the number
of chemical species if spectra are similar or linearly dependent, an intermediate
barely accumulates, or part of the reaction falls outside the selected window.
Conversely, baseline drift, scattering or instrumental artifacts can introduce
extra factors. Check the rank after preprocessing and interpret it alongside
the chemistry, time coverage and signal-to-noise level.

### From factors to a chemical model

SVD describes the independent information present in the data; the kinetic
model, nonnegativity, concentration conservation and spectral references
constrain its chemical interpretation. Factor count alone cannot distinguish
A → B → C from A ⇌ B → C, since both can contain three absorbing species.
Compare the recovered spectra, concentration profiles and residual map when
assessing a mechanism. Extra model complexity needs support from the data and
chemistry, even if it lowers the residual error.

## Choose a general mechanism

The main menu offers three general models:

| Reaction | CLI identifier | Fitted rate constants | Use when |
| --- | --- | --- | --- |
| A → B | `a_to_b` | `k` | A reactant converts irreversibly into one product. |
| A → B → C | `a_to_b_to_c` | `k1`, `k2` | An intermediate forms and then converts irreversibly into product. |
| A ⇌ B → C | `a_rev_b_to_c` | `k1`, `k_1`, `k2` | Reactant and intermediate interconvert before irreversible product formation. |

A is the reactant, B is the product in the two-species model or the
intermediate in the three-species models, and C is the final product.
`k1` describes A → B, `k_1` describes B → A, and `k2` describes B → C.

All three models start with only A at the chosen reaction time zero:
`[A](0) = c0`, with the other species initially absent. Total concentration
is conserved. The rate laws are first order; they can also represent
pseudo-first-order conditions when another reagent remains effectively constant.
With times in seconds, the fitted constants are in s^-1.

### Model equations

**A → B** describes exponential reactant decay and product accumulation:

```text
d[A]/dt = -k*[A]
d[B]/dt =  k*[A]
```

**A → B → C** allows the intermediate to rise and fall while product
accumulates. Both consecutive steps are fitted, including the equal-rate case:

```text
d[A]/dt = -k1*[A]
d[B]/dt =  k1*[A] - k2*[B]
d[C]/dt =  k2*[B]
```

**A ⇌ B → C** adds return from the intermediate to the reactant:

```text
d[A]/dt = -k1*[A] + k_1*[B]
d[B]/dt =  k1*[A] - (k_1 + k2)*[B]
d[C]/dt =  k2*[B]
```

Its spectra and individual rates can be ambiguous when the species spectra
are unknown: similar residuals can accompany different rate/spectrum
combinations. Known spectra and chemical evidence help assess those solutions.

By default, the module optimizes the rate constants while estimating
nonnegative species spectra by NNLS (nonnegative least squares). You can fit
all spectra freely or supply references for individual species. The spectral
model is additive: absorbance is the sum of each species concentration times
its molar absorptivity, using a 1 cm optical path convention.

## Prepare and fit an experiment

1. Open the module with `tolkien-tools 4 filename` and choose a model.
2. Inspect the spectra and singular values, then choose baseline correction,
   wavelength range, reaction start/end times, and any spectra to exclude.
   Recheck the factor count in the preprocessed window.
3. Enter the initial concentration in mol/L and choose which species spectra
   to fit freely or fix to references.
4. Run the fit, inspect spectra, concentration profiles and residuals, and
   revise preprocessing if needed.

### Input data

`.KD` files are converted automatically to a sibling `*-converted.txt` file.
Text input is a semicolon-separated matrix: the first row contains times,
the first column contains wavelengths in nm, and the remaining cells contain
absorbance. Use a numeric placeholder in the top-left cell:

```text
0;0;1;2
400;0.120;0.110;0.100
401;0.125;0.115;0.105
402;0.130;0.120;0.110
```

R `write.csv2` tables with row names and comma decimals are also supported.
Use a consistent time unit; rates are expressed in its inverse. Enter the
actual concentration to obtain molar concentrations and meaningful epsilon
values; the CLI concentration default of 1 M is only a numerical default.

### Baseline, wavelength range and reaction time zero

General models use 320–820 nm by default. Adjust the window to the spectral
region relevant to your experiment. Baseline correction can use an automatic
flat region, a chosen wavelength region or selected points, or be disabled
for data that are already corrected. Exclude individual spectra or ranges
only when justified by the experiment.

The reaction-start value is the exact time zero (`t0`) in the original
acquisition clock. Spectra before it are discarded and retained times become:

```text
t_corrected = t_original - t0
```

For example, a frame acquired at 20.9 s has corrected time 0.49 s if you enter
`t0 = 20.41 s`. There need not be a measured frame at t0. Excluding the first
retained frame does not move this origin. The end cutoff also uses original
acquisition times. Without an explicit t0, the first retained frame is the
default origin; the legacy CLI `--initial-spectrum-index` can instead set
that origin. Interactive spectral-reference selections do not change t0.

### Species spectra and references

Each species has one reference question. Choose:

| Answer | Effect |
| --- | --- |
| `no` | Fit the spectrum freely. |
| `yes` | Fix the reactant to the first retained spectrum, or product to the last. |
| An original index, e.g. `23` | Fix the spectrum to that measured frame. |
| A range, e.g. `23-25` | Fix the spectrum to the mean of those frames. |
| A filename | Import a reference spectrum. |

Indices are 1-based and refer to the original acquisition, before trimming.
Measured references receive the same baseline correction and wavelength crop
as the fitted data. Excluded frames cannot be used as references.
Intermediates require an index, range or file; they have no automatic endpoint.
Use endpoint references only if those frames adequately represent pure species.

**Reference files must contain baseline-corrected molar absorptivity epsilon
in M^-1 cm^-1, for the 1 cm optical path convention, rather than absorbance.**
A plain whitespace-separated file can contain wavelength followed by one
spectrum, or multiple columns in model species order (A, B, C). Labeled files
with `# wavelength A B C` are also accepted. Numeric pure-spectra exports can
be reused directly. References are interpolated over common wavelength
coverage without extrapolation.

For each imported file, the interactive dialog offers a fixed amplitude
(multiplier 1) or an adjustable positive multiplier between 0.7 and 1.3.
Spectral shape remains fixed. The legacy `--known-spectrum` CLI import retains
its separate, broader adjustable-scale behavior.

## Read the results

Each individual fit writes to `results_<input-stem>` beside the input file.
Inspect the fitted species spectra, concentration profiles and the full
wavelength/time residual map, as well as the overall RMSE. Structured residuals
can expose behavior that a small overall error hides. The plotted residual
scale is fixed at ±20 mAU, with exceedances indicated.

The exports include:

| File suffix | Contents |
| --- | --- |
| `_experimental.csv` | Experimental absorbance after accepted preprocessing. |
| `_simulated.csv` | Fitted absorbance on the same axes. |
| `_residuals_mAU.csv` | Experimental minus fitted absorbance, in mAU. |
| `_pure_spectra.dat` | Wavelength followed by species epsilon columns. |
| `_concentrations.dat` | Model concentrations and sparse spectral projections. |
| `_spectral_concentrations.dat` | Sparse spectral projections alone. |
| `_export_layout.json` | Units, species order and numeric table layouts. |

Numeric exports have no text headers. Spectral matrices have times in the
first row, wavelengths in the first column and an empty top-left cell.
Concentration tables have times in the first row; `_concentrations.dat`
uses numeric row identifiers in the first column, followed by the model
concentration rows and then the sparse projection rows. Concentrations are
in mol/L. Empty projection cells mean no display point, not zero concentration.

Spectral projections estimate concentrations using the fitted species spectra;
they are nonnegative and sum to c0. They help compare the spectral data with
the kinetic profiles, but are not independent concentration measurements.
A low RMSE alone does not establish a mechanism or uniquely identify its rates.

## Command-line examples

For automated fitting without the interactive menu or plots:

```bash
tolkien-tools 4 experiment.txt --model a_to_b \
  --c0 1e-5 --baseline-mode none --no-plot

tolkien-tools 4 experiment.txt --model a_to_b_to_c \
  --c0 1e-5 --reaction-start-time 20.41 \
  --lambda-min 400 --lambda-max 650 --baseline-mode none --no-plot

tolkien-tools 4 experiment.txt --model a_rev_b_to_c \
  --c0 1e-5 --fix-initial-spectrum --fix-final-spectrum \
  --baseline-mode none --no-plot
```

These examples disable baseline correction because they assume corrected
input. Omit or change `--baseline-mode none` when correction is needed.
The endpoint flags fix the first/last retained experimental spectra.

An experimental MCR-ALS method is available for the two three-species models
with `--fit-method mcr_als`. It alternates nonnegative spectral and
concentration estimation with concentration closure and a soft kinetic
constraint (`--mcr-kinetic-weight`, default 1). Its reported rates are a
projection onto the chosen mechanism; use NNLS for the standard kinetic fit.

## Specialized mechanisms and studies

The **Special mechanisms and studies** menu contains project-specific presets
and a temperature/pH analysis. General reactions should use the models above.

### FeMb + disulfides

The reversible preset (`femb_disulfide_reversible`) is a variant of
A ⇌ B → C that looks for reactant/intermediate/product Soret maxima at
405–414 / 423–430 / 432–438 nm, respectively, and an intermediate/product
peak epsilon ratio between 0.50 and 0.95. It explores candidate solutions
whose global spectral RMSE lies within 0.1% of the lowest found and provides
an interactive viewer to assess their chemical plausibility and select a
complete fit. The viewer requires an interactive Matplotlib backend;
noninteractive runs retain automatic selection. These spectral criteria are
specific to this system.

The irreversible preset (`femb_disulfide_irreversible`) uses A → B → C,
with an intermediate Soret window of 427–433 nm and the same reactant/product
windows and peak ratio. It does not use the reversible candidate viewer.

### Other FeMb presets

Sulfide presets describe autocatalytic reduction after binding, or with
irreversible/reversible binding included. The acceleration is modeled using
the reduced fraction `x = [MbFeII]/c0`:

```text
Reduction rate = (k_slow + k_auto*x)*[MbFeIII-HS]

Without binding in the fitted window:
dx/dt = (k_slow + k_auto*x)*(1 - x)
```

`k_slow` sets the initial slow reduction and `k_auto` describes the increase
in apparent rate as reduction proceeds (named `k_cat` in the binding presets).
The reduced fraction is a phenomenological proxy for unobserved reactive sulfur
species; the autocatalytic constant is not directly an elementary HSS- reduction
constant. Binding presets add association/dissociation to this reduction law.

The HSS- postbinding preset uses first-order conversion without autocatalysis.
These presets use 400–650 nm by default. See the
[kinetic model manual](modelos_cineticos_tolkien_tools.md) (Spanish) for the
complete models and interpretation.

### Temperature and pH analysis

This module analyzes measured rate constants for a two-component acid/base
system in rapid equilibrium. Both forms generate the same product, but with
different intrinsic rate constants:

```text
HA  <=========>  A- + H+       pKa(T)
 |              |
 | k1(T)        | k2(T)
 v              v
 P              P            same product
```

The equilibrium must be fast compared with product formation. At each
temperature and pH, the observed rate constant is the population-weighted
sum of the two pathways:

```text
alpha_A  = 1 / (1 + 10^(pKa(T) - pH))
alpha_HA = 1 - alpha_A
k_obs(T, pH) = alpha_HA*k1(T) + alpha_A*k2(T)
```

The module combines the Eyring equation, written relative to each rate
constant at a reference temperature `Tref`, with the van't Hoff equation
for the temperature dependence of the pKa:

```text
k1(T) = k1_ref*(T/Tref)*exp[-H_A/R*(1/T - 1/Tref)]
k2(T) = k2_ref*(T/Tref)*exp[-H_B/R*(1/T - 1/Tref)]
pKa(T) = pKa_ref + delta_H_acidity/(R*ln(10))*(1/T - 1/Tref)
```

Here `H_A` and `H_B` denote the activation enthalpies of the acid (HA) and
base (A-) pathways, respectively. `delta_H_acidity` is the enthalpy of
HA → A- + H+. Enthalpies are assumed constant over the fitted temperature
range. Use consistent energy units for the enthalpies and R.

A joint fit across temperatures and pH values estimates six parameters:
`k1_ref`, `k2_ref`, `H_A`, `H_B`, `pKa_ref`, and `delta_H_acidity`.
The two reference rates are fitted from the measurements, rather than
requiring separate pure-form measurements at Tref. In the outputs these
parameters are named `k_HA_ref`, `k_A_ref`, `H_HA`, `H_A`, `pKa_ref`, and
`H_acidity`, respectively. Tref defaults to 298.15 K and can be changed with
`--reference`; an independently known pKa can be fixed with `--fixed-pka`.

### Fit quality and parameter correlations

The fit minimizes squared log-rate residuals, with optional uncertainty
weights, from multiple starting points. Alongside fitted curves and residuals,
the module reports local parameter correlations and a graphical matrix
crossing all six fitted variables (using logarithms of the reference rates).
Diagonal panels vary one parameter while refitting the other five;
off-diagonal panels vary a pair while refitting the other four. With fixed
pKa, the matrix instead covers five free parameters.

These profiles and maps help assess whether the solution occupies a well-defined
minimum or lies in a broad valley of compensating parameters. They can reveal
nonlinear correlations and flag a lower objective found during exploration,
which calls for a new fit. They assess the minimum over the explored region;
they do not prove that it is the global minimum or that all six parameters
are uniquely determined. Map contours are not confidence regions.

Choose **Thermal analysis of disulfide acid-base speciation** in the special
menu, or run:

```bash
tolkien-tools kinetics activation measurements.csv --output results_activation
```

CSV/TSV columns are `T_K,pH,k`, with temperature in kelvin and positive observed
rate constants in s^-1; optional columns are `id` and `sigma_log_k`.
This analysis uses rate constants, not spectral time series. Include multiple
temperatures and pH values, ideally spanning both sides of the pKa, to help
separate the two pathways and their temperature dependences.

Profiling can take several minutes; `--no-profiles` skips it. See the
[technical manual](activacion_acido_base.md) (Spanish) for input requirements,
options, output details and limitations.
