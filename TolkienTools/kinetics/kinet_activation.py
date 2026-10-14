"""Joint Eyring/van't Hoff analysis of two rapidly equilibrating forms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.special import expit

R = 0.00831446261815324  # kJ mol^-1 K^-1
LN10 = np.log(10.0)
NAMES = ('ln_k_HA_ref', 'ln_k_A_ref', 'H_HA_kJ_mol', 'H_A_kJ_mol',
         'H_acidity_kJ_mol', 'pKa_ref')
LOW = np.array([-30., -30., -200., -200., -300., -5.])
HIGH = np.array([30., 30., 400., 400., 300., 20.])


def validate_records(records):
    if not records:
        raise ValueError('No measurements supplied.')
    for row in records:
        for key in ('T_K', 'pH', 'k'):
            if not np.isfinite(row[key]):
                raise ValueError(f'Non-finite {key} in {row["id"]}.')
        if row['T_K'] <= 0 or row['k'] <= 0:
            raise ValueError('T_K and k must be strictly positive.')
        if row.get('sigma_log_k') is not None and (
            not np.isfinite(row['sigma_log_k']) or row['sigma_log_k'] <= 0
        ):
            raise ValueError('sigma_log_k must be finite and positive.')
    provided = [r.get('sigma_log_k') is not None for r in records]
    if any(provided) and not all(provided):
        raise ValueError('Supply sigma_log_k for every row, or omit it everywhere.')
    return records


def read_records(path):
    """Read a headered comma- or tab-separated table; retain replicates."""
    path = Path(path)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        first = stream.readline()
        stream.seek(0)
        reader = csv.DictReader(stream, delimiter='\t' if '\t' in first else ',')
        if not {'T_K', 'pH', 'k'} <= set(reader.fieldnames or []):
            raise ValueError('Required header: T_K,pH,k (optional: id,sigma_log_k).')
        records = []
        for line, row in enumerate(reader, 2):
            try:
                records.append(dict(id=row.get('id') or str(line - 1),
                    **{key: float(row[key]) for key in ('T_K', 'pH', 'k')},
                    sigma_log_k=float(row['sigma_log_k']) if row.get('sigma_log_k') else None))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'Invalid numeric value on line {line}. Use decimal points.') from exc
    return validate_records(records)


def evaluate(parameters, temperature, ph, reference=298.15):
    """Return log(k), analytic Jacobian, A population and A flux fraction."""
    p = np.asarray(parameters)
    temperature, ph = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(ph, float))
    du = 1 / temperature - 1 / reference
    q = LN10 * (ph - p[5]) - p[4] * du / R
    alpha = expit(q)
    lh = p[0] + np.log(temperature / reference) - p[2] * du / R
    la = p[1] + np.log(temperature / reference) - p[3] * du / R
    hterm = lh - np.logaddexp(0, q)
    aterm = la - np.logaddexp(0, -q)
    logk = np.logaddexp(hterm, aterm)
    flux = np.exp(aterm - logk)
    jac = np.stack((1-flux, flux, -(1-flux)*du/R, -flux*du/R,
                    -(flux-alpha)*du/R, -LN10*(flux-alpha)), axis=-1)
    return logk, jac, alpha, flux


def fit_records(records, reference=298.15, fixed_pka=None, starts=50, seed=0):
    validate_records(records)
    if not np.isfinite(reference) or reference <= 0 or starts < 1:
        raise ValueError('Reference temperature must be positive; starts must be >= 1.')
    if fixed_pka is not None and (not np.isfinite(fixed_pka) or not LOW[5] < fixed_pka < HIGH[5]):
        raise ValueError('Fixed pKa must lie strictly between -5 and 20.')
    ids = np.arange(6 if fixed_pka is None else 5)
    if len(records) <= len(ids):
        raise ValueError('More measurements than free parameters are required.')
    t = np.array([r['T_K'] for r in records])
    ph = np.array([r['pH'] for r in records])
    if np.ptp(t) == 0 or len(np.unique(ph)) < 2:
        raise ValueError('At least two temperatures and two pH values are required.')
    observed = np.log([r['k'] for r in records])
    sigma = np.array([r.get('sigma_log_k') or 1.0 for r in records])
    def unpack(v):
        return np.r_[v, fixed_pka] if fixed_pka is not None else v
    def fun(v):
        return (evaluate(unpack(v), t, ph, reference)[0] - observed) / sigma
    def jac(v):
        return evaluate(unpack(v), t, ph, reference)[1][:, ids] / sigma[:, None]
    rng = np.random.default_rng(seed)
    center = np.median(observed)
    initial = np.array([center-1, center+1, 80., 30., 20., np.median(ph)])
    fits = []
    for index in range(starts):
        start = initial.copy()
        if index:
            start[:2] = center + rng.uniform(-5, 3, 2)
            start[2:4] = rng.uniform(-50, 200, 2)
            start[4] = rng.uniform(-150, 150)
            start[5] = rng.uniform(max(LOW[5]+.01, ph.min()-1), min(HIGH[5]-.01, ph.max()+1))
        result = least_squares(fun, np.clip(start[ids], LOW[ids]+1e-8, HIGH[ids]-1e-8),
            jac=jac, bounds=(LOW[ids], HIGH[ids]), xtol=1e-12, ftol=1e-12,
            gtol=1e-12, max_nfev=3000)
        fits.append(dict(start=index, initial=start[ids].tolist(), parameters=unpack(result.x).tolist(),
            objective=float(result.fun @ result.fun), success=bool(result.success),
            status=int(result.status), message=result.message, nfev=int(result.nfev)))
    converged = [f for f in fits if f['success'] and np.isfinite(f['objective'])]
    if not converged:
        raise RuntimeError('No multistart converged. Inspect data and numerical bounds.')
    best = min(converged, key=lambda f: f['objective'])
    p = np.array(best['parameters'])
    logk, j, alpha, flux = evaluate(p, t, ph, reference)
    singular = np.linalg.svd(j[:, ids]/sigma[:, None], compute_uv=False)
    rank = int(np.linalg.matrix_rank(j[:, ids]/sigma[:, None]))
    warnings = ['Parameter profiles and numerical convergence do not supply experimental confidence intervals.']
    if rank < len(ids):
        warnings.append('Jacobian is rank deficient: parameters are locally unidentifiable.')
    if np.any(np.minimum(p[ids]-LOW[ids], HIGH[ids]-p[ids]) < 1e-5):
        warnings.append('A fitted parameter is near a numerical bound.')
    if len(converged) != starts:
        warnings.append('Some multistarts did not converge; see multistart.json.')
    entropy = 1000*R*(p[:2] - np.log((1.380649e-23/6.62607015e-34)*reference)) + 1000*p[2:4]/reference
    predictions = []
    for i, row in enumerate(records):
        predictions.append(dict(**row, k_pred=float(np.exp(logk[i])),
            log_residual=float(logk[i]-observed[i]), alpha_A=float(alpha[i]), flux_A=float(flux[i]),
            pKa_T=float(p[5]+p[4]*(1/t[i]-1/reference)/(R*LN10)),
            H_apparent_local_kJ_mol=float((1-flux[i])*p[2]+flux[i]*p[3]+(flux[i]-alpha[i])*p[4])))
    return dict(parameters=dict(zip(NAMES, p.tolist())),
        k_HA_ref=float(np.exp(p[0])), k_A_ref=float(np.exp(p[1])),
        S_HA_J_mol_K=float(entropy[0]), S_A_J_mol_K=float(entropy[1]),
        objective=best['objective'], RMS_log=float(np.sqrt(np.mean((logk-observed)**2))),
        jacobian_rank=rank, jacobian_singular_values=singular.tolist(),
        converged_starts=len(converged), matching_starts=sum(abs(f['objective']-best['objective']) <= 1e-8 for f in converged),
        warnings=warnings, predictions=predictions, multistart=fits,
        configuration=dict(reference_K=reference, fixed_pka=fixed_pka, starts=starts, seed=seed,
            bounds=dict(zip(NAMES, zip(LOW.tolist(), HIGH.tolist()))),
            objective='sum(((ln(k_pred)-ln(k))/sigma_log_k)^2)',
            weighting='supplied sigma_log_k' if records[0].get('sigma_log_k') else 'equal',
            solver='scipy.optimize.least_squares', max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12))


def write_table(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def export_result(result, output, plot=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output/'input.tsv', [{k: '' if v is None else v for k, v in r.items()}
                                     for r in result['records']])
    write_table(output/'predictions.tsv', result['predictions'])
    (output/'multistart.json').write_text(json.dumps(result['multistart'], indent=2)+'\n')
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['Joint acid-base activation analysis', json.dumps(result['parameters'], indent=2),
        f"Intrinsic activation entropies (J/mol/K): HA={result['S_HA_J_mol_K']:.8g}; A={result['S_A_J_mol_K']:.8g}",
        f"RMS log residual: {result['RMS_log']:.8g}", *result['warnings']]
    (output/'summary.txt').write_text('\n'.join(lines)+'\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        rows = result['predictions']
        for ph in sorted({r['pH'] for r in rows}):
            group = [r for r in rows if r['pH'] == ph]
            t = np.array([r['T_K'] for r in group])
            axes[0].scatter(1000/t, np.log([r['k'] for r in group])-np.log(t), label=f'pH {ph:g}')
            grid = np.linspace(t.min(), t.max(), 150)
            p = [result['parameters'][name] for name in NAMES]
            axes[0].plot(1000/grid, evaluate(p, grid, ph, result['configuration']['reference_K'])[0]-np.log(grid))
            axes[1].scatter(t, [r['log_residual'] for r in group], label=f'pH {ph:g}')
        axes[0].set(xlabel='1000/T (K^-1)', ylabel='ln(k/T)')
        axes[1].set(xlabel='T (K)', ylabel='ln(k_pred/k)')
        axes[1].axhline(0, color='gray', lw=.7)
        axes[0].legend()
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output/f'activation_fit.{ext}', dpi=250)
        plt.close(fig)


def interactive_records():
    print('Columnas: T_K pH k; opcional cuarta columna sigma_log_k.')
    print('Ejemplo: 298.15 7.4 0.0085. Punto decimal; k en s^-1. Enter termina.')
    rows = []
    while True:
        line = input(f'Medicion {len(rows)+1}: ').strip()
        if not line:
            break
        try:
            values = list(map(float, line.split()))
            if len(values) not in (3, 4):
                raise ValueError('Ingresar tres o cuatro numeros.')
            row = dict(id=str(len(rows)+1), T_K=values[0], pH=values[1], k=values[2],
                       sigma_log_k=values[3] if len(values)==4 else None)
            validate_records([row])
            rows.append(row)
        except ValueError as exc:
            print(f'Entrada invalida: {exc}')
    return validate_records(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', nargs='?', help='CSV/TSV header: T_K,pH,k; optional id,sigma_log_k')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--reference', type=float, default=298.15)
    parser.add_argument('--fixed-pka', type=float, default=None)
    parser.add_argument('--starts', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.interactive:
            source = input('Archivo CSV/TSV? Enter = ingresar mediciones a mano: ').strip()
            args.table = source or None
            pka = input('pKa fijo a Tref? Enter = ajustarlo libre: ').strip()
            args.fixed_pka = float(pka) if pka else None
            ref = input('Tref (K)? Enter = 298.15: ').strip()
            args.reference = float(ref) if ref else 298.15
            args.output = input('Carpeta de resultados? Enter = results_activation: ').strip() or 'results_activation'
        records = read_records(args.table) if args.table else interactive_records() if args.interactive else None
        if records is None:
            parser.error('Supply a table or use --interactive.')
        result = fit_records(records, args.reference, args.fixed_pka, args.starts, args.seed)
        result['records'] = records
        result['environment'] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
        result['source_sha256'] = hashlib.sha256(Path(args.table).read_bytes()).hexdigest() if args.table else None
        output = args.output or str(Path(args.table).parent/f'results_activation_{Path(args.table).stem}')
        export_result(result, output, not args.no_plot)
        print('Resultados:', Path(output).resolve())
        print('Parametros:', result['parameters'])
        for warning in result['warnings']:
            print('Nota:', warning)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
