"""FeMb/disulfide reversible fits with explicit spectrokinetic selection.

Selection explores rate families using spectral and concentration consistency.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from kinet_common import FitResult, MODEL_SPECIES
from kinet_fitting import (solve_small_nnls_batch, fit_direct_spectra,
                          direct_parameter_names_and_bounds, known_scale_parameters,
                          extract_known_scale_report)
from kinet_linalg import factor_analysis

MODEL = "femb_disulfide_reversible"
IRREVERSIBLE_MODEL = "femb_disulfide_irreversible"
RATE_BOUNDS = {"k_on": (1e-5, 5.), "k_slow": (1e-6, 1.), "k_off": (1e-7, 5.)}
PEAK_WINDOWS = ((405., 414.), (423., 430.), (432., 438.))
IRREVERSIBLE_PEAK_WINDOWS = ((405., 414.), (427., 433.), (432., 438.))
RELATIVE_RMSE_TOLERANCE = .001
INTERMEDIATE_PRODUCT_PEAK_RATIO = (.5, .95)


def fractions(t, kon, kred, koff):
    if koff == 0:
        from kinet_models import concentration_profile_a_to_b_to_c
        return concentration_profile_a_to_b_to_c(np.asarray(t), kon, kred)
    total = kon + kred + koff
    delta = np.sqrt(max(total*total-4*kon*kred, 1e-30))
    fast = (total+delta)/2
    slow = kon*kred/fast
    es, ef = np.exp(-slow*t), np.exp(-fast*t)
    b = kon*(es-ef)/delta
    a = ((fast-kon)*es+(kon-slow)*ef)/delta
    return np.vstack((a,b,1-a-b))


def spectral_fractions(spectra, observed):
    """Project each observed frame onto the nonnegative sum-one simplex."""
    g = spectra.T@spectra
    h = observed.T@spectra
    contrast = spectra[:,1:]-spectra[:,0,None]
    bc = np.linalg.lstsq(contrast, observed-spectra[:,0,None], rcond=None)[0].T
    f = np.column_stack((1-bc.sum(axis=1),bc))
    error = np.einsum('ij,jk,ik->i',f,g,f)-2*np.sum(h*f,axis=1)
    error[(f<0).any(axis=1)] = np.inf
    for i,j in ((0,1),(0,2),(1,2)):
        denominator = max(g[i,i]+g[j,j]-2*g[i,j],1e-30)
        x = np.clip((h[:,i]-h[:,j]-g[i,j]+g[j,j])/denominator,0,1)
        trial = np.zeros_like(f); trial[:,i]=x; trial[:,j]=1-x
        e = np.einsum('ij,jk,ik->i',trial,g,trial)-2*np.sum(h*trial,axis=1)
        better = e<error; f[better]=trial[better]; error[better]=e[better]
    return f.T


def select_candidate(candidates):
    """Keep absorption-equivalent candidates, then prioritize profile agreement."""
    minimum = min(r['rmse_mAU'] for r in candidates)
    comparable = [r for r in candidates if r['rmse_mAU'] <= minimum*(1+RELATIVE_RMSE_TOLERANCE)+1e-9]
    chemical = [r for r in comparable if r['chemical_pass']]
    pool = chemical or comparable
    chosen = min(pool,key=lambda r:(r['profile_score'],r['rmse_mAU']))
    return chosen, minimum, len(comparable), len(chemical)


def fit_femb_disulfide(experiment, c0=1., n_components=3, method='nnls',
        known_spectra=None, known_species=(), known_spectrum_scale_bounds=None,
        fix_initial_spectrum=False, fix_final_spectrum=False,
        initial_spectrum_weight=0., progress_callback=None, irreversible=False):
    """Explore reversible eigenvalue families and rank concentration consistency."""
    model = IRREVERSIBLE_MODEL if irreversible else MODEL
    rate_bounds = {k:v for k,v in RATE_BOUNDS.items() if not (irreversible and k=='k_off')}
    peak_windows = IRREVERSIBLE_PEAK_WINDOWS if irreversible else PEAK_WINDOWS
    if method!='nnls' or n_components!=3:
        raise ValueError('FeMb + disulfides binding requires NNLS and three species')
    if not np.isfinite(c0) or c0<=0:
        raise ValueError('c0 must be positive and finite')
    t=experiment.t; wave=experiment.wavelength; y=experiment.absorbance
    if len(t)<2 or not np.isfinite(t).all() or np.any(t<0) or np.any(np.diff(t)<=0):
        raise ValueError("FeMb fit times must be finite, nonnegative and increasing")
    soret=(wave>=400)&(wave<=450)
    if not soret.any() or wave.min()>405 or wave.max()<438:
        raise ValueError('FeMb selection requires spectral coverage of 405-438 nm')
    q,w,sv=factor_analysis(y,3)
    names,bounds=direct_parameter_names_and_bounds(tuple(rate_bounds),known_species,
        (1e-7,5.),known_spectrum_scale_bounds)
    bounds.update(rate_bounds)
    lower=np.log([bounds[n][0] for n in names]); upper=np.log([bounds[n][1] for n in names])
    known_abs=None if known_spectra is None else known_spectra*c0
    fast_free=(known_abs is None and not fix_initial_spectrum and not fix_final_spectrum
               and initial_spectrum_weight==0)
    late=t>=5
    if not late.any(): late=np.ones(len(t),dtype=bool)

    def build(parameters):
        f=fractions(t,parameters['k_on'],parameters['k_slow'],0. if irreversible else parameters['k_off'])
        if fast_free:
            spectra=solve_small_nnls_batch(f.T,y)
        else:
            spectra=fit_direct_spectra(y,f,'nnls',initial_spectrum_weight=initial_spectrum_weight,
                known_spectra=known_abs,
                known_spectrum_scales=known_scale_parameters(parameters,model,known_species),
                fix_initial_spectrum=fix_initial_spectrum,fix_final_spectrum=fix_final_spectrum)
        pred=spectra@f
        mse=float(np.mean((y-pred)**2))
        return mse,spectra,f,pred

    def objective(log_parameters):
        parameters=dict(zip(names,np.exp(log_parameters)))
        mse=build(parameters)[0]
        if progress_callback: progress_callback(parameters,float(np.sqrt(mse*y.size)))
        return mse

    seeds=((.4,.005,.1),(.8,.04,.01),(.2,.02,.4),(.1,.002,.05))
    optimized=[]
    for rates in seeds:
        start=dict(zip(RATE_BOUNDS,rates))
        initial=np.array([start.get(name,1.) for name in names])
        result=minimize(objective,np.clip(np.log(initial),lower,upper),method='L-BFGS-B',
            bounds=list(zip(lower,upper)),options=dict(maxiter=350,ftol=1e-14,gtol=1e-11,maxls=30))
        if np.isfinite(result.fun): optimized.append(dict(zip(names,np.exp(result.x))))
    if not optimized: raise RuntimeError('No finite FeMb fit from the deterministic starting points')

    records=[]
    preview_spectra=[]
    def evaluate(parameters,origin):
        mse,spectra,f,pred=build(parameters)
        est=spectral_fractions(spectra,y)
        rms=np.sqrt(np.mean((est[:,late]-f[:,late])**2,axis=1))
        peaks=[float(wave[soret][np.argmax(spectra[soret,i])]) for i in range(3)]
        peak_pass=all(lo<=peak<=hi for peak,(lo,hi) in zip(peaks,peak_windows))
        peak_heights=np.max(spectra[soret],axis=0)
        ratio=float(peak_heights[1]/max(peak_heights[2],1e-30))
        amplitude_pass=INTERMEDIATE_PRODUCT_PEAK_RATIO[0]<=ratio<=INTERMEDIATE_PRODUCT_PEAK_RATIO[1]
        chemical=peak_pass and amplitude_pass
        r=dict(parameters={k:float(v) for k,v in parameters.items()},source=origin,
            rmse_mAU=1000*np.sqrt(mse),profile_rms=rms.tolist(),profile_score=float(np.mean(rms*rms)),
            peaks_nm=peaks,chemical_pass=chemical,peak_positions_pass=peak_pass,
            B_peak_over_C_peak=ratio,amplitude_ratio_pass=amplitude_pass,
            B409_over_max=float(np.interp(409,wave,spectra[:,1])/max(spectra[:,1].max(),1e-30)))
        records.append(r)
        preview_spectra.append(spectra/c0)
        return r

    for p in optimized: evaluate(p,'multistart')
    best_mse=min(r['rmse_mAU'] for r in records)
    families=[]
    if not irreversible:
        for p,r in zip(optimized,records[:]):
            if r['rmse_mAU']>best_mse*(1+RELATIVE_RMSE_TOLERANCE)+1e-9: continue
            total=p['k_on']+p['k_slow']+p['k_off']; product=p['k_on']*p['k_slow']
            invariant=np.array([total,product])
            if any(np.allclose(invariant,old,rtol=1e-4,atol=0) for old in families): continue
            families.append(invariant)
            fast=(total+np.sqrt(max(total*total-4*product,1e-30)))/2
            slow=product/fast
            lo=max(RATE_BOUNDS['k_slow'][0],slow*(1+1e-6))
            hi=min(RATE_BOUNDS['k_slow'][1],fast*(1-1e-6))
            if lo>=hi: continue
            for kred in np.unique(np.r_[np.geomspace(lo,hi,241),p['k_slow']]):
                kon=product/kred; koff=total-kon-kred
                if not RATE_BOUNDS['k_on'][0]<=kon<=RATE_BOUNDS['k_on'][1]: continue
                if not RATE_BOUNDS['k_off'][0]<=koff<=RATE_BOUNDS['k_off'][1]: continue
                trial=dict(p,k_on=kon,k_slow=kred,k_off=koff)
                evaluate(trial,'eigenvalue_family')

    chosen,minimum,n_comparable,n_chemical=select_candidate(records)
    diagnostics=dict(preset=model,rate_bounds=rate_bounds,
        fixed_parameters={'k_off':0.} if irreversible else {},
        relative_rmse_tolerance=RELATIVE_RMSE_TOLERANCE,peak_windows_nm=peak_windows,
        intermediate_product_peak_ratio_bounds=INTERMEDIATE_PRODUCT_PEAK_RATIO,
        profile_time_min_s=5.,profile_metric='mean of three squared species RMS fraction discrepancies',
        shoulder_metric='B409/max diagnostic only, not selection penalty',
        n_candidates=len(records),n_comparable=n_comparable,n_chemical=n_chemical,
        chemical_filter_passed=bool(n_chemical),minimum_rmse_mAU=minimum,selected=chosen,candidates=records)
    preview_spectra = np.asarray(preview_spectra)

    def reconstruct(index):
        record = records[index]
        parameters = record['parameters']
        candidate_mse, candidate_spectra, candidate_f, candidate_pred = build(parameters)
        candidate_est = spectral_fractions(candidate_spectra, y)
        candidate_diagnostics = dict(diagnostics, selected=record,
                                     selection_mode='automatic', selected_candidate_index=int(index))
        return FitResult(method='nnls', model=model,
            params={k:parameters[k] for k in rate_bounds}, species_labels=MODEL_SPECIES[model],
            c=candidate_f*c0, spectra=candidate_spectra/c0,
            absorbance_calc=candidate_pred, residuals=y-candidate_pred,
            singular_values=sv, q=q, w=w, error=float(np.sqrt(candidate_mse*y.size)),
            known_species=known_species,
            known_spectrum_scales=extract_known_scale_report(parameters,known_species),
            fixed_initial_spectrum=fix_initial_spectrum, fixed_final_spectrum=fix_final_spectrum,
            selection_diagnostics=candidate_diagnostics,
            estimated_concentrations=candidate_est*c0,
            candidate_reconstructor=reconstruct, candidate_spectra=preview_spectra)

    return reconstruct(next(i for i, record in enumerate(records) if record is chosen))
