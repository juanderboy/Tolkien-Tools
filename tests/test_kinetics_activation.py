import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'TolkienTools'/'kinetics'))
from kinet_activation import evaluate, fit_records, validate_records

class ActivationTests(unittest.TestCase):
    def test_jacobian_and_local_enthalpy(self):
        p=np.array([-7., -4., 115., 27., 24., 7.55])
        t=np.array([291., 301., 315.]); ph=np.array([6.8, 7.4, 8.])
        lk,j,a,w=evaluate(p,t,ph)
        for i in range(6):
            d=np.zeros(6);d[i]=1e-5
            np.testing.assert_allclose(j[:,i],(evaluate(p+d,t,ph)[0]-evaluate(p-d,t,ph)[0])/2e-5,atol=1e-8)
        u=1/t;d=1e-7;tp=1/(u+d);tm=1/(u-d)
        from kinet_activation import R
        numeric=-R*((evaluate(p,tp,ph)[0]-np.log(tp))-(evaluate(p,tm,ph)[0]-np.log(tm)))/(2*d)
        np.testing.assert_allclose(numeric,(1-w)*p[2]+w*p[3]+(w-a)*p[4],rtol=1e-6)

    def test_recover_known_parameters_free_and_fixed(self):
        p=np.array([-7., -4., 115., 27., 24., 7.55])
        rows=[dict(id=str(i),T_K=t,pH=ph,k=float(np.exp(evaluate(p,t,ph)[0])))
              for i,(t,ph) in enumerate(( (t,ph) for t in (285.,295.,305.,315.) for ph in (6.,7.,8.,9.) ))]
        for fixed in (None,7.55):
            result=fit_records(rows,fixed_pka=fixed,starts=8,seed=12)
            np.testing.assert_allclose(list(result['parameters'].values()),p,atol=1e-5)
            self.assertLess(result['RMS_log'],1e-9)

    def test_invalid_values_and_partial_weights(self):
        for bad in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):
                validate_records([dict(id='a',T_K=298.,pH=7.,k=bad)])
        with self.assertRaises(ValueError):
            validate_records([dict(id='a',T_K=298.,pH=7.,k=.01,sigma_log_k=.1),dict(id='b',T_K=300.,pH=8.,k=.02)])
