"""Bounded regressions for the September 2026 independent audit."""
import unittest
import numpy as np
import pandas as pd
from didintpy import Design, InferenceError
from didintpy.fitted_didint import prepare_fitted_first_stage
from didintpy.didint import run_didint
from unittest.mock import patch


def fixture(missing_level=False):
    rng = np.random.default_rng(20260930)
    rows = []
    for state in range(8):
        for time in range(1, 4):
            for person in range(18):
                category = person % (2 if missing_level and state == 5 and time == 2 else 3)
                x = rng.normal() + state / 10
                g = 3 if state < 3 else 0
                y = state / 4 + time / 3 + (.4 + state / 20) * x + .2 * category
                y += (g > 0 and time >= g) + rng.normal(scale=.1)
                rows.append((state,time,g,x,y,category))
    d = pd.DataFrame(rows,columns=['state','time','g','x','y','category'])
    for k in range(3):
        d[f'd{k}'] = (d.category == k).astype(float)
    d['twox'] = 2*d.x
    return d


def design(d, covs):
    return Design(d,outcome='y',cluster='state',time='time',
                  assignment={s: 3 if s < 3 else 0 for s in range(8)},covariates=covs)


class AuditRegressions(unittest.TestCase):
    def test_saturation_counts_absorbed_intercepts(self):
        # One slope plus the cell intercept exactly fits two observations.
        d=pd.DataFrame(dict(state=[1,1],time=[1,1],g=[0,0],x=[0.,1.],y=[2.,3.]))
        for ccc in ('int','hom','state','time'):
            with self.subTest(ccc=ccc), self.assertRaisesRegex(InferenceError,'saturated'):
                prepare_fitted_first_stage(d,outcome_col='y',state_col='state',time_col='time',
                                          gvar_col='g',covariates=['x'],ccc=ccc)

    def test_legacy_api_warns_and_labels_its_target(self):
        with patch('didintpy.didint.run_undid',return_value={}), self.assertWarnsRegex(FutureWarning,'Legacy intercept'):
            result=run_didint(pd.DataFrame())
        self.assertEqual(result['estimator_variant'],'legacy_intercept')
        self.assertFalse(result['target_standardized'])

    def test_missing_category_excludes_only_ineligible_control(self):
        data=fixture(True)
        values=[]
        for reference in range(3):
            covs=['x']+[f'd{k}' for k in range(3) if k!=reference]
            fit=design(data,covs)
            values.append(fit.point())
            self.assertTrue(all(cell[2]==4 for cell in fit.last_support))
            sample=data.loc[data.state!=5]
            check=Design(sample,outcome='y',cluster='state',time='time',covariates=covs,
                         assignment={s:3 if s<3 else 0 for s in sample.state.unique()})
            self.assertAlmostEqual(values[-1],check.point(),places=10)
        np.testing.assert_allclose(values,values[0],atol=1e-11,rtol=0)

    def test_full_rank_reference_and_location_invariance(self):
        d = fixture()
        values = [design(d,['x']+[f'd{k}' for k in range(3) if k!=ref]).point() for ref in range(3)]
        np.testing.assert_allclose(values,values[0],atol=1e-11,rtol=0)
        shifted = d.copy();shifted['x'] += 10
        self.assertAlmostEqual(design(shifted,['x','d1','d2']).point(),values[0],11)

    def test_all_slope_specifications_reject_collinearity(self):
        for ccc in ('int','hom','state','time'):
            with self.subTest(ccc=ccc), self.assertRaisesRegex(InferenceError,'numerical_failure'):
                prepare_fitted_first_stage(fixture(),outcome_col='y',state_col='state',time_col='time',
                                          gvar_col='g',covariates=['x','twox'],ccc=ccc)

    def test_missing_values_not_silently_dropped(self):
        for column in ('y','x','state','time'):
            d = fixture();d.loc[0,column] = np.nan
            with self.subTest(column=column), self.assertRaisesRegex(InferenceError,'invalid_input'):
                design(d,['x'])

    def test_direct_fitted_api_rejects_missing_values(self):
        d=fixture();d.loc[0,'x']=np.nan
        with self.assertRaisesRegex(InferenceError,'invalid_input'):
            prepare_fitted_first_stage(d,outcome_col='y',state_col='state',time_col='time',gvar_col='g',covariates=['x'])

    def test_roundoff_jackknife_is_not_significant(self):
        d=design(fixture(),[])
        d.point=lambda *a,omit=None,**kw: 1.0 if omit is None else 1.0+(omit-3)*1e-16
        fit=d.jackknife()
        self.assertEqual(fit['diagnostics']['status'],'zero_standard_error')
        self.assertTrue(np.isnan(fit['p_value']))

if __name__ == '__main__':
    unittest.main()
