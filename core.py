import math as mt
import pyfits as pf
import seaborn as sb
import pandas as pd
import numpy as np

from os.path import join
from IPython.display import HTML
from scipy.ndimage import binary_erosion as be

from pytransit.orbits_f import orbits as of
from pytransit import MandelAgol as MA

from exotk.de import DiffEvol
from exotk.priors import PriorSet, UP, NP, JP
from exotk.utils.orbits import as_from_rhop
from exotk.utils.likelihood import ll_normal_es
from exotk.utils.misc_f import utilities as uf
from k2sc.utils import fold

from emcee import EnsembleSampler
cp = sb.color_palette()

from matplotlib import rc
from numpy import *

rc('figure', figsize=(14,6))
N = lambda a: a/nanmedian(a)

sb.set_style('white')
random.seed(0)

bjdref           = pf.getval('data/raw/ktwo211916756-c05_llc.fits', 'bjdrefi', 1)
tc_bjd           = 2457150.8786
zero_epoch = tc  = tc_bjd - bjdref
period     = p   = 10.1344
duration   = dur = 0.3

from exotk.constants import rsun
from scipy.constants import G
G_cgs = 6.674e-8
Rs = 0.43 * rsun
logg, logg_e = 4.82, 0.06

def logg(rho, r):
    """
    Parameters
    
     rho : mean density [g/cm^3]
       r : radius [Rsun]
    """
    return mt.log10(4*mt.pi/3.*G_cgs*rho*(r*rsun*1e2))


class HNPrior(NP):
    def __call__(self, x, pv=None):
        return exp(self.log(x))

    def log(self, x, pv=None):
        if isinstance(x, np.ndarray):
            return np.select([x <= self.a, (self.a < x) & (x < self.mean), x > self.mean],
                            [-1e18, self._lf1 - (x-self.mean)**2 * self._f2, self._lf1])
        else:
            if x < self.a:
                return -1e18
            elif x > self.mean:
                return self._lf1
            else:
                return self._lf1 -(x-self.mean)**2*self._f2

HP = HNPrior
            
class LPFunction(object):
    """A simple log posterior function class.
    """
    def __init__(self, time, flux, tc, p, nthreads=2, constrain_rho=0, eccentric=False):
        self.tm = MA(lerp=False, supersampling=10, nthr=nthreads) 
        self.nthr = nthreads
        self.eccentric = eccentric
        
        self.time     = time.copy()
        self.flux_o   = flux.copy()
        self.npt      = flux.size
        self._wrk_lc  = zeros_like(time)  
        self._wrk_ld  = zeros(2)        

        
        self.priors = [NP(   tc,   0.02,   'tc'), ##  0  - Transit centre
                       NP(    p,   0.01,    'p'), ##  1  - Period
                       UP( 0.06**2,  0.12**2,  'k2'), ##  2  - planet-star area ratio
                       HP(  1.25,   0.45, 'rho', lims=(0.05,10)), ##  3  - Stellar density
                       UP(     0,   0.99,   'b'), ##  4  - Impact parameter
                       UP(  1e-4,  15e-4,   'e'), ##  5  - White noise std
                       NP(   1.0,   0.01,   'c'), ##  6  - Baseline constant
                       UP(     0,    1.0,  'q1'), ##  7  - limb darkening q1
                       UP(     0,    1.0,  'q2')] ##  8  - limb darkening q2
        if constrain_rho==1:
            self.priors[3] = NP(4.9, 0.5, 'rho', lims=(0.05,10))
        elif constrain_rho == 2:
            self.priors[3] = NP(15.0, 0.5, 'rho', lims=(1.0,20))
        elif constrain_rho >= 3:
            self.priors[3] = UP(1.0, 25.0, 'rho')

        self.logg_prior = NP(4.82, 0.06, lims=(4,6))
        if constrain_rho == 4:
            self.logg_prior = NP(4.5, 0.5, lims=(1,8))
                
        if self.eccentric:
            self.priors.extend([UP(-0.7, 0.7, 'secw'),   ##  9 - sqrt e cos w
                                UP(-0.7, 0.7, 'sesw')])  ## 10 - sqrt e sin w
               
        self.ps = PriorSet(self.priors)
        
        
    def compute_baseline(self, pv):
        """Simple constant baseline model"""
        self._wrk_lc.fill(pv[6])
        return self._wrk_lc

    
    def compute_transit(self, pv):
        """Transit model"""
        _a  = as_from_rhop(pv[3], pv[1])  # Scaled semi-major axis from stellar density and orbital period
        _i  = mt.acos(pv[4]/_a)           # Inclination from impact parameter and semi-major axis
        _k  = mt.sqrt(pv[2])              # Radius ratio from area ratio
        
        a,b = mt.sqrt(pv[7]), 2*pv[8]
        self._wrk_ld[:] = a*b, a*(1.-b)   # Quadratic limb darkening coefficients

        _e, _w = 0., 0.
        if self.eccentric:
            _e = pv[9]**2+pv[10]**2
            _w = mt.atan2(pv[10], pv[9])
            
        return self.tm.evaluate(self.time, _k, self._wrk_ld, pv[0], pv[1], _a, _i, _e, _w)

    
    def compute_lc_model(self, pv):
        """Combined baseline and transit model"""
        return self.compute_baseline(pv) * self.compute_transit(pv)


    def __call__(self, pv):
        """Log posterior density"""
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return -inf
        flux_m = self.compute_lc_model(pv)
        return self.ps.c_log_prior(pv) + ll_normal_es(self.flux_o, flux_m, pv[5]) + self.logg_prior.log(logg(pv[3], 0.43))
