import math as mt
import astropy.io.fits as pf
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

from exotk.constants import rsun
from scipy.constants import G

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

G_cgs = 6.674e-8
Rs = 0.43 * rsun
logg, logg_e = 4.82, 0.06

def T14(p,a,k,I):
    return p/pi*arcsin( (1./a)*sqrt( ((1+k)**2 - (a*cos(I))**2) / (1-cos(I)**2)))

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
    def __init__(self, nthreads=2, logg_prior=None):
        self.tm = MA(lerp=False, supersampling=10, nthr=nthreads) 
        self.nthr = nthreads

        ## Import the K2 data
        ## ------------------
        d  = pf.getdata('data/red/EPIC_211916756_mast.fits', 1)
        time = d.time
        flux_raw = d.flux_1
        trend_t  = d.trend_t_1 - np.nanmedian(d.trend_t_1)
        trend_p  = d.trend_p_1 - np.nanmedian(d.trend_p_1)
        flux = flux_raw - trend_t - trend_p
        flux /= np.nanmedian(flux)

        mask_q = d.quality == 0
        mask_f = np.isfinite(flux)
        mask_o = flux < 1.004
        mask = mask_f & mask_o

        tid_arr  = np.round((time - tc) / p).astype(np.int)
        tid_arr -= tid_arr.min()
        tids     = np.unique(tid_arr)

        phase = p*(fold(time, p, tc, shift=0.5) - 0.5)
        pmask = np.abs(phase) < 6*0.5*dur
        phases  = [phase[tid_arr==tt]  for tt in np.unique(tids)]
        times   = [time[tid_arr==tt]   for tt in np.unique(tids)]
        fluxes  = [flux[tid_arr==tt]   for tt in np.unique(tids)]

        self.time     = time[pmask&mask].copy()
        self.flux     = flux[pmask&mask].copy()
        self.npt      = self.flux.size

        ## Define priors
        ## -------------
        self.priors = [NP(     tc,   0.02,   'tc'), ##   0 - Transit centre
                       NP(      p,   0.01,    'p'), ##   1 - Period
                       UP( 0.0035, 0.0145,   'k2'), ##   2 - planet-star area ratio
                       HP(   1.25,   0.45,  'rho', lims=(0.05,15)), ##  3 - Stellar density
                       UP(      0,   0.99,    'b'), ##   4 - Impact parameter
                       UP(   1e-4,  15e-4,    'e'), ##   5 - White noise std
                       NP(    1.0,   0.01,    'c'), ##   6 - Baseline constant
                       UP(      0,    1.0,   'q1'), ##   7 - limb darkening q1
                       UP(      0,    1.0,   'q2'), ##   8 - limb darkening q2
                       UP(   -0.8,    0.8, 'secw'), ##   9 - sqrt(e) cos(w)
                       UP(   -0.8,    0.8, 'sesw')] ##  10 - sqrt(e) sin(w)
        self.ps = PriorSet(self.priors)
        self.logg_prior = logg_prior or UP(0, 10, 'logg')
        
                
    def compute_baseline(self, pv):
        """Constant baseline model"""
        return pv[6]

    
    def compute_transit(self, pv, times=None):
        """Transit model"""
        _a  = as_from_rhop(pv[3], pv[1])  # Scaled semi-major axis from stellar density and orbital period
        _i  = mt.acos(pv[4]/_a)           # Inclination from impact parameter and semi-major axis
        _k  = mt.sqrt(pv[2])              # Radius ratio from area ratio
        
        a,b = mt.sqrt(pv[7]), 2*pv[8]     # Mapping the limb darkening coefficients from the Kipping (2014) 
        _uv = np.array([a*b, a*(1.-b)])   # parameterisation to quadratic MA coefficients

        _e = pv[9]**2+pv[10]**2           # Eccentricity
        _w = mt.atan2(pv[10], pv[9])      # Argument of periastron

        times = self.time if times is None else times
        return self.tm.evaluate(times, _k, _uv, pv[0], pv[1], _a, _i, _e, _w)

    
    def compute_lc_model(self, pv, times=None):
        """Combined baseline and transit model"""
        return self.compute_baseline(pv) * self.compute_transit(pv, times)


    def __call__(self, pv):
        """Log posterior density"""
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return -inf
        flux_m = self.compute_lc_model(pv)
        return self.ps.c_log_prior(pv) + ll_normal_es(self.flux, flux_m, pv[5]) + self.logg_prior.log(logg(pv[3], 0.43))
