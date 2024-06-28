import sys
import os

import numpy as np
import pandas as pd
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt
from scipy import stats
from scipy.integrate import quad
from scipy.signal import argrelmin, argrelmax

# My packages
sys.path.insert(0, "../scs/")
import data_plotting as dplt
import data_preparation as dp
from prepare_datasets_for_training import extract
from data_preparation import extract_dataframe

from icecream import ic
from importlib import reload

rng = np.random.RandomState(1415)


def wavelength_to_velocity(lambda_obs, lambda_em):
    vel = c.c * ((lambda_obs / lambda_em) - 1)
    return vel.to(u.km / u.s)


def find_spectral_line(wvl, flx, spectral_line_center, wvl_lookback, plot=False):
    wvl_range = spectral_line_center - wvl_lookback, spectral_line_center
    ind = np.where(np.logical_and(wvl >= wvl_range[0], wvl <= wvl_range[1]))[0]
    
    spectral_min_ind = argrelmin(flx[ind])[0]
    if spectral_min_ind.size > 1:
        warning_flag = True
    else:
        warning_flag = False
    ic(spectral_min_ind)
    
    wvl_min = wvl[ind][spectral_min_ind[-1]]
    flx_min = flx[ind][spectral_min_ind[-1]]
    wvl_min = wvl[ind][spectral_min_ind[-5]]
    flx_min = flx[ind][spectral_min_ind[-5]]
    ic(wvl_min)
    ic(flx_min)
    
    line_velocity = wavelength_to_velocity(wvl_min, spectral_line_center)
    ic(line_velocity)
    
    if plot:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title(f"Measured line velocity: {line_velocity}")
        ax.plot(wvl, flx, c="k", label="Original spectrum")
        ax.plot(wvl[ind], flx[ind], c="tab:red", label="Lookback area")
        
        ax.axvline(x=spectral_line_center, c="k", ls=":", label=f"Theortical Spectral Line: {spectral_line_center}")
        ax.axvline(x=wvl_min, c="tab:red", ls=":", label=f"Measured Spectral Line: {wvl_min}")
        
        if spectral_min_ind.size > 1:
            for line_ind in spectral_min_ind[:-1]:
                line = wvl[ind][line_ind]
                ax.axvline(x=line, c="tab:green", ls=":", label=f"Additional minimum: {line}")
        
        ax.set_xlim(4000, 7500)
        ax.legend()
        fig.show()
    
    
    return wvl_min, flx_min, line_velocity, warning_flag


def find_spectral_shoulders(wvl, flx, wvl_min, plot=False):
    wvl_min_ind = np.where(wvl == wvl_min)[0][0]
    flx_after = flx[wvl_min_ind:]
    flx_befor = flx[:wvl_min_ind]
    
    # shoulder_red_ind = argrelmax(flx_after)[0][0]
    shoulder_red_ind = argrelmax(flx_after)[0][4]
    shoulder_blu_ind = argrelmax(flx_befor)[0][-8]
    
    wvl_shoulder_red = wvl[wvl_min_ind + shoulder_red_ind]
    wvl_shoulder_blu = wvl[shoulder_blu_ind]
    ic(wvl_shoulder_red)
    ic(wvl_shoulder_blu)
    

    if plot:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title("Finding spectral shoulders")
        ax.plot(wvl, flx, c="k", label="Original spectrum")
        ax.axvline(x=wvl_min, c="k", ls=":", label=f"Measured Spectral Line: {wvl_min}")
        
        ax.axvline(x=wvl_shoulder_red, c="tab:red", label=f"Red spectral shoulder at: {wvl_shoulder_red}")
        ax.axvline(x=wvl_shoulder_blu, c="tab:blue", label=f"Blue spectral shoulder at: {wvl_shoulder_blu}")
        
        ax.set_xlim(4000, 7500)
        ax.legend()
        fig.show()
        
    return wvl_shoulder_blu, wvl_shoulder_red


def calc_pEW(wvl, flx, wvl_min, flx_min, wvl_shoulder_blu, wvl_shoulder_red, plot=False):
    shoulder_red_ind = np.where(wvl == wvl_shoulder_red)[0][0]
    shoulder_blu_ind = np.where(wvl == wvl_shoulder_blu)[0][0]
    flx_shoulder_red = flx[shoulder_red_ind]
    flx_shoulder_blu = flx[shoulder_blu_ind]
    
    interp_ind = np.where(np.logical_and(wvl >= wvl_shoulder_blu, wvl <= wvl_shoulder_red))[0]
    x = wvl[interp_ind]
    xp = [wvl_shoulder_blu, wvl_shoulder_red]
    fp = [flx_shoulder_blu, flx_shoulder_red]
    pseudo_cont = np.interp(x, xp, fp)
    
    # Calc psuedo equivalent width (pEW)
    flux_change = (pseudo_cont - flx[interp_ind]) / (pseudo_cont)
    pEW = np.trapz(flux_change, x)
    ic(pEW)
    
    delta_lam = np.diff(x)
    pEW = np.sum(delta_lam * flux_change[:-1])
    ic(pEW)
    
    
    if plot:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title("Construct pseudo-continuum and calculate pEW")
        ax.plot(wvl, flx, c="k", label="Original spectrum")
        ax.axvline(x=wvl_min, c="k", ls=":", label=f"Measured Spectral Line: {wvl_min}")
        ax.axvline(x=wvl_shoulder_red, c="tab:red", label=f"Red spectral shoulder at: {wvl_shoulder_red}")
        ax.axvline(x=wvl_shoulder_blu, c="tab:blue", label=f"Blue spectral shoulder at: {wvl_shoulder_blu}")
        ax.plot(x, pseudo_cont, c="tab:orange", label="Pseudo-continuum")
        ax.fill_between(x, y1=pseudo_cont, y2=flx[interp_ind], color="tab:blue", alpha=0.5, label=f"pEW = {pEW}")
        
        
        ax.set_xlim(4000, 7500)
        ax.legend()
        fig.show()


    return pEW


def find_spectral_line_from_df(df, df_ind, spectral_line_center, wvl_lookback):
    extraction = extract_dataframe(df)

    index = extraction[0]
    wvl = extraction[1]
    flux_columns = extraction[2]
    metadata_columns = extraction[3]
    df_fluxes = extraction[4]
    df_metadata = extraction[5]
    fluxes = extraction[6]
    
    flx = fluxes[df_ind]

    wvl_min, flx_min, line_velocity, warning_flag = find_spectral_line(wvl, flx, spectral_line_center, wvl_lookback)
    
    wvl_shoulder_blu, wvl_shoulder_red = find_spectral_shoulders(wvl, flx, wvl_min)
    
    pEW = calc_pEW(wvl, flx, wvl_min, flx_min, wvl_shoulder_blu, wvl_shoulder_red)
    
    return pEW

