import os
import sys
import pandas as pd
from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt
from information_metrics import *
import xarray as xr

import warnings
warnings.simplefilter("ignore")


data_dir = 'M:/Original_data/FLUXNET/XAI/summa_sa_nn1w_nn2w_output'
plotdir =  '../../PROJECTS/InfoFlow_SourceDep/WRR_figures'

sites_df = pd.read_csv('site_list.csv')


def air_vapor_pressure_sat(air_temp):
    #Pa
    eStar = 610.8 * np.exp((17.27 * air_temp) / (air_temp + 237.3)) 
    return eStar


def sfe_hyp(air_temp, qa):
    rv = 461.5           # (J kg−1 K−1) is the gas constant for water vapor.
    cp_air = 1005        # (J kg−1 K−1) is the specific heat capacity of air at constant pressure
    Lv = 2.5008 * 10**6  # (J kg−1) is the latent heat of vaporization of water
    mwratio = 0.622      # (-) Ratio molecular weight of water vapor/dry air
    Bo = rv * cp_air * (air_temp + 273.25) ** 2 / (Lv ** 2 * qa)
    return Bo


def get_data(site, daytime=True, daily_resample=True, filter_conditions=False):
    fd = '%s.nc' % site
    data = xr.open_dataset(os.path.join(data_dir, fd))
    data = data.to_dataframe()
    data = data.unstack()
    aridity_index  = np.nansum(data[('Qle_cor', 'SUMMA-SA')] + data[('Qh_cor', 'SUMMA-SA')]) / np.nansum(data[('pptrate', 'SUMMA-SA')] * 2.5008 * 10**6)

    if daytime:
        data = data[data[('SWRadAtm', 'SUMMA-SA')] > 50] # select daytime
    data_c = data.resample('D').count()
    if daily_resample:
        data = data.resample('D').mean()
        data = data[data_c[('Qle_cor', 'SUMMA-SA')]>=12]

    data['Qn'] = data[('Qle_cor', 'SUMMA-SA')] + data[('Qh_cor', 'SUMMA-SA')]
    data['LE'] = data[('Qle_cor', 'SUMMA-SA')]
    data['H'] = data[('Qh_cor', 'SUMMA-SA')]
    data['B'] = data[('Qh_cor', 'SUMMA-SA')] / data[('Qle_cor', 'SUMMA-SA')]
    data['qa'] = data[('spechum', 'SUMMA-SA')]
    data['P_air'] = data[('airpres', 'SUMMA-SA')]
    data['T_air'] = data[('airtemp', 'SUMMA-SA')] - 273
    data['RH'] = data[('scalarVPair', 'SUMMA-SA')]  / air_vapor_pressure_sat(data['T_air'])

    data['B_SFE'] = sfe_hyp(data['T_air'], data['qa'])
    data['B_SUMMA'] = data[('scalarSenHeatTotal', 'SUMMA-SA')] / data[('scalarLatHeatTotal', 'SUMMA-SA')]
    data['B_SUMMA_NN1'] = data[('scalarSenHeatTotal', 'SUMMA-NN1W')] / data[('scalarLatHeatTotal', 'SUMMA-NN1W')]
    data['B_SUMMA_NN2'] = data[('scalarSenHeatTotal', 'SUMMA-NN2W')] / data[('scalarLatHeatTotal', 'SUMMA-NN2W')]

    data['EF'] = 1 / (1 + data['B'])
    for k in ['SFE', 'SUMMA', 'SUMMA_NN1', 'SUMMA_NN2']:
        data['EF_%s' % k] = 1 / (1 + data['B_%s' % k])
    data['LE_SFE'] = data['EF_SFE'] * (data[('scalarNetRadiation', 'SUMMA-SA')] - data[('scalarGroundNetNrgFlux', 'SUMMA-SA')])
    data['LE_SUMMA'] = - data[('scalarLatHeatTotal', 'SUMMA-SA')]
    data['LE_SUMMA_NN1'] = - data[('scalarLatHeatTotal', 'SUMMA-NN1W')]
    data['LE_SUMMA_NN2'] = - data[('scalarLatHeatTotal', 'SUMMA-NN2W')]

    data['H_SFE'] = data['B_SFE'] * data['LE_SFE'] 
    data['H_SUMMA'] = - data[('scalarSenHeatTotal', 'SUMMA-SA')]
    data['H_SUMMA_NN1'] = - data[('scalarSenHeatTotal', 'SUMMA-NN1W')]
    data['H_SUMMA_NN2'] = - data[('scalarSenHeatTotal', 'SUMMA-NN2W')]

    data['SWC'] = (data[('scalarTotalSoilWat', 'SUMMA-SA')] - np.nanmin(data[('scalarTotalSoilWat', 'SUMMA-SA')]))\
                        / (np.nanmax(data[('scalarTotalSoilWat', 'SUMMA-SA')]) - np.nanmin(data[('scalarTotalSoilWat', 'SUMMA-SA')]))
    
    data['month'] = data.index.month
    
    data = data[['B', 'B_SFE', 'B_SUMMA', 'B_SUMMA_NN1', 'B_SUMMA_NN2',
                 'EF', 'EF_SFE', 'EF_SUMMA', 'EF_SUMMA_NN1', 'EF_SUMMA_NN2',
                 'LE', 'LE_SFE', 'LE_SUMMA', 'LE_SUMMA_NN1', 'LE_SUMMA_NN2',
                 'H', 'H_SFE', 'H_SUMMA', 'H_SUMMA_NN1', 'H_SUMMA_NN2',
                 'P_air', 'qa', 'RH', 'T_air', 'SWC', 'Qn', 'month']].dropna()
    
    if filter_conditions:
        data = data[(data['H']>0) & (data['LE']>0) & (data['B']>0)] # select unstable BL & LE>0
    return data, aridity_index


def global_normalize(data_i, target_o, target_m, source1, source2, global_bounds):
    [[pl_t, pu_t], [pl_s1, pu_s1], [pl_s2, pu_s2]] = global_bounds
    
    data_i[source1] = (data_i[source1] - pl_s1) / (pu_s1 - pl_s1)
    data_i[source1] = [e if e>0 else 0 for e in data_i[source1]]
    data_i[source1] = [e if e<1 else 1 for e in data_i[source1]]

    data_i[source2] = (data_i[source2] - pl_s2) / (pu_s2 - pl_s2)
    data_i[source2] = [e if e>0 else 0 for e in data_i[source2]]
    data_i[source2] = [e if e<1 else 1 for e in data_i[source2]]

    data_i[target_o] = (data_i[target_o] - pl_t) / (pu_t - pl_t)
    data_i[target_o] = [e if e>0 else 0 for e in data_i[target_o]]
    data_i[target_o] = [e if e<1 else 1 for e in data_i[target_o]]

    data_i[target_m] = (data_i[target_m] - pl_t) / (pu_t - pl_t)
    data_i[target_m] = [e if e>0 else 0 for e in data_i[target_m]]
    data_i[target_m] = [e if e<1 else 1 for e in data_i[target_m]]

    return data_i


def local_normalize(data_i, target_o, target_m, source1, source2):
    data_i[source1] = (data_i[source1] - np.nanmin(data_i[source1])) \
                        / (np.nanmax(data_i[source1]) - np.nanmin(data_i[source1]))
    
    data_i[source2] = (data_i[source2] - np.nanmin(data_i[source2])) \
                        / (np.nanmax(data_i[source2]) - np.nanmin(data_i[source2]))
    
    data_i[target_m] = (data_i[target_m] - np.nanmin(data_i[target_o])) \
                        / (np.nanmax(data_i[target_o]) - np.nanmin(data_i[target_o]))
    data_i[target_m] = [i if i>0 else 0 for i in data_i[target_m]]
    data_i[target_m] = [i if i<1 else 1 for i in data_i[target_m]]

    data_i[target_o] = (data_i[target_o] - np.nanmin(data_i[target_o])) \
                            / (np.nanmax(data_i[target_o]) - np.nanmin(data_i[target_o]))
    
    return data_i


def get_decomp_results(target_o, source1, source2, model_list, global_bounds, nbins=50, binning='global', n_lim=100, daily_resample=True):
    
    bins = np.linspace(0, 1, nbins + 1)
    results_models = {}
    result_dict = {}
    for model_test in model_list:
        results_models[model_test] = []
        result_dict[model_test] = {}

    for site in sites_df['id']:
        data, aridity_index = get_data(site, filter_conditions=True, daily_resample=daily_resample)
        
        for m in range(1, 13):
            for model_test in model_list:  
                target_m = '%s_%s' % (target_o, model_test)
                data_i = data[data['month']==m]   
                swc_i = np.nanmean(data_i['SWC'])
                
                data_i = global_normalize(data_i, target_o, target_m, source1, source2, global_bounds)
                ll = len(data_i.index)
                if ll >= n_lim:
                    if binning == 'local':
                        data_i = local_normalize(data_i, target_o, target_m, source1, source2)
                   
                    data_i = data_i[[source1,  source2, target_o, target_m]]

                    isource = mutual_information(data_i, source1,  source2,  bins)
                    H_s1 = shannon_entropy(data_i[source1].values, [bins])
                    H_s2 = shannon_entropy(data_i[source2].values, [bins])
                    H_tar = shannon_entropy(data_i[target_o].values, [bins])
                    H_tar_m = shannon_entropy(data_i[target_m].values, [bins])
                    mi_t = mutual_information(data_i, target_o,  target_m,  bins)

                    isource_nmin = isource / (np.min([H_s1, H_s2]))
                    isource_nmax = isource / (np.max([H_s1, H_s2]))

                    ap = 1 -  mi_t / H_tar

                    tot, unique_1, unique_2, red, syn = information_partitioning(data_i, source1, source2, target_o, bins)
                    unique_1 = unique_1 / tot
                    unique_2 = unique_2 / tot
                    red = red / tot
                    syn = syn / tot
                    tot = tot / H_tar

                    tot_m, unique_1_m, unique_2_m, red_m, syn_m = information_partitioning(data_i, source1, source2, target_m, bins)
                    unique_1_m = unique_1_m / tot_m
                    unique_2_m = unique_2_m / tot_m
                    red_m = red_m / tot_m
                    syn_m = syn_m / tot_m
                    tot_m = tot_m / H_tar_m
                    
                    l_x = len([i for i in [ap, isource, tot, unique_1, unique_2, red, syn,
                                        tot_m, unique_1_m, unique_2_m, red_m, syn_m] 
                                        if np.isnan(i) or np.isinf(i)])
                    if l_x == 0:
                        results_models[model_test].append([site, ll, aridity_index,  swc_i, 
                                        ap, isource, isource_nmin, isource_nmax,
                                        tot, unique_1, unique_2, red, syn,
                                        tot_m, unique_1_m, unique_2_m, red_m, syn_m])

    for model_test in model_list: 
        site, ll, aridity_index, swc_i, \
        ap, isource, isource_nmin, isource_nmax,\
        tot, unique_1, unique_2, red, syn,\
        tot_m, unique_1_m, unique_2_m, red_m, syn_m = zip(*results_models[model_test])

        results_i = {}
        for v, vn in zip([site, ll, aridity_index, swc_i, 
                         ap, isource, isource_nmin, isource_nmax,
                         tot, unique_1, unique_2, red, syn,
                         tot_m, unique_1_m, unique_2_m, red_m, syn_m], \
                        ['site', 'll', 'AI', 'SWC_i', \
                        'Ap', 'I_source', 'I_source_nmin', 'I_source_nmax', \
                        'T', 'u1', 'u2', 'R', 'S', \
                        'T_m', 'u1_m', 'u2_m', 'R_m', 'S_m']):
            results_i[vn] = np.array(v)

        results_i['Af_Tot'] = (results_i['T_m'] - results_i['T']) 
        results_i['Af_u1'] = results_i['u1_m'] - results_i['u1']
        results_i['Af_u2'] = results_i['u2_m'] - results_i['u2']
        results_i['Af_S'] = results_i['S_m'] - results_i['S']
        results_i['Af_R'] = results_i['R_m'] - results_i['R']
        results_i['Af_Part'] = np.abs(results_i['Af_u1']) + np.abs(results_i['Af_u2']) + \
                             np.abs(results_i['Af_S']) + np.abs(results_i['Af_R'])
        result_dict[model_test] = results_i
    return result_dict


def map_sites(fig_name):
    fig_name = os.path.join(plotdir, fig_name)
    import geopandas
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    world.plot(zorder=1, color='lavender', ax=ax)
    ax.scatter(sites_df['lon'].values, sites_df['lat'].values, edgecolor='none', color='k', s=5)
    ax.set_ylim([-60, 75])
    ax.set_xlim([-175, 190])
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-40, 0, 40])
    #plt.show()
    plt.savefig(fig_name)


def fig_site_year_Ts(site, year):
    data, aridity_index = get_data(site)
    data_i = data[data.index.year == year]

    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    ax2.plot(data_i['RH'], color='blue')
    ax.plot(data_i['T_air'], color='red')
    ax.set_ylabel('Ta', color='red', fontsize=16)
    ax.set_ylim([-5, 25])
    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels([0, 10, 20], fontsize=10)
    ax2.set_ylim([0,1])
    ax2.set_ylabel('RH', color='blue', fontsize=16)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(['0', 0.5, '1'], fontsize=10)
    ax.set_xlim(['%s-05-01'%year, '%s-10-01'%(year)])
    ax.set_xticks(['%s-05-01'%year, '%s-06-01'%year, '%s-07-01'%year,'%s-08-01'%year, '%s-09-01'%(year), '%s-10-01'%(year)])
    ax.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'], fontsize=12)
    fig_name = os.path.join(plotdir, 'ts_sources_%s.svg' %(site))
    plt.savefig(fig_name)
    fig_name = os.path.join(plotdir, 'ts_sources_%s.png' %(site))
    plt.savefig(fig_name)
    
    fig = plt.figure(figsize=(7.5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data_i['B_SUMMA'], color='dodgerblue',  label='SUMMA-SA')
    ax.plot(data_i['B_SUMMA_NN1'], color='darkred', label='SUMMA-NN1')
    ax.plot(data_i['B_SUMMA_NN2'], color='tomato', label='SUMMA-NN2')
    ax.plot(data_i['B_SFE'], color='gold', label='SFE')
    ax.plot(data_i['B'], color='k', linestyle='--', label='Obs', lw=2)
    ax.set_ylabel('Bowen ratio', fontsize=14)
    ax.set_ylim([0, 6])
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticklabels([0, 2, 4, 6], fontsize=10)
    ax.set_xlim(['%s-05-01'%year, '%s-10-01'%(year)])
    ax.set_xticks(['%s-05-01'%year, '%s-06-01'%year, '%s-07-01'%year,'%s-08-01'%year, '%s-09-01'%(year), '%s-10-01'%(year)])
    ax.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'], fontsize=12)
    ax.legend(frameon=False, fontsize=12)
    
    fig_name = os.path.join(plotdir, 'ts_B_%s.svg' %(site))
    plt.savefig(fig_name)
    fig_name = os.path.join(plotdir, 'ts_B_%s.png' %(site))
    plt.tight_layout()
    plt.savefig(fig_name)


def fig_results(results, model_list, fig_name, color_list = ['dodgerblue', 'darkred', 'tomato', 'gold'], xbin=np.linspace(0.5, 3.5, 7)):
    
    plt.rcParams['xtick.labelsize']=6
    plt.rcParams['ytick.labelsize']=6
    fig = plt.figure(figsize=(8, 7))
    x = 'I_source'
    xlabel = 'I($Ta$;$RH$)'
    xstep = xbin[1] - xbin[0]
    xticks = [1, 2, 3]
    xlim = [xbin[0] - 0.23, 3.33]
    ms = 3.5
    ii = 1    

    for y, y2, ylabel1, ylabel2 in zip(['T', 'S', 'R', 'u1', 'u2'], ['Af_Tot', 'Af_S', 'Af_R', 'Af_u1', 'Af_u2'],
                                    ['$I_{tot}$', 'S', 'R', '$U_{Ta}$', '$U_{RH}$'],
                                    ['$A_{f, Itot}$', '$A_{f,S}$', '$A_{f,R}$', '$A_{f,U_{Ta}}$', '$A_{f,U_{RH}}$']):
        ax = fig.add_subplot(5, 4, ii)
        ax.tick_params(direction='inout')
        for m, color, ls in zip(model_list, color_list, ['-', '-', '-', '-']):
            ybinned = [[yi for xi, yi in zip(results[m][x], results[m]['%s_m' % y])
                                 if (xi>=bi-xstep) and (xi<bi+xstep)] for bi in xbin]
            ybin = [np.nanmean(mi) if len(mi)>5 else np.nan for mi in ybinned ]
            ax.plot(xbin, ybin, color=color, linestyle=ls, label=m)
            for xxx, yyy in zip(xbin, ybin):
                ax.plot(xxx, yyy, color=color, markersize=(xxx) * ms, marker='o')

        m = 'SUMMA'
        ybinned = [[yi for xi, yi in zip(results[m][x], results[m][y])
                             if (xi>=bi-xstep) and (xi<bi+xstep)] for bi in xbin]

        ybin = [np.nanmean(mi) if len(mi)>5 else np.nan for mi in ybinned ]
        ax.plot(xbin, ybin, color='k', linestyle=':', label='Obs')
        ax.set_ylabel(ylabel1)
        if y == 'T':
            ax.set_title('a)')
        if y == 'u2':
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])
        if y == 'u1':
            ax.set_yticks([0.15, 0.2, 0.25])
        ax.set_xlim(xlim)
        ax.get_yaxis().set_label_coords(-0.27,0.5)

        ii = ii + 1
        ax = fig.add_subplot(5, 4, ii)
        ax.tick_params(direction='inout')
        for m, color, ls in zip(model_list, color_list, ['-', '-', '-', '-']):
            ybinned = [[yi for xi, yi in zip(results[m][x], results[m][y2])
                                 if (xi>=bi-xstep) and (xi<bi+xstep)] for bi in xbin]
            ybin = [np.nanmean(mi) if len(mi)>5 else np.nan for mi in ybinned]
            ax.plot(xbin, ybin, color=color, linestyle=ls, label=m)
            for xxx, yyy in zip(xbin, ybin):
                ax.plot(xxx, yyy, color=color, markersize=(xxx) * ms, marker='o')
        if y2=='Af_Tot':
            ax.axhline(0, color='k', linestyle=':')
            if np.nanmax(yyy)>0.1:
                ax.set_yticks([ 0, 0.1, 0.2])
        else:
            ax.axhline(0, color='k', linestyle=':')
            ax.set_yticks([-0.05, 0, 0.05])
        ax.set_ylabel(ylabel2)
        if y == 'u2':
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])
        if y == 'T':
            ax.set_title('b)')
        ax.set_xlim(xlim)
        ax.get_yaxis().set_label_coords(-0.27,0.5)
        ii = ii + 3

    ax = fig.add_subplot(2, 2, 2)
    y2 = 'Ap'
    ylabel2 = '$A_{p}$'
    ax.tick_params(direction='inout')
    ax.plot([], [], color='k', linestyle=':', label='Obs')
    if 'SFE' in model_list:
        ax.plot([], [], color='gold', linestyle='-', label='SFE')
    ax.plot([], [], color='dodgerblue', linestyle='-', label='SUMMA-SA')
    ax.plot([], [], color='darkred', linestyle='-', label='SUMMA-NN1')
    ax.plot([], [], color='tomato', linestyle='-', label='SUMMA-NN2')

    for m, color, ls in zip(model_list, color_list, ['-', '-', '-', '-']):
        ybinned = [[yi for xi, yi in zip(results[m][x], results[m][y2])
                             if (xi>=bi-xstep) and (xi<bi+xstep)] for bi in xbin]
        ybin = [np.nanmean(mi) if len(mi)>5 else np.nan for mi in ybinned]
        ax.plot(xbin, ybin, color=color, linestyle=ls)
        for xxx, yyy in zip(xbin, ybin):
            ax.plot(xxx, yyy, color=color, markersize=(xxx) * ms *2, marker='o')
    ax.set_title('c)')
    ax.set_xlim(xlim)
    ax.get_yaxis().set_label_coords(-0.15,0.5)
    ax.set_ylabel(ylabel2, fontsize=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels([], fontsize=8)
    plt.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(2, 2, 4)
    y2 = 'Af_Part'
    ylabel2 = '$A_{f, Ipart}$'
    ax.tick_params(direction='inout')
    
    for m, color, ls in zip(model_list, color_list, ['-', '-', '-', '-']):
        ybinned = [[yi for xi, yi in zip(results[m][x], results[m][y2])
                             if (xi>=bi-xstep) and (xi<bi+xstep)] for bi in xbin]
        ybin = [np.nanmean(mi) if len(mi)>5 else np.nan for mi in ybinned]
        ax.plot(xbin, ybin, color=color, linestyle=ls, label=m)
        for xxx, yyy in zip(xbin, ybin):
            ax.plot(xxx, yyy, color=color, markersize=(xxx) * ms*2, marker='o')
    
    ax.set_title('d)')
    ax.set_xlim(xlim)
    ax.get_yaxis().set_label_coords(-0.15,0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel2, fontsize=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=8)
    fig.subplots_adjust(wspace=0.55, hspace=0.17)
    
    plt.savefig(fig_name)


def fig_SI_flux_CDF(results, fig_name, title, model_list=['SUMMA', 'SUMMA_NN1', 'SUMMA_NN2' ], color_list = ['dodgerblue', 'darkred', 'tomato']):
    
    fig = plt.figure(figsize=(3, 6.5))

    ax = fig.add_subplot(2, 1, 1)
    v = 'Ap'
    ax.tick_params(direction='inout')
    if 'SFE' in model_list:
        ax.plot([], [], color='gold', linestyle='-', label='SFE')
    ax.plot([], [], color='dodgerblue', linestyle='-', label='SUMMA-SA')
    ax.plot([], [], color='darkred', linestyle='-', label='SUMMA-NN1')
    ax.plot([], [], color='tomato', linestyle='-', label='SUMMA-NN2')

    for m, color in zip(model_list, color_list):
        y = results[m][v]
        y.sort()
        x = [i/np.float(len(y)+1) for i in range(len(y))]
        ax.plot(x, y, color=color)
    ax.set_ylim([0.3, 1])
    ax.set_title(title, fontsize=10)
    ax.get_yaxis().set_label_coords(-0.15,0.5)
    ax.set_ylabel('$A_{p}$', fontsize=12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlim([0, 1])
    ax.set_xticklabels([], fontsize=8)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_yticklabels([0.4, '', 0.6, '', 0.8, '', 1], fontsize=8)

    plt.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(2, 1, 2)
    v = 'Af_Part'
    ax.tick_params(direction='inout')
    
    for m, color in zip(model_list, color_list):
        y = results[m][v]
        y.sort()
        #y = np.flip(y)
        x = [i/np.float(len(y)+1) for i in range(len(y))]
        ax.plot(x, y, color=color)
    ax.set_ylim([0, 1])
    ax.get_yaxis().set_label_coords(-0.15,0.5)
    ax.set_ylabel('$A_{f, Ipart}$', fontsize=12)
    ax.set_xlabel('Non-exceedance probability', fontsize=10)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0', 0.25, 0.5, 0.75, '1'], fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], fontsize=8)
    ax.set_xlim([0, 1])

    fig.subplots_adjust(hspace=0.11)
    
    plt.savefig(fig_name)


def results_ranks(results_all, model_list):
    def __data2rank(data):
        data_rank = []
        for x in data:
            x_s = list(np.sort(x))
            data_rank.append([1 - x_s.index(xi) / np.float(len(x) - 1) for xi in x])
        data_rank = list(zip(*data_rank))
        return [np.mean(r) for r in data_rank]

    print(model_list)
    x = 'I_source'
    print(x)
    print(np.mean(results['SUMMA'][x]), np.median(results['SUMMA'][x]))
    for y in ['Ap',  'Af_Part', 'Af_Tot', 'Af_S', 'Af_R', 'Af_u1', 'Af_u2']:
        print(y)
        y_data = [list(np.abs(results[m][y])) for m in model_list]
        print(np.mean(y_data[0]), np.mean(y_data[1]), np.mean(y_data[2]), np.mean(y_data[3]))
        y_data = list(zip(*y_data))
        ranks = __data2rank(y_data)
        print(ranks)


def figs_checkdata():
    plotdir_i =  '../../PROJECTS/InfoFlow_SourceDep/WRR_figures/check_data'
    sites = os.listdir(data_dir)
    sites = [s.split('.')[0] for s in sites]

    nbins=50
    bin_bounds = [[0, 6], [0, 1], [-20, 40], [0, 1]]
    [[pl_t, pu_t], [pl_t2, pu_t2], [pl_1, pu_1], [pl_2, pu_2]] = bin_bounds

    T_air = np.array([])
    RH = np.array([])
    Bo = np.array([])
    EF = np.array([])

    for site in sites:
        #print(site, 'check data')
        data, aridity_index = get_data(site, filter_conditions=True)
        pft = sites_df[sites_df['id']==site]['pft'].values[0]
        
        for k in ['EF', 'EF_SFE', 'EF_SUMMA', 'EF_SUMMA_NN1', 'EF_SUMMA_NN2']:
            data[k] = [ii  if ii<1 else 1 for ii in data[k]]
            data[k] = [ii  if ii>0 else 0 for ii in data[k]]
        years = list(set(data.index.year))
        T_air = np.concatenate((T_air, data['T_air'].values), axis=None)
        RH = np.concatenate((RH, data['RH'].values), axis=None)
        Bo = np.concatenate((Bo, data['B'].values), axis=None)
        EF = np.concatenate((EF, data['EF'].values), axis=None)
        
        if len(years)>5:
            years = years[:5]
        i = 0
        fig = plt.figure(figsize=(8, 10))
        for year in years:
            i = i + 1
            ax = fig.add_subplot(5, 1, i)
            data_i = data[data.index.year == year].resample('D').mean()
            
            ax.plot(data_i['B_SFE'], color='gold')
            ax.plot(data_i['B_SUMMA'], color='dodgerblue')
            ax.plot(data_i['B_SUMMA_NN1'], color='darkred')
            ax.plot(data_i['B_SUMMA_NN2'], color='tomato')
            ax.plot(data_i['B'], color='k', linestyle='-')
            ax.axhline(0, color='cyan', linestyle='-')
            ax.axhline(pl_t, color='k', linestyle=':')
            ax.axhline(pu_t, color='k', linestyle=':')
            #ax.set_ylim([np.max([np.min(data_i['B']), -5]), np.min([np.max(data_i['B']), 20])])
            ax.set_ylim([0, 10])
            ax.set_ylabel('B')
            ax.set_xlim(['%s-01-01'%year, '%s-01-01'%(year+1)])
            if i == 1:
                ax.set_title('%s %s ' % (site, pft))
        fig_name = os.path.join(plotdir_i, 'B_%s.png' %(site))
        plt.savefig(fig_name)

        i = 0
        fig = plt.figure(figsize=(8, 10))
        for year in years:
            i = i + 1
            ax = fig.add_subplot(5, 1, i)
            data_i = data[data.index.year == year].resample('D').mean()
            
            ax.plot(data_i['EF_SFE'], color='gold')
            ax.plot(data_i['EF_SUMMA'], color='dodgerblue')
            ax.plot(data_i['EF_SUMMA_NN1'], color='darkred')
            ax.plot(data_i['EF_SUMMA_NN2'], color='tomato')
            ax.plot(data_i['EF'], color='k', linestyle='-')
            ax.set_ylim([0, 1])
            ax.set_ylabel('EF')
            ax.set_xlim(['%s-01-01'%year, '%s-01-01'%(year+1)])
            if i == 1:
                ax.set_title('%s %s ' % (site, pft))
        fig_name = os.path.join(plotdir_i, 'EF_%s.png' %(site))
        plt.savefig(fig_name)

        
        Bo_m = [b if b<pu_t else pu_t for b in data['B']]
        Bo_m = [b if b>pl_t else pl_t for b in Bo_m]
        Bo_m.append(pl_t)
        Bo_m.append(pu_t)

        EF_m = [b if b<pu_t2 else pu_t2 for b in data['EF']]
        EF_m = [b if b>pl_t2 else pl_t2 for b in EF_m]
        EF_m.append(pl_t2)
        EF_m.append(pu_t2)

        T_air_m = [b if b<pu_1 else pu_1 for b in data['T_air']]
        T_air_m = [b if b>pl_1 else pl_1 for b in T_air_m]
        T_air_m.append(pl_1)
        T_air_m.append(pu_1)
        
        RH_m = [b if b<pu_2 else pu_2 for b in data['RH']]
        RH_m = [b if b>pl_2 else pl_2 for b in RH_m]
        RH_m.append(pl_2)
        RH_m.append(pu_2)

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(2, 4, 1)
        ax.hist(Bo_m, bins= nbins)
        ax.set_title('B')
        ax = fig.add_subplot(2, 4, 2)
        ax.hist(EF_m, bins=nbins)
        ax.set_title('EF')

        ax = fig.add_subplot(2, 4, 3)
        ax.hist(T_air_m, bins=nbins)
        ax.set_title('Tair')
        ax = fig.add_subplot(2, 4, 4)
        ax.hist(RH_m, bins=nbins)
        ax.set_title('RH')
        ax = fig.add_subplot(2, 3, 4)
        ax.hist2d(RH_m, T_air_m, bins=nbins)
        ax.set_ylabel('Tair')
        ax.set_xlabel('RH')
        ax = fig.add_subplot(2, 3, 5)
        ax.hist2d(T_air_m, EF_m, bins=nbins)
        ax.set_ylabel('EF')
        ax.set_xlabel('Tair')
        ax = fig.add_subplot(2, 3, 6)
        ax.hist2d(RH_m, EF_m, bins=nbins)
        ax.set_ylabel('EF')
        ax.set_xlabel('RH')
        fig_name = os.path.join(plotdir_i, 'hist_%s.png' %(site))
        plt.savefig(fig_name)

        plt.savefig(fig_name)

    Bo_m = [b if b<pu_t else pu_t for b in Bo]
    Bo_m = [b if b>pl_t else pl_t for b in Bo_m]

    EF_m = [b if b<pu_t2 else pu_t2 for b in EF]
    EF_m = [b if b>pl_t2 else pl_t2 for b in EF_m]

    T_air_m = [b if b<pu_1 else pu_1 for b in T_air]
    T_air_m = [b if b>pl_1 else pl_1 for b in T_air_m]
    
    RH_m = [b if b<pu_2 else pu_2 for b in RH]
    RH_m = [b if b>pl_2 else pl_2 for b in RH_m]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 4, 1)
    ax.hist(Bo_m, bins=np.linspace(pl_t, pu_t, nbins))
    ax.set_title('B')
    ax = fig.add_subplot(1, 4, 2)
    ax.hist(EF_m, bins=np.linspace(pl_t2, pu_t2, nbins))
    ax.set_title('EF')
    ax = fig.add_subplot(1, 4, 3)
    ax.hist(T_air_m, bins=np.linspace(pl_1, pu_1, nbins))
    ax.set_title('Tair')
    ax = fig.add_subplot(1, 4, 4)
    ax.hist(RH_m, bins=np.linspace(pl_2, pu_2, nbins))
    ax.set_title('RH')
    fig_name = os.path.join(plotdir, 'hist_variables.png')
    plt.savefig(fig_name)
    

if __name__ == "__main__":
    pass
