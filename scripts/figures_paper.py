# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:24:12 2022
@author: Alex Garcia-Duran
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from scipy.stats import linregress, sem
from matplotlib.lines import Line2D
from statsmodels.stats.proportion import proportion_confint
from matplotlib.colors import LogNorm
from scipy import interpolate
import scipy
import types

import extended_ddm_v2 as edd2
import different_models as model_variations
import figure_2 as fig_2
import analyses_humans as ah
import matplotlib
import matplotlib.pylab as pl



matplotlib.rcParams['font.size'] = 12
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['xtick.labelsize']= 12
plt.rcParams['ytick.labelsize']= 12
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3

# ---GLOBAL VARIABLES
SV_FOLDER = '...'
DATA_FOLDER = '...'

FRAME_RATE = 14
BINS_RT = np.linspace(1, 301, 11)
xpos_RT = int(np.diff(BINS_RT)[0])
COLOR_COM = 'coral'
COLOR_NO_COM = 'tab:cyan'

# ---FUNCTIONS
def rm_top_right_lines(ax, right=True):
    """
    Function to remove top and right (or left) lines from panels.
    """
    if right:
        ax.spines['right'].set_visible(False)
    else:
        ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)


def add_inset(ax, inset_sz=0.2, fgsz=(4, 8), marginx=0.01, marginy=0.05,
              right=True):
    """
    adds inset to an axis
    """
    ratio = fgsz[0]/fgsz[1]
    pos = ax.get_position()
    ax_inset = plt.axes([pos.x1-inset_sz-marginx, pos.y0+marginy, inset_sz,
                         inset_sz*ratio])
    rm_top_right_lines(ax_inset, right=right)
    return ax_inset


def binning_mt_prior(df, bins):
    """
    Returns MT binned by the prior, with some specified bins.
    """
    # matrix with rows for subjects and columns for bins
    mat_mt = np.empty((len(df.subjid.unique()), len(bins)-1))
    for i_s, subject in enumerate(df.subjid.unique()):
        df_sub = df.loc[df.subjid == subject]
        for bin in range(len(bins)-1):
            mt_sub = df_sub.loc[(df_sub.choice_x_prior >= bins[bin]) &
                                (df_sub.choice_x_prior < bins[bin+1]), 'resp_len']
            mat_mt[i_s, bin] = np.nanmedian(mt_sub)
            if np.isnan(mat_mt[i_s, bin]):
                print(1)
    return mat_mt  # if you want mean across subjects, np.nanmean(mat_mt, axis=0)


def get_bin_info(df, condition, prior_limit=0.25, after_correct_only=True, rt_lim=50,
                 fpsmin=29, num_bins_prior=5, rtmin=0, silent=True):
    """
    For a given condition:
        - choice_x_coh (stimulus evidence towards response)
        - choice_x_prior (prior evidence towards response)
        - origidx (trial index)
    Returns bins (equipopulated for choice_x_prior) and info about the bins.
    """
    # after correct condition
    ac_cond = df.aftererror == False if after_correct_only else (df.aftererror*1) >= 0
    # common condition 
    # put together all common conditions: prior, reaction time and framerate
    common_cond = ac_cond & (df.norm_allpriors.abs() <= prior_limit) &\
        (df.sound_len < rt_lim) & (df.framerate >= fpsmin) & (df.sound_len >= rtmin)
    # define bins, bin type, trajectory index and colormap depending on condition
    if condition == 'choice_x_coh':
        bins = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
        bintype = 'categorical'
        indx_trajs = common_cond & (df.special_trial == 0) 
        n_iters = len(bins)
        colormap = pl.cm.coolwarm(np.linspace(0., 1, n_iters))
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["mediumblue","plum","firebrick"])
        colormap = colormap(np.linspace(0, 1, n_iters))
    elif condition == 'choice_x_prior':
        if silent:
            indx_trajs = common_cond & (df.special_trial == 2)
        if not silent:
            indx_trajs = common_cond & (df.special_trial == 0)
        bins_zt = [-1.01]
        percentiles = [1/num_bins_prior*i for i in range(1, num_bins_prior)]
        for perc in percentiles:
            bins_zt.append(df.loc[indx_trajs, 'choice_x_prior'].quantile(perc))
        bins_zt.append(1.01)
        bins = np.array(bins_zt)
        bintype = 'edges'
        n_iters = len(bins)-1
        colormap = pl.cm.copper(np.linspace(0., 1, n_iters))
    elif condition == 'origidx':
        bins = np.linspace(0, 1e3, num=6)
        bintype = 'edges'
        n_iters = len(bins) - 1
        indx_trajs = common_cond & (df.special_trial == 0)
        colormap = pl.cm.jet(np.linspace(0., 1, n_iters))
    return bins, bintype, indx_trajs, n_iters, colormap


def get_lb():
    """
    Returns list with hard lower bounds (LB) for BADS optimization.

    Returns
    -------
    list
        List with hard lower bounds.

    """
    lb_aff = 3
    lb_eff = 3
    lb_t_a = 4
    lb_w_zt = 0.05
    lb_w_st = 0
    lb_e_bound = 1.3
    lb_com_bound = 0
    lb_w_intercept = 0.01
    lb_w_slope = 1e-6
    lb_a_bound = 0.1
    lb_1st_r = 75
    lb_2nd_r = 75
    lb_leak = 0
    lb_mt_n = 1
    lb_mt_int = 120
    lb_mt_slope = 0.01
    return [lb_w_zt, lb_w_st, lb_e_bound, lb_com_bound, lb_aff,
            lb_eff, lb_t_a, lb_w_intercept, lb_w_slope, lb_a_bound,
            lb_1st_r, lb_2nd_r, lb_leak, lb_mt_n,
            lb_mt_int, lb_mt_slope]


def get_ub():
    """
    Returns list with hard upper bounds (UB) for BADS optimization.

    Returns
    -------
    list
        List with hard upper bounds.

    """
    ub_aff = 12
    ub_eff = 12
    ub_t_a = 22
    ub_w_zt = 1
    ub_w_st = 0.18
    ub_e_bound = 4
    ub_com_bound = 0.3
    ub_w_intercept = 0.12
    ub_w_slope = 1e-3
    ub_a_bound = 4
    ub_1st_r = 500
    ub_2nd_r = 500
    ub_leak = 0.1
    ub_mt_n = 12
    ub_mt_int = 370
    ub_mt_slope = 0.6
    return [ub_w_zt, ub_w_st, ub_e_bound, ub_com_bound, ub_aff,
            ub_eff, ub_t_a, ub_w_intercept, ub_w_slope, ub_a_bound,
            ub_1st_r, ub_2nd_r, ub_leak, ub_mt_n,
            ub_mt_int, ub_mt_slope]


def tachometric_data(coh, hit, sound_len, subjid, ax, label='Data',
                     legend=True, rtbins=np.arange(0, 201, 3)):
    """
    Plots tachometric curve with the function tachometric().
    """
    rm_top_right_lines(ax)
    df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit,
                                 'sound_len': sound_len, 'subjid': subjid})
    tachometric(df_plot_data, ax=ax, fill_error=True, cmap='gist_yarg',
                rtbins=rtbins, evidence_bins=[0, 0.25, 0.5, 1])
    ax.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title(label)
    ax.set_ylim(0.24, 1.04)
    if legend:
        colormap = pl.cm.gist_gray_r(np.linspace(0.4, 1, 4))
        legendelements = [Line2D([0], [0], color=colormap[0], lw=2,
                                 label='0'),
                          Line2D([0], [0], color=colormap[1], lw=2,
                                 label='0.25'),
                          Line2D([0], [0], color=colormap[2], lw=2,
                                 label='0.5'),
                          Line2D([0], [0], color=colormap[3], lw=2,
                                 label='1')]
        ax.legend(handles=legendelements, fontsize=7)
    return ax.get_position()


def add_text(ax, letter, x=-0.1, y=1.2, fontsize=16):
    """function to add letters to panel"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top', ha='right')


def groupby_binom_ci(x, method="beta"):
    # so we can plot groupby with errorbars in binomial vars in 2 lines
    return [abs(x.mean() - ci) for ci in
            proportion_confint(x.sum(), len(x), method=method)]


def run_model(stim, zt, coh, gt, trial_index, human=False, sv_folder='',
              subject=None, num_tr=None, load_params=True, params_to_explore=[],
              extra_label=''):
    """
    Function to run the model specified by the string in extra_label.
    Default: extra_label = '', which will run the full model.
    
    Returns:
        - Hit (correct/incorrect)
        - Reaction time, ms
        - Detected CoM, reversals
        - Final response
        - Original CoM
        - Proactive or reactive trial
        - Total trajectory
        - Deflection point
        - First MT
    """
    if extra_label == '':
        model = edd2.trial_ev_vectorized  # full model
    if type(extra_label) is not int:
        if 'neg' in extra_label:  # negative starting point
            model = model_variations.trial_ev_vectorized_neg_starting_point
        if '_1_ro' in extra_label:  # only with 1st readout
            model = model_variations.trial_ev_vectorized_without_2nd_readout
        if '_1_ro' in extra_label and 'com_modulation_' in extra_label:  # only with 1st readout (RESULAJ)
            model = model_variations.trial_ev_vectorized_CoM_without_update
        if '_2_ro' in extra_label and 'rand' not in extra_label:  # only with 2nd readout
            model = model_variations.trial_ev_vectorized_without_1st_readout
        if '_2_ro' in extra_label and 'rand' in extra_label:  # random choice in 1st readout
            model = model_variations.trial_ev_vectorized_without_1st_readout_random_1st_choice
        if 'virt' in extra_label:  # used for parameter recovery, full model
            model = edd2.trial_ev_vectorized
        if '_prior_sign_1_ro_' in extra_label:  # only prior in first readout
            model = model_variations.trial_ev_vectorized_only_prior_1st_choice
        if 'silent' in extra_label or 'redo' in extra_label:  # full model, to run only with silent trials
            model = edd2.trial_ev_vectorized
        if 'continuous' in extra_label:  # continuous readouts
            model = model_variations.trial_ev_vectorized_n_readouts
        if 'orig' in extra_label:  # original model
            model = edd2.trial_ev_vectorized
    if type(extra_label) is int:  # if not string, run original model
        model = edd2.trial_ev_vectorized
    if num_tr is not None:
        num_tr = num_tr
    else:
        num_tr = int(len(zt))
    data_augment_factor = 10  # 50 for 1 ms precision, 10 for 5 ms precision, dt = 5 ms
    if not human:  # CoM detection threshold
        detect_CoMs_th = 8
    if human:
        detect_CoMs_th = 100
    if not load_params:  # default parameters to run simulation if there are no parameters to load
        p_t_aff = 5
        p_t_eff = 4
        p_t_a = 14
        p_w_zt = 0.1
        p_w_stim = 0.08
        p_e_bound = .6
        p_com_bound = 0.1
        p_w_a_intercept = 0.056
        p_w_a_slope = 2e-5
        p_a_bound = 2.6
        p_1st_readout = 50
        p_2nd_readout = 90
        p_leak = 0.5
        p_mt_noise = 35
        p_MT_intercept = 320
        p_MT_slope = 0.07
        conf = [p_w_zt, p_w_stim, p_e_bound, p_com_bound, p_t_aff,
                p_t_eff, p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound, p_1st_readout,
                p_2nd_readout, p_leak, p_mt_noise, p_MT_intercept, p_MT_slope]
        jitters = len(conf)*[0]
    else:
        # load parameters
        if human:
            conf = np.load(sv_folder +
                           'parameters_MNLE_BADS_human_subj_' + str(subject) + '.npy')
        else:
            if 'virt' not in str(extra_label):  # not parameter recovery
                conf = np.load(sv_folder + 'parameters_MNLE_BADS' + subject + '.npy')
            if 'virt' in str(extra_label):  # parameter recovery (another folder)
                conf = np.load(sv_folder + 'virt_params/' + 'parameters_MNLE_BADS_prt_n50_' +
                               str(extra_label) + '.npy')
        jitters = len(conf)*[0]
        # check if there are params to explore
        if len(params_to_explore) != 0:
            # update conf with params to explore
            for i, index in enumerate(params_to_explore[0]):
                conf[index] = params_to_explore[1][i]

    print('Number of trials: ' + str(stim.shape[1]))
    factor = 10 / data_augment_factor  # to normalize parameters
    # add jitters option, but jitters set to 0 by default
    p_w_zt = conf[0]+jitters[0]*np.random.rand()
    p_w_stim = conf[1]*factor+jitters[1]*np.random.rand()
    p_e_bound = conf[2]+jitters[2]*np.random.rand()
    p_com_bound = conf[3]*p_e_bound+jitters[3]*np.random.rand()
    p_t_aff = int(round(conf[4]/factor+jitters[4]*np.random.rand()))
    p_t_eff = int(round(conf[5]/factor++jitters[5]*np.random.rand()))
    p_t_a = int(round(conf[6]/factor+jitters[6]*np.random.rand()))
    p_w_a_intercept = conf[7]*factor+jitters[7]*np.random.rand()
    p_w_a_slope = -conf[8]*factor+jitters[8]*np.random.rand()
    p_a_bound = conf[9]+jitters[9]*np.random.rand()
    p_1st_readout = conf[10]+jitters[10]*np.random.rand()
    p_2nd_readout = conf[11]+jitters[11]*np.random.rand()
    p_leak = conf[12]*factor+jitters[12]*np.random.rand()
    p_mt_noise = conf[13]+jitters[13]*np.random.rand()
    p_MT_intercept = conf[14]+jitters[14]*np.random.rand()
    p_MT_slope = conf[15]+jitters[15]*np.random.rand()
    # augment data so that stimulus (previously with 50 ms resolution)
    # has now the desired resolution (dt = 5 ms)
    stim = edd2.data_augmentation(stim=stim.reshape(stim.shape[0], num_tr),
                                  daf=data_augment_factor)
    stim_res = 50/data_augment_factor
    compute_trajectories = True
    all_trajs = True
    stim_temp =\
        np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                        stim.shape[1]))))
    # run model
    E, A, com_model, first_ind, second_ind, resp_first, resp_fin,\
        pro_vs_re, matrix, total_traj, init_trajs, final_trajs,\
        frst_traj_motor_time, x_val_at_updt  =\
        model(zt=zt, stim=stim_temp, coh=coh,
              trial_index=trial_index,
              p_MT_slope=p_MT_slope,
              p_MT_intercept=p_MT_intercept,
              p_w_zt=p_w_zt, p_w_stim=p_w_stim,
              p_e_bound=p_e_bound, p_com_bound=p_com_bound,
              p_t_aff=p_t_aff, p_t_eff=p_t_eff, p_t_a=p_t_a,
              num_tr=num_tr, p_w_a_intercept=p_w_a_intercept,
              p_w_a_slope=p_w_a_slope,
              p_a_bound=p_a_bound,
              p_1st_readout=p_1st_readout,
              p_2nd_readout=p_2nd_readout, p_leak=p_leak,
              p_mt_noise=p_mt_noise,
              compute_trajectories=compute_trajectories,
              stim_res=stim_res, all_trajs=all_trajs,
              human=human)
    hit_model = resp_fin == gt  # compute accuracy (1 - True -  is right, 0 - False - is wrong)
    # compute RT = (first_hit_index - fixation + t_efferent)*dt
    reaction_time = (first_ind-int(300/stim_res) + p_t_eff)*stim_res
    detected_com = np.abs(x_val_at_updt) > detect_CoMs_th  # wether CoMs are detected
    return hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj, x_val_at_updt, frst_traj_motor_time


def trajs_splitting_stim_all(df, ax, color, threshold=300, par_value=None,
                             rtbins=np.linspace(0, 150, 16),
                             trajectory="trajectory_y", plot=True):
    """
    Computes and plots (if plot=True) the splitting time.
    Returns the minimum splitting time and the RT at which that happens.
    """
    # split time/subject by coherence
    splitfun = fig_2.get_splitting_mat_simul
    df['traj'] = df.trajectory_y.values
    out_data = []
    for subject in df.subjid.unique():
        out_data_sbj = []
        for i in range(rtbins.size-1):
            evs = [0, 0.25, 0.5, 1]
            for iev, ev in enumerate(evs):
                matatmp =\
                    splitfun(df=df.loc[(df.subjid == subject)],
                             side=0, rtbin=i, rtbins=rtbins, coh=ev,
                             align="sound")
                
                if iev == 0:
                    mat = matatmp
                    evl = np.repeat(0, matatmp.shape[0])
                else:
                    mat = np.concatenate((mat, matatmp))
                    evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
            max_mt = 800
            current_split_index =\
                fig_2.get_split_ind_corr(mat, evl, pval=0.0001, max_MT=max_mt,
                                         startfrom=0)+5
            if current_split_index >= rtbins[i]:
                out_data_sbj += [current_split_index]
            else:
                out_data_sbj += [np.nan]
        out_data += [out_data_sbj]

    out_data = np.array(out_data).reshape(
        df.subjid.unique().size, rtbins.size-1, -1)
    # set axes: rtbins, subject, sides
    out_data = np.swapaxes(out_data, 0, 1)

    # change the type so we can have NaNs
    out_data = out_data.astype(float)

    out_data[out_data > threshold] = np.nan

    binsize = rtbins[1]-rtbins[0]
    error_kws = dict(ecolor=color, capsize=2,
                     color=color, marker='o', label=str(par_value*5))
    xvals = binsize/2 + binsize * np.arange(rtbins.size-1)
    if plot:
        ax.errorbar(
            xvals,
            # we do the mean across rtbin axis
            np.nanmean(out_data.reshape(rtbins.size-1, -1), axis=1),
            yerr=sem(out_data.reshape(rtbins.size-1, -1),
                     axis=1, nan_policy='omit'),
            **error_kws)
    # get minimum splitting time and the RT at which that happens
    min_st = np.nanmin(np.nanmean(out_data.reshape(rtbins.size-1, -1), axis=1))
    rt_min_split = xvals[np.where(np.nanmean(out_data.reshape(rtbins.size-1, -1), axis=1) == min_st)[0]]
    if rt_min_split.shape[0] > 1:
        rt_min_split = rt_min_split=[0]
    if sum(min_st.shape) > 1:
        min_st = min_st[0]
    return min_st, rt_min_split


def supp_parameter_analysis(stim, zt, coh, gt, trial_index, subjects,
                            subjid, sv_folder, idx=None):
    """
    Plots supplementary figure 8.
    """
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.45)
    ax = ax.flatten()
    labs = ['a', 'b', 'c', 'd', '', '']
    for i_a, a in enumerate(ax):
        rm_top_right_lines(a)
        a.text(-0.1, 1.17, labs[i_a], transform=a.transAxes, fontsize=16,
               fontweight='bold', va='top', ha='right')
    # changing t_aff and t_eff, 4, 5 are the indices of the parameters to change
    # np.arange(7) are the values (recall that this is with a precision of 5 ms, so t_eff=7 means 35 ms)
    params_to_explore_aff = [[4]] + [np.arange(7)]
    params_to_explore_eff = [[5]] + [np.arange(7)]
    params_to_explore = [params_to_explore_aff, params_to_explore_eff]
    # plots minimum splitting time vs t_eff
    plot_splitting_for_param(stim, zt, coh, gt, trial_index, subjects,
                             subjid, params_to_explore, ax=[ax[0], ax[2]])
    # plots the cartoon of minimum splitting time vs t_eff
    plot_min_st_vs_t_eff_cartoon(ax[1], offset=10)
    # changing theta_AI
    params_to_explore = [[9]] + [np.round(np.linspace(0.1, 5, 11), 2)]
    # plots p(CoM) vs p(proactive)
    plot_pcom_vs_proac(stim, zt, coh, gt, trial_index, subjects,
                       subjid, params_to_explore, ax=ax[3],
                       param_title=r'$\theta_{AI}$', idx=idx)
    # save figure
    fig.savefig(sv_folder+'/supp_model_params.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'/supp_model_params.png', dpi=400, bbox_inches='tight')


def plot_min_st_vs_t_eff_cartoon(ax, offset=15):
    """
    Creates cartoon of minimum splitting time vs t_eff.
    """
    if ax is None:
        fig, ax = plt.subplots(1)
        rm_top_right_lines(ax)
    t_aff = np.arange(7)*5
    t_eff = np.copy(t_aff)
    colormap = pl.cm.BrBG(np.linspace(0.1, 1, len(t_eff)))[::-1]
    for i_taff, t_aff_val in enumerate(t_aff[::-1]):
        min_st = t_aff_val + t_eff + offset
        ax.plot(t_eff, min_st, color=colormap[i_taff], label=str(t_aff_val))
    ax.set_ylabel('Minimum splitting time (ms)')
    ax.legend(title=r'$t_{aff} \;\; (ms)$', loc='upper right',
              frameon=False, bbox_to_anchor=(1.21, 1.145),
              labelspacing=0.35)
    ax.plot([0, 30], [0, 30], color='gray', linewidth=0.8)
    ax.set_xlabel(r'Efferent time $t_{eff} \;\; (ms)$')
    ax.annotate(text='', xy=(30, 31), xytext=(30, 39), arrowprops=dict(arrowstyle='<->'))
    ax.text(30.5, 31.5, 'offset')
    ax.text(15, 10, 'y=x', rotation=17, color='grey')
    ax.set_title(r'Prediction: $min(ST) = t_{aff}+t_{eff}+offset$',
                 fontsize=10)
    ax.set_ylim(-2, 71)


def plot_splitting_for_param(stim, zt, coh, gt, trial_index, subjects,
                             subjid, params_to_explore, ax=None):
    """
    Tunes panel and plots minimum splitting time vs t_eff.
    """
    # tune panel
    for a in ax:
        rm_top_right_lines(a)
    ax[0].plot([0, 155], [0, 155], color='k')
    ax[0].fill_between([0, 155], [0, 155], [0, 0],
                    color='grey', alpha=0.2)
    ax[0].set_xlim(-5, 155)
    ax[0].set_yticks([0, 100, 200, 300])
    ax[0].set_xlabel('Reaction time (ms)')
    ax[0].set_ylabel('Splitting time (ms)')
    # plot
    change_params_exploration_and_plot(stim, zt, coh, gt, trial_index, subjects,
                                       subjid, params_to_explore, [ax[0], ax[1]])
    ax[0].set_ylim(-1, 275)


def plot_pcom_vs_proac(stim, zt, coh, gt, trial_index, subjects,
                       subjid, params_to_explore, ax=None,
                       param_title=r'$\theta_{AI}$', idx=None):
    """
    Plots p(CoM) vs p(proactive) changing theta_AI (bound for action initiation process).
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    rm_top_right_lines(ax)
    ax.set_ylabel('P(CoM)')
    ax.set_xlabel('P(proactive)')
    num_tr = int(len(coh))
    param_ind = params_to_explore[0][0]
    com_all = []
    reversals = []
    proac = []
    acc_list = []
    colormap = pl.cm.magma(np.linspace(0., 0.9, len(params_to_explore[1])))
    for ind in range(len(params_to_explore[1])):  # for each possible value of theta_AI
        param_value = params_to_explore[1][ind]        
        param_iter = str(param_ind)+'_'+str(param_value)+'_v2'
        # run simulation
        hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
            pro_vs_re, trajs, x_val_at_updt =\
                run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                       trial_index=trial_index, num_tr=num_tr,
                                       subject_list=subjects, subjid=subjid,
                                       simulate=False,
                                       params_to_explore=[[param_ind], [param_value]],
                                       change_param=True,
                                       param_iter=param_iter)
        if idx is None:  # append the simulated data
            com_all.append(np.nanmean(com_model[(reaction_time >= 0)]))
            acc_list.append(np.nanmean(hit_model[(reaction_time >= 0)]))
            reversals.append(np.nanmean(com_model_detected[(reaction_time >= 0)]))
            proac.append(1-np.nanmean(pro_vs_re[(reaction_time >= 0)]))
        else:
            com_all.append(np.nanmean(com_model[(reaction_time >= 0) & idx &
                                                (reaction_time < 1000)]))
            reversals.append(np.nanmean(com_model_detected[(reaction_time >= 0) &
                                                           idx & (reaction_time < 1000)]))
            acc_list.append(np.nanmean(hit_model[(reaction_time >= 0) & 
                                                 idx & (reaction_time < 1000)]))
            proac.append(1-np.nanmean(pro_vs_re[(reaction_time >= 0) &
                                                idx & (reaction_time < 1000)]))
    # plot proportions of com vs proac
    ax.plot(proac, com_all, color='k', linewidth=0.9)
    for i_val, pcom in enumerate(com_all):
        ax.plot(proac[i_val], pcom,
                color=colormap[i_val], marker='o', markersize=7)
    # plot a fake image to get the colorbar
    fig2, ax2 = plt.subplots(1)
    img = ax2.imshow(np.array([[min(params_to_explore[1]),
                                max(params_to_explore[1])]]), cmap="magma")
    img.set_visible(False)
    plt.close(fig2)
    # tune colorbar and panel
    cbar = plt.colorbar(img, orientation="vertical", ax=ax, cmap='magma', fraction=0.08,
                        aspect=12)
    cbar.ax.set_title(r'$\theta_{AI}$')
    ax.set_xticks([0, 0.4, 0.8])
    ind = np.where(np.round(params_to_explore[1], 3) == 1.08)[0][0]
    ax.arrow(proac[ind]-0.075, com_all[ind], 0.045, 0, color='k', head_width=0.001,
             width=0.0001, head_length=0.015)
    ax.text(proac[ind]-0.12, com_all[ind], r'$\theta_{AI}^*$')


def change_params_exploration_and_plot(stim, zt, coh, gt, trial_index, subjects,
                                       subjid, params_to_explore, ax, matrix=False):
    """
    Computes the minimum splititng time for different combinations of t_aff, t_eff 
    given by params_to_explore. Then plots it against t_eff, fixing t_aff.
    """
    num_tr = int(len(coh))
    ind_stim = np.sum(stim, axis=0) != 0
    params_to_explore_aff, params_to_explore_eff = params_to_explore
    param_aff = params_to_explore_aff[0][0]
    param_eff = params_to_explore_eff[0][0]
    colormap = pl.cm.BrBG(np.linspace(0.1, 1, len(params_to_explore_eff[1])))
    subject = str(np.unique(subjid)[0])
    sim_data = DATA_FOLDER + subject + '/sim_data/' + subject + '_splitting_matrix_params_2_5ms_bin_final.npy'
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(sim_data), exist_ok=True)
    if os.path.exists(sim_data):  # load data
        mat_min_st = np.load(sim_data)
    else:
        mat_min_st = np.empty((7, 7))
        mat_min_st[:] = np.nan
        for ind_aff in range(len(params_to_explore_aff[1])):
            min_st_list = []
            rt_min_sp = []
            aff_value = params_to_explore_aff[1][ind_aff]
            for ind_eff in range(len(params_to_explore_eff[1])):
                eff_value = params_to_explore_aff[1][ind_eff]
                param_iter = str(param_aff)+'_'+str(aff_value)+'_'+str(param_eff)+'_'+str(eff_value)+'_res_1_ms'
                # smiulate value with some combination of t_aff, t_eff
                hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
                    _, trajs, x_val_at_updt =\
                        run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                               trial_index=trial_index, num_tr=num_tr,
                                               subject_list=subjects, subjid=subjid,
                                               simulate=False,
                                               params_to_explore=[[param_aff, param_eff], [aff_value, eff_value]],
                                               change_param=True,
                                               param_iter=param_iter)
                MT = [len(t) for t in trajs]  # given that for trajs dt=1ms, MT (in ms) is just the length of the trajectory
                df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                                        'sound_len': reaction_time,
                                        'rewside': (gt + 1)/2,
                                        'R_response': (resp_fin+1)/2,
                                        'resp_len': np.array(MT)*1e-3,
                                        'subjid': subjid})
                df_sim = df_sim.loc[ind_stim]
                # compute minimum splitting time
                min_st, rt_min_split = trajs_splitting_stim_all(df=df_sim, ax=ax[0], color=colormap[ind_eff], threshold=300,
                                                                rtbins=np.linspace(0, 150, 61),
                                                                trajectory="trajectory_y", par_value=param_eff, plot=False)
                min_st_list.append(min_st)
                rt_min_sp.append(rt_min_split)
                mat_min_st[ind_aff, ind_eff] = min_st
            i = 0
            for min_st, rt_min_split in zip(min_st_list, rt_min_sp):  # plot example for supplementary panel a
                ax[0].plot(rt_min_split, min_st, marker='o', color=colormap[i],
                           markersize=8)
                i += 1
        np.save(sim_data, mat_min_st)
    # example ST vs RT eff=3
    eff_value = 6
    min_st_list = []
    rt_min_sp = []
    for ind_aff in range(len(params_to_explore_aff[1])):
        aff_value = params_to_explore_aff[1][ind_aff]
        param_iter = str(param_aff)+'_'+str(aff_value)+'_'+str(param_eff)+'_'+str(eff_value)+'_res_1ms'
        hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
            _, trajs, x_val_at_updt =\
                run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                        trial_index=trial_index, num_tr=num_tr,
                                        subject_list=subjects, subjid=subjid,
                                        simulate=False,
                                        params_to_explore=[[param_aff, param_eff], [aff_value, eff_value]],
                                        change_param=True,
                                        param_iter=param_iter)
        MT = [len(t) for t in trajs]  # given that for trajs dt=1ms, MT (in ms) is just the length of the trajectory
        df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                                'sound_len': reaction_time,
                                'rewside': (gt + 1)/2,
                                'R_response': (resp_fin+1)/2,
                                'resp_len': np.array(MT)*1e-3,
                                'subjid': subjid})
        df_sim = df_sim.loc[ind_stim]
        # to plot the splitting time curves in panel a
        min_st, rt_min_split = trajs_splitting_stim_all(df=df_sim, ax=ax[0], color=colormap[ind_aff], threshold=300,
                                                        rtbins=np.linspace(0, 150, 16),
                                                        trajectory="trajectory_y", par_value=aff_value, plot=True)
        min_st_list.append(min_st)
        rt_min_sp.append(rt_min_split)
    i = 0
    # plot the round markers in panel a
    for min_st, rt_min_split in zip(min_st_list, rt_min_sp):
        ax[0].plot(rt_min_split, min_st, marker='o', color=colormap[i],
                    markersize=8)
        i += 1
    # tune panels and plot the other panel
    ax[0].text(70, 200, r'$t_{eff} = $ '+ str(eff_value*5) + ' ms')
    vals = (np.arange(7)*5)[::-1]
    legendelements = []
    for col, val in zip(colormap[::-1], vals):
        legendelements.append(Line2D([0], [0], color=col, lw=2,
                             label=val))
    ax[0].legend(handles=legendelements,
                 title=r'$t_{aff}$ (ms)', loc='upper right',
                 frameon=False, bbox_to_anchor=(1.24, 1.175))
    if matrix:  # if result should be plotted as a matrix, much less intuitive
        im = ax[1].imshow(np.flipud(mat_min_st))
        cbar = plt.colorbar(im, ax=ax[1], fraction=0.04)
        cbar.ax.set_title('   Min. ST (ms)')
        ax[1].set_yticks([6, 3, 0], [0, 15, 30])
        ax[1].set_xticks([0, 3, 6], [0, 15, 30])
        ax[1].set_ylabel(r'$t_{aff}$ (ms)')
        ax[1].set_xlabel(r'$t_{eff}$ (ms)')
    else:
        # plot actual minimum splitting time vs t_eff fixing t_aff
        for i_aff, min_st_list in enumerate(mat_min_st[::-1]):
            ax[1].plot(params_to_explore_eff[1]*5, min_st_list, color=colormap[::-1][i_aff],
                       label=str(params_to_explore_aff[1][::-1][i_aff]*5))
    # tune panels
    ax[1].set_ylabel('Minimum splitting time (ms)')
    ax[1].annotate(text='', xy=(30, 31), xytext=(30, 43), arrowprops=dict(arrowstyle='<->'))
    ax[1].text(15, 10, 'y=x', rotation=17, color='grey')
    ax[1].text(30.5, 31.5, 'offset')
    ax[1].plot([0, 30], [0, 30], color='gray', linewidth=0.8)
    ax[1].set_xlabel(r'Efferent time $t_{eff} \;\; (ms)$')
    ax[1].legend(title=r'$t_{aff} \;\; (ms)$', loc='upper right',
                 frameon=False, bbox_to_anchor=(1.24, 1.175),
                 labelspacing=0.35)
    ax[1].set_ylim(-2, 71)


def stim_generation(coh, data_folder=DATA_FOLDER, stim_res=5, sigma=0.1,
                    new_stim=False):
    """
    Function to create false stimulus with given resolution and standard deviation from
    putative stimulus given by the variable coh.
    """
    stim_data = data_folder + 'artificial_stimulus.npy'
    os.makedirs(os.path.dirname(stim_data), exist_ok=True)
    if os.path.exists(stim_data) and not new_stim:
        stim = np.load(stim_data)
    else:
        timepoints = int(1e3/stim_res)
        stim = np.clip(coh + sigma*np.random.randn(timepoints, len(coh)), -1, 1)
        np.save(stim_data, stim)
    return stim


def get_simulated_data_extra_lab(subjects, subjid, stim, zt, coh, gt, trial_index,
                                 special_trial, extra_label='_1_ro', sv_folder='',
                                 data_folder=''):
    """
    Simulate data given an extra label. Returns a dataframe with the simulated data.
    """
    num_tr = len(gt)
    hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
        _, trajs, x_val_at_updt =\
        run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                       trial_index=trial_index, num_tr=num_tr,
                                       subject_list=subjects, subjid=subjid,
                                       simulate=False, extra_label=extra_label,
                                       sv_folder=sv_folder, data_folder=data_folder)
    MT = [len(t) for t in trajs]  # given that for trajs dt=1ms, MT (in ms) is just the length of the trajectory
    df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                           'sound_len': reaction_time,
                           'rewside': (gt + 1)/2,
                           'R_response': (resp_fin+1)/2,
                           'resp_len': np.array(MT)*1e-3})
    df_sim['CoM_sugg'] = com_model.astype(bool)
    df_sim['traj_d1'] = [np.diff(t) for t in trajs]
    df_sim['subjid'] = subjid
    df_sim['origidx'] = trial_index
    df_sim['special_trial'] = special_trial
    df_sim['traj'] = df_sim['trajectory_y']
    df_sim['com_detected'] = com_model_detected.astype(bool)
    df_sim['peak_com'] = np.array(x_val_at_updt)
    df_sim['hithistory'] = np.array(resp_fin == gt)
    df_sim['allpriors'] = np.copy(zt)
    df_sim['norm_allpriors'] = np.copy(norm_allpriors_per_subj(df_sim))
    df_sim['normallpriors'] = np.copy(df_sim['norm_allpriors'])
    df_sim['framerate']=200
    df_sim['dW_lat'] = 0
    df_sim['dW_trans'] = np.copy(zt)
    df_sim['aftererror'] = False
    return df_sim


def run_simulation_different_subjs(stim, zt, coh, gt, trial_index, subject_list,
                                   subjid, human=False, num_tr=None, load_params=True,
                                   simulate=True, change_param=False, param_iter=None,
                                   params_to_explore=[], extra_label='',
                                   data_folder='', sv_folder=''):
    """
    Runs simulations for different subjects and returns the simulated data for all subjects.
    """
    # initialize arrays, which will be concatenated with simulated data
    hit_model = np.empty((0))
    reaction_time = np.empty((0))
    detected_com = np.empty((0))
    resp_fin = np.empty((0))
    com_model = np.empty((0))
    pro_vs_re = np.empty((0))
    total_traj = []
    x_val_at_updt = np.empty((0))
    for subject in subject_list:  # for each subject
        if subject_list[0] is not None:
            index = subjid == subject
        else:
            index = range(num_tr)
        if not human:
            if change_param:
                sim_data = data_folder + subject + '/sim_data/' + subject + '_simulation_' + str(param_iter) + '.pkl'
            else:
                sim_data = data_folder + subject + '/sim_data/' + subject + '_simulation' + str(extra_label) + '.pkl'
        if human:
            if change_param:
                sim_data = data_folder + '/Human/' + str(subject) + '/sim_data/' + str(subject) + '_simulation_' + str(param_iter) + '.pkl'
            else:
                sim_data = data_folder + '/Human/' + str(subject) + '/sim_data/' + str(subject) + '_simulation.pkl'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(sim_data), exist_ok=True)
        if os.path.exists(sim_data) and not simulate:  # load data if exists
            print('Loading simulated data')
            data_simulation = np.load(sim_data, allow_pickle=True)
            hit_model_tmp = data_simulation['hit_model_tmp']
            reaction_time_tmp = data_simulation['reaction_time_tmp']
            detected_com_tmp = data_simulation['detected_com_tmp']
            resp_fin_tmp = data_simulation['resp_fin_tmp']
            com_model_tmp = data_simulation['com_model_tmp']
            pro_vs_re_tmp = data_simulation['pro_vs_re_tmp']
            total_traj_tmp = data_simulation['total_traj_tmp']
            x_val_at_updt_tmp = data_simulation['x_val_at_updt_tmp']
        else:  # simulate data
            hit_model_tmp, reaction_time_tmp, detected_com_tmp, resp_fin_tmp,\
                com_model_tmp, pro_vs_re_tmp, total_traj_tmp, x_val_at_updt_tmp, frst_traj_motor_time_tmp=\
                run_model(stim=stim[:, index], zt=zt[index], coh=coh[index],
                          gt=gt[index], trial_index=trial_index[index],
                          subject=subject, load_params=load_params, human=human,
                          params_to_explore=params_to_explore, extra_label=extra_label,
                          sv_folder=sv_folder)
            data_simulation = {'hit_model_tmp': hit_model_tmp, 'reaction_time_tmp': reaction_time_tmp,
                               'detected_com_tmp': detected_com_tmp, 'resp_fin_tmp': resp_fin_tmp,
                               'com_model_tmp': com_model_tmp, 'pro_vs_re_tmp': pro_vs_re_tmp,
                               'total_traj_tmp': total_traj_tmp, 'x_val_at_updt_tmp': x_val_at_updt_tmp,
                               'frst_traj_motor_time_tmp':frst_traj_motor_time_tmp}
            pd.to_pickle(data_simulation, sim_data)
        hit_model = np.concatenate((hit_model, hit_model_tmp))
        reaction_time = np.concatenate((reaction_time, reaction_time_tmp))
        detected_com = np.concatenate((detected_com, detected_com_tmp))
        resp_fin = np.concatenate((resp_fin, resp_fin_tmp))
        com_model = np.concatenate((com_model, com_model_tmp))
        pro_vs_re = np.concatenate((pro_vs_re, pro_vs_re_tmp))
        total_traj = total_traj + total_traj_tmp
        x_val_at_updt = np.concatenate((x_val_at_updt, x_val_at_updt_tmp))
    return hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj, x_val_at_updt


def norm_allpriors_per_subj(df):
    """
    Normalize priors for each subject to the maximum value.
    """
    norm_allpriors = np.empty((0,))
    for subj in df.subjid.unique():  # for each subject
        df_1 = df.loc[df.subjid == subj]
        zt_tmp = df_1.allpriors.values  # raw prior
        norm_allpriors = np.concatenate((norm_allpriors,
                                         zt_tmp/np.nanmax(abs(zt_tmp))))
    return norm_allpriors  # normalized prior


def supp_parameter_recovery_test(margin=.6, n_sims=100):
    """
    Plots supplementary figure 16.
    """
    # create figure
    fig, ax = plt.subplots(4, 4, figsize=(14, 12))
    ax = ax.flatten()
    ax[0].text(-0.2, 1.26, 'a', transform=ax[0].transAxes, fontsize=16,
               fontweight='bold', va='top', ha='right')
    # plot
    plot_param_recovery_test(fig, ax,
            subjects=['Virtual_rat_random_params' for _ in range(n_sims)],
            sv_folder=SV_FOLDER, corr=True)
    # tune axes position
    pos = ax[12].get_position()
    cbar_ax = fig.add_axes([pos.x0+pos.width/1.3, pos.y0-pos.height-margin/1.8,
                            pos.width*2.5, pos.height*2.5])
    cbar_ax.text(-0.06, 1.09, 'b', transform=cbar_ax.transAxes, fontsize=16,
                 fontweight='bold', va='top', ha='right')
    cbar_ax_2 = fig.add_axes([pos.x0+pos.width*3.5, pos.y0-pos.height-margin/1.8,
                              pos.width*2.5, pos.height*2.5])
    cbar_ax_2.text(-0.06, 1.09, 'c', transform=cbar_ax_2.transAxes, fontsize=16,
                   fontweight='bold', va='top', ha='right')
    # correlation matrix plot
    plot_corr_matrix_prt(cbar_ax_2, cbar_ax,
            subjects=['Virtual_rat_random_params' for _ in range(n_sims)],
            sv_folder=SV_FOLDER)
    # save figure
    fig.savefig(SV_FOLDER + 'supp_prt.png', dpi=500, bbox_inches='tight')
    fig.savefig(SV_FOLDER + 'supp_prt.svg', dpi=500, bbox_inches='tight')


def supp_plot_params_all_subs(subjects, sv_folder=SV_FOLDER, diff_col=False):
    """
    Plots supplementary figure 13. Loads the fitted parameters and plots the distributions
    as violins together with individual points.
    """
    fig, ax = plt.subplots(4, 4, figsize=(12, 10))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92,
                        hspace=0.5, wspace=0.4)
    if diff_col:
        colors = pl.cm.jet(np.linspace(0., 1, len(subjects)))
    else:
        colors = ['k' for _ in range(len(subjects))]
    ax = ax.flatten()
    for a in ax:
        rm_top_right_lines(a)
    labels = [r'Prior weight, $z_P$', r'Stimulus drift, $a_P$',
              r'EA bound, $\theta_{DV}$',
              r'CoM bound, $\theta_{COM}$',
              r'Afferent time, $t_{aff}$',
              r'Efferent time, $t_{eff}$',
              r'AI time offset, $t_{AI}$',
              r'AI drift offset, $v_{AI}$',
              r'AI drift slope, $w_{AI}$',
              r'AI bound, $\theta_{AI}$',
              r'DV weight 1st readout, $\beta_{DV}$',
              r'DV weight update, $\beta_u$',
              r'Leak, $\lambda$',
              r'MT offset, $\beta_0$', r'MT slope, $\beta_{TI}$']
    conf_mat = np.empty((len(labels), len(subjects)))
    upper_bounds = np.delete(get_ub(), -3)
    lower_bounds = np.delete(get_lb(), -3)
    for i_b in [4, 5, 6]:
        upper_bounds[i_b] = upper_bounds[i_b]*5
        lower_bounds[i_b] = lower_bounds[i_b]*5
    for i_b in [1, 7, 8]:
        upper_bounds[i_b] = upper_bounds[i_b]/5
        lower_bounds[i_b] = lower_bounds[i_b]/5
    for i_s, subject in enumerate(subjects):  # for each subject, load params
        conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
        conf[1] = conf[1] / 5
        conf[7] = conf[7] / 5
        conf[8] = conf[8] / 5  # over 5 ms resolution
        conf_mat[:, i_s] = np.delete(conf, -3)
    for i in range(len(labels)):  # for each parameter, plot and tune
        if i == 4 or i == 5 or i == 6:
            sns.violinplot(conf_mat[i, :]*5, ax=ax[i], orient='h', color='lightskyblue',
                           fmt='g', linewidth=0, bw_adjust=10)
            for i_s in range(len(subjects)):
                ax[i].plot(conf_mat[i, i_s]*5,
                           0.1*np.random.randn(),
                           color=colors[i_s], marker='o', linestyle='',
                           markersize=4)
            ax[i].set_xlabel(labels[i] + str(' (ms)'))
        else:
            sns.violinplot(conf_mat[i, :], ax=ax[i], orient='h', color='lightskyblue',
                           fmt='g', linewidth=0, bw_adjust=10)
            for i_s in range(len(subjects)):
                ax[i].plot(conf_mat[i, i_s],
                           0.1*np.random.randn(),
                           color=colors[i_s], marker='o', linestyle='',
                           markersize=4)
            ax[i].set_xlabel(labels[i])
        ax[i].set_yticks([])
        ax[i].spines['left'].set_visible(False)
    ax[-1].axis('off')
    # save figures
    fig.savefig(sv_folder+'/supp_params_distro.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'/supp_params_distro.png', dpi=400, bbox_inches='tight')


def plot_corr_matrix_prt(ax, ax2,
        subjects=['Virtual_rat_random_params' for _ in range(50)],
        sv_folder=SV_FOLDER):
    """
    Plots cross-correlation and correlation matrix of recovered vs original parameters.
    """
    labels = [r'Prior weight, $z_P$', r'Stimulus drift, $a_P$',
              r'EA bound, $\theta_{DV}$',
              r'CoM bound, $\theta_{COM}$',
              r'Afferent time, $t_{aff}$', r'Efferent time, $t_{eff}$',
              r'AI time offset, $t_{AI}$',
              r'AI drift offset, $v_{AI}$',
              r'AI drift slope, $w_{AI}$',
              r'AI bound, $\theta_{AI}$',
              r'DV weight 1st readout, $\beta_{DV}$',
              r'DV weight update, $\beta_u$', r'Leak, $\lambda$',
              r'MT noise variance, $\sigma_{MT}$',
              r'MT offset, $\beta_0$', r'MT slope, $\beta_{TI}$']
    labels_reduced = [r'$z_P$', r'$a_P$',
                      r'$\theta_{DV}$',
                      r'$\theta_{COM}$',
                      r'$t_{aff}$', r'$t_{eff}$',
                      r'$t_{AI}$',
                      r'$v_{AI}$',
                      r'$w_{AI}$',
                      r'$\theta_{AI}$',
                      r'$\beta_{DV}$',
                      r'$\beta_u$', r'$\lambda$',
                      r'$\sigma_{MT}$',
                      r'$\beta_0$', r'$\beta_{TI}$']
    conf_mat = np.empty((len(labels), len(subjects)))
    conf_mat_rec = np.empty((len(labels), len(subjects)))
    for i_s, subject in enumerate(subjects):  # for each subject, load virt (original) and rec (recovered) parameters
        conf = np.load(sv_folder + 'virt_params/' +
                       'parameters_MNLE_BADS_prt_n50_' + 'virt_sim_' + str(i_s) + '.npy')
        conf_rec =  np.load(sv_folder + 'virt_params/' +
                            'parameters_MNLE_BADS_prt_n50_prt_' + str(i_s) + '.npy')
        conf_mat[:, i_s] = conf
        conf_mat_rec[:, i_s] = conf_rec
    # define correlation matrix
    corr_mat = np.empty((len(labels), len(labels)))
    corr_mat[:] = np.nan
    for i in range(len(labels)):
        for j in range(len(labels)):
            # compute cross-correlation matrix
            corr_mat[i, j] = np.corrcoef(conf_mat[i, :], conf_mat_rec[j, :])[1][0]
    # plot cross-correlation matrix
    im = ax.imshow(corr_mat.T, cmap='bwr', vmin=-1, vmax=1)
    # tune panels
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_xticks(np.arange(16), labels, rotation='270', fontsize=12)
    ax.set_yticks(np.arange(16), labels_reduced, fontsize=12)
    ax.set_xlabel('Original parameters', fontsize=14)
    # compute correlation matrix
    mat_corr = np.corrcoef(conf_mat_rec, rowvar=True)
    mat_corr *= np.tri(*mat_corr.shape, k=-1)
    # plot correlation matrix
    im = ax2.imshow(mat_corr,
                    cmap='bwr', vmin=-1, vmax=1)
    # tune panels
    ax2.step(np.arange(0, 17)-0.5, np.arange(0, 17)-0.5, color='k',
             linewidth=.7)
    rm_top_right_lines(ax2)
    ax2.set_xticks(np.arange(16), labels, rotation='270', fontsize=12)
    ax2.set_yticks(np.arange(16), labels, fontsize=12)
    ax2.set_xlabel('Inferred parameters', fontsize=14)
    ax2.set_ylabel('Inferred parameters', fontsize=14)


def plot_param_recovery_test(fig, ax,
        subjects=['Virtual_rat_random_params' for _ in range(50)],
        sv_folder=SV_FOLDER, corr=True):
    """
    Plots the recovered parameters vs the original.
    """
    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.06, right=0.94,
                        hspace=0.6, wspace=0.5)
    for a in ax:
        rm_top_right_lines(a)
        pos = a.get_position()
        a.set_position([pos.x0, pos.y0, pos.height, pos.height])
    labels = [r'Prior weight, $z_P$', r'Stimulus drift, $a_P$',
              r'EA bound, $\theta_{DV}$',
              r'CoM bound, $\theta_{COM}$',
              r'Afferent time, $t_{aff}$ (ms)', r'Efferent time, $t_{eff}$ (ms)',
              r'AI time offset, $t_{AI}$ (ms)',
              r'AI drift offset, $v_{AI}$',
              r'AI drift slope, $w_{AI}$',
              r'AI bound, $\theta_{AI}$',
              r'DV weight 1st readout, $\beta_{DV}$',
              r'DV weight update, $\beta_u$', r'Leak, $\lambda$',
              r'MT noise variance, $\sigma_{MT}$',
              r'MT offset, $\beta_0$', r'MT slope, $\beta_{TI}$']
    conf_mat = np.empty((len(labels), len(subjects)))
    conf_mat_rec = np.empty((len(labels), len(subjects)))
    for i_s, subject in enumerate(subjects):  # for each subject, load parameters
        conf = np.load(SV_FOLDER + 'virt_params/' +
                       'parameters_MNLE_BADS_prt_n50_' + 'virt_sim_' + str(i_s) + '.npy')
        conf_rec =  np.load(SV_FOLDER + 'virt_params/' +
                            'parameters_MNLE_BADS_prt_n50_prt_' + str(i_s) + '.npy')
        conf_mat[:, i_s] = conf
        conf_mat_rec[:, i_s] = conf_rec
    mlist = []
    rlist = []
    for i in range(len(labels)):
        max_val = max(conf_mat_rec[i, :])
        max_val_rec = max(conf_mat[i, :])
        max_total = max(max_val, max_val_rec)
        min_total = 0
        ax[i].set_title(labels[i], pad=12)
        # plot recovered vs original parameter
        if i == 4 or i == 5 or i == 6:  # recall dt=5ms
            ax[i].plot(conf_mat[i, :]*5, conf_mat_rec[i, :]*5,
                       marker='o', color='k', linestyle='')
            ax[i].plot([5*min_total*0.4,
                        5*max_total*1.6],
                       [5*min_total*0.4,
                        5*max_total*1.6])
            ax[i].set_xlim(5*min_total*0.4, 5*max_total*1.6)
            ax[i].set_ylim(5*min_total*0.4, 5*max_total*1.6)
            out = linregress(conf_mat[i, :]*5, 5*conf_mat_rec[i, :])
        else:
            ax[i].plot(conf_mat[i, :], conf_mat_rec[i, :],
                       marker='o', color='k', linestyle='')
            ax[i].plot([min_total*0.4, max_total*1.6],
                       [min_total*0.4, max_total*1.6])
            ax[i].set_xlim(min_total*0.4, max_total*1.6)
            ax[i].set_ylim(min_total*0.4, max_total*1.6)
            out = linregress(conf_mat[i, :], conf_mat_rec[i, :])  # linear regression
        r = out.rvalue  # correlation value
        m = out.slope
        if i != 13:
            mlist.append(m)
            rlist.append(r)
        xtcks = ax[i].get_yticks()
        if i == 8:
            ax[i].set_xticks(xtcks[:-1], [f'{x:.0e}' for x in xtcks[:-1]])
            ax[i].set_yticks(xtcks[:-1], [f'{x:.0e}' for x in xtcks[:-1]])
        else:
            ax[i].set_xticks(xtcks[:-1])    
        if i > 11:
            ax[i].set_xlabel('Original parameter')
        if i % 4 == 0:
            ax[i].set_ylabel('Inferred parameter')


def supp_plot_rt_distros_data_model(df, df_sim, sv_folder):
    """
    Plots supplementary figure 7.
    Plots the RT distributions for each subject, for both data and model.
    """
    fig, ax = plt.subplots(6, 5, figsize=(9, 10))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.06, right=0.95,
                        hspace=0.4, wspace=0.4)
    ax = ax.flatten()
    ev_vals = [0, 1]
    colormap = pl.cm.gist_gray_r(np.linspace(0.4, 1, len(ev_vals)))
    cmap_model = pl.cm.Reds(np.linspace(0.4, 1, len(ev_vals)))
    subjects = df.subjid.unique()
    labs_data = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    labs_model = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29]
    for i_s, subj in enumerate(subjects):  # for each subject
        # tune panels
        rm_top_right_lines(ax[labs_model[i_s]])
        rm_top_right_lines(ax[labs_data[i_s]])
        pos_ax_mod = ax[labs_model[i_s]].get_position()
        ax[labs_model[i_s]].set_position([pos_ax_mod.x0,
                                          pos_ax_mod.y0 + pos_ax_mod.height/12.5,
                                          pos_ax_mod.width,
                                          pos_ax_mod.height])
        pos_ax_dat = ax[labs_data[i_s]].get_position()
        ax[labs_data[i_s]].set_position([pos_ax_dat.x0,
                                          pos_ax_dat.y0 - pos_ax_dat.height/12.5,
                                          pos_ax_dat.width,
                                          pos_ax_dat.height])
        if (i_s+1) % 5 == 0:
            axmod = ax[labs_model[i_s]].twinx()
            axdat = ax[labs_data[i_s]].twinx()
            axdat.set_ylabel('Data')
            axmod.set_ylabel('Model')
            axmod.set_yticks([])
            axdat.set_yticks([])
            axmod.spines['bottom'].set_visible(False)
            axdat.spines['bottom'].set_visible(False)
            rm_top_right_lines(axdat)
            rm_top_right_lines(axmod)
        df_1 = df[df.subjid == subj]
        df_sim_1 = df_sim[df_sim.subjid == subj]
        coh_vec = df_1.coh2.values
        coh = df_sim_1.coh2.abs().values
        ax[labs_data[i_s]].set_ylim(-0.0001, 0.011)
        ax[labs_model[i_s]].set_ylim(-0.0001, 0.011)
        for ifb, fb in enumerate(df_1.fb):
            for j in range(len(fb)):
                coh_vec = np.append(coh_vec, [df_1.coh2.values[ifb]])
        # get fixation break (fb) trials
        fix_breaks =\
            np.vstack(np.concatenate([df_1.sound_len/1000,
                                      np.concatenate(df_1.fb.values)-0.3]))
        for iev, ev in enumerate(ev_vals):
            index = np.abs(coh_vec) == ev
            fix_breaks_2 = fix_breaks[index]*1e3
            rt_model = df_sim_1.sound_len.values[coh == ev]
            # plot distributions
            sns.kdeplot(fix_breaks_2.reshape(-1),
                        color=colormap[iev], ax=ax[labs_data[i_s]],
                        bw_adjust=1.5)
            sns.kdeplot(rt_model,
                        color=cmap_model[iev], ax=ax[labs_model[i_s]],
                        bw_adjust=1.5)
        # tune panels
        ax[labs_data[i_s]].set_xticks([])
        ax[labs_data[i_s]].set_title(subj)
        ax[labs_data[i_s]].set_xlim(-205, 410)
        ax[labs_model[i_s]].set_xlim(-205, 410)
        if (i_s) % 5 != 0:
            axmod = ax[labs_model[i_s]]
            axdat = ax[labs_data[i_s]]
            axdat.set_ylabel('')
            axmod.set_ylabel('')
            axdat.set_yticks([])
            axmod.set_yticks([])
        if i_s < 10:
            ax[labs_model[i_s]].set_xticks([])        
        if i_s >= 10:
            ax[labs_model[i_s]].set_xlabel('RT (ms)')
    # save figure
    fig.savefig(sv_folder+'supp_fig_8.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'supp_fig_8.png', dpi=400, bbox_inches='tight')


def supp_mt_per_rat(df, df_sim, title='', sv_folder=SV_FOLDER):
    """
    Plots supplementary figure 6.
    Plots the MT distribution for each subject, for both model and data.
    """
    fig, ax = plt.subplots(5, 3, figsize=(8, 10))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.2, wspace=0.3)
    ax = ax.flatten()
    for a in ax:
        rm_top_right_lines(a)
    subjects = df.subjid.unique()
    for i_s, subj in enumerate(subjects):
        df_1 = df[df.subjid == subj]
        df_sim_1 = df_sim[df_sim.subjid == subj]
        mt_rat = df_1.resp_len.values*1e3
        mt_model = df_sim_1.resp_len.values*1e3
        # plot MT distribution
        sns.kdeplot(mt_rat, color='k', ax=ax[i_s],
                    label='Rats', bw_adjust=3)
        sns.kdeplot(mt_model, color='r', ax=ax[i_s],
                    label='Model', bw_adjust=3)
        # tune panel
        if i_s >= 12:
            ax[i_s].set_xlabel('MT (ms)')
        else:
            ax[i_s].set_xticks([])
        if i_s % 3 != 0:
            ax[i_s].set_ylabel('')
            ax[i_s].set_yticks([])
        ax[i_s].set_title(subj)
        ax[i_s].set_xlim(-5, 725)
        ax[i_s].set_ylim(0, 0.0095)
    ax[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    # save figure
    fig.savefig(sv_folder+'supp_fig_7.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'supp_fig_7.png', dpi=400, bbox_inches='tight')


def plot_model_density(df_sim, sv_folder=SV_FOLDER, df=None, offset=0,
                       plot_data_trajs=False, n_trajs_plot=150,
                       pixel_precision=1, cmap='Reds', max_ms=400):
    """
    Plots density of the position of the model, it can plot rat trajectories on top.
    Plots supplementary figure 4.

    Parameters
    ----------
    df_sim : data frame
        Data frame with simulations.
    df : data frame, optional
        Data frame with rat data. The default is None.
    offset : int, optional
        Padding. The default is 0.
    plot_data_trajs : bool, optional
        Whereas to plot rat trajectories on top or not. The default is False.
    n_trajs_plot : int, optional
        In case of plotting the trajectories, how many. The default is 150.
    pixel_precision : float, optional
        Pixel precision for the density (the smaller the cleaner the plot).
        The default is 5.
    cmap : str, optional
        Colormap. The default is 'Reds'.

    Returns
    -------
    None.

    """
    n_steps = int(max_ms/5)
    fig2, ax2 = plt.subplots(nrows=3, ncols=3, figsize=(10, 7))
    np.random.seed(seed=5)  # set seed
    fig2.tight_layout()
    ax2 = ax2.flatten()
    coh = df_sim.coh2.values
    zt = np.round(df_sim.norm_allpriors.values, 1)
    coh_vals = [-1, 0, 1]
    zt_vals = [-np.max(np.abs(zt)), -np.max(np.abs(zt))*0.4,
               -0.05, 0.05,
               np.max(np.abs(zt))*0.4, np.max(np.abs(zt))]
    i = 0
    ztlabs = [-1, 0, 1]
    gkde = scipy.stats.gaussian_kde  # we define gkde that will generate the kde
    if plot_data_trajs:
        bins = np.array([-1.1, 1.1])  # for data plotting
        bintype = 'edges'
        trajectory = 'trajectory_y'
        df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    for ie, ev in enumerate(coh_vals):
        for ip, pr in enumerate(zt_vals):
            ip2 = 2*ip
            if ip == 3:
                break
            index = (zt >= zt_vals[ip2]) & (zt < zt_vals[ip2+1]) & (coh == ev)  # index of filtered
            max_len = max([len(t) for t in df_sim.traj[index].values])
            mat_fin = np.empty((sum(index), max_len+offset))
            mat_fin[:] = np.nan
            trajs = df_sim.traj[index].values
            for j in range(sum(index)):
                mat_fin[j, :len(trajs[j])] = trajs[j]  # mat_fin contains trajectories by rows
                mat_fin[j, len(trajs[j]):-1] = trajs[j][-1]  # set the last value (-75 or 75) until the end
            values = np.arange(-80, 81, pixel_precision)
            mat_final_density = np.empty((len(values), n_steps))  # matrix that will contain density by columns
            mat_final_density[:] = np.nan
            for j in range(2, n_steps):
                yvalues = np.nanmean(mat_fin[:, j*5:(j+1)*5], axis=1)  # we get the trajectory values
                kernel_1 = gkde(yvalues)  # we create the kernel using gkde
                vals_density = kernel_1(values)  # we evaluate the values defined before
                mat_final_density[:, j] = vals_density / np.nansum(vals_density)  # we normalize the density
            ax2[i].imshow(np.flipud(mat_final_density), cmap=cmap, aspect='auto',
                          norm=LogNorm(vmin=0.001, vmax=0.6))  # plot the matrix
            ax2[i].set_xlim(0, n_steps+0.2)
            ax2[i].set_ylim(len(values), 0)
            if i == 2 or i == 5 or i == 8:
                ax1 = ax2[i].twinx()
                ax1.set_yticks([])
                ax1.set_ylabel('stimulus = {}'.format(ztlabs[int((i-2) // 3)]),
                               rotation=90, labelpad=5, fontsize=12)
            if i >= 6:
                ax2[i].set_xticks(np.arange(0, n_steps+1, 20), np.arange(0, n_steps+1, 20)*5)
            else:
                ax2[i].set_xticks([])
            if i % 3 == 0:
                ax2[i].set_yticks(np.arange(0, len(values), int(80/pixel_precision)),
                                  ['5.6', '0', '-5.6'])
            else:
                ax2[i].set_yticks([])
            if i % 3 == 0:
                ax2[i].set_ylabel('y position (cm)')
            if i >= 6:
                ax2[i].set_xlabel('Time (ms)')
            if plot_data_trajs:
                index = (zt >= zt_vals[ip2]) & (zt < zt_vals[ip2+1]) & (coh == ev) &\
                    (df.subjid == 'LE38') # index of filtered
                # to extract interpolated trajs in mat --> these aren't signed
                _, _, _, mat, idx, _ =\
                trajectory_thr(df.loc[index], 'choice_x_prior', bins,
                               collapse_sides=True, thr=30, ax=None, ax_traj=None,
                               return_trash=True, error_kwargs=dict(marker='o'),
                               cmap=None, bintype=bintype,
                               trajectory=trajectory, plotmt=False, alpha_low=False)
                mat_0 = mat[0]
                # we multiply by response to have the sign
                mat_0 = mat_0*(df.loc[idx[0]].R_response.values*2-1).reshape(-1, 1)
                mtime = df.loc[idx[0]].resp_len.values
                n_trajs = mat_0.shape[0]
                # we select the number of trajectories that we want
                np.random.seed(3)
                index_trajs_plot = np.random.choice(np.arange(n_trajs), n_trajs_plot)
                for ind in index_trajs_plot:
                    traj = mat_0[ind, :]
                    traj = traj - np.nanmean(traj[500:700])
                    # we do some filtering
                    if sum(np.abs(traj[700:700+int(max_ms)]) > 80) > 1:
                        continue
                    if np.abs(traj[700]) > 5:
                        continue
                    if mtime[ind] > 0.4:
                        continue
                    if np.abs(traj[920]) < 10:
                        continue
                    ax2[i].plot(np.arange(0, n_steps, 0.2), (-traj[700:700+int(max_ms)]+80)/160*len(values),
                                color='blue', linewidth=0.3)
            i += 1
    ax2[0].set_title('prior = -1')
    ax2[1].set_title('prior = 0')
    ax2[2].set_title('prior = 1')
    # save figure
    fig2.savefig(sv_folder+'supp_fig_6.svg', dpi=400, bbox_inches='tight')
    fig2.savefig(sv_folder+'supp_fig_6.png', dpi=400, bbox_inches='tight')


def get_human_mt(df_data):
    """
    Returns the MT from human given the human data frame.
    """
    motor_time = []
    times = df_data.times.values
    for tr in range(len(df_data)):
        ind_time = [True if t != '' else False for t in times[tr]]
        time_tr = np.array(times[tr])[np.array(ind_time)].astype(float)
        mt = time_tr[-1]
        if mt > 2:
            mt = 2
        motor_time.append(mt*1e3)
    return motor_time


def get_human_data(user_id, sv_folder=SV_FOLDER, nm='300'):
    """
    Returns human data.
    """
    if user_id == 'alex':
        folder = 'C:\\Users\\alexg\\Onedrive\\Escritorio\\CRM\\Human\\80_20\\'+nm+'ms\\'
    if user_id == 'alex_CRM':
        folder = 'C:/Users/agarcia/Desktop/CRM/human/'
    if user_id == 'idibaps':
        folder =\
            '/home/molano/Dropbox/project_Barna/psycho_project/80_20/'+nm+'ms/'
    if user_id == 'idibaps_alex':
        folder = '/home/jordi/DATA/Documents/changes_of_mind/humans/'+nm+'ms/'
    if user_id == 'sara':
        folder = 'C:\\Users\\Sara Fuentes\\OneDrive - Universitat de Barcelona\\Documentos\\EBM\\4t\\IDIBAPS\\80_20\\'+nm+'ms\\'
    subj = ['general_traj_all']
    steps = [None]
    # retrieve data
    df = ah.traj_analysis(data_folder=folder,
                          subjects=subj, steps=steps, name=nm,
                          sv_folder=sv_folder)
    return df


def simulate_model_humans(df_data, stim, load_params, params_to_explore=[]):
    """
    Simulates model for humans given the data frame and the generated stimulus.
    """
    choice = df_data.R_response.values*2-1
    hit = df_data.hithistory.values*2-1
    subjects = df_data.subjid.unique()
    subjid = df_data.subjid.values
    gt = (choice*hit+1)/2
    coh = df_data.avtrapz.values*5
    zt = df_data.norm_allpriors.values*3
    trial_index = df_data.origidx.values
    num_tr = len(trial_index)
    # simulate
    hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj, x_val_at_updt =\
        run_simulation_different_subjs(
            stim=stim, zt=zt, coh=coh, gt=gt,
            trial_index=trial_index, num_tr=num_tr, human=True,
            subject_list=subjects, subjid=subjid, simulate=True,
            load_params=load_params, params_to_explore=params_to_explore)
    return hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj, x_val_at_updt


def simulate_random_rat(n_samples, sv_folder, stim, zt, coh, gt, trial_index, num_tr):
    """
    Simulate rat data for the parameter recovery test.
    It loads/saves a certain distribution of parameters, and simulates the model
    with these configurations.
    """
    subjid = np.array(['Virtual_rat_random_params' for _ in range(len(coh))])
    subjects = ['Virtual_rat_random_params']
    for n in range(n_samples):
        extra_label = 'virt_sim_' + str(n)
        parameter_string = sv_folder + 'virt_params/' + 'parameters_MNLE_BADS_prt_n50_' +\
            extra_label + '.npy'
        os.makedirs(os.path.dirname(parameter_string), exist_ok=True)
        if os.path.exists(parameter_string):
            pars = np.load(parameter_string)
            simulate = False
        else:
            # parameters sampled from uniform distributions
            pars = np.array([np.random.uniform(1e-2, .5),
                             np.random.uniform(1e-2, .15),
                             np.random.uniform(1e-1, 4),
                             np.random.uniform(1e-6, .3),
                             np.random.uniform(3, 6),
                             np.random.uniform(3, 6),
                             np.random.uniform(5, 16),
                             np.random.uniform(0.015, 0.06),
                             np.random.uniform(1e-6, 2e-5),
                             np.random.uniform(1.5, 3),
                             np.random.uniform(50, 200),
                             np.random.uniform(200, 400),
                             np.random.uniform(0.05, 0.15),
                             np.random.uniform(18, 22),
                             np.random.uniform(200, 300),
                             np.random.uniform(0.05, 0.2)])
            np.save(sv_folder + 'virt_params/' + 'parameters_MNLE_BADS_prt_n50_' +
                    extra_label + '.npy',
                    pars)
            simulate = True
        # simulate, simulations are saved
        _, _, _, _, _,\
            _, _, _ =\
            run_simulation_different_subjs(
                stim=stim, zt=zt, coh=coh, gt=gt,
                trial_index=trial_index, num_tr=num_tr, human=False,
                subject_list=subjects, subjid=subjid, simulate=simulate,
                load_params=True, extra_label=extra_label)
    return


def supp_p_com_vs_rt_silent(df, ax, bins_rt=BINS_RT, label='', column='CoM_sugg',
                            color='k', adjusted_error=False):
    """
    Plots p(CoM) or p(reversal) vs RT for silent trials across subjects.
    """
    subjid = df.subjid
    subjects = np.unique(subjid)
    com_data = np.empty((len(subjects), len(bins_rt)-1))
    com_data[:] = np.nan
    xpos_plot = (bins_rt[:-1] + bins_rt[1:]) / 2
    for i_s, subject in enumerate(subjects):  # for each subject
        df_plot = df.loc[subjid == subject]
        mean_pcom_mod_det = []
        for i_rt, rt in enumerate(bins_rt[:-1]):  # for each RT bin
            df_filt = df_plot.loc[(df_plot.sound_len >= rt) &
                                  (df_plot.sound_len < bins_rt[i_rt+1])]
            mean_pcom_mod_det.append(np.nanmean((df_filt[column])))
        com_data[i_s, :len(mean_pcom_mod_det)] = mean_pcom_mod_det
    if adjusted_error:
        yval = np.nanmean(com_data, axis=0)
        adj_factor_per_sub = - com_data + yval
        mean_val = com_data + adj_factor_per_sub
        error = np.nanstd(mean_val, axis=0)/np.sqrt(len(subjects))
        
    else:
        yval = np.nanmedian(com_data, axis=0)
        error = np.nanstd(com_data, axis=0)/np.sqrt(len(subjects))
    # plot
    ax.errorbar(xpos_plot, yval,
                yerr=error,
                color=color, label=label)
    ax.set_ylabel('p(reversal) - silent')
    ax.set_xlabel('Reaction time (ms)')


def prev_vs_prior_evidence(df, df_sim, ax):
    """
    Plots p(reversal) and p(CoM) vs prior evidence in silent trials.
    Special trial = 2 is silent trials.
    """
    nsubs = len(df.subjid.unique())
    df_prior = df.copy().loc[df.special_trial == 2]
    df_prior['choice_x_prior'] = df_prior.norm_allpriors
    df_prior_sim = df_sim.copy().loc[df_sim.special_trial == 2]
    df_prior_sim['choice_x_prior'] = df_prior_sim.norm_allpriors
    # bin data with equipopulated bins
    bins_data, _, _, _, _ = \
        get_bin_info(df_prior,
                     condition='choice_x_prior', prior_limit=1,
                     after_correct_only=True, rt_lim=300,
                     fpsmin=29, num_bins_prior=8, rtmin=0, silent=True)
    bins_sim, _, _, _, _ = \
        get_bin_info(df_prior_sim,
                     condition='choice_x_prior', prior_limit=1,
                     after_correct_only=True, rt_lim=300,
                     fpsmin=29, num_bins_prior=8, rtmin=0, silent=True)
    bins_data = np.linspace(0, 1, 4)
    bins_sim = bins_data
    # initialize arrays
    p_rev_sil = np.empty((nsubs, len(bins_data)-1))
    p_rev_sil_sim = np.empty((nsubs, len(bins_sim)-1))
    p_com_sil_sim = np.empty((nsubs, len(bins_sim)-1))
    for i_s, subject in enumerate(df_prior.subjid.unique()):  # for each subject
        df_sub = df_prior.loc[df_prior.subjid == subject]
        df_sub_sim = df_prior_sim.loc[df_prior_sim.subjid == subject]
        for bin in range(len(bins_data)-1):  # for each prior bin
            prev_sub = df_sub.loc[(df_sub.choice_x_prior >= bins_data[bin]) &
                                    (df_sub.choice_x_prior < bins_data[bin+1]) &
                                    (df_sub.special_trial == 2),
                                    'CoM_sugg']
            p_rev_sil[i_s, bin] = np.nanmean(prev_sub)
            prev_sub = df_sub_sim.loc[
                (df_sub_sim.choice_x_prior >= bins_sim[bin]) &
                (df_sub_sim.choice_x_prior < bins_sim[bin+1]) &
                (df_sub_sim.special_trial == 2), 'com_detected']
            p_rev_sil_sim[i_s, bin] = np.nanmean(prev_sub)  # average to get proportion
            prev_sub = df_sub_sim.loc[
                (df_sub_sim.choice_x_prior >= bins_sim[bin]) &
                (df_sub_sim.choice_x_prior < bins_sim[bin+1]) &
                (df_sub_sim.special_trial == 2), 'CoM_sugg']
            p_com_sil_sim[i_s, bin] = np.nanmean(prev_sub)  # average to get proportion
    prior_x_vals = bins_data[:-1] + np.diff(bins_data)/2
    prior_x_vals_mod = bins_sim[:-1] + np.diff(bins_sim)/2 + 0.02
    ax2 = ax
    # average across subjects
    p_rev_sil_all = np.nanmean(p_rev_sil, axis=0)
    p_rev_sil_all_err = np.nanstd(p_rev_sil, axis=0) / np.sqrt(6)
    p_rev_sil_all_sim = np.nanmean(p_rev_sil_sim, axis=0)
    p_rev_sil_all_sim_err = np.nanstd(p_rev_sil_sim, axis=0) / np.sqrt(6)
    p_com_sil_all_sim = np.nanmean(p_com_sil_sim, axis=0)
    p_com_sil_all_sim_err = np.nanstd(p_com_sil_sim, axis=0) / np.sqrt(6)
    # plot p(reversal) for data and model
    ax2.errorbar(prior_x_vals, p_rev_sil_all, p_rev_sil_all_err,
                  color='k', label='Rats')
    ax2.errorbar(prior_x_vals_mod, p_rev_sil_all_sim, p_rev_sil_all_sim_err,
                  color='gray', label='Model, reversals')
    axtwin = ax2.twinx()
    # plot p(CoM) for model in a twin vertical axis
    axtwin.errorbar(prior_x_vals_mod+0.02, p_com_sil_all_sim,
                    p_com_sil_all_sim_err,
                    color='r', label='Model, CoMs')
    # tune panel
    axtwin.set_ylim(0, 0.33)
    axtwin.spines['right'].set_color('red')
    axtwin.tick_params(axis='y', colors='red')
    ax2.spines['top'].set_visible(False)
    axtwin.set_ylabel('p(CoM)', color='red')
    axtwin.spines['top'].set_visible(False)
    axtwin.yaxis.label.set_color('red')
    rm_top_right_lines(ax2)
    ax2.set_xlabel('Prior evidence')
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_ylim(0, 0.085)
    ax2.set_yticks([0, 0.02, 0.04, 0.06])
    axtwin.set_yticks([0, 0.1, 0.2, 0.3])
    legendelements = [Line2D([0], [0], color='k', lw=2,
                             label='Rats'),
                      Line2D([0], [0], color='grey', lw=2,
                             label='Model reversals'),
                      Line2D([0], [0], color='r', lw=2,
                             label='Model CoMs')]
    ax.legend(handles=legendelements, loc='upper right',
              bbox_to_anchor=(0.9, 1.05), frameon=False)


def supp_silent(df, df_sim, sv_folder):
    """
    Plots supplementary figure 11.
    a) p(reversal | silent) vs p(reversal | stim=0)
    b) p(reversal) vs RT
    c) p(reversal) and p(CoM) vs prior evidence.
    d) p(right response) vs prior evidence.
    """
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(9, 8))
    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.45)
    labs = ['a', 'b', 'c', 'd', 'e', '']
    ax = ax.flatten()
    for i_a, a in enumerate(ax):
        rm_top_right_lines(a)
        a.text(-0.12, 1.1, labs[i_a], transform=a.transAxes, fontsize=16,
               fontweight='bold', va='top', ha='right')
    # a)
    ax_a = ax[0]
    subjects = df.loc[df.special_trial == 2].subjid.unique()
    column_sim = 'com_detected'
    column = 'CoM_sugg'
    com_coh_0 = []
    com_silent = []
    com_coh_0_sim = []
    com_silent_sim = []
    i = 0
    colormap = pl.cm.jet(np.linspace(0., 1, len(df.subjid.unique())))
    for subj in subjects:  # for each subject
        df_1 = df.copy().loc[(df.subjid == subj) & (df.special_trial == 2)]
        df_sim_1 = df_sim.copy().loc[(df_sim.subjid == subj) &
                                     (df_sim.special_trial == 2)]
        mean_coms = np.nanmean(df_1[column].values)
        mean_coms_sim = np.nanmean(df_sim_1[column_sim].values)
        com_silent.append(mean_coms)
        com_silent_sim.append(mean_coms_sim)
        df_1 = df.copy().loc[(df.subjid == subj) & (df.special_trial == 0) &
                             (df.coh2 == 0)]
        df_sim_1 = df_sim.copy().loc[
            (df_sim.subjid == subj) & (df_sim.special_trial == 0) &
            (df_sim.coh2 == 0)]
        mean_coms = np.nanmean(df_1[column].values)
        mean_coms_sim = np.nanmean(df_sim_1[column_sim].values)
        com_coh_0.append(mean_coms)
        com_coh_0_sim.append(mean_coms_sim)
        # plot p(reversal | silent) vs p(reversal | stim=0) for data and model
        ax_a.plot(com_coh_0[i], com_silent[i], marker='o',
                  linestyle='', color=colormap[i], markersize=6)
        ax_a.plot(com_coh_0_sim[i], com_silent_sim[i], marker='^',
                  linestyle='', color=colormap[i], markersize=6)
        i += 1
    # tune panel
    legendelements = [Line2D([0], [0], color='k', linestyle='', marker='o',
                             markersize=7, label='Rats'),
                      Line2D([0], [0], color='k', linestyle='', marker='^',
                             markersize=7, label='Model')]
    ax_a.legend(handles=legendelements, frameon=False)
    max_val = np.max((com_silent_sim, com_coh_0_sim))
    ax_a.plot([0, max_val+0.005], [0, max_val+0.005], color='grey', alpha=0.4)
    ax_a.set_xticks([0, 0.03, 0.06])
    ax_a.set_yticks([0, 0.03, 0.06])
    ax_a.set_ylabel('p(reversal | silent)')
    ax_a.set_xlabel('p(reversal | stim=0)')
    # b)
    ax_b = ax[1]
    # plot p(reversal) vs RT for data and model
    supp_p_com_vs_rt_silent(df.loc[(df.special_trial == 2)],
                            ax_b, bins_rt=BINS_RT, label='Rats',
                            column=column, color='k')
    supp_p_com_vs_rt_silent(df_sim.loc[(df_sim.special_trial == 2)],
                            ax_b, bins_rt=BINS_RT, label='Model',
                            column=column_sim, color='silver')
    # tune panel
    ax_b.set_ylabel('p(reversal)')
    ax_b.set_xlim(-1, 301)
    ax_b.set_yticks([0, 0.02, 0.04])
    ax_b.legend(frameon=False)
    # c)
    # plot p(rev) vs prior
    prev_vs_prior_evidence(df, df_sim, ax[2])
    ax[2].set_ylabel('p(reversal)')
    # d) 
    ax_d = ax[3]
    df_prior = df.copy().loc[df.special_trial == 2]
    df_prior['choice_x_prior'] = df_prior.norm_allpriors
    df_prior_sim = df_sim.copy().loc[df_sim.special_trial == 2]
    df_prior_sim['choice_x_prior'] = df_prior_sim.norm_allpriors
    # define equipopulated prior bins
    bins, _, _, _, _ = \
        get_bin_info(df_prior,
                     condition='choice_x_prior', prior_limit=1,
                     after_correct_only=True, rt_lim=300,
                     fpsmin=29, num_bins_prior=8, rtmin=0, silent=True)
    nsubs = len(df_prior.subjid.unique())
    p_right = np.empty((nsubs, len(bins)-1))
    p_right_sim = np.empty((nsubs, len(bins)-1))
    # get p(right response) for each prior bin
    for i_s, subject in enumerate(df_prior.subjid.unique()):
        df_sub = df_prior.loc[df_prior.subjid == subject]
        df_sub_sim = df_prior_sim.loc[df_prior_sim.subjid == subject]
        for bin in range(len(bins)-1):
            pright_sub = df_sub.loc[(df_sub.choice_x_prior >= bins[bin]) &
                                    (df_sub.choice_x_prior < bins[bin+1]),
                                    'R_response']
            p_right[i_s, bin] = np.nanmean(pright_sub)
            pright_sub = df_sub_sim.loc[
                (df_sub_sim.choice_x_prior >= bins[bin]) &
                (df_sub_sim.choice_x_prior < bins[bin+1]), 'R_response']
            p_right_sim[i_s, bin] = np.nanmean(pright_sub)
    prior_x_vals = bins[:-1] + np.diff(bins)/2
    # average across subjects
    p_right_all = np.nanmean(p_right, axis=0)
    p_right_all_sim = np.nanmean(p_right_sim, axis=0)
    p_right_error_all = np.nanstd(p_right, axis=0) / np.sqrt(nsubs)
    p_right_error_all_sim = np.nanstd(p_right_sim, axis=0) / np.sqrt(nsubs)
    # plot p(right response) vs prior evidence
    ax_d.errorbar(prior_x_vals, p_right_all, yerr=p_right_error_all, color='k',
                  label='Rats')
    ax_d.errorbar(prior_x_vals, p_right_all_sim, yerr=p_right_error_all_sim,
                  color='silver', label='Model')
    # tune panel
    ax_d.set_xlabel('Prior evidence')
    ax_d.set_ylabel('p(Right response)')
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.set_xticks([prior_x_vals[0], 0, prior_x_vals[-1]], ['L', '0', 'R'])
    # save figure
    fig.savefig(sv_folder+'/supp_silent.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'/supp_silent.png', dpi=400, bbox_inches='tight')


def com_heatmap(x, y, com, flip=False, annotate=True, predefbins=None,
                return_mat=False, annotate_div=1, **kwargs):
    """
    Produces matrix of the variable com (could be other, like choice, if defined like so),
    with respect to the variables x (usually prior) and y (usually stimulus).
    If return_mat is True, most of the function (plotting) is not used.
    """
    tmp = pd.DataFrame(np.array([x, y, com]).T,
                       columns=["prior", "stim", "com"])

    # make bins
    tmp["binned_prior"] = np.nan
    if predefbins is None:  # there are no predefined bins
        predefbinsflag = False
        bins = [-1.01]
        # bins by quantiles regarding the abs value
        for i_p, perc in enumerate([0.75, 0.5, 0.25, 0.25, 0.5, 0.75]):
            if i_p < 3:
                bins.append(-np.quantile(tmp.prior.abs(), perc))
            else:
                bins.append(np.quantile(tmp.prior.abs(), perc))
        bins.append(1.01)
        bins = np.array(bins)
    else:
        predefbinsflag = True
        bins = np.asarray(predefbins[0])

    # bin prior
    tmp.loc[:, "binned_prior"], priorbins = pd.cut(
        tmp.prior, bins=bins, retbins=True, labels=np.arange(bins.size-1),
        include_lowest=True)

    tmp.loc[:, "binned_prior"] = tmp.loc[:, "binned_prior"].astype(int)
    # define labels (plotting)
    priorlabels = [round((priorbins[i] + priorbins[i + 1]) / 2, 2)
                   for i in range(bins.size-1)]

    tmp["binned_stim"] = np.nan
    maxedge_stim = tmp.stim.abs().max()
    if not predefbinsflag:  # there are no predefined bins
        bins = np.linspace(-maxedge_stim - 0.01, maxedge_stim + 0.01, 8)
    else:
        bins = np.asarray(predefbins[1])
    # bin stimulus
    tmp.loc[:, "binned_stim"], stimbins = pd.cut(
        tmp.stim, bins=bins, retbins=True, labels=np.arange(bins.size-1),
        include_lowest=True)
    tmp.loc[:, "binned_stim"] = tmp.loc[:, "binned_stim"].astype(int)
    # define labels (plotting)
    stimlabels = [round((stimbins[i] + stimbins[i + 1]) / 2, 2)
                  for i in range(bins.size-1)]

    # populate matrices
    matrix = np.zeros((len(stimlabels), len(priorlabels)))  # p(com) matrix
    nmat = matrix.copy()  # number of observations matrix
    plain_com_mat = matrix.copy()
    for i in range(len(stimlabels)):  # for each bin (assumed SQUARE matrix)
        # count for each bin (prior, stim) the number of com
        switch = (tmp.loc[(tmp.com == True) & (tmp.binned_stim == i)]
                  .groupby("binned_prior")["binned_prior"]
                  .count())
        # count for each bin (prior, stim) the number of observations (trials)
        nobs = (switch + tmp.loc[(tmp.com == False) & (tmp.binned_stim == i)]
            .groupby("binned_prior")["binned_prior"].count())
        # fill where there are no CoM (instead it will be nan)
        nobs.loc[nobs.isna()] = (tmp.loc[(tmp.com == False) & (tmp.binned_stim == i)]
                                 .groupby("binned_prior")["binned_prior"].count()
                                 .loc[nobs.isna()])
        # the value for matrix will be the count / #observations
        crow = switch / nobs
        nmat[i, nobs.index.astype(int)] = nobs
        plain_com_mat[i, switch.index.astype(int)] = switch.values
        matrix[i, crow.index.astype(int)] = crow


    if return_mat:
        # matrix is com/obs, nmat is number of observations
        return matrix, nmat

    if isinstance(annotate, str):  # wether to annotate or not some numbers in the matrix plot
        if annotate == 'com':  # annotate #coms
            annotate = True
            annotmat = plain_com_mat/annotate_div
        if annotate == 'counts':  # annotate #observations
            annotate = True
            annotmat = nmat/annotate_div
    else:
        annotmat = nmat/annotate_div  # annotate #observations

    if not kwargs:
        kwargs = dict(cmap="viridis", fmt=".0f")
    # plotting
    if flip:
        if annotate:
            g = sns.heatmap(np.flipud(matrix), annot=np.flipud(annotmat),
                            **kwargs).set(xlabel="prior", ylabel="average stim",
                                          xticklabels=priorlabels,
                                          yticklabels=np.flip(stimlabels))
        else:
            g = sns.heatmap(np.flipud(matrix), annot=None, **kwargs).set(
                xlabel="prior", ylabel="average stim",
                xticklabels=priorlabels, yticklabels=np.flip(stimlabels))
    else:
        if annotate:
            g = sns.heatmap(matrix, annot=annotmat, **kwargs).set(
                xlabel="prior", ylabel="average stim", xticklabels=priorlabels,
                yticklabels=stimlabels)
        else:
            g = sns.heatmap(matrix, annot=None, **kwargs).set(
                xlabel="prior", ylabel="average stim", xticklabels=priorlabels,
                yticklabels=stimlabels)
    return g


def binned_curve(df, var_toplot, var_tobin, bins, errorbar_kw={}, ax=None, sem_err=True,
                 xpos=None, subplot_kw={}, legend=True, traces=None, traces_kw={},
                 traces_rolling=0, xoffset=True, median=False, return_data=False):
    """ bins a var and plots a var according to those bins
    df: dataframe.
    var_toplot: str, col in df.
    var_tobin: str, col in df.
    bins: edges to bin.
    """
    mdf = df.copy()
    # bin variable
    mdf["tmp_bin"] = pd.cut(mdf[var_tobin], bins,
                            include_lowest=True, labels=False)
    # plotting options
    traces_default_kws = {"color": "grey", "alpha": 0.15}
    traces_default_kws.update(traces_kw)

    if ax is None and not return_data:
        f, ax = plt.subplots(**subplot_kw)

    if sem_err:  # error
        errfun = sem  # standard error of the mean
    else:
        errfun = groupby_binom_ci  # confidence intervals
    if median:  # median or mean
        tmp = mdf.groupby("tmp_bin")[var_toplot].agg(m="median", e=errfun)
    else:
        tmp = mdf.groupby("tmp_bin")[var_toplot].agg(m="mean", e=errfun)

    if isinstance(xoffset, (int, float)):
        xoffsetval = xoffset
        xoffset = False
    elif isinstance(xoffset, bool):
        if not xoffset:
            xoffsetval = 0

    if xpos is None:
        xpos_plot = tmp.index
        if xoffset:
            xpos_plot += (tmp.index[1] - tmp.index[0]) / 2
        else:
            xpos_plot += xoffsetval
    elif isinstance(xpos, (list, np.ndarray) ):
        xpos_plot = np.array(xpos) + xoffsetval
        if xoffset:
            xpos_plot += (xpos_plot[1] - xpos_plot[0]) / 2
    elif isinstance(xpos, (int, float)):
        if xoffset:
            xpos_plot += (xpos_plot[1] - xpos_plot[0]) / 2
        else:
            xpos_plot = tmp.index * xpos + xoffsetval
    elif isinstance(xpos, (types.FunctionType, types.LambdaType)):
        xpos_plot = xpos(tmp.index)

    # define error to plot as errorbars
    if sem_err:
        yerrtoplot = tmp["e"]
    else:
        yerrtoplot = [tmp["e"].apply(
            lambda x: x[0]), tmp["e"].apply(lambda x: x[1])]

    if "label" not in errorbar_kw.keys():
        errorbar_kw["label"] = var_toplot

    if return_data:
        # returns x, y and errors. Function stops here if return_data.
        return xpos_plot, tmp["m"], yerrtoplot

    # plot
    ax.errorbar(xpos_plot, tmp["m"], yerr=yerrtoplot, **errorbar_kw)
    if legend:
        ax.legend()

    if traces is not None:  # plot individual traces for each subject (traces = 'subjid')
        traces_tmp = mdf.groupby([traces, "tmp_bin"])[var_toplot].mean()
        for tr in mdf[traces].unique():
            if xpos is not None:
                if isinstance(xpos, (int, float)):
                    if not xoffset:
                        xpos_tr = traces_tmp[tr].index * xpos + xoffsetval
                    else:
                        xpos_tr = traces_tmp[tr].index * xpos
                        xpos_tr += (xpos_tr[1] - xpos_tr[0]) / 2
                else:
                    raise NotImplementedError(
                        "traces just work with xpos=None/float/int, offsetval")
            if traces_rolling:
                y_traces = (traces_tmp[tr].rolling(
                    traces_rolling, min_periods=1).mean().values)
            else:
                y_traces = traces_tmp[tr].values
            ax.plot(xpos_tr, y_traces, **traces_default_kws)
    return ax


def interpolapply(row, stamps="trajectory_stamps", ts_fix_onset="fix_onset_dt",
                  trajectory="trajectory_y", resp_side="R_response", collapse_sides=False,
                  interpolatespace=np.linspace(-700000, 1000000, 1701),
                  fixation_us=300000, align="action", interp_extend=False, discarded_tstamp=0):
    """
    Performs linear interpolation of trajectories given the stamps and the value at
    each timestamp.
    Align:
        - 'action': aligns to movement onset
        - 'sound': aligns to sound onset
        - 'response': aligns to response onset
    row is a df row, with columns named as the strings given by
    stamps, ts_fix_onset and trajectory.
    """
    x_vec = []
    y_vec = []
    try:
        x_vec = row[stamps] - np.datetime64(row[ts_fix_onset])
        # aligned to fixation onset (0) using timestamps
        # by def 0 aligned to fixation
        if align == "sound":
            x_vec = (x_vec -
                     np.timedelta64(int(fixation_us),
                                    "us")).astype(float)
        elif align == "action":
            x_vec = (x_vec - int(fixation_us + (row["sound_len"] * 1e3))).astype(float)
            # shift it in order to align 0 with motor-response/action onset
        elif align == "response":
            x_vec = (x_vec - np.timedelta64(int(fixation_us + (row["sound_len"] * 10 ** 3)
                                                + (row["resp_len"] * 10 ** 6)),
                                            "us",)).astype(float)
        else:
            x_vec = x_vec.astype(float)

        x_vec = x_vec[discarded_tstamp:]

        if isinstance(trajectory, tuple):
            y_vec = row[trajectory[0]][:, trajectory[1]]
        else:  # is a column name
            y_vec = row[trajectory]
        if collapse_sides:
            if (row[resp_side] == 0):
                y_vec = y_vec * -1
        if interp_extend:  # interpolate, fill value with last value
            f = interpolate.interp1d(
                x_vec, y_vec, bounds_error=False, fill_value=(y_vec[0], y_vec[-1]))
        else:
            f = interpolate.interp1d(
                x_vec, y_vec, bounds_error=False)
            # should fill everything else with NaNs
        out = f(interpolatespace)  # interpolate
        return out
    except Exception:
        return np.array([np.nan] * interpolatespace.size)


def tachometric(df, ax=None,
    hits='hithistory',  # column name
    evidence='avtrapz',  # column name
    evidence_bins=np.array([0, 0.15, 0.30, 0.60, 1.05]),  # bins for stimulus evidence
    rt='sound_len',  # column
    rtbins=np.arange(0, 201, 3),  # bins for reaction time
    fill_error=False,  # if true it uses fill between instead of errorbars
    error_kws={}, cmap='inferno', subplots_kws={},  # ignored if ax is provided
    labels=None, linestyle='solid', plot=True):
    """
    Plots tachometric curves (accuracy vs RT conditioned on stimulus strength).
    """
    cmap = pl.cm.get_cmap(cmap)
    rtbinsize = rtbins[1]-rtbins[0]
    rtbins_diff = np.diff(rtbins)
    error_kws_ = dict(marker='', capsize=3)
    error_kws_.update(error_kws)

    tmp_df = df
    tmp_df['rtbin'] = pd.cut(
        tmp_df[rt], rtbins, labels=np.arange(rtbins.size-1),
        retbins=False, include_lowest=True, right=True
    ).astype(float)
    if ax is None and plot:
        f, ax = plt.subplots(**subplots_kws)
    n_subs = len(df.subjid.unique())
    vals_tach_per_sub = np.empty((len(rtbins)-1, n_subs))
    vals_tach_per_sub[:] = np.nan
    evidence_bins = np.sort(np.round(tmp_df[evidence].abs(), 2).unique())
    for i in range(evidence_bins.size):  # for each value of stimulus
        for i_s, subj in enumerate(df.subjid.unique()):  # for each subject
            tmp = (
                tmp_df.loc[(tmp_df[evidence].abs() == evidence_bins[i]) &
                    (df.subjid == subj)]  # select stim str
                .groupby('rtbin')[hits].mean()).values
            vals_tach_per_sub[:len(tmp), i_s] = tmp
        # average across subjects
        vals_total = np.nanmean(vals_tach_per_sub, axis=1)            
        error_total = np.nanstd(vals_tach_per_sub, axis=1)/np.sqrt(n_subs)
        clabel = evidence_bins[i]
        if fill_error and plot:
            # plot and panel tuning
            ax.plot(rtbins[:-1] + rtbins_diff/2, vals_total, label=clabel,
                    c=cmap((i+1)/(evidence_bins.size)), marker=error_kws.get('marker', ''),
                    linestyle=linestyle)
            cmp_face = list(cmap(1/(evidence_bins.size)))
            cmp_face[-1] = 0.5
            cmp_edge = list(cmap(1/(evidence_bins.size)))
            cmp_edge[-1] = 0.
            if n_subs > 1:  # more than one subject
                ax.fill_between(
                    np.arange(len(rtbins)-1) * rtbinsize + 0.5 * rtbinsize,
                    vals_total[:(len(rtbins)-1)] +
                    error_total[:(len(rtbins)-1)],
                    y2=vals_total[:(len(rtbins)-1)] -
                    error_total[:(len(rtbins)-1)],
                    facecolor=cmp_face, edgecolor=cmp_edge, alpha=0.6)
            else:
                ax.fill_between(
                    rtbins[:-1] + rtbins_diff/2,
                    vals_total + error_total,
                    y2=vals_total - error_total,
                    facecolor=cmp_face, edgecolor=cmp_edge)
        if not fill_error and plot:  # not fillerror --> errorbars
            ax.errorbar(
                tmp_df.rtbin.values * rtbinsize + 0.5 * rtbinsize,
                tmp['mean'].values,
                yerr=[tmp.groupby_binom_ci.apply(lambda x: x[0]),
                      tmp.groupby_binom_ci.apply(lambda x: x[1])],
                label=clabel, c=cmap(i/(evidence_bins.size-1)), **error_kws_)
        if not plot:
            return


def trajectory_thr(df, bincol, bins, thr=40, trajectory="trajectory_y",
                   stamps="trajectory_stamps", ax=None, fpsmin=29, fixation_us=300000,
                   collapse_sides=False, return_trash=False,
                   interpolatespace=np.linspace(-700000, 1000000, 1700),
                   zeropos_interp=700, fixation_delay_offset=0,
                   error_kwargs={"ls": "none", 'markersize': 12}, ax_traj=None,
                   traj_kws={}, ts_fix_onset="fix_onset_dt", align="action",
                   interp_extend=False, discarded_tstamp=0, cmap='viridis',
                   rollingmeanwindow=0, bintype="edges", xpoints=None,
                   raiseerrors=False, plotmt=True, plot_traj=True, color_tr='olive',
                   alpha_low=False):
    """
    Preprocesses (and - optional - plots) trajectories.
    """

    if bintype not in ["edges", "categorical", "dfmask"]:
        raise ValueError(
            'bintype can take values: "edges", "categorical" and "dfmask"')

    categorical_bins = False
    if bintype != "edges":
        categorical_bins = True
    if align not in ["action", "sound", "response"]:
        raise ValueError('align must be "action","sound" or "response"')
    if (fixation_us != 300000) or (fixation_delay_offset != 0):
        print("fixation and delay offset should be adressed and you should " +
              "avoid tweaking defaults")

    if isinstance(cmap, str):
        cmap = pl.cm.get_cmap(cmap)
        if traj_kws is not None:
            traj_kws.pop("c", None)
            traj_kws.pop("color", None)
        if error_kwargs is not None:
            error_kwargs.pop("c", None)
            error_kwargs.pop("color", None)

    matrix_dic = {}  # will contain average trajectories per bin
    idx_dic = {}

    if xpoints is None:
        if bintype == "edges":
            xpoints = (bins[:-1] + bins[1:]) / 2
        elif bintype == "categorical":
            xpoints = bins
        else:
            try:
                xpoints = [float(x) for x in bins.keys()]
            except Exception as e:
                if raiseerrors:
                    raise e
                xpoints = np.arange(len(bins.keys()))
    y_points = []
    y_err = []
    mt_time = []
    mt_time_err = []

    test = df.loc[df.framerate >= fpsmin]

    if bintype == "dfmask":
        bkeys = list(bins.keys())
        niters = len(bkeys)
    elif bintype == "categorical":
        niters = len(bins)
    elif bintype == "edges":
        niters = len(bins) - 1  # we iterate 1 less because of edges!

    for b in range(niters):  # for each bin
        if isinstance(thr, (list, tuple, np.ndarray)):
            cthr = thr[b]  # beware if list passes crashes
        else:
            cthr = thr

        if bintype == "dfmask":
            idx_dic[b] = test.loc[bins[bkeys[b]]].index.values
        elif (len(bins) > 1) & (not categorical_bins):
            idx_dic[b] = test.loc[(test[bincol] > bins[b]) &
                                  (test[bincol] < bins[b + 1])].index.values
        else:
            idx_dic[b] = test.loc[(test[bincol] == bins[b])].index.values
        matrix_dic[b] = np.zeros((idx_dic[b].size, interpolatespace.size))
        # index filtering
        selected_trials = test.loc[idx_dic[b]]
        # interpolate trajectories
        arrays = (selected_trials.apply(lambda x: interpolapply(
                        x, collapse_sides=collapse_sides, interpolatespace=interpolatespace,
                        align=align, interp_extend=interp_extend, trajectory=trajectory,
                        discarded_tstamp=discarded_tstamp), axis=1,).values)
        if arrays.size > 0:
            matrix_dic[b] = np.concatenate(arrays).reshape(-1, interpolatespace.size)

        tmp_mat = matrix_dic[b][:, zeropos_interp:]
        if (cthr > 0) or collapse_sides:
            r, c = np.where(tmp_mat > cthr)
        else:
            r, c = np.where(tmp_mat < cthr)

        _, idxes = np.unique(r, return_index=True)
        if alpha_low:
            y_point = np.median(c[idxes])
            y_err += [sem(c[idxes], nan_policy="omit")]
        else:
            y_point = np.nanmedian(np.nanmax(tmp_mat, axis=1))
            y_err += [np.nanstd(np.nanmax(tmp_mat, axis=1)) /
                      np.sqrt(len(np.nanmax(tmp_mat, axis=1)))]
        y_points += [y_point]

        extra_kw = {}

        # get motor time
        mt_time += [np.median(selected_trials.resp_len)*1e3]
        mt_time_err += [sem(selected_trials.resp_len*1e3, nan_policy="omit")]

        # plot section
        if ax_traj is not None:  # original stuff
            if cmap is not None:
                traj_kws.pop("color", None)
                traj_kws["color"] = cmap(b / (niters-1))
            if "label" not in traj_kws.keys():
                if bintype == "categorical":
                    extra_kw["label"] = f"{bincol}={round(bins[b],2)}"
                elif bintype == "dfmask":
                    extra_kw["label"] = f"{bincol}={bkeys[b]}"
                else:  # edges
                    extra_kw["label"] = f"{round(bins[b],2)}<{bincol}<{round(bins[b+1],2)}"
            if rollingmeanwindow:
                ytoplot = (pd.Series(np.nanmedian(matrix_dic[b], axis=0))
                           .rolling(rollingmeanwindow, min_periods=1).mean().values)
            if cmap is None and alpha_low:
                traj_kws["color"] = color_tr
                traj_kws["alpha"] = 0.3
            ytoplot = np.nanmedian(matrix_dic[b], axis=0)
            if plot_traj:
                ax_traj.plot((interpolatespace) / 1000,
                             ytoplot-np.nanmean(
                                 ytoplot[(interpolatespace > -100000) *
                                         (interpolatespace < 0)]), **traj_kws,
                             **extra_kw)
    y_points = np.array(y_points)
    y_err = np.array(y_err)
    # plot section again
    if (ax is not None) & return_trash:
        if cmap is not None:
            extra_kw = {}
            for i in range(len(xpoints)):
                if "label" not in error_kwargs.keys():
                    if bintype == "categorical":
                        extra_kw["label"] = f"{bincol}={round(bins[i],2)}"
                    elif bintype == "dfmask":
                        extra_kw["label"] = f"{bincol}={bkeys[b]}"
                    else:
                        extra_kw[
                            "label"
                        ] = f"{round(bins[i],2)}<{bincol}<{round(bins[i+1],2)}"
                if plotmt:
                    ax.errorbar(xpoints[i], mt_time[i], yerr=mt_time_err[i],
                                **error_kwargs, color=cmap(i / (niters-1)),
                                **extra_kw,)
                else:
                    ax.errorbar(xpoints[i], y_points[i] + fixation_delay_offset,
                                yerr=y_err[i], **error_kwargs, color=cmap(i / (niters-1)),
                                **extra_kw,)
        else:
            if plotmt:
                ax.errorbar(xpoints, mt_time, yerr=mt_time_err,
                                 **error_kwargs)
            if not plotmt and not alpha_low:
                ax.errorbar(xpoints, y_points + fixation_delay_offset,
                            yerr=y_err, **error_kwargs)
            if alpha_low:
                ax.errorbar(xpoints, y_points + fixation_delay_offset,
                            yerr=y_err, **error_kwargs)

        return xpoints, y_points + fixation_delay_offset, y_err, matrix_dic,\
            idx_dic, mt_time, mt_time_err
    elif not return_trash:
        return xpoints, y_points + fixation_delay_offset, y_err
    else:
        return xpoints, y_points + fixation_delay_offset, y_err, matrix_dic,\
            idx_dic, mt_time



def pcom_vs_rt_plot(df, ax, average=False, rtbins= np.arange(0,201,10), sv_folder=None, dist=False):
    """p(com) and RT distribution"""
    sv_folder = sv_folder
    if not average:
        binned_curve(df[df.special_trial == 0], 'CoM_sugg', 'sound_len',
                     rtbins, sem_err=False, legend=False, xpos=10,
                     xoffset=5, errorbar_kw={'color': 'coral',
                                             'label': 'p(CoM)', 'zorder': 3,
                                             'lw': 2},
                     traces='subjid', traces_kw=dict(alpha=0.3), ax=ax)
    else:  # mean of means + sem
        rtbinsize = rtbins[1]-rtbins[0]
        df['rtbin'] = pd.cut(df.sound_len, rtbins, labels=np.arange(rtbins.size-1),
                             retbins=False, include_lowest=True, right=True).astype(float)
        f, ax = plt.subplots()
        # traces
        tmp = (df.groupby(['subjid', 'rtbin'])['CoM_sugg'].mean().reset_index())
        for subject in tmp.subjid.unique():
            ax.plot(tmp.loc[tmp.subjid == subject, 'rtbin'] * rtbinsize + 0.5 * rtbinsize,
                    tmp.loc[tmp.subjid == subject, 'CoM_sugg'], ls=':',
                    color='gray', alpha=0.6)
        # average
        tmp = tmp.groupby(['rtbin'])['CoM_sugg'].agg(['mean', sem])
        ax.errorbar(tmp.index * rtbinsize + 0.5 * rtbinsize, tmp['mean'],
                    yerr=tmp['sem'], label='p(CoM)', color='coral', lw=2)
    ax.set_ylim(0, 0.075)
    if dist:  # plot distributions
        hist_list = []
        for subject in df.subjid.unique():
            counts, bns = np.histogram(df[(df.subjid == subject) & (
                df.special_trial == 0)].sound_len.dropna().values, bins=rtbins)
            hist_list += [counts]
        ax.set_ylim(0, 0.075)
        _, ymax = ax.get_ylim()
        counts = np.stack(hist_list).mean(axis=0)
        ax.hist(bns[:-1], bns, weights=0.5*ymax * counts /
                counts.max(), alpha=.4, label='RT distribution')
    # tune panel
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylabel('p(detected CoM)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 0.075)
    ax.spines['bottom'].set_bounds(0, 200)
    plt.gcf().patch.set_facecolor('white')
    plt.gca().set_facecolor('white')
