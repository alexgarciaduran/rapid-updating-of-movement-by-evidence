#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:28:46 2020
@author: molano
"""
import pandas as pd
from scipy.optimize import curve_fit
from numpy import logical_and as and_
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from numpy import concatenate as conc
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score

# GLOBAL VARIABLES
FIX_TIME = 0.5
NUM_BINS_RT = 6
NUM_BINS_MT = 7
MT_MIN = 0.05
MT_MAX = 1
START_ANALYSIS = 0  # trials para ignorar
RESP_W = 0.3
START_ANALYSIS = 0  # trials to ignore
GREEN = np.array((77, 175, 74))/255
PURPLE = np.array((152, 78, 163))/255
model_cols = ['evidence',
              'L+', 'L-', 'T+-', 'T-+', 'T--', 'T++', 'intercept']
afterc_cols = [x for x in model_cols if x not in ['L+2', 'L-1', 'L-2']]
aftere_cols = [x for x in model_cols if x not in ['L+1', 'L+2', 'L-2']]


def tune_panel(ax, xlabel, ylabel, font=10):
    ax.set_xlabel(xlabel, fontsize=font)
    ax.set_ylabel(ylabel, fontsize=font)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def extract_vars_from_dict(data, steps=None):
    """
    Extracts data
    """
    steps = get_steps(steps, num_tr=len(data['correct']))
    ev = data['soundPlay_object1_leftRightBalance'][steps[0]:steps[1]]
    choice = data['answer_response'][steps[0]:steps[1]]
    perf = data['correct'][steps[0]:steps[1]]
    valid = data['respondedInTime'][steps[0]:steps[1]] == 1
    reaction_time = data['soundPlay_responseTime'][steps[0]:steps[1]]-FIX_TIME
    answ_rt = data['answer_responseTime'][steps[0]:steps[1]]
    sound_dur = data['soundPlay_duration'][steps[0]:steps[1]]-FIX_TIME
    blocks = data['block'][steps[0]:steps[1]]
    return ev, choice, perf, valid, reaction_time, blocks, answ_rt, sound_dur


def get_steps(steps, num_tr):
    if steps is None:
        steps = [0, num_tr]
    else:
        if steps < 0:
            steps = [num_tr+steps, num_tr]
        else:
            steps = [0, steps]
    return steps


def data_processing(data_tr, data_traj, rgrss_folder, sv_folder,
                   com_threshold=50):
    """


    Parameters
    ----------
    data_tr : dataframe
        must contain all trial's data already extracted from 2AFC
        human task.
    data_traj : dataframe
        must contain all trial's trajectory data already extracted from 2AFC
        human task.
    com_threshold : int or float, optional
        Threshold in pixels to detect CoMs in trajectories. The default is 50.
    plot : bool, optional
        Wether to plot or not. The default is False.

    Returns
    -------
    com_list : list
        Boolean list that indicates the CoMs in its corresponding index.

    """
    ev, choice, perf, valid, reaction_time, blocks, answ_rt, sound_dur =\
        extract_vars_from_dict(data_tr, steps=None)
    subjid = data_tr['subjid']
    choice_12 = choice + 1
    choice_12[~valid] = 0
    data = {'signed_evidence': ev, 'choice': choice_12,
            'performance': perf}
    if rgrss_folder is None:
        df_regressors = get_GLM_regressors(data, tau=2)
        df_regressors.to_csv(sv_folder + 'df_regressors_all_sub.csv')
    else:
        df_regressors = pd.read_csv(rgrss_folder+'df_regressors_all_sub.csv')
    ind_af_er = df_regressors['aftererror'] == 0
    subjid = subjid[ind_af_er]
    ev = ev[ind_af_er]
    perf = perf[ind_af_er]
    valid = valid[ind_af_er]
    prior = np.nansum((df_regressors['T++'], df_regressors['T++']), axis=0)/2
    prior = prior[ind_af_er]
    blocks = blocks[ind_af_er]
    pos_x = data_traj['answer_positionsX']
    pos_y = data_traj['answer_positionsY']
    answer_times = [x for x in data_traj['answer_times']
                    if x not in [np.nan]]
    for inde in range(len(choice_12)):
        answer_times[inde] = answer_times[inde].split(';')
        for i in range(len(pos_x[inde])):
            if i == 0:
                answer_times[inde][i] = 0
            else:
                answer_times[inde][i] = float(answer_times[inde][i])
    answer_times = np.array(answer_times, dtype=object)[ind_af_er]
    pos_x = pos_x[ind_af_er]
    choice = choice[ind_af_er]
    pos_y = pos_y[ind_af_er]
    choice_signed = choice*2 - 1
    reaction_time = reaction_time[ind_af_er]
    com_list = []
    com_peak = []
    time_com = []
    for i, traj in enumerate(pos_x):
        traj = np.array(traj)
        max_val = max((traj) * (-choice_signed[i]))
        com_list.append(max_val > com_threshold)
        if max_val > 0:
            com_peak.append(max_val)
        else:
            com_peak.append(0)
        if answer_times[i][-1] != '' and max_val > com_threshold:
            time_com_ind = np.array(answer_times[i])[traj == max_val]
            try:
                if len(time_com_ind) >= 1:
                    time_com.append(time_com_ind[0])
                else:
                    time_com.append(-1)
            except Exception:
                time_com.append(time_com_ind)
        else:
            time_com.append(-1)
    indx = ~np.isnan(ev)
    com_list = np.array(com_list)
    avtrapz = ev[indx]
    CoM_sugg = com_list[indx]
    norm_allpriors = prior[indx]/max(abs(prior[indx]))
    R_response = choice[indx]
    blocks = blocks[indx]
    for i, e_val in enumerate(avtrapz):
        if abs(e_val) > 1:
            avtrapz[i] = np.sign(e_val)
    df_data = pd.DataFrame({'avtrapz': avtrapz, 'CoM_sugg': CoM_sugg,
                            'norm_allpriors': norm_allpriors,
                            'R_response': R_response,
                            'sound_len': reaction_time[indx]*1e3,
                            'hithistory': perf[indx],
                            'trajectory_y': pos_x[indx],
                            'times': answer_times[indx],
                            'traj_y': pos_y[indx],
                            'subjid': subjid[indx],
                            'com_peak': np.array(com_peak)[indx],
                            'time_com': np.array(time_com)[indx],
                            'blocks': blocks})
    return df_data


def traj_analysis(data_folder, sv_folder, subjects, steps=[None], name=''):
    for i_s, subj in enumerate(subjects):
        print('-----------')
        print(subj)
        folder = data_folder+subj+'/'
        for i_stp, stp in enumerate(steps):
            data_tr, data_traj = get_data_traj(folder=folder)
            df_data = data_processing(data_tr, data_traj, com_threshold=100,
                                      rgrss_folder=data_folder,
                                      sv_folder=sv_folder)
    return df_data


def get_data_traj(folder):
    """
    Extracts trajectories and psychometric data.
    Inputs: subject name (subj) and main_folder
    Outputs: psychometric data and trajectories data in two different dictionaries
    """
    # subject folder
    # folder = main_folder+'\\'+subj+'\\'  # Alex
    # find all data files
    files_trials = glob.glob(folder+'*trials.csv')
    files_traj = glob.glob(folder+'*trials-trajectories.csv')
    # take files names
    file_list_trials = [os.path.basename(x) for x in files_trials
                        if x.endswith('trials.csv')]
    file_list_traj = [os.path.basename(x) for x in files_traj
                      if x.endswith('trials-trajectories.csv')]
    # sort files
    sfx_tls = [x[x.find('202'):x.find('202')+15] for x in file_list_trials]
    sfx_trj = [x[x.find('202'):x.find('202')+15] for x in file_list_traj]

    sorted_list_tls = [x for _, x in sorted(zip(sfx_tls, file_list_trials))]
    sorted_list_trj = [x for _, x in sorted(zip(sfx_trj, file_list_traj))]
    # create data
    data_tls = {'correct': np.empty((0,)), 'answer_response': np.empty((0,)),
                'soundPlay_object1_leftRightBalance': np.empty((0,)),
                'respondedInTime': np.empty((0,)), 'block': np.empty((0,)),
                'soundPlay_responseTime': np.empty((0,)),
                'soundPlay_duration': np.empty((0,)),
                'answer_responseTime': np.empty((0,))}
    # go over all files
    subjid = np.empty((0,))
    for i_f, f in enumerate(sorted_list_tls):
        # read file
        df1 = pd.read_csv(folder+'/'+f, sep=',')  # Manuel
        # df1 = pd.read_csv(folder+'\\'+f, sep=',')  # Alex
        if np.mean(df1['correct']) < 0.4:
            continue
        else:
            for k in data_tls.keys():
                values = df1[k].values[START_ANALYSIS:]
                if k == 'soundPlay_object1_leftRightBalance':
                    values = values-.5
                    values[np.abs(values) < 0.01] = 0
                data_tls[k] = np.concatenate((data_tls[k], values))
            subjid = np.concatenate((subjid, np.repeat(i_f+1, len(values))))
    data_tls['subjid'] = subjid
    num_tr = len(data_tls['correct'])
    data_trj = {'answer_positionsX': np.empty((0,)),
                'answer_positionsY': np.empty((0,)),
                'answer_times': np.empty((0,))}
    for i_f, f in enumerate(sorted_list_trj):
        # read file
        df1 = pd.read_csv(folder+'/'+f, sep=',')  # Manuel
        df2 = pd.read_csv(folder+'/'+sorted_list_tls[i_f], sep=',')
        # df1 = pd.read_csv(folder+'\\'+f, sep=',')  # Alex
        if np.mean(df2['correct']) < 0.4:
            print('subject discarded: ' + str(i_f+1))
            print('acc: ' + str(np.mean(df2['correct'])))
            continue
        else:
            pos_x = df1['answer_positionsX'].dropna().values[:num_tr]
            pos_y = df1['answer_positionsY'].dropna().values[:num_tr]
            cont = 0
            for ind_trl in range(len(pos_x)):
                if cont == 1 and df1['trial'][ind_trl] == 1:
                    break
                if df1['trial'][ind_trl] == 1:
                    cont = 1
                pos_x[ind_trl] = [float(x) for x in pos_x[ind_trl].split(';')]
                pos_y[ind_trl] = [float(x) for x in pos_y[ind_trl].split(';')]
            k = 'answer_positionsX'
            data_trj[k] = np.concatenate((data_trj[k], pos_x))
            k = 'answer_positionsY'
            data_trj[k] = np.concatenate((data_trj[k], pos_y))
            k = df1.columns[-1]
            values = df1[k].dropna().values
            k = 'answer_times'
            data_trj[k] = np.concatenate((data_trj[k], values))
    return data_tls, data_trj


def get_repetitions(mat):
    """
    Return mask indicating the repetitions in mat.
    Makes diff of the input vector, mat, to obtain the repetition vector X,
    i.e. X will be 1 at t if the value of mat at t is equal to that at t-1
    Parameters
    ----------
    mat : array
        array of elements.
    Returns
    -------
    repeats : array
        mask indicating the repetitions in mat.
    """
    mat = mat.flatten()
    values = np.unique(mat)
    # We need to account for size reduction of np.diff()
    rand_ch = np.array(np.random.choice(values, size=(1,)))
    repeat_choice = conc((rand_ch, mat))
    diff = np.diff(repeat_choice)
    repeats = (diff == 0)*1.
    repeats[np.isnan(diff)] = np.nan
    return repeats


def nanconv(vec_1, vec_2):
    """
    This function returns a convolution result of two vectors without
    considering nans
    """
    mask = ~np.isnan(vec_1)
    return np.nansum(np.multiply(vec_2[mask], vec_1[mask]))


def get_GLM_regressors(data, tau, chck_corr=False):
    """
    Compute regressors.
    Parameters
    ----------
    data : dict
        dictionary containing behavioral data.
    chck_corr : bool, optional
        whether to check correlations (False)
    Returns
    -------
    df: dataframe
        dataframe containg evidence, lateral and transition regressors.
    """
    ev = data['signed_evidence'][START_ANALYSIS::]  # coherence/evidence with sign
    perf = data['performance'].astype(float)  # performance (0/1)
    ch = data['choice'][START_ANALYSIS::].astype(float)  # choice (1, 2)
    # discard (make nan) non-standard-2afc task periods
    if 'std_2afc' in data.keys():
        std_2afc = data['std_2afc'][START_ANALYSIS::]
    else:
        std_2afc = np.ones_like(ch)
    inv_choice = and_(ch != 1., ch != 2.)
    nan_indx = np.logical_or.reduce((std_2afc == 0, inv_choice))

    ev[nan_indx] = np.nan
    perf[nan_indx] = np.nan
    ch[nan_indx] = np.nan
    ch = ch-1  # choices should belong to {0, 1}
    prev_perf = ~ (conc((np.array([True]), data['performance'][:-1])) == 1)
    prev_perf = prev_perf.astype('int')
    prevprev_perf = (conc((np.array([False]), prev_perf[:-1])) == 1)
    ev /= np.nanmax(ev)
    rep_ch_ = get_repetitions(ch)
    # variables:
    # 'origidx': trial index within session
    # 'rewside': ground truth
    # 'hithistory': performance
    # 'R_response': choice (right == 1, left == 0, invalid == nan)
    # 'subjid': subject
    # 'sessid': session
    # 'res_sound': stimulus (left - right) [frame_i, .., frame_i+n]
    # 'sound_len': stim duration
    # 'frames_listened'
    # 'aftererror': not(performance) shifted
    # 'rep_response'
    df = {'origidx': np.arange(ch.shape[0]),
          'R_response': ch,
          'hit': perf,
          'evidence': ev,
          'aftererror': prev_perf,
          'rep_response': rep_ch_,
          'prevprev_perf': prevprev_perf}
    df = pd.DataFrame(df)

    # Lateral module
    df['L+1'] = np.nan  # np.nan considering invalids as errors
    df.loc[(df.R_response == 1) & (df.hit == 1), 'L+1'] = 1
    df.loc[(df.R_response == 0) & (df.hit == 1), 'L+1'] = -1
    df.loc[df.hit == 0, 'L+1'] = 0
    df['L+1'] = df['L+1'].shift(1)
    df.loc[df.origidx == 1, 'L+1'] = np.nan
    # L-
    df['L-1'] = np.nan
    df.loc[(df.R_response == 1) & (df.hit == 0), 'L-1'] = 1
    df.loc[(df.R_response == 0) & (df.hit == 0), 'L-1'] = -1
    df.loc[df.hit == 1, 'L-1'] = 0
    df['L-1'] = df['L-1'].shift(1)
    df.loc[df.origidx == 1, 'L-1'] = np.nan

    # pre transition module
    df.loc[df.origidx == 1, 'rep_response'] = np.nan
    df['rep_response_11'] = df.rep_response
    df.loc[df.rep_response == 0, 'rep_response_11'] = -1
    df.rep_response_11.fillna(value=0, inplace=True)
    df.loc[df.origidx == 1, 'aftererror'] = np.nan

    # transition module
    df['T++1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 0) & (df.hit == 1), 'T++1'] =\
        df.loc[(df.aftererror == 0) & (df.hit == 1), 'rep_response_11']
    df.loc[(df.aftererror == 1) | (df.hit == 0), 'T++1'] = 0
    df['T++1'] = df['T++1'].shift(1)

    df['T+-1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 0) & (df.hit == 0), 'T+-1'] =\
        df.loc[(df.aftererror == 0) & (df.hit == 0), 'rep_response_11']
    df.loc[(df.aftererror == 1) | (df.hit == 1), 'T+-1'] = 0
    df['T+-1'] = df['T+-1'].shift(1)

    df['T-+1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hit == 1), 'T-+1'] =\
        df.loc[(df.aftererror == 1) & (df.hit == 1), 'rep_response_11']
    df.loc[(df.aftererror == 0) | (df.hit == 0), 'T-+1'] = 0
    df['T-+1'] = df['T-+1'].shift(1)

    df['T--1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hit == 0), 'T--1'] =\
        df.loc[(df.aftererror == 1) & (df.hit == 0), 'rep_response_11']
    df.loc[(df.aftererror == 0) | (df.hit == 1), 'T--1'] = 0
    df['T--1'] = df['T--1'].shift(1)

    # exponential fit for T++
    decay_tr = np.exp(-np.arange(10)/tau)  # exp(-x/tau)
    regs = [x for x in model_cols if x != 'intercept' and x != 'evidence']
    N = len(decay_tr)
    for reg in regs:  # all regressors (T and L)
        df[reg] = df[reg+str(1)]
        for j in range(N, len(df[reg+str(1)])):
            df[reg][j-1] = nanconv(df[reg+str(1)][j-N:j], decay_tr[::-1])
            # its j-1 for shifting purposes

    # transforming transitions to left/right space
    for col in [x for x in df.columns if x.startswith('T')]:
        df[col] = df[col] * (df.R_response.shift(1)*2-1)
        # {-1 = Left; 1 = Right, nan=invalid}

    df['intercept'] = 1

    df.loc[:, model_cols].fillna(value=0, inplace=True)
    # check correlation between regressors

    return df  # resulting df with lateralized T


def glm(df):
    """
    Compute GLM weights for data in df conditioned on previous outcome.

    Parameters
    ----------
    df : dataframe
        dataframe containing regressors and response.

    Returns
    -------
    Lreg_ac : LogisticRegression model
        logistic model fit to after correct trials.
    Lreg_ae : LogisticRegression model
        logistic model fit to after error trials.

    """
    not_nan_indx = df['R_response'].notna()
    X_df_ac, y_df_ac = df.loc[(df.aftererror == 0) & not_nan_indx,
                              afterc_cols].fillna(value=0),\
        df.loc[(df.aftererror == 0) & not_nan_indx, 'R_response']
    X_df_ae, y_df_ae =\
        df.loc[(df.aftererror == 1) & not_nan_indx,
               aftere_cols].fillna(value=0),\
        df.loc[(df.aftererror == 1) & not_nan_indx, 'R_response']
    if len(np.unique(y_df_ac.values)) == 2 and len(np.unique(y_df_ae.values)) == 2:
        Lreg_ac = LogisticRegression(C=1, fit_intercept=False, penalty='l2',
                                     solver='saga', random_state=123,
                                     max_iter=10000000, n_jobs=-1)
        Lreg_ac.fit(X_df_ac.values, y_df_ac.values)
        vals_ac = np.concatenate([Lreg_ac.intercept_,
                                  Lreg_ac.coef_.flatten()])
        smodel_ac = sm.Logit(y_df_ac,
                             X_df_ac).fit(start_params=vals_ac[1::], max_iter=0)
        summary_ac = smodel_ac.summary().tables[1].as_html()
        summary_ac = pd.read_html(summary_ac, header=0, index_col=0)[0]
        p_z_ac = summary_ac['P>|z|']
        score_ac = cross_val_score(Lreg_ac, X=X_df_ac, y=y_df_ac, cv=5)
        Lreg_ae = LogisticRegression(C=1, fit_intercept=False, penalty='l2',
                                     solver='saga', random_state=123,
                                     max_iter=10000000, n_jobs=-1)
        Lreg_ae.fit(X_df_ae.values, y_df_ae.values)
        vals_ae = np.concatenate([Lreg_ae.intercept_,
                                  Lreg_ae.coef_.flatten()])
        score_ae = cross_val_score(Lreg_ae, X=X_df_ae, y=y_df_ae, cv=5)
        smodel_ae = sm.Logit(y_df_ae,
                             X_df_ae).fit(start_params=vals_ae[1::], max_iter=0)
        summary_ae = smodel_ae.summary().tables[1].as_html()
        summary_ae = pd.read_html(summary_ae, header=0, index_col=0)[0]
        p_z_ae = summary_ae['P>|z|']
    else:
        Lreg_ac = None
        Lreg_ae = None

    return Lreg_ac, Lreg_ae, score_ac, score_ae, p_z_ac, p_z_ae


def psycho_curves_rep_alt(df_data, ax):
    # MEAN PSYCHO-CURVES FOR REP/ALT, AFTER CORRECT/ERROR
    rojo = np.array((228, 26, 28))/255
    azul = np.array((55, 126, 184))/255
    colors = [rojo, azul]
    ttls = ['']
    bias_final = []
    slope_final = []
    subjects = df_data.subjid.unique()
    fig2, ax2 = plt.subplots(ncols=3)
    lbs = ['after error', 'after correct']
    median_rep_alt = np.empty((len(subjects), 2, 7))
    cohs_rep_alt = np.empty((len(subjects), 2, 7))
    for i_s, subj in enumerate(subjects):
        df_sub = df_data.loc[df_data.subjid == subj]
        ev = df_sub.avtrapz
        choice_12 = df_sub.R_response.values + 1
        blocks = df_sub.blocks
        perf = df_sub.hithistory
        prev_perf = np.concatenate((np.array([0]), perf[:-1]))
        all_means = []
        all_xs = []
        biases = []
        for i_b, blk in enumerate([1, 2]):  # blk = 1 --> alt / blk = 2 --> rep
            p = 1
            alpha = 1 if p == 0 else 1
            lnstyl = '-' if p == 0 else '-'
            plt_opts = {'color': colors[i_b],
                        'alpha': alpha, 'linestyle': lnstyl}
            # rep/alt
            popt, pcov, ev_mask, repeat_mask =\
                bias_psychometric(choice=choice_12.copy(), ev=-ev.copy(),
                                     mask=and_(prev_perf == p,
                                               blocks == blk),
                                     maxfev=100000)
            # this is to avoid rounding differences
            ev_mask = np.round(ev_mask, 2)
            d =\
                plot_psycho_curve(ev=ev_mask, choice=repeat_mask,
                                     popt=popt, ax=ax2[p],
                                     color_scatter=colors[i_b],
                                     label=lbs[p], plot_errbars=True,
                                     **plt_opts)
            means = d['means']
            xs = d['xs']
            all_means.append(means)
            all_xs.append(xs)
            biases.append(popt[1])
        median_rep_alt[i_s] = np.array([np.array(a) for a in all_means])
        cohs_rep_alt[i_s] = np.array([np.array(a) for a in all_xs])
    plt.close(fig2)
    labels = ['Alternating', 'Repeating']
    for i_b, blk in enumerate([1, 2]):
        ip = 0
        p = 1
        if i_b == 0:
            ax.axvline(x=0., linestyle='--', lw=0.2,
                             color=(.5, .5, .5))
            ax.axhline(y=0.5, linestyle='--', lw=0.2,
                             color=(.5, .5, .5))
            ax.set_title(ttls[ip])
            ax.set_yticks([0, 0.5, 1])
            tune_panel(ax=ax, xlabel='Repeating stimulus evidence',
                       ylabel='p(repeat response)')
        ax.plot(cohs_rep_alt[:, i_b, :].flatten(),
                median_rep_alt[:, i_b, :].flatten(),
                color=colors[i_b], alpha=0.2, linestyle='',
                marker='+')
        medians = np.median(median_rep_alt, axis=0)[i_b]
        sems = np.std(median_rep_alt, axis=0)[i_b] /\
            np.sqrt(median_rep_alt.shape[0])
        ax.errorbar(cohs_rep_alt[0][0], medians, sems,
                    color=colors[i_b], marker='.', linestyle='')
        ev_gen = cohs_rep_alt[0, i_b, :].flatten()
        popt, pcov = curve_fit(probit_lapse_rates,
                               ev_gen,
                               np.median(median_rep_alt, axis=0)[i_b],
                               maxfev=10000)
        bias_final.append(popt[1])
        slope_final.append(popt[0])
        x_fit = np.linspace(-np.max(ev), np.max(ev), 20)
        y_fit = probit_lapse_rates(x_fit, popt[0], popt[1], popt[2],
                                   popt[3])
        ax.plot(x_fit, y_fit, color=colors[i_b], label=labels[i_b])
    ax.legend(title='Context', bbox_to_anchor=(0.5, 1.12), frameon=False,
              handlelength=1.2)


def probit(x, beta, alpha):
    from scipy.special import erf
    """
    Return probit function with parameters alpha and beta.

    Parameters
    ----------
    x : float
        independent variable.
    beta : float
        sensitiviy.
    alpha : TYPE
        bias term.

    Returns
    -------
    probit : float
        probit value for the given x, beta and alpha.

    """
    probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
    return probit


def probit_lapse_rates(x, beta, alpha, piL, piR):
    """
    Return probit with lapse rates.

    Parameters
    ----------
    x : float
        independent variable.
    beta : float
        sensitiviy.
    alpha : TYPE
        bias term.
    piL : float
        lapse rate for left side.
    piR : TYPE
        lapse rate for right side.

    Returns
    -------
    probit : float
        probit value for the given x, beta and alpha and lapse rates.

    """
    piL = 0
    piR = 0
    probit_lr = piR + (1 - piL - piR) * probit(x, beta, alpha)
    return probit_lr


def plot_psycho_curve(ev, choice, popt, ax, color_scatter, plot_errbars=False,
                      **plt_opts):
    """
    Plot psycho-curves (fits and props) using directly the fit parameters.

    THIS FUNCTION ASSUMES PUTATIVE EVIDENCE (it will compute response proportions
                                             for all values of ev)

    Parameters
    ----------
    ev : array
        array with **putative** evidence for each trial.
    choice : array
        array with choices made by agent.
    popt : list
        list containing fitted parameters (beta, alpha, piL, piR).
    ax : axis
        where to plot.
    **plt_opts : dict
        plotting options.

    Returns
    -------
    means : list
        response means for each evidence value.
    sems : list
        sem for the responses.
    x : array
        evidences values for which the means/sems are computed.
    y_fit : array
        y values for the fit.
    x_fit : array
        x values for the fit.

    """
    x_fit = np.linspace(np.min(ev), np.max(ev), 20)
    y_fit = probit_lapse_rates(x_fit, popt[0], popt[1], popt[2], popt[3])
    ax.plot(x_fit, y_fit, markersize=6, **plt_opts)
    means = []
    sems = []
    n_samples = []
    for e in np.unique(ev):
        means.append(np.mean(choice[ev == e]))
        sems.append(np.std(choice[ev == e])/np.sqrt(np.sum(ev == e)))
        n_samples.append(np.sum(ev == e))
    x = np.unique(ev)
    plt_opts['linestyle'] = ''
    if 'label' in plt_opts.keys():
        del plt_opts['label']
    if plot_errbars:
        ax.errorbar(x, means, sems, **plt_opts)
    ax.scatter(x, means, marker='.', alpha=1, s=60, c=color_scatter)
    ax.plot([0, 0], [0, 1], '--', lw=0.2, color=(.5, .5, .5))
    d_list = [means, sems, x, y_fit, x_fit, n_samples]
    d_str = ['means, sems, xs, y_fit, x_fit, n_samples']
    d = list_to_dict(d_list, d_str)
    return d


def standard_glm(ev, choice_12, perf, tau):
    """
    Performs GLM and returns weights after correct and after error ws_ac,
    ws_ae respectively. It returns the df with the regressors preprocessed
    and the scores (accuracy) of the GLM altogether with the p-values of the weights
    p_z_ac, p_z_ae.
    """
    data = {'signed_evidence': ev, 'choice': choice_12,
            'performance': perf}
    df = get_GLM_regressors(data, tau, chck_corr=False)
    fit_ac, fit_ae, score_ac, score_ae, p_z_ac, p_z_ae = glm(df)
    ws_ac = fit_ac.coef_[None, :, :]
    ws_ae = fit_ae.coef_[None, :, :]
    return ws_ac, ws_ae, df, score_ac, score_ae, p_z_ac, p_z_ae


def list_to_dict(lst, string):
    """
    Transform a list of variables into a dictionary.

    Parameters
    ----------
    lst : list
        list with all variables.
    string : str
        string containing the names, separated by commas.

    Returns
    -------
    d : dict
        dictionary with items in which the keys and the values are specified
        in string and lst values respectively.

    """
    string = string[0]
    string = string.replace(']', '')
    string = string.replace('[', '')
    string = string.replace('\\', '')
    string = string.replace(' ', '')
    string = string.replace('\t', '')
    string = string.replace('\n', '')
    string = string.split(',')
    d = {s: v for s, v in zip(string, lst)}
    return d

def bias_psychometric(choice, ev, mask=None, maxfev=10000):
    """
    Compute repeating bias by fitting probit function.

    Parameters
    ----------
    choice : array
        array of choices made bythe network.
    ev : array
        array with (signed) stimulus evidence.
    mask : array, optional
        array of booleans indicating the trials on which the bias
    # should be computed (None)

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of probit(xdata) - ydata is minimized
    pcov : 2d array
        The estimated covariance of popt. The diagonals provide the variance
        of the parameter estimate. To compute one standard deviation errors
        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.

        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.

        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix.

    """
    choice = choice.astype(float)
    choice[and_(choice != 1, choice != 2)] = np.nan
    repeat = get_repetitions(choice).astype(float)
    repeat[np.isnan(choice)] = np.nan
    # choice_repeating is just the original right_choice mat
    # but shifted one element to the left.
    choice_repeating = conc(
        (np.array(np.random.choice([1, 2])).reshape(1, ),
         choice[:-1]))
    # the rep. evidence is the original evidence with a negative sign
    # if the repeating side is the left one
    rep_ev = ev*(-1)**(choice_repeating == 2)
    if mask is None:
        mask = ~np.isnan(repeat)
    else:
        mask = and_(~np.isnan(repeat), mask)
    rep_ev_mask = rep_ev[mask]  # xdata
    repeat_mask = repeat[mask]  # ydata
    try:
        # Use non-linear least squares to fit probit to xdata, ydata
        popt, pcov = curve_fit(probit_lapse_rates, rep_ev_mask,
                               repeat_mask, maxfev=maxfev)
    except RuntimeError as err:
        print(err)
        popt = [np.nan, np.nan, np.nan, np.nan]
        pcov = 0
    return popt, pcov, rep_ev_mask, repeat_mask