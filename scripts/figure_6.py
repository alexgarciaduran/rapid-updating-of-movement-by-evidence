# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 01:01:54 2023

@author: Alex Garcia-Duran
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pl
from matplotlib.lines import Line2D
import matplotlib as mtp
import figures_paper as fp
import figure_3 as fig_3
import figure_2 as fig_2
import figure_1 as fig_1


# ---GLOBAL VARIABLES
VAR_INC = fig_1.VAR_INC
VAR_CON = fig_1.VAR_CON
VAR_INC_SHORT = fig_1.VAR_INC_SHORT
VAR_CON_SHORT = fig_1.VAR_CON_SHORT
VAR_L = fig_1.VAR_L
VAR_R = fig_1.VAR_R
FRAME_RATE = 14
BINS_RT = np.linspace(1, 301, 11)
xpos_RT = int(np.diff(BINS_RT)[0])

# ---FUNCTIONS
def create_figure_5_model(fgsz):
    mtp.rcParams['font.size'] = 13
    plt.rcParams['legend.title_fontsize'] = 10.5
    plt.rcParams['xtick.labelsize'] = 10.5
    plt.rcParams['ytick.labelsize'] = 10.5
    fig, ax = plt.subplots(ncols=4, nrows=4,
                           gridspec_kw={'top': 0.95, 'bottom': 0.055, 'left': 0.07,
                                        'right': 0.95, 'hspace': 0.6, 'wspace': 0.6},
                           figsize=fgsz)
    ax = ax.flatten()
    labs = ['a', 'b', 'c', 'd',
            'e', '', '', 'g',
            'f', '', '', 'h',
            'i', '', 'j', 'k']
    # set correct size of some panels
    pos_ax_0 = ax[0].get_position()
    pos_ax_0 = ax[4].get_position()
    ax[4].set_position([pos_ax_0.x0, pos_ax_0.y0, pos_ax_0.width*0.9,
                        pos_ax_0.height*0.9])
    pos_ax_0 = ax[8].get_position()
    ax[8].set_position([pos_ax_0.x0, pos_ax_0.y0, pos_ax_0.width*0.9,
                        pos_ax_0.height*0.9])
    pos_ax_0 = ax[2].get_position()
    ax[2].set_position([pos_ax_0.x0, pos_ax_0.y0, pos_ax_0.width*0.9,
                        pos_ax_0.height*0.9])
    for i in [5, 9]:
        pos_ax_0 = ax[i].get_position()
        ax[i].set_position([pos_ax_0.x0-pos_ax_0.width/4, pos_ax_0.y0, pos_ax_0.width*1.15,
                            pos_ax_0.height])
    for i in [6, 10]:
        pos_ax_0 = ax[i].get_position()
        ax[i].set_position([pos_ax_0.x0-pos_ax_0.width/6, pos_ax_0.y0, pos_ax_0.width*1.15,
                            pos_ax_0.height])    
    # letters for panels
    for n, ax_1 in enumerate(ax):
        fp.rm_top_right_lines(ax_1)
        if n == 2:
            ax_1.text(-0.1, 1.35, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')
        elif n == 4 or n == 8:
            ax_1.text(-0.1, 1.3, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')
        else:
            ax_1.text(-0.1, 1.2, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')
    ax[0].set_ylabel('Stimulus evidence')
    return fig, ax, ax[13], pos_ax_0


def plot_com_vs_rt_f5(df_plot_pcom, ax, ax2, eps=0):
    """
    Plots proportion of changes of mind and trajectory reversals vs RT, whose
    bins are BINS_RT. It also plots data p(reversal) vs RT.
    """
    subjid = df_plot_pcom.subjid
    subjects = np.unique(subjid)
    com_data = np.empty((len(subjects), len(BINS_RT)-1))
    com_data[:] = np.nan
    com_model_all = np.empty((len(subjects), len(BINS_RT)-1))
    com_model_all[:] = np.nan
    com_model_det = np.empty((len(subjects), len(BINS_RT)-1))
    com_model_det[:] = np.nan
    for i_s, subject in enumerate(subjects):
        df_plot = df_plot_pcom.loc[subjid == subject]
        xpos_plot, median_pcom_dat, _ =\
            fp.binned_curve(df_plot, 'com', 'sound_len', bins=BINS_RT, xpos=xpos_RT,
                            errorbar_kw={'label': 'Data', 'color': 'k'}, ax=ax,
                            legend=False, return_data=True)
        xpos_plot, median_pcom_mod_det, _ =\
            fp.binned_curve(df_plot, 'com_model_detected', 'rt_model', bins=BINS_RT,
                            xpos=xpos_RT, errorbar_kw={'label': 'Model detected',
                                                       'color': 'red'}, ax=ax,
                            legend=False, return_data=True)
        xpos_plot, median_pcom_mod_all, _ =\
            fp.binned_curve(df_plot, 'com_model', 'rt_model', bins=BINS_RT,
                            xpos=xpos_RT,
                            errorbar_kw={'label': 'Model all',
                                         'color': 'saddlebrown',
                                         'linestyle': '--'},
                            ax=ax2, legend=False, return_data=True)
        com_data[i_s, :len(median_pcom_dat)] = median_pcom_dat
        com_model_all[i_s, :len(median_pcom_mod_all)] = median_pcom_mod_all
        com_model_det[i_s, :len(median_pcom_mod_det)] = median_pcom_mod_det
    xpos_plot = (BINS_RT[:-1] + BINS_RT[1:]) / 2
    ax.errorbar(xpos_plot-eps, np.nanmedian(com_data, axis=0),
                yerr=np.nanstd(com_data, axis=0)/len(subjects), color='k',
                linestyle='--')
    ax.errorbar(xpos_plot, np.nanmedian(com_model_det, axis=0),
                yerr=np.nanstd(com_model_det, axis=0)/len(subjects), color='k')
    ax2.errorbar(xpos_plot+eps, np.nanmedian(com_model_all, axis=0),
                 yerr=np.nanstd(com_model_all, axis=0)/len(subjects), color='r')
    ax.xaxis.tick_top()
    ax.xaxis.tick_bottom()
    legendelements = [Line2D([0], [0], color='k', lw=2, linestyle='--',
                             label='Rats reversals'),
                      Line2D([0], [0], color='k', lw=2,
                             label='Model reversals'),
                      Line2D([0], [0], color='r', lw=2,
                             label='Model CoMs')]
    ax.legend(handles=legendelements, loc='upper left',
              bbox_to_anchor=(0.12, 1.3), frameon=False)
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylabel('p(reversal)')
    ax2.set_ylabel('p(CoM)')


def supp_com_analysis(df_sim, sv_folder, pcom_rt):
    """
    Plots supplementary figure 10.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 9))
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.45)
    ax = ax.flatten()
    labs = ['a', 'b', '', '', 'c', '']
    # add panel letters
    for i_a, a in enumerate(ax):
        fp.rm_top_right_lines(a)
        if i_a == 5:
            a.text(0.7, 1.1, labs[i_a], transform=a.transAxes, fontsize=16,
                   fontweight='bold', va='top', ha='right')
        else:
            a.text(-0.1, 1.2, labs[i_a], transform=a.transAxes, fontsize=16,
                   fontweight='bold', va='top', ha='right')
    # to plot p(CoM) in stim/prior matrix
    supp_prob_vs_prior(df_sim, ax=ax[0], fig=fig, column='CoM_sugg', com=False,
                       title_col=r'$p(CoM)$')
    # to plot p(detect CoM) in stim/prior matrix
    supp_prob_vs_prior(df_sim, ax=ax[1], fig=fig, column='com_detected', com=True,
                       title_col=r'$\;\;\;\;\;\;\;\;\;\;\;\;\;\;p(detection)$')
    # load image for CoM detection schematics (panel c)
    pcom_img = plt.imread(pcom_rt)
    # adjust axes and text
    ax[3].axis('off')
    ax[2].text(0, 1., 'c', transform=ax[2].transAxes, fontsize=16,
               fontweight='bold', va='top', ha='right')
    pos_ax_2 = ax[2].get_position()
    ax[2].set_position([pos_ax_2.x0-0.04, pos_ax_2.y0-0.1,
                        pos_ax_2.width*3, pos_ax_2.height*1.4])
    # show schematics
    ax[2].imshow(pcom_img, aspect='equal')
    ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    # save figures
    fig.savefig(sv_folder+'/supp_model_com.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'/supp_model_com.png', dpi=400, bbox_inches='tight')


def supp_prob_vs_prior(df_sim, ax, title_col, fig, column='com_detected', com=False,
                       margin=.05):
    """
    Plots proportion of CoMs in stim/prior matrix. If com=True, will average 
    across CoM trials (i.e. to get detected proportion).
    """
    ax_mat = ax
    if com:
        # focus only on CoMs
        df_prob = df_sim.loc[df_sim.CoM_sugg]
    else:
        # all trials
        df_prob = df_sim.copy()
    df_prob['CoM_sugg'] = df_prob[column]
    subjects = df_prob.subjid.unique()
    # initialize matrices M0 and M1
    mat_side_0_all = np.zeros((7, 7, len(subjects)))
    mat_side_1_all = np.zeros((7, 7, len(subjects)))
    for i_s, subj in enumerate(df_prob.subjid.unique()):
        # compute proportions
        matrix_side_0 =\
            fig_3.com_heatmap_marginal_pcom_side_mat(
                df=df_prob.loc[df_prob.subjid == subj], side=0)
        matrix_side_1 =\
            fig_3.com_heatmap_marginal_pcom_side_mat(
                df=df_prob.loc[df_prob.subjid == subj], side=1)
        mat_side_0_all[:, :, i_s] = matrix_side_0
        mat_side_1_all[:, :, i_s] = matrix_side_1
    # average matrices M0 and M1
    matrix_side_0 = np.nanmean(mat_side_0_all, axis=2)
    matrix_side_1 = np.nanmean(mat_side_1_all, axis=2)
    # plot average matrix as (M1 + flip(M0)) / 2
    im = ax_mat.imshow(0.5*(matrix_side_1+np.flip(matrix_side_0)),
                       vmin=0, cmap='magma')
    # tune panel
    for ax_i in [ax_mat]:
        ax_i.set_xlabel('Prior evidence')
        ax_i.set_xticks([0, 3, 6], [VAR_L, '0', VAR_R])
    ax_mat.set_yticks([0, 3, 6], [VAR_R, '0', VAR_L])
    ax_mat.set_ylabel('Stimulus evidence')
    pos = ax_mat.get_position()
    cbar_ax = fig.add_axes([pos.x0+pos.width*1.09, pos.y0+margin/6,
                            pos.width/15, pos.height/1.5])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title(title_col)


def plot_pright_model(df_sim, sound_len_model, decision_model, subjid, coh,
                      zt_model, ax):
    """
    Plots proportion of right choices in a stim/prior matrix.
    """
    subjects = np.unique(subjid)
    coh_model = coh[sound_len_model >= 0]
    decision_01_model = (decision_model+1)/2
    # initialize matrix
    mat_pright = np.zeros((7, 7, len(subjects)))
    for i_s, subject in enumerate(subjects):
        mat_per_subj, _ = fp.com_heatmap(zt_model[subjid == subject],
                                         coh_model[subjid == subject],
                                         decision_01_model[subjid == subject],
                                         return_mat=True, annotate=False)
        mat_pright[:, :, i_s] = mat_per_subj
    # average
    mat_pright_avg = np.nanmean(mat_pright, axis=2)
    ax_pright = ax
    # plot matrix
    im = ax_pright.imshow(np.flipud(mat_pright_avg), vmin=0., vmax=1, cmap='PRGn_r')
    # tune panel
    plt.sca(ax_pright)
    cbar = plt.colorbar(im, fraction=0.04)
    cbar.ax.set_title('p(right)', pad=17, fontsize=10)
    ax_pright.set_yticks([0, 3, 6])
    ax_pright.set_yticklabels([VAR_R, '0', VAR_L])
    ax_pright.set_xticks([0, 3, 6])
    ax_pright.set_xticklabels([VAR_L, '0', VAR_R])
    ax_pright.set_xlabel('Prior evidence')
    ax_pright.set_ylabel('Stimulus evidence')


def plot_pcom_matrices_model(df_model, n_subjs, ax_mat, f, nbins=7, pos_ax_0=[],
                             margin=.03, title='p(reversal)',
                             mat_titles=['Left to right reversal',
                                         'Right to left reversal'],
                             return_matrix=False):
    """
    Plots p(reversal | choice) in a stim/prior matrix.
    """
    # initialize a matrix for each side (choice = R or choice = L)
    mat_side_0_all = np.zeros((7, 7, n_subjs))
    mat_side_1_all = np.zeros((7, 7, n_subjs))
    for i_s, subj in enumerate(df_model.subjid.unique()):
        # compute reversal proportion for each choice
        matrix_side_0 =\
            fig_3.com_heatmap_marginal_pcom_side_mat(
                df=df_model.loc[df_model.subjid == subj], side=0)
        matrix_side_1 =\
            fig_3.com_heatmap_marginal_pcom_side_mat(
                df=df_model.loc[df_model.subjid == subj], side=1)
        mat_side_0_all[:, :, i_s] = matrix_side_0
        mat_side_1_all[:, :, i_s] = matrix_side_1
    # average and substitute nan by 0
    matrix_side_0 = np.nanmean(mat_side_0_all, axis=2)
    matrix_side_0[np.isnan(matrix_side_0)] = 0
    matrix_side_1 = np.nanmean(mat_side_1_all, axis=2)
    matrix_side_1[np.isnan(matrix_side_1)] = 0
    if return_matrix:
        # in case that the matrices are to be returned, function ends here
        return matrix_side_0, matrix_side_1
    # plotting
    # get second matrix closer
    pos_ax_0 = ax_mat[1].get_position()
    ax_mat[1].set_position([pos_ax_0.x0-pos_ax_0.width/3, pos_ax_0.y0,
                            pos_ax_0.width, pos_ax_0.height])
    # tune panels
    vmax = max(np.nanmax(matrix_side_0), np.nanmax(matrix_side_1))
    if vmax == 0:
        vmax = 1.1e-2
    pcomlabel_1 = mat_titles[1]
    pcomlabel_0 = mat_titles[0]
    ax_mat[0].set_title(pcomlabel_0, fontsize=11.5)
    ax_mat[1].set_title(pcomlabel_1, fontsize=11.5)
    # plot M1
    ax_mat[0].imshow(matrix_side_1, vmin=0, vmax=vmax, cmap='magma')
    # plot M0
    im = ax_mat[1].imshow(matrix_side_0, vmin=0, vmax=vmax, cmap='magma')
    # tune panels
    ax_mat[1].yaxis.set_ticks_position('none')
    for ax_i in [ax_mat[0], ax_mat[1]]:
        ax_i.set_xlabel('Prior evidence')
        ax_i.set_xticks([0, 3, 6], [VAR_L, '0', VAR_R])
    ax_mat[0].set_yticks([0, 3, 6], [VAR_R, '0', VAR_L])
    ax_mat[1].set_yticks([0, 3, 6], ['']*3)
    ax_mat[0].set_ylabel('Stimulus evidence')
    pos = ax_mat[1].get_position()
    cbar_ax = f.add_axes([pos.x0+pos.width+margin/2, pos.y0+margin/6,
                      pos.width/15, pos.height/1.5])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title('         '+title, fontsize=8.5)


def plot_trajs_cond_on_prior_and_stim(df_sim, ax, inset_sz, fgsz, marginx, marginy,
                                      new_data, save_new_data, data_folder, extra_label=''):
    """
    Plots trajectories and velocities as in figure 2 fashion,
    depending on stimulus and prior evidence towards the final response.
    """
    # arrange axes
    ax_cohs = np.array([ax[9], ax[10], ax[8]])
    ax_zt = np.array([ax[5], ax[6], ax[4]])

    ax_inset = fp.add_inset(ax=ax_cohs[1], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy, right=True)
    ax_cohs = np.insert(ax_cohs, 3, ax_inset)

    ax_inset = fp.add_inset(ax=ax_zt[1], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy, right=True)
    ax_zt = np.insert(ax_zt, 3, ax_inset)
    # first for prior
    if sum(df_sim.special_trial == 2) > 0:
        # if there are silent trials
        traj_cond_coh_simul(df_sim=df_sim[df_sim.special_trial == 2], ax=ax_zt,
                            new_data=new_data, data_folder=data_folder,
                            save_new_data=save_new_data,
                            median=True, prior=True, rt_lim=300, extra_label=extra_label)
    else:
        print('No silent trials')
        traj_cond_coh_simul(df_sim=df_sim, ax=ax_zt, new_data=new_data,
                            save_new_data=save_new_data,
                            data_folder=data_folder, median=True, prior=True, extra_label=extra_label)
    # finally for stimulus
    traj_cond_coh_simul(df_sim=df_sim, ax=ax_cohs, median=True, prior=False,
                        save_new_data=save_new_data,
                        new_data=new_data, data_folder=data_folder,
                        prior_lim=np.quantile(df_sim.norm_allpriors.abs(), 0.1),
                        rt_lim=50, extra_label=extra_label)


def mean_com_traj_simul(df_sim, data_folder, new_data, save_new_data, ax):
    """
    Plots mean reversal and non-reversal trajectory across simulated subjects.
    """
    raw_com = df_sim.CoM_sugg.values
    index_com = df_sim.com_detected.values
    trajs_all = df_sim.trajectory_y.values
    dec = df_sim.R_response.values*2-1
    max_ind = 800  # max MT
    subjects = df_sim.subjid.unique()
    # prepare arrays
    matrix_com_tr = np.empty((len(subjects), max_ind))
    matrix_com_tr[:] = np.nan
    matrix_com_und_tr = np.empty((len(subjects), max_ind))
    matrix_com_und_tr[:] = np.nan
    matrix_nocom_tr = np.empty((len(subjects), max_ind))
    matrix_nocom_tr[:] = np.nan
    for i_s, subject in enumerate(subjects):
        # for each subject
        traj_data_path = data_folder+subject+'/sim_data/'+subject +\
            '_mean_com_trajs.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data_path), exist_ok=True)
        if os.path.exists(traj_data_path) and not new_data:
            # load data if there is data saved
            traj_data = np.load(traj_data_path, allow_pickle=True)
            mean_com_und_traj = traj_data['mean_com_und_traj']
            mean_nocom_tr = traj_data['mean_nocom_tr']
            mean_com_traj = traj_data['mean_com_traj']
        else:
            # compute mean trajectories
            it_subs = np.where(df_sim.subjid.values == subject)[0][0]
            i_com = 0
            i_nocom = 0
            i_und_com = 0
            # define arrays
            mat_nocom_erase = np.empty((sum(~(raw_com)), max_ind))
            mat_nocom_erase[:] = np.nan
            mat_com_erase = np.empty((sum(index_com), max_ind))
            mat_com_erase[:] = np.nan
            mat_com_und_erase = np.empty((sum((~index_com) & (raw_com)), max_ind))
            mat_com_und_erase[:] = np.nan
            for i_t, traj in enumerate(trajs_all[df_sim.subjid == subject]):
                if index_com[i_t+it_subs]:
                    mat_com_erase[i_com, :len(traj)] = traj*dec[i_t+it_subs]
                    i_com += 1
                if not index_com[i_t+it_subs] and not raw_com[i_t]:
                    mat_nocom_erase[i_nocom, :len(traj)] = traj*dec[i_t+it_subs]
                    i_nocom += 1
                if raw_com[i_t+it_subs] and not index_com[i_t+it_subs]:
                    mat_com_und_erase[i_und_com, :len(traj)] = traj*dec[i_t+it_subs]
                    i_und_com += 1
            # average trajectories across trials
            mean_com_traj = np.nanmean(mat_com_erase, axis=0)
            mean_nocom_tr = np.nanmean(mat_nocom_erase, axis=0)
            mean_com_und_traj = np.nanmean(mat_com_und_erase, axis=0)
        if save_new_data:
            # if we want to save new data
            data = {'mean_com_traj': mean_com_traj, 'mean_nocom_tr': mean_nocom_tr,
                    'mean_com_und_traj': mean_com_und_traj}
            np.savez(traj_data_path, **data)
        matrix_com_tr[i_s, :len(mean_com_traj)] = mean_com_traj
        matrix_nocom_tr[i_s, :len(mean_nocom_tr)] = mean_nocom_tr
        matrix_com_und_tr[i_s, :len(mean_com_und_traj)] = mean_com_und_traj
        # plot mean trajectory for each subject
        ax.plot(np.arange(len(mean_com_traj)), mean_com_traj, color=fig_3.COLOR_COM,
                linewidth=1.4, alpha=0.25)
    # average across subjects
    mean_com_traj = np.nanmean(matrix_com_tr, axis=0)
    mean_nocom_traj = np.nanmean(matrix_nocom_tr, axis=0)
    mean_com_all_traj = np.nanmean(matrix_com_und_tr, axis=0)
    # plot trajectories
    ax.plot(np.arange(len(mean_com_traj)), mean_com_traj, color=fig_3.COLOR_COM,
            linewidth=2)
    ax.plot(np.arange(len(mean_com_all_traj)), mean_com_all_traj, color='saddlebrown',
            linewidth=1.4, linestyle='--')
    ax.plot(np.arange(len(mean_nocom_traj)), mean_nocom_traj, color=fig_3.COLOR_NO_COM,
            linewidth=2)
    # tune panels
    legendelements = [Line2D([0], [0], color=fig_3.COLOR_COM, lw=2,
                             label='Detected reversal'),
                      Line2D([0], [0], color='saddlebrown', lw=1.5,  linestyle='--',
                             label='All CoMs'),
                      Line2D([0], [0], color=fig_3.COLOR_NO_COM, lw=2,
                             label='No-reversal')]
    ax.legend(handles=legendelements, loc='upper left',
              bbox_to_anchor=(0.05, 1.32), handlelength=1.2, frameon=False)
    ax.set_xlabel('Time from movement onset (ms)')
    ax.set_ylabel('x position (cm)')
    ax.set_xlim(-25, 400)
    ax.set_ylim(-25, 80)
    ax.axhline(-8, color='r', linestyle=':')
    ax.text(200, -19, "Detection threshold", color='r')
    conv_factor = 0.07
    ticks = np.array([-2, 0, 2, 4, 6])/conv_factor
    ax.set_yticks(ticks, np.int64(np.round(ticks*conv_factor, 2)))


def traj_cond_coh_simul(df_sim, data_folder, new_data, save_new_data,
                        ax=None, median=True, prior=True, prior_lim=1, rt_lim=200,
                        extra_label=''):
    """
    Plots trajectories depending on:
        - stimulus evidence towards response: prior = False
        - prior evidence towards response: prior = True
    """
    ax[2].axvline(x=0, color='k', linestyle='--', linewidth=0.6)
    df_sim = df_sim[df_sim.sound_len >= 0]
    if median:
        # in case we use median instead of mean
        func_final = np.nanmedian
    if not median:
        func_final = np.nanmean
    # define prior/stim. evs. towards response
    df_sim['choice_x_coh'] = (df_sim.R_response*2-1) * df_sim.coh2
    df_sim['choice_x_prior'] = (df_sim.R_response*2-1) * df_sim.norm_allpriors
    bins_coh = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    if prior:
        condition = 'choice_x_prior'
        # get equipopulated bins
        bins_zt, _, _, _, _ =\
              fp.get_bin_info(df=df_sim, condition=condition, prior_limit=1,
                              after_correct_only=True)
        xvals_zt = (bins_zt[:-1] + bins_zt[1:]) / 2
    else:
        xvals_zt = bins_coh
    signed_response = df_sim.R_response.values
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
    labels_zt = [VAR_INC_SHORT, ' ', '0', ' ', VAR_CON_SHORT]
    labels_coh = [VAR_INC_SHORT, ' ', ' ', '0', ' ', ' ', VAR_CON_SHORT]
    if prior:  # define bins and colormaps for each condition
        bins_ref = bins_zt
        colormap = pl.cm.copper(np.linspace(0, 1, len(bins_zt)-1))
    else:
        bins_ref = bins_coh
        colormap = mtp.colors.LinearSegmentedColormap.from_list("", ["mediumblue","plum","firebrick"])
        colormap = colormap(np.linspace(0, 1, len(bins_coh)))
    subjects = df_sim.subjid
    max_mt = 1200
    # initialize arrays
    mat_trajs_subs = np.empty((len(bins_ref), max_mt,
                               len(subjects.unique())))  # trajectories
    mat_vel_subs = np.empty((len(bins_ref), max_mt,
                             len(subjects.unique())))  # velocities
    mat_trajs_indsub = np.empty((len(bins_ref), max_mt))  # MT
    mat_vel_indsub = np.empty((len(bins_ref), max_mt))  # peak velocity
    if prior:
        val_mt_subs = np.empty((len(bins_ref)-1, len(subjects.unique())))
        val_vel_subs = np.empty((len(bins_ref)-1, len(subjects.unique())))
        label_save = 'prior'+extra_label
    else:
        val_mt_subs = np.empty((len(bins_ref), len(subjects.unique())))
        val_vel_subs = np.empty((len(bins_ref), len(subjects.unique())))
        label_save = 'stim'+extra_label
    for i_s, subject in enumerate(subjects.unique()):  # for each subject
        traj_data = data_folder+subject+'/sim_data/'+subject +\
            '_traj_sim_pos_'+label_save+'.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data), exist_ok=True)
        if os.path.exists(traj_data) and not new_data:
            # load data if there is data saved
            traj_data = np.load(traj_data, allow_pickle=True)
            vals_thr_vel = traj_data['vals_thr_vel']
            vals_thr_traj = traj_data['vals_thr_traj']
            mat_trajs_indsub = traj_data['mat_trajs_indsub']
            mat_vel_indsub = traj_data['mat_vel_indsub']
        else:
            # compute trajectories
            vals_thr_traj = []
            vals_thr_vel = []
            lens = []
            for i_ev, ev in enumerate(bins_ref):
                if not prior:
                    # indexing for stimulus
                    index = (df_sim.choice_x_coh.values == ev) &\
                        (df_sim.normallpriors.abs() <= prior_lim) &\
                        (df_sim.special_trial == 0) & (~np.isnan(df_sim.allpriors)) *\
                        (df_sim.sound_len >= 0) & (df_sim.sound_len <= rt_lim) &\
                        (subjects == subject)
                if prior:
                    # indexing for prior
                    if i_ev == len(bins_ref)-1:
                        break
                    index = (df_sim.choice_x_prior.values >= bins_ref[i_ev]) &\
                        (df_sim.choice_x_prior.values < bins_ref[i_ev + 1]) &\
                        (df_sim.sound_len >= 0) & (df_sim.sound_len <= rt_lim) &\
                        (subjects == subject)
                    if sum(index) == 0:
                        continue
                lens.append(max([len(t) for t in df_sim.trajectory_y[index].values]))
                traj_all = np.empty((sum(index), max_mt))
                traj_all[:] = np.nan
                vel_all = np.empty((sum(index), max_mt))
                vel_all[:] = np.nan
                for tr in range(sum(index)):  # group trajectories and compute velocities
                    vals_traj = df_sim.traj[index].values[tr] *\
                        (signed_response[index][tr]*2 - 1)
                    if sum(vals_traj) == 0:
                        continue
                    vals_traj = np.concatenate((vals_traj,
                                                np.repeat(75, max_mt-len(vals_traj))))
                    vals_vel = df_sim.traj_d1[index].values[tr] *\
                        (signed_response[index][tr]*2 - 1)
                    vals_vel = np.diff(vals_traj)
                    traj_all[tr, :len(vals_traj)] = vals_traj
                    vel_all[tr, :len(vals_vel)] = vals_vel
                try:
                    index_vel = np.where(np.sum(np.isnan(traj_all), axis=0)
                                          > traj_all.shape[0] - 50)[0][0]
                    mean_traj = func_final(traj_all[:, :index_vel], axis=0)
                    std_traj = np.nanstd(traj_all[:, :index_vel],
                                          axis=0) / np.sqrt(len(subjects.unique()))
                except Exception:  # if there is some error
                    mean_traj = func_final(traj_all, axis=0)
                    std_traj = np.nanstd(traj_all, axis=0) /\
                        np.sqrt(len(subjects.unique()))
                # compute averages and save them in arrays
                val_mt = np.mean(df_sim['resp_len'].values[index])*1e3
                vals_thr_traj.append(val_mt)
                mean_vel = func_final(vel_all, axis=0)
                std_vel = np.nanstd(vel_all, axis=0) / np.sqrt(len(subjects.unique()))
                val_vel = np.nanmax(mean_vel)
                vals_thr_vel.append(val_vel)
                mat_trajs_indsub[i_ev, :len(mean_traj)] = mean_traj
                mat_vel_indsub[i_ev, :len(mean_vel)] = mean_vel
            if save_new_data:
                # save data if desired
                data = {'mat_trajs_indsub': mat_trajs_indsub,
                        'mat_vel_indsub': mat_vel_indsub,
                        'vals_thr_traj': vals_thr_traj, 'vals_thr_vel': vals_thr_vel}
                np.savez(traj_data, **data)
        mat_trajs_subs[:, :, i_s] = mat_trajs_indsub
        mat_vel_subs[:, :, i_s] = mat_vel_indsub
        val_mt_subs[:len(vals_thr_traj), i_s] = vals_thr_traj
        val_vel_subs[:len(vals_thr_vel), i_s] = vals_thr_vel
    for i_ev, ev in enumerate(bins_ref):
        # plotting each trajectory/velocity
        if prior and ev == 1.01:
            break
        val_mt = np.nanmean(val_mt_subs[i_ev, :])
        std_mt = np.nanstd(val_mt_subs[i_ev, :]) /\
            np.sqrt(len(subjects.unique()))
        val_vel = np.nanmean(val_vel_subs[i_ev, :])
        std_vel_points = np.nanstd(val_vel_subs[i_ev, :]) /\
            np.sqrt(len(subjects.unique()))
        mean_traj = np.nanmean(mat_trajs_subs[i_ev, :, :], axis=1)
        std_traj = np.std(mat_trajs_subs[i_ev, :, :], axis=1) /\
            np.sqrt(len(subjects.unique()))
        mean_vel = np.nanmean(mat_vel_subs[i_ev, :, :], axis=1)
        std_vel = np.std(mat_vel_subs[i_ev, :, :], axis=1) /\
            np.sqrt(len(subjects.unique()))
        if prior:
            xval = xvals_zt[i_ev]
        else:
            xval = ev
        ax[2].errorbar(xval, val_mt, std_mt, color=colormap[i_ev], marker='o')
        ax[3].errorbar(xval, val_vel, std_vel_points, color=colormap[i_ev],
                       marker='o')
        if not prior:
            label = labels_coh[i_ev]
        if prior:
            label = labels_zt[i_ev]
        ax[0].plot(np.arange(len(mean_traj)), mean_traj, label=label,
                   color=colormap[i_ev])
        ax[0].fill_between(x=np.arange(len(mean_traj)),
                           y1=mean_traj - std_traj, y2=mean_traj + std_traj,
                           color=colormap[i_ev], alpha=0.1)
        ax[1].plot(np.arange(len(mean_vel)), mean_vel, label=label,
                   color=colormap[i_ev])
        ax[1].fill_between(x=np.arange(len(mean_vel)),
                           y1=mean_vel - std_vel, y2=mean_vel + std_vel,
                           color=colormap[i_ev], alpha=0.1)
    # panel tuning
    ax[0].set_xlim(-5, 335)
    ax[0].set_yticks([0, 25, 50, 75])
    ax[0].set_ylim(-8, 85)
    ax[1].set_ylim(-0.06, 0.6)
    ax[1].set_xlim(-5, 335)
    ax[2].set_xlim(-1.2, 1.2)
    if prior:
        leg_title = 'Prior'
        ax[2].plot(xvals_zt, np.nanmean(val_mt_subs, axis=1),
                   color='k', ls='-', lw=0.5)
        ax[3].plot(xvals_zt, np.nanmean(val_vel_subs, axis=1),
                   color='k', ls='-', lw=0.5)
        ax[2].text(-0.4, 290, r'$\longleftarrow $', fontsize=10)
        ax[2].text(-0.92, 292.3, r'$\it{incongruent}$', fontsize=7.5)
        ax[2].text(0.09, 290, r'$\longrightarrow $', fontsize=10)
        ax[2].text(0.09, 292.3, r'$\it{congruent}$', fontsize=7.5)
        ax[2].set_xlabel('Prior evidence \ntowards response')
        ax[3].set_xlabel('Prior')
    if not prior:
        leg_title = 'Stimulus'
        ax[2].plot(bins_coh, np.nanmean(val_mt_subs, axis=1),
                   color='k', ls='-', lw=0.5)
        ax[3].plot(bins_coh,  np.nanmean(val_vel_subs, axis=1),
                   color='k', ls='-', lw=0.5)
        ax[2].text(-0.4, 302, r'$\longleftarrow $', fontsize=10)
        ax[2].text(-0.92, 306, r'$\it{incongruent}$', fontsize=7.5)
        ax[2].text(0.09, 302, r'$\longrightarrow $', fontsize=10)
        ax[2].text(0.09, 306, r'$\it{congruent}$', fontsize=7.5)
        ax[2].set_xlabel('Stimulus evidence \ntowards response')
        ax[3].set_xlabel('Stimulus')
    ax[2].set_xticks([-1, 0, 1], [VAR_INC, '0', VAR_CON])
    ax[0].legend(title=leg_title, labelspacing=0.15,
                 loc='center left', bbox_to_anchor=(0.7, 0.45), handlelength=1.5,
                 frameon=False)
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles[::-1], labels[::-1], title=leg_title, labelspacing=0.15,
                 loc='center left', bbox_to_anchor=(0.7, 0.45), handlelength=1.5,
                 frameon=False)
    ax[0].set_ylabel('y position (cm)')
    ax[0].set_xlabel('Time from movement onset (ms)')
    ax[1].set_ylabel('y velocity (cm/s)')
    ax[1].set_xlabel('Time from movement onset (ms)')
    ax[2].set_ylabel('Movement time (ms)')
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_ylabel('Peak')
    conv_factor = 0.07
    ticks = np.array([0, 2, 4, 6])/conv_factor
    ax[0].set_yticks(ticks,np.int64(np.round(ticks*conv_factor, 2)))
    conv_factor = 0.07*1e3  # *1000 to have cm/s
    ticks = np.array([0, 0.28571429, 0.57142857])
    labs = np.int64(np.round(ticks*conv_factor))
    ax[1].set_yticks(ticks, labs)



def fig_6_model(sv_folder, data_folder, new_data, save_new_data,
                coh, sound_len, hit_model, sound_len_model, zt,
                decision_model, com, com_model, com_model_detected,
                df_sim, inset_sz=.06,
                marginx=-0.02, marginy=0.08, fgsz=(13, 12),
                extra_label=''):
    """
    Create and plot figure 6.
    """
    fig, ax, ax_inset, pos_ax_0 = create_figure_5_model(fgsz=fgsz)
    # select RT > 0 (no FB, as in data)
    hit_model = hit_model[sound_len_model >= 0]
    com_model_detected = com_model_detected[sound_len_model >= 0]
    decision_model = decision_model[sound_len_model >= 0]
    com_model = com_model[sound_len_model >= 0]
    subjid = df_sim.subjid.values
    # Tachometrics
    _ = fp.tachometric_data(coh=coh[sound_len_model >= 0], hit=hit_model,
                            sound_len=sound_len_model[sound_len_model >= 0],
                            subjid=subjid, ax=ax[1], label='', legend=False,
                            rtbins=np.arange(0, 201, 10))
    colormap = pl.cm.gist_gray_r(np.linspace(0.4, 1, 4))
    # tune panel
    legendelements = [Line2D([0], [0], color=colormap[3], lw=2,
                             label='1'),
                      Line2D([0], [0], color=colormap[2], lw=2,
                             label='0.5'),
                      Line2D([0], [0], color=colormap[1], lw=2,
                             label='0.25'),
                      Line2D([0], [0], color=colormap[0], lw=2,
                             label='0')]
    ax[1].legend(handles=legendelements, fontsize=8, loc='center left',
                 bbox_to_anchor=(0.92, 1.2), title='Stimulus',
                 handlelength=1.3, frameon=False)
    ax[1].set_ylim(0.4, 1)
    # PCoM vs RT
    df_plot_pcom = pd.DataFrame({'com': com[sound_len_model >= 0],
                                 'sound_len': sound_len[sound_len_model >= 0],
                                 'rt_model': sound_len_model[sound_len_model >= 0],
                                 'com_model': com_model, 'subjid': subjid,
                                 'com_model_detected': com_model_detected})
    zt_model = df_sim.norm_allpriors.values
    plot_com_vs_rt_f5(df_plot_pcom=df_plot_pcom, ax=ax[-1], ax2=ax[-1])
    ax[-1].set_ylim(-0.005, 0.121)
    ax[-1].set_yticks([0, 0.04, 0.08, 0.12])
    # slowing in MT
    fig_1.plot_mt_vs_stim(df=df_sim, ax=ax[3], prior_min=0.8, rt_max=50,
                          sim=True)
    ax[3].set_yticks([200, 240, 280])
    # P(right) matrix
    plot_pright_model(df_sim=df_sim, sound_len_model=sound_len_model,
                      decision_model=decision_model, subjid=subjid, coh=coh,
                      zt_model=zt_model, ax=ax[0])
    df_model = pd.DataFrame({'avtrapz': coh[sound_len_model >= 0],
                             'CoM_sugg': com_model_detected,
                             'norm_allpriors': df_sim.norm_allpriors.values,
                             'R_response': (decision_model+1)/2, 'subjid': subjid})
    df_model = df_model.loc[~df_model.norm_allpriors.isna()]
    nbins = 7
    # plot Pcoms matrices
    ax_mat = [ax[12], ax_inset]
    n_subjs = len(df_sim.subjid.unique())
    # PCoM matrices
    plot_pcom_matrices_model(df_model=df_model, n_subjs=n_subjs,
                             ax_mat=ax_mat, pos_ax_0=pos_ax_0, nbins=nbins,
                             f=fig)
    # MT matrix vs stim/prior
    fig_1.mt_matrix_ev_vs_zt(df_sim, ax[2], f=fig, silent_comparison=False,
                             collapse_sides=True, margin=0.03)
    # MT distributions
    fig_3.mt_distros(df=df_sim, ax=ax[14], xlmax=625, sim=True)
    ax[14].get_legend().remove()
    # plot trajs and MT conditioned on stim/prior
    plot_trajs_cond_on_prior_and_stim(df_sim=df_sim, ax=ax, new_data=new_data,
                                      save_new_data=save_new_data,
                                      inset_sz=inset_sz, data_folder=data_folder,
                                      fgsz=fgsz, marginx=marginx, marginy=marginy,
                                      extra_label=extra_label)
    # plot splitting time vs RT
    fig_2.trajs_splitting_stim(df_sim.loc[df_sim.special_trial == 0],
                               data_folder=data_folder, ax=ax[7], collapse_sides=True,
                               threshold=800, sim=True, rtbins=np.linspace(0, 150, 16),
                               connect_points=True, trajectory="trajectory_y",
                               p_val=0.05, extra_label=extra_label)
    
    subjects = np.unique(subjid)
    ta_te = []
    for subject in subjects:
        conf = np.load(sv_folder + 'parameters_MNLE_BADS' + subject + '.npy')
        ta_te.append(conf[4]+conf[5])
    ax[7].axhline(np.nanmean(ta_te)*5, color='k', alpha=0.5, linestyle='--')
    ax[7].text(90, np.nanmean(ta_te)*5-25, r'$t_{aff}+t_{eff}$', fontsize=9.5)
    ax[7].annotate(text='', xy=(75, 0.5),
                   xytext=(75, np.nanmean(ta_te)*5-0.5),
                   arrowprops=dict(arrowstyle='<->'))
    ax[7].set_ylim(0, 205)
    # plot mean com traj
    
    if len(df_sim.subjid.unique()) > 1:
        subject = ''
    else:
        subject = df_sim.subjid.unique()[0]
    # first save, mean_com_traj_simul can be slow so you can take a look at
    # the figure without the panel
    fig.savefig(sv_folder+subject+'/fig5.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+subject+'/fig5.png', dpi=400, bbox_inches='tight')
    mean_com_traj_simul(df_sim, ax=ax[11], data_folder=data_folder, new_data=new_data,
                        save_new_data=save_new_data)
    fig.savefig(sv_folder+subject+'/fig6.png', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+subject+'/fig6.svg', dpi=400, bbox_inches='tight')


def shortpad2(row, upto=1000, align='sound', pad_value=np.nan,
              pad_pre=0):
    """pads nans to trajectories so it can be stacked in a matrix
    align can be either 'movement' (0 is movement onset), or 'sound'
    """
    if align == 'movement':
        missing = upto - row.traj.size
        return np.pad(row.traj, ((0, missing)), "constant",
                    constant_values=pad_value)
    elif align == 'sound':
        missing_pre = int(row.sound_len)
        missing_after = upto - missing_pre - row.traj.size
        return np.pad(row.traj, ((missing_pre, missing_after)), "constant",
                    constant_values=(pad_pre, pad_value))
