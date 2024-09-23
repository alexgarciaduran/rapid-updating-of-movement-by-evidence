# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:18:25 2023

@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import figure_1 as fig_1
import figure_6 as fig_6
import figures_paper as fp


# ---FUNCTIONS
def plot_mt_vs_stim_cong_and_prev_pcom_mats_different_models(subjects, subjid, stim,
                                                             zt, coh, gt, trial_index,
                                                             special_trial,
                                                             data_folder, ax, fig,
                                                             sv_folder,
                                                             extra_labels=['_2_ro',
                                                                           '_1_ro',''],
                                                             alpha_list=[1, 0.3],
                                                             margin=0.03):
    """
    Loads data (or simulates) alternative models of figure 7. Then, it plots
    MT vs stim. in express responses (RT < 50 ms) as in Figure 1f, and the 
    p(CoM|choice) and p(reversal|choice) matrices against prior/stimulus.
    """
    mat_titles_rev = ['L to R reversal',
                      'R to L reversal']
    mat_titles_com = ['L to R CoM',
                      'R to L CoM']
    for i_l, lab in enumerate(extra_labels):  # for each alternative model
        # load/simulate data given extra lab
        df_data = fp.get_simulated_data_extra_lab(subjects, subjid, stim, zt, coh, gt, trial_index,
                                                  special_trial, extra_label=lab,
                                                  data_folder=data_folder,
                                                  sv_folder=sv_folder)
        df_mt = df_data.copy()
        # plot MT vs stim as in Figure 1f
        fig_1.plot_mt_vs_evidence(df=df_mt, ax=ax[1+i_l*4], prior_limit=0.1,  # 10% quantile
                                  condition='choice_x_coh', rt_lim=50, alpha=1,
                                  write_arrows=False)
        del df_mt
        if i_l >= 1:
            mat_titles_rev = ['', '']
            mat_titles_com = ['', '']
        # compute CoM matrices
        mat0, mat1 = fig_6.plot_pcom_matrices_model(df_model=df_data,
                                                    n_subjs=len(df_data.subjid.unique()),
                                                    ax_mat=[ax[0], ax[0]],
                                                    pos_ax_0=[], nbins=7,
                                                    f=fig, title='p(CoM)',
                                                    mat_titles=mat_titles_com,
                                                    return_matrix=True)
        vmax = 0.55
        bone_cmap = matplotlib.cm.get_cmap('bone', 256)
        # tune colormap
        newcolors = bone_cmap(9*np.linspace(0, 1, 256)/
                              (1+np.linspace(0, 1, 256)*8))
        cmap = matplotlib.colors.ListedColormap(newcolors)
        # plot matrices and tune panels
        im = ax[4*i_l+2].imshow(mat0+mat1, vmin=0, vmax=vmax, cmap=cmap)
        if i_l >= 3:
            ax[4*i_l+2].set_xticks([0, 3, 6], ['L', '0', 'R'])
            ax[4*i_l+2].set_xlabel('Prior evidence')
            ax[4*i_l+3].set_xlabel('Prior evidence')
            ax[4*i_l+3].set_xticks([0, 3, 6], ['L', '0', 'R'])
        else:
            ax[4*i_l+2].set_xticks([0, 3, 6], ['', '', ''])
            ax[4*i_l+3].set_xticks([0, 3, 6], ['', '', ''])
        if 4*i_l+2 in [2, 6, 10, 14]:
            ax[4*i_l+2].set_yticks([0, 3, 6], ['R', '0', 'L'])
            ax[4*i_l+2].set_ylabel('Stimulus evidence')
        else:
            ax[4*i_l+2].set_yticks([0, 3, 6], ['', '', ''])
        ax[4*i_l+3].set_yticks([0, 3, 6], ['', '', ''])
        pos = ax[4*i_l+2].get_position()
        cbar_ax = fig.add_axes([pos.x0+pos.width+margin/2, pos.y0+margin/6,
                                pos.width/15, pos.height/1.5])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.ax.set_title('         '+'p(CoM)', fontsize=8.5, pad=10)
        cbar.ax.set_yticks([0, 0.25, 0.5])
        df_data['CoM_sugg'] = df_data.com_detected
        # compute reversal matrices
        mat0, mat1 = fig_6.plot_pcom_matrices_model(df_model=df_data,
                                                    n_subjs=len(df_data.subjid.unique()),
                                                    ax_mat=[ax[0], ax[0]],
                                                    pos_ax_0=[], nbins=7,
                                                    f=fig, title='p(reversal)',
                                                    mat_titles=mat_titles_rev,
                                                    return_matrix=True)
        # plot matrices and tune panels
        vmax = np.nanmax(mat0+mat1)
        if vmax == 0:
            vmax = 1.1e-2
        im = ax[4*i_l+3].imshow(mat0+mat1, vmin=0, vmax=vmax, cmap='magma')
        pos = ax[4*i_l+3].get_position()
        cbar_ax = fig.add_axes([pos.x0+pos.width+margin/2, pos.y0+margin/6,
                                pos.width/15, pos.height/1.5])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.ax.set_title('         '+'p(reversal)', fontsize=8.5, pad=10)


def fig_7(subjects, subjid, stim, zt, coh, gt, trial_index,
          special_trial, data_folder, sv_folder,
          extra_labels=['',
                        '_2_ro_rand_',
                        '_1_ro_',
                        '_1_ro__com_modulation_']):
    '''
    Plots figure 7 for all 4 different models:
        1. Full model.
        2. Random initial choice (2 read-outs but one is random).
        3. No trajectory update (only 1st read-out).
        4. Only update when CoM (no vigor update).
    '''
    fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(12, 12))
    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.5, wspace=0.5)
    ax = ax.flatten()
    labs = ['i', 'ii', 'iii', 'iv',
            'i', 'ii', 'iii', 'iv',
            'i', 'ii', 'iii', 'iv',
            'i', 'ii', 'iii', 'iv']
    general_labels = ['a', '', '', '',
                      'b', '', '', '',
                      'c', '', '', '',
                      'd', '', '', '']
    # tune panels
    for i_ax, a in enumerate(ax):
        if (i_ax-1) % 4 == 0 or (i_ax) % 4 == 0:
            fp.rm_top_right_lines(a)
            a.text(-0.31, 1.12, labs[i_ax], transform=a.transAxes, fontsize=14,
                   fontweight='bold', va='top', ha='right')
        else:
            a.text(-0.31, 1.17, labs[i_ax], transform=a.transAxes, fontsize=14,
                   fontweight='bold', va='top', ha='right')
        a.text(-0.41, 1.22, general_labels[i_ax], transform=a.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='right')
    titles = ['Full model',
              'Random initial choice',
              'No trajectory update',
              'No vigor update']
    # plots
    plot_mt_vs_stim_cong_and_prev_pcom_mats_different_models(
        subjects, subjid, stim, zt, coh, gt, trial_index,
        special_trial, data_folder=data_folder, ax=ax, fig=fig,
        extra_labels=extra_labels, sv_folder=sv_folder)
    # tune panels
    yvals_text = [0.7, 1.1, 1.1, 1.1]
    for i in range(4):
        ax[i*4].text(-0.68, yvals_text[i], titles[i], transform=ax[i*4].transAxes,
                     fontsize=14, va='top', rotation='vertical')
        ax[i*4].axis('off')
        pos = ax[i*4+1].get_position()
        ax[i*4+1].set_position([pos.x0, pos.y0+(pos.height-pos.width)/2,
                                pos.width, pos.width])
        ax[i*4+1].set_xlabel('')
    ax[1].set_ylim(240, 280)
    ax[1].set_yticks([240, 260, 280])
    ax[5].set_ylim(190, 240)
    ax[5].set_yticks([200, 220, 240])
    ax[9].set_ylim(210, 260)
    ax[9].set_yticks([220, 240, 260])
    ax[13].set_ylim(240, 280)
    ax[13].set_yticks([240, 260, 280])
    ax[13].set_xlabel('Stimulus evidence\ntowards response')
    ax[1].text(-0.4, 283, r'$\longleftarrow $', fontsize=10)
    ax[1].text(-0.98, 287, r'$\it{incongruent}$', fontsize=8)
    ax[1].text(0.07, 283, r'$\longrightarrow $', fontsize=10)
    ax[1].text(0.07, 287, r'$\it{congruent}$', fontsize=8)
    # save figure
    fig.savefig(sv_folder+'/fig7.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'/fig7.png', dpi=400, bbox_inches='tight')


def supp_prior_only_continuous(subjects, subjid, stim, zt, coh, gt, trial_index,
                               special_trial, data_folder, sv_folder,
                               extra_labels=['_prior_sign_1_ro_',
                                             '_50_ms_continuous_']):
    """
    Plots supplementary figure 12, same as figure 7 for two alternative models:
        1. First choice ONLY driven by prior.
        2. Continuous read-outs.
    """
    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(12, 7))
    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.5, wspace=0.5)
    ax = ax.flatten()
    labs = ['i', 'ii', 'iii', 'iv',
            'i', 'ii', 'iii', 'iv']
    general_labels = ['a', '', '', '',
                      'b', '', '', '']
    # tune panels
    for i_ax, a in enumerate(ax):
        if (i_ax-1) % 4 == 0 or (i_ax) % 4 == 0:
            fp.rm_top_right_lines(a)
            a.text(-0.31, 1.12, labs[i_ax], transform=a.transAxes, fontsize=14,
                   fontweight='bold', va='top', ha='right')
        else:
            a.text(-0.31, 1.17, labs[i_ax], transform=a.transAxes, fontsize=14,
                   fontweight='bold', va='top', ha='right')
        a.text(-0.41, 1.22, general_labels[i_ax], transform=a.transAxes,
               fontsize=16, fontweight='bold', va='top', ha='right')
    titles = ['Continuous update', 'Prior-based initial trajectory']
    # plot
    plot_mt_vs_stim_cong_and_prev_pcom_mats_different_models(
        subjects, subjid, stim, zt, coh, gt, trial_index,
        special_trial, data_folder, ax=ax, fig=fig,
        extra_labels=extra_labels, data_folder=data_folder)
    # tune panels
    yvals_text = [0.7, 1.1, 1.1, 1.1]
    for i in range(2):
        ax[i*4].text(-0.68, yvals_text[i], titles[i], transform=ax[i*4].transAxes,
                     fontsize=14, va='top', rotation='vertical')
        ax[i*4].axis('off')
    ax[1].set_ylim(240, 290)
    ax[1].set_yticks([240, 260, 280])
    ax[5].set_ylim(240, 290)
    ax[5].set_yticks([240, 260, 280])
    ax[5].set_xlabel('Stimulus evidence\ntowards response')
    ax[1].text(-0.4, 283, r'$\longleftarrow $', fontsize=10)
    ax[1].text(-0.98, 287, r'$\it{incongruent}$', fontsize=8)
    ax[1].text(0.07, 283, r'$\longrightarrow $', fontsize=10)
    ax[1].text(0.07, 287, r'$\it{congruent}$', fontsize=8)
    ax[6].set_xticks([0, 3, 6], ['L', '0', 'R'])
    ax[7].set_xticks([0, 3, 6], ['L', '0', 'R'])
    ax[6].set_xlabel('Prior evidence')
    ax[7].set_xlabel('Prior evidence')
    # save figure
    fig.savefig(sv_folder+'/supp_alternative.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'/supp_alternative.png', dpi=400, bbox_inches='tight')

