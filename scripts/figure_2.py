import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.stats import pearsonr
import matplotlib as mtp
from matplotlib.lines import Line2D
from scipy.stats import sem
import figures_paper as fp


# ---FUNCTIONS
def plots_trajs_conditioned(df, ax, data_folder, condition='choice_x_coh', cmap='viridis',
                            prior_limit=0.25, rt_lim=50,
                            after_correct_only=True,
                            trajectory="trajectory_y",
                            velocity=("traj_d1", 1)):
    """
    Plots mean trajectories, MT, velocity and peak velocity
    conditioning on stimulus/prior/trial index. This depends on the variable condition:
        - condition == "choice_x_coh": stimulus congruency
        - condition == "choice_x_prior": prior congruency
        - condition == "origidx": trial index
    """
    interpolatespace = np.linspace(-700000, 1000000, 1700)  # space of interpolation
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1) == 2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan
    df['norm_allpriors'] = fp.norm_allpriors_per_subj(df)
    # transform variables into congruent/incongruent with final response
    df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    df['choice_x_coh'] = (df.R_response*2-1) * df.coh2
    # get bins of prior or stimmulus
    bins, bintype, indx_trajs, n_iters, colormap =\
          fp.get_bin_info(df=df, condition=condition, prior_limit=prior_limit,
                          after_correct_only=after_correct_only,
                          rt_lim=rt_lim)
    # POSITION
    subjects = df['subjid'].unique()
    # initialize matrix
    mat_all = np.empty((n_iters, 1700, len(subjects)))
    mt_all = np.empty((n_iters, len(subjects)))
    for i_subj, subj in enumerate(subjects):  # for each subject
        traj_data_path = data_folder+subj+'/traj_data/'+subj+'_traj_pos_'+condition+'.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data_path), exist_ok=True)
        if os.path.exists(traj_data_path):  # load data if possible
            traj_data = np.load(traj_data_path, allow_pickle=True)
            mean_traj = traj_data['mean_traj']
            xpoints = traj_data['xpoints']
            mt_time = traj_data['mt_time']
        else:
            # extract trajectories given
            # mat will be a matrix with all trajectories (1700 timepoints),
            # aligned to movement onset (700 represents movement onset, 0 ms)
            # and for different conditions given by "condition" and "bins"
            xpoints, _, _, mat, _, mt_time =\
                fp.trajectory_thr(df.loc[(indx_trajs) & (df.subjid == subj)],
                                  condition, bins, collapse_sides=True,
                                  ax=None, ax_traj=None, return_trash=True,
                                  error_kwargs=dict(marker='o'), cmap=cmap, bintype=bintype,
                                  trajectory=trajectory, plotmt=True, alpha_low=False)
            # compute mean trajectory for each condition
            mean_traj = np.array([np.nanmean(mat[m], axis=0) for m in mat])
            data = {'xpoints': xpoints, 'mean_traj': mean_traj, 'mt_time': mt_time}
            # save data
            np.savez(traj_data_path, **data)
        # save mean trajectory
        mat_all[:, :, i_subj] = mean_traj
        mt_all[:, i_subj] = mt_time
    # mean trajectory across subjects and S.E.M.
    all_trajs = np.nanmean(mat_all, axis=2)
    all_trajs_err = np.nanstd(mat_all, axis=2) / np.sqrt(len(subjects))
    mt_time = np.nanmedian(mt_all, axis=1)  # median MT
    for i_tr, traj in enumerate(all_trajs):  # for each average trajectory
        col_face = colormap[i_tr].copy()
        col_face[-1] = 0.2
        col_edge = [0, 0, 0, 0]
        # center trajectory to value between -100 and 0 ms, so it starts at 0
        traj -= np.nanmean(traj[(interpolatespace > -100000) * (interpolatespace < 0)])
        # plot each mean trajectory
        ax[0].plot(interpolatespace/1000, traj, color=colormap[i_tr], linewidth=1.2)
        ax[0].fill_between(interpolatespace/1000, traj-all_trajs_err[i_tr],
                           traj+all_trajs_err[i_tr], facecolor=col_face,
                           edgecolor=col_edge)
    # tune panels depending on condition (legend mainly)
    if condition == 'choice_x_coh':
        legendelements = [Line2D([0], [0], color=colormap[6], lw=2, label='cong.'),
                          Line2D([0], [0], color=colormap[5], lw=2, label=''),
                          Line2D([0], [0], color=colormap[4], lw=2, label=''),
                          Line2D([0], [0], color=colormap[3], lw=2, label='0'),
                          Line2D([0], [0], color=colormap[2], lw=2, label=''),
                          Line2D([0], [0], color=colormap[1], lw=2, label=''),
                          Line2D([0], [0], color=colormap[0], lw=2, label='inc.')]
        title = 'Stimulus'
    if condition == 'choice_x_prior':
        legendelements = [Line2D([0], [0], color=colormap[4], lw=2,
                                 label='cong.'),
                          Line2D([0], [0], color=colormap[3], lw=2,
                                 label=''),
                          Line2D([0], [0], color=colormap[2], lw=2,
                                 label='0'),
                          Line2D([0], [0], color=colormap[1], lw=2, label=''),
                          Line2D([0], [0], color=colormap[0], lw=2,
                                 label='inc.')]
        title = 'Prior'
    if condition == 'origidx':
        legendelements = []
        labs = ['1-200', '201-400', '401-600', '601-800', '801-1000']
        for i in range(len(colormap)):
            legendelements.append(Line2D([0], [0], color=colormap[i], lw=2,
                                  label=labs[i]))
        title = 'Trial index'
        ax[1].set_xlabel('Trial index')
    # tune panels
    ax[0].legend(handles=legendelements, loc='upper left', title=title,
                labelspacing=.1, bbox_to_anchor=(0., 1.1), handlelength=1.5,
                frameon=False)
    ax[1].set_xlabel(title)
    ax[0].set_xlim([-20, 450])
    ax[0].set_xticklabels('')
    ax[0].axhline(0, c='gray')
    ax[0].set_ylabel('y-position (cm)')
    ax[0].set_xlabel('Time from movement onset (ms)')
    ax[0].set_ylim([-10, 85])
    conv_factor = 0.07
    ticks = np.array([0, 2, 4, 6])/conv_factor
    labs = np.int64(np.round(ticks*conv_factor, 2))
    ax[0].set_yticks(ticks, labs)
    # VELOCITIES (same process as trajectories)
    mat_all = np.empty((n_iters, 1700, len(subjects)))
    mt_all = np.empty((n_iters, len(subjects)))
    for i_subj, subj in enumerate(subjects):
        traj_data = data_folder + subj + '/traj_data/' + subj + '_traj_vel_'+condition+'.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data), exist_ok=True)
        if os.path.exists(traj_data):
            traj_data = np.load(traj_data, allow_pickle=True)
            mean_traj = traj_data['mean_traj']
            xpoints = traj_data['xpoints']
            mt_time = traj_data['mt_time']
            ypoints = traj_data['ypoints']
        else:
            xpoints, ypoints, _, mat, _, mt_time =\
                fp.trajectory_thr(df.loc[(indx_trajs) & (df.subjid == subj)],
                                  condition, bins,
                                  collapse_sides=True, thr=30, ax=None, ax_traj=None,
                                  return_trash=True, error_kwargs=dict(marker='o'),
                                  cmap=cmap, bintype=bintype,
                                  trajectory=velocity, plotmt=True, alpha_low=False)
            mean_traj = np.array([np.nanmean(mat[m], axis=0) for m in mat])
            data = {'xpoints': xpoints, 'ypoints': ypoints, 'mean_traj': mean_traj, 'mt_time': mt_time}
            np.savez(traj_data, **data)
        mat_all[:, :, i_subj] = mean_traj
        mt_all[:, i_subj] = ypoints
    all_trajs = np.nanmean(mat_all, axis=2)  # average velocity
    all_trajs_err = np.nanstd(mat_all, axis=2) / np.sqrt(len(subjects))
    peak = np.nanmedian(mt_all, axis=1)  # median peak velocity
    peak_err = np.nanstd(mt_all, axis=1) / np.sqrt(len(subjects))
    for i_tr, traj in enumerate(all_trajs):  # plot
        col_face = colormap[i_tr].copy()
        col_face[-1] = 0.2
        col_edge = [0, 0, 0, 0]
        traj -= np.nanmean(traj[(interpolatespace > -100000) * (interpolatespace < 0)])
        ax[2].plot(interpolatespace/1000, traj, color=colormap[i_tr], linewidth=1.2)
        ax[2].fill_between(interpolatespace/1000, traj-all_trajs_err[i_tr],
                           traj+all_trajs_err[i_tr],
                           facecolor=col_face, edgecolor=col_edge)
        ax[1].errorbar(xpoints[i_tr], peak[i_tr], yerr=peak_err[i_tr],
                       color=colormap[i_tr], marker='o')
    # tune panels
    ax[2].set_xlim([-20, 450])
    ax[1].set_ylabel('Peak')
    ax[2].set_ylim([-0.05, 0.5])
    ax[2].axhline(0, c='gray')
    ax[2].set_ylabel('y-velocity (cm/s)')
    ax[2].set_xlabel('Time from movement onset (ms)')
    conv_factor = 0.07*1e3  # *1000 to have cm/s
    ticks = np.array([0, 0.28571429, 0.57142857])
    labs = np.int64(np.round(ticks*conv_factor))
    ax[2].set_yticks(ticks, labs)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].plot(xpoints, peak, color='k', ls='-', lw=0.5)


def get_split_ind_corr(mat, evl, pval=0.01, max_MT=400, startfrom=700, sim=True):
    """
    Splitting time computation.
    Returns index at which the trajectories (mat) and stimulus/prior evidence (evl)
    vector become correlated.
    """
    # mat: trajectories (n trials x time)
    for i in range(max_MT):
        trajs = mat[:, startfrom + i]  # trajectories at each point in time
        nan_idx = ~np.isnan(trajs)
        evidence = evl[nan_idx]  # stimulus evidence
        trajs = trajs[nan_idx]
        try:
            _, p2 = pearsonr(trajs, evidence)  # p2 = pvalue from pearson corr
            if p2 < pval and not np.isnan(p2):
                return i
        except ValueError:
            continue
    return np.nan


def get_corr_coef(mat, evl, pval=0.05, max_MT=400, startfrom=700, sim=True):
    # Returns correlation coefficient 
    # mat: trajectories (n trials x time)
    rlist = []
    for i in reversed(range(max_MT)):  # reversed so it goes backwards in time (matrix plotting purposes)
        trajs = mat[:, startfrom + i]
        nan_idx = ~np.isnan(trajs)
        evidence = evl[nan_idx]
        trajs = trajs[nan_idx]
        r, p2 = pearsonr(trajs, evidence)  # p2 = pvalue from pearson corr
        rlist.append(r)
    return rlist


def corr_rt_time_prior(df, fig, ax, data_folder, rtbins=np.linspace(0, 150, 16, dtype=int),
                       trajectory='trajectory_y', threshold=300):
    """
    Computes and plots the correlation for all time-points between position and prior evidence
    for different bins of RT.
    """
    # split time/subject by prior
    cmap = mtp.colors.LinearSegmentedColormap.from_list("", ["chocolate", "white", "olivedrab"])
    kw = {"trajectory": trajectory, "align": "sound"}
    zt = df.allpriors.values
    # initialize matrix
    out_data = np.empty((400, len(rtbins)-1, 15))
    out_data[:] = np.nan
    df_1 = df.copy()
    split_data = data_folder + 'prior_matrix.npy'
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(split_data), exist_ok=True)
    if os.path.exists(split_data):
        out_data = np.load(split_data, allow_pickle=True)
    else:
        for i_s, subject in enumerate(df_1.subjid.unique()):  # for each subject
            for i in range(rtbins.size-1):  # for each RT bin
                dat = df_1.loc[(df_1.subjid == subject) &
                            (df_1.sound_len < rtbins[i + 1]) &
                            (df_1.sound_len >= rtbins[i]) &
                            (~np.isnan(zt))]
                ztl = zt[(df_1.subjid == subject) &
                        (df_1.sound_len < rtbins[i + 1]) &
                        (df_1.sound_len >= rtbins[i]) &
                        (~np.isnan(zt))]
                # extract interpolated trajectories
                mat = np.vstack(
                    dat.apply(lambda x: fp.interpolapply(x, **kw), axis=1).values.tolist())
                ztl = ztl[~np.isnan(mat).all(axis=1)]
                mat = mat[~np.isnan(mat).all(axis=1)]
                # get correlation coefficient between trajectory and prior evidence
                corr_coef = get_corr_coef(mat, ztl, pval=0.01, max_MT=400,
                                          startfrom=700)
                # save corr_coef
                out_data[:, i, i_s] = corr_coef
        # save matrix for each subject
        np.save(split_data, out_data) 
    # average across subjects
    r_coef_mean = np.nanmean(out_data, axis=2)
    # tune panel
    ax.set_title('Prior-position \ncorrelation', fontsize=12)
    ax.plot([0, 14], [0, 150], color='k', linewidth=2)
    # plot matrix
    im = ax.imshow(r_coef_mean, aspect='auto', cmap=cmap,
                   vmin=-0.5, vmax=0.5, extent=[0, 14, 0, 304])
    # tune panel
    ax.set_xlabel('Reaction time (ms)')
    pos = ax.get_position()
    ax.set_xticks([0, 4, 9, 14], [rtbins[0], rtbins[5], rtbins[10], rtbins[15]])
    pright_cbar_ax = fig.add_axes([pos.x0+pos.width,
                                   pos.y0 + pos.height/10,
                                   pos.width/20, pos.height/1.3])
    cbar = plt.colorbar(im, cax=pright_cbar_ax)
    cbar.ax.set_title('Corr.\ncoef.')


def corr_rt_time_stim(df, ax, split_data_all_s, data_folder, rtbins=np.linspace(0, 150, 16, dtype=int),
                      trajectory='trajectory_y', threshold=300):
    """
    Computes and plots the correlation for all time-points between position and stimulus evidence
    for different bins of RT.
    """
    # split time/subject by prior
    cmap = mtp.colors.LinearSegmentedColormap.from_list("", ["chocolate", "white", "olivedrab"])
    # initialize matrix
    out_data = np.empty((400, len(rtbins)-1, 15))
    out_data[:] = np.nan
    splitfun = get_splitting_mat_data
    df_1 = df.copy()
    evs = [0, 0.25, 0.5, 1]
    split_data = data_folder + 'stim_matrix.npy'
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(split_data), exist_ok=True)
    if os.path.exists(split_data):
        out_data = np.load(split_data, allow_pickle=True)
    else:
        for i_s, subject in enumerate(df_1.subjid.unique()):  # for each subject
            for i in range(rtbins.size-1):  # for each RT bin
                for iev, ev in enumerate(evs):  # for each stim. strength
                    # extract trajectories
                    matatmp =\
                        splitfun(df=df.loc[(df.special_trial == 0)
                                           & (df.subjid == subject)],
                                 side=0,
                                 rtbin=i, rtbins=rtbins, coh1=ev,
                                 trajectory=trajectory, align="sound")
                    if iev == 0:
                        mat = matatmp
                        evl = np.repeat(0, matatmp.shape[0])
                    else:
                        mat = np.concatenate((mat, matatmp))
                        evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
                # get correlation coefficient between trajectory and stimulus evidence
                corr_coef = get_corr_coef(mat, evl, pval=0.05, max_MT=400,
                                          startfrom=700)
                # save corr_coef
                out_data[:, i, i_s] = corr_coef
        # save matrix for each subject
        np.save(split_data, out_data)
    # average across subjects
    r_coef_mean = np.nanmean(out_data, axis=2)
    # tune panel
    ax.set_title('Stimulus-position \ncorrelation', fontsize=12)
    ax.plot([0, 14], [0, 150], color='k', linewidth=2)
    # plot splitting time (significant correlation)
    ax.plot(np.arange(len(split_data_all_s)),
            split_data_all_s, color='firebrick', linewidth=1.4, alpha=0.5)
    # plot correlation matrix
    ax.imshow(r_coef_mean, aspect='auto', cmap=cmap,
              vmin=-0.5, vmax=0.5, extent=[0, 14, 0, 304])
    # tune panel
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylim(0, 304)
    ax.set_ylabel('Time from stimulus onset (ms)')
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xticks([0, 4, 9, 14], [rtbins[0], rtbins[5], rtbins[10], rtbins[15]])


def get_splitting_mat_data(df, side, rtbin=0, rtbins=np.linspace(0, 150, 7),
                           align='movement', trajectory="trajectory_y",
                           coh1=1):
    """
    Create matrix that will be used to compute splitting time.
    
    df= dataframe
    side= {0,1} left or right,
    rtbins = defined bins of RT
    startfrom= index to start checking diffs, for rats, 700 is the 0 in movement;
    align: whether to align 0 to movement(action) or sound onset
    """
    kw = {"trajectory": trajectory}

    # get matrices
    if side == 0:
        coh1 = -coh1
    else:
        coh1 = coh1
    # get df for the selected RT bin
    dat = df.loc[(df.sound_len < rtbins[rtbin + 1]) & (df.sound_len >= rtbins[rtbin])]
    if align == 'movement':
        kw["align"] = "action"
    elif align == 'sound':
        kw["align"] = "sound"
    # interpolate trajectories for each side and flip one to get everything on same positive space
    idx = (dat.coh2 == coh1) & (dat.rewside == 0)
    mata_0 = np.vstack(dat.loc[idx].apply(lambda x: fp.interpolapply(x, **kw), axis=1).values.tolist())
    idx = (dat.coh2 == -coh1) & (dat.rewside == 1)
    mata_1 = np.vstack(dat.loc[idx].apply(lambda x: fp.interpolapply(x, **kw), axis=1).values.tolist())
    mata = np.vstack([mata_0*-1, mata_1])
    # exclude NaNs
    mata = mata[~np.isnan(mata).all(axis=1)]
    return mata


def get_splitting_mat_simul(df, side, rtbin=0, rtbins=np.linspace(0, 150, 7),
                            align='movement', coh=1, flip=True, pad_end=True):
    """
    Create matrix that will be used to compute splitting time for simulation data.
    """
    def shortpad2(row, upto=1400, align='movement', pad_value=np.nan,
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
    
    # get matrices
    if coh is not None:
        if side == 0:
            coh1 = -coh
        else:
            coh1 = coh
    shortpad_kws = {}
    if align == 'sound':
        lentraj = []
        for traj in df.traj:
            lentraj.append(len(lentraj))
        shortpad_kws = dict(upto=max(lentraj)+int(max(df.sound_len))+1,
                            align='sound')
    dat = df.loc[(df.sound_len < rtbins[rtbin + 1])& (df.sound_len >= rtbins[rtbin])]
    if coh is not None:
        idx = (dat.coh2 == coh1) & (dat.rewside == 0)
        mata_0 = np.vstack(dat.loc[idx].apply(
        lambda row: shortpad2(row, **shortpad_kws), axis=1).values.tolist())
        idx = (dat.coh2 == -coh1) & (dat.rewside == 1)
        mata_1 = np.vstack(dat.loc[idx].apply(
        lambda row: shortpad2(row, **shortpad_kws), axis=1).values.tolist())
        if pad_end:
            for mat in [mata_0, mata_1]:
                for i_t, t in enumerate(mat):
                    ind_last_val = np.where(t == t[~np.isnan(t)][-1])[0][0]
                    mat[i_t, ind_last_val:-1] = np.repeat(t[ind_last_val],
                                                          len(t)-ind_last_val-1)
        mata = np.vstack([mata_0*((-1)**flip), mata_1])
    if coh is None:
        mata = np.vstack(dat.apply(
        lambda row: shortpad2(row, **shortpad_kws), axis=1).values.tolist())
    # discard all nan rows
    idx_nan = ~np.isnan(mata).all(axis=1)
    mata = mata[idx_nan]
    if coh is None:
        return mata, idx_nan
    else:
        return mata


def plot_trajs_splitting_example(df, ax, rtbin=0, rtbins=np.linspace(0, 150, 2),
                                 subject='LE37', xlabel='', ylabel='', show_legend=False,
                                 startfrom=700, fix_per_offset_subtr=50):
    """
    Plot trajectories depending on COH and the corresponding Splitting Time as arrow,
    Panel 2h.
    """
    assert startfrom == 700, 'startfrom must be 700, which is the stimulus onset'
    indx = (df.special_trial == 0) & (df.subjid == subject)
    assert np.sum(indx) > 0, 'No trials for subject ' + subject + ' with special_trial == 0'
    lbl = 'RTs: ['+str(rtbins[rtbin])+'-'+str(rtbins[rtbin+1])+']'
    evs = [0, 0.25, 0.5, 1]
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
    medians = []
    for iev, ev in enumerate(evs):
        matatmp =\
            get_splitting_mat_data(df=df[indx], side=0, rtbin=rtbin,
                                    rtbins=rtbins, coh1=ev, align='sound')
        median_plt = np.nanmedian(matatmp, axis=0) -\
                np.nanmedian(matatmp[:,
                                     startfrom-fix_per_offset_subtr:
                                         startfrom+int(rtbins[rtbin])])
        ax.plot(np.arange(matatmp.shape[1]) - startfrom,
                median_plt, color=colormap[iev], label=lbl)
        medians.append(median_plt)

        if iev == 0:
            mat = matatmp
            evl = np.repeat(0, matatmp.shape[0])
        else:
            mat = np.concatenate((mat, matatmp))
            evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
    ind = get_split_ind_corr(mat, evl, pval=0.05, max_MT=400, startfrom=startfrom)
    ind_y = np.max([m[ind+startfrom] for m in medians])
    ax.set_xlim(-10, 255)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([-0.5, 4])
    
    # plot horizontal line
    ax.axhline(0, color='k', lw=0.5, ls='--')
    col_face = [0.7, 0.7, 0.7, 0.4]
    col_edge = [0.7, 0.7, 0.7, 0]
    if rtbins[1] == 300:
        ax.set_title('RT > 150 ms', fontsize=9.5)
        mean_stim_dur = 150
    if rtbins[1] == 65:
        ax.set_title('RT = 50 ms', fontsize=9.5)
        mean_stim_dur = 50
    if rtbins[1] == 15:
        ax.set_title('RT < 15 ms', fontsize=9.5)
        mean_stim_dur = 15
    ax.fill_between([0, mean_stim_dur], [3.3, 3.3], [4, 4],
                     facecolor=col_face, edgecolor=col_edge)
    # plot arrow
    al = 0.5
    hl = 0.4
    ax.arrow(ind, ind_y+al+3*hl, 0, -al-hl,  color='k', width=1, head_width=8,
             head_length=hl)
    ax.axvline(mean_stim_dur, color='k', linestyle='--', linewidth=0.8,
               alpha=0.6)
    if rtbins[1] == 300:
        ax.text(ind-20, ind_y+al+3.7*hl, 'Splitting\ntime', fontsize=8.5)
        ax.text(35, 3.5, 'Stimulus', fontsize=8.5)
        ax.text(mean_stim_dur-30, 1.22, 'mvmt.', fontsize=8.5,
                rotation='vertical', style='italic')
    if show_legend:
        labels = ['0', '0.25', '0.5', '1']
        legendelements = []
        for i_l, lab in enumerate(labels[::-1]):
            legendelements.append(Line2D([0], [0], color=colormap[::-1][i_l], lw=2,
                                  label=lab))
        ax.legend(handles=legendelements, fontsize=8, loc='lower right',
                  labelspacing=0.1, frameon=False, bbox_to_anchor=(1.2, 0.12),
                  title='Stimulus\nstrength')
    conv_factor = 0.07
    ticks = np.array([0, 0.1, 0.2]) / conv_factor
    labs = np.round(ticks*conv_factor, 2)
    ax.set_yticks(ticks, labs)


def retrieve_trajs(df, rtbins=np.linspace(0, 150, 16),
                   rtbin=0, align='sound', trajectory='trajectory_y',
                   flip=True):
    """
    Function to extract trajectories from dataframe from a specific RT bin.
    """
    kw = {"trajectory": trajectory}
    dat = df.loc[
        (df.sound_len < rtbins[rtbin + 1])
        & (df.sound_len >= rtbins[rtbin])
    ]
    if align == 'movement':
        kw["align"] = "action"
    elif align == 'sound':
        kw["align"] = "sound"
    mata = np.vstack(dat.apply(lambda x: fp.interpolapply(x, **kw), axis=1).values.tolist())
    if flip:
        mata = mata * (dat.rewside.values*2-1).reshape(-1, 1)
    index_nan = ~np.isnan(mata).all(axis=1)
    mata = mata[index_nan]
    return mata, index_nan


def trajs_splitting_stim(df, ax, data_folder, collapse_sides=True, threshold=300,
                         sim=False,
                         rtbins=np.linspace(0, 150, 16), connect_points=False,
                         trajectory="trajectory_y", p_val=0.05, extra_label=''):
    """
    Panel 2i, plots splitting time for each subject and the median.
    This function can be used for data (sim=False) and simulations (sim=True).
    """
    # split time/subject by coherence
    if sim:
        splitfun = get_splitting_mat_simul
        df['traj'] = df.trajectory_y.values
    if not sim:
        splitfun = get_splitting_mat_data
    out_data = []
    for subject in df.subjid.unique():
        out_data_sbj = []
        if not sim:
            split_data = data_folder + str(subject) + '/traj_data/' + str(subject) + '_traj_split_stim_005.npz'
        if sim:
            split_data = data_folder + str(subject) + '/sim_data/' + str(subject) + '_traj_split_stim_005_forward'+extra_label+'.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(split_data), exist_ok=True)
        if os.path.exists(split_data):
            split_data = np.load(split_data, allow_pickle=True)
            out_data_sbj = split_data['out_data_sbj']
        else:
            for i in range(rtbins.size-1):
                if collapse_sides:
                    evs = [0, 0.25, 0.5, 1]
                    for iev, ev in enumerate(evs):
                        if not sim:
                            matatmp =\
                                splitfun(df=df.loc[(df.special_trial == 0)
                                                   & (df.subjid == subject)],
                                         side=0,
                                         rtbin=i, rtbins=rtbins, coh1=ev,
                                         trajectory=trajectory, align="sound")
                        if sim:
                            matatmp =\
                                splitfun(df=df.loc[(df.special_trial == 0)
                                                   & (df.subjid == subject)],
                                         side=0, rtbin=i, rtbins=rtbins, coh=ev,
                                         align="sound")
                        
                        if iev == 0:
                            mat = matatmp
                            evl = np.repeat(0, matatmp.shape[0])
                        else:
                            mat = np.concatenate((mat, matatmp))
                            evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
                    if not sim:
                        current_split_index =\
                            get_split_ind_corr(mat, evl, pval=p_val, max_MT=400,
                                            startfrom=700)
                    if sim:
                        max_mt = 800
                        current_split_index =\
                            get_split_ind_corr(mat, evl, pval=p_val, max_MT=max_mt,
                                            startfrom=0)
                    if current_split_index >= rtbins[i]:
                        out_data_sbj += [current_split_index]
                    else:
                        out_data_sbj += [np.nan]
                else:
                    for j in [0, 1]:  # side values
                        current_split_index, _, _ = splitfun(
                            df.loc[df.subjid == subject], j,
                            rtbin=i, rtbins=rtbins, align='sound')
                        out_data_sbj += [current_split_index]
            np.savez(split_data, out_data_sbj=out_data_sbj)
        out_data += [out_data_sbj]
    out_data = np.array(out_data).reshape(
        df.subjid.unique().size, rtbins.size-1, -1)
    # set axes: rtbins, subject, sides
    out_data = np.swapaxes(out_data, 0, 1)
    # change the type so we can have NaNs
    out_data = out_data.astype(float)

    out_data[out_data > threshold] = np.nan

    binsize = rtbins[1]-rtbins[0]

    scatter_kws = {'color': (.6, .6, .6, .3), 'edgecolor': (.6, .6, .6, 1)}
    if collapse_sides:
        nrepeats = df.subjid.unique().size  # n subjects
    else:
        nrepeats = df.subjid.unique().size * 2  # two responses per subject
    if not connect_points:
        ax.scatter(  # add some offset/shift on x axis based on binsize
            binsize/2 + binsize * (np.repeat(
                np.arange(rtbins.size-1), nrepeats
            ) + np.random.normal(loc=0, scale=0.02, size=out_data.size)),  # jitter
            out_data.flatten(),
            **scatter_kws,
        )
    else:
        for i in range(df.subjid.unique().size):
            for j in range(out_data.shape[2]):
                ax.plot(
                    binsize/2 + binsize * np.arange(rtbins.size-1),
                    out_data[:, i, j],
                    marker='o', mfc=(.6, .6, .6, .3), mec=(.6, .6, .6, 1),
                    mew=1, color=(.6, .6, .6, .3)
                )

    error_kws = dict(ecolor='firebrick', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='firebrick', marker='o', label='mean & SEM')
    ax.errorbar(
        binsize/2 + binsize * np.arange(rtbins.size-1),
        # we do the mean across rtbin axis
        np.nanmedian(out_data.reshape(rtbins.size-1, -1), axis=1),
        yerr=sem(out_data.reshape(rtbins.size-1, -1),
                 axis=1, nan_policy='omit'),
        **error_kws
    )
    ax.plot([0, 155], [0, 155], color='k')
    ax.fill_between([0, 155], [0, 155], [0, 0],
                    color='grey', alpha=0.2)
    ax.set_xlim(-5, 155)
    ax.set_ylim(-1, 305)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylabel('Splitting time (ms)')
    return np.nanmedian(out_data.reshape(rtbins.size-1, -1), axis=1)


def fig_2_trajs(df, rat_nocom_img, data_folder, sv_folder, st_cartoon_img, fgsz=(8, 12),
                inset_sz=.1, marginx=-.04, marginy=0.1, subj='LE46'):
    """
    Altogether, plots figure 2 with all panels.
    """
    f, ax = plt.subplots(4, 3, figsize=fgsz)
    # add letters to panels
    letters = 'abcdehfgXij'
    ax = ax.flatten()
    for lett, a in zip(letters, ax):
        if lett != 'X' and lett != 'h' and lett != 'j':
            fp.add_text(ax=a, letter=lett, x=-0.1, y=1.2)
        if lett == 'h':
            fp.add_text(ax=a, letter=lett, x=-0.1, y=.7)
        if lett == 'j':
            fp.add_text(ax=a, letter=lett, x=-0.1, y=1.34)
    ax[8].axis('off')
    # adjust panels positions
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.5, wspace=0.4)
    factor = 1.
    for i_ax in [3, 4, 6, 7]:
        pos = ax[i_ax].get_position()
        if i_ax in [3, 6]:
            ax[i_ax].set_position([pos.x0, pos.y0, pos.width*factor,
                                    pos.height])
        else:
            ax[i_ax].set_position([pos.x0+pos.width/8, pos.y0, pos.width*factor,
                                    pos.height])
    # add insets
    ax = f.axes
    ax_zt = np.array([ax[3], ax[6]])
    ax_cohs = np.array([ax[4], ax[7]])
    ax_inset = fp.add_inset(ax=ax_cohs[1], inset_sz=inset_sz, fgsz=fgsz,
                            marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    # ax_cohs contains in this order the axes for:
    # index 0: mean position of rats conditioned on stim. evidence,
    # index 1: the inset for the velocity panel (peak)
    # index 2: mean velocity  of rats conditioned on stim. evidence
    ax_cohs = np.insert(ax_cohs, 1, ax_inset)
    ax_inset = fp.add_inset(ax=ax_zt[1], inset_sz=inset_sz, fgsz=fgsz,
                            marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    ax_zt = np.insert(ax_zt, 1, ax_inset)
    # ax_zt contains in this order the axes for:
    # index 0: mean position of rats conditioned on prior evidence,
    # index 1: the inset for the velocity panel  (peak)
    # index 2: mean velocity  of rats conditioned on priors evidence
    ax_weights = ax[2]
    pos = ax_weights.get_position()
    ax_weights.set_position([pos.x0, pos.y0+pos.height/4, pos.width,
                             pos.height*1/2])
    for i_a, a in enumerate(ax):
        if i_a != 8:
            fp.rm_top_right_lines(a)
    margin = 0.05
    # tune screenshot panel
    ax_scrnsht = ax[0]
    ticks = np.array([0, 206])
    conv_factor = 5.25/180
    labs = np.int64(np.round(ticks*conv_factor, 2))
    ax_scrnsht.set_xticks(ticks, labs)
    right_port_y = 70
    center_port_y = 250
    left_port_y = 440
    conv_factor = 0.07
    ticks = np.array([-5, 0, 5])/conv_factor
    labs = np.int64(np.round(ticks*conv_factor, 2))
    ticks_scrn = [78, center_port_y, 432]
    ax_scrnsht.set_yticks(ticks_scrn, labs)
    ax_scrnsht.set_xlabel('x dimension (cm)')
    ax_scrnsht.set_ylabel('y dimension (cm)')
    # add colorbar for screenshot
    n_stps = 100
    pos = ax_scrnsht.get_position()
    ax_clbr = plt.axes([pos.x0+margin/2, pos.y0+pos.height+margin/8,
                        pos.width*0.7, pos.height/15])
    ax_clbr.imshow(np.linspace(0, 1, n_stps)[None, :], aspect='auto')
    x_tcks = np.linspace(0, n_stps, 5)
    ax_clbr.set_xticks(x_tcks)
    x_tcks_str = ['0', '', '', '', str(int(2.5*n_stps))]
    x_tcks_str[-1] += ' ms'
    ax_clbr.set_xticklabels(x_tcks_str)
    ax_clbr.tick_params(labelsize=8)
    ax_clbr.set_yticks([])
    ax_clbr.xaxis.set_ticks_position("top")
    # tune trajectories panels
    ax_rawtr = ax[1]
    ax_ydim = ax[2]
    x_lim = [-80, 20]
    y_lim = [-100, 100]
    ax_rawtr.set_xlim(x_lim)
    ax_rawtr.set_ylim(y_lim)
    ticks = np.array([-80, -20])
    conv_factor = 0.07
    labs = [0, 6]
    ax_rawtr.set_xticks(ticks, labs)
    conv_factor = 0.07
    ticks = np.array([-5, 0, 5])/conv_factor
    labs = np.int64(np.round(ticks*conv_factor, 2))
    ax_rawtr.set_yticks(ticks, labs)
    ax_rawtr.set_xlabel('x dimension (cm)')
    ax_rawtr.set_ylabel('y dimension (cm)')
    pos_coh = ax_cohs[2].get_position()
    pos_rawtr = ax_rawtr.get_position()
    ax_rawtr.set_position([pos_coh.x0, pos_rawtr.y0,
                           pos_rawtr.width/1.3, pos_rawtr.height])
    ax_rawtr.text(x=0.4, y=1., s='rat LE46', transform=ax_rawtr.transAxes,
                  fontsize=10, va='top', ha='right')
    x_lim = [-100, 800]
    y_lim = [-100, 100]
    ax_ydim.set_xlim(x_lim)
    ax_ydim.set_ylim(y_lim)
    ax_ydim.set_yticks([])
    ax_ydim.set_xlabel('Time from movement onset (ms)')
    pos_ydim = ax_ydim.get_position()
    ax_ydim.set_position([pos_ydim.x0, pos_rawtr.y0,
                          pos_ydim.width, pos_rawtr.height])
    ax_ydim.text(x=0.32, y=1., s='rat LE46', transform=ax_ydim.transAxes,
                 fontsize=10, va='top', ha='right')
    # tune splitting time panels
    factor_y = 0.622
    factor_x = 0.8
    ax_cartoon = ax[5]
    ax_cartoon.axis('off')
    pos = ax_cartoon.get_position()
    ax_cartoon.set_position([pos.x0+pos.width/8, pos.y0+pos.height*factor_y*0.9,
                              pos.width*0.9, pos.height*0.9])
    ax_top = plt.axes([.1, .1, .1, .1])
    ax_middle = plt.axes([.2, .2, .1, .1])
    ax_bottom = plt.axes([.3, .3, .1, .1])
    for i_a, a in enumerate([ax_top, ax_middle, ax_bottom]):
        a.set_position([pos.x0+pos.width/4, 0.06+pos.y0-(i_a)*pos.height*factor_y*1.5,
                        pos.width*factor_x, pos.height*factor_y])
        fp.rm_top_right_lines(a)
    # move ax[10] (spllting time coherence) to the right
    pos = ax[10].get_position()
    ax[10].set_position([pos_coh.x0, pos.y0+pos.height/12,
                         pos.width*0.9, pos.height*0.9])
    # TRACKING SCREENSHOT
    rat = plt.imread(rat_nocom_img)
    img = rat[80:576, 120:-10, :]
    ax_scrnsht.imshow(np.flipud(img)) # rat.shape = (796, 596, 4)
    ax_scrnsht.axhline(y=left_port_y, linestyle='--', color='k', lw=.5)
    ax_scrnsht.axhline(y=right_port_y, linestyle='--', color='k', lw=.5)
    ax_scrnsht.axhline(center_port_y, color='k', lw=.5)
    ax_scrnsht.set_ylim([0, img.shape[0]])


    # TRAJECTORIES
    df_subj = df[df.subjid == subj]
    ran_max = 100
    for tr in range(ran_max):
        if tr > (ran_max/2):
            trial = df_subj.iloc[tr]
            traj_x = trial['trajectory_x']
            traj_y = trial['trajectory_y']
            ax_rawtr.plot(traj_x, traj_y, color='grey', lw=.5, alpha=0.6)
            time = trial['time_trajs']
            ax_ydim.plot(time, traj_y, color='grey', lw=.5, alpha=0.6)
    # plot dashed lines
    for i_a in [1, 2]:
        ax[i_a].axhline(y=75, linestyle='--', color='k', lw=.5)
        ax[i_a].axhline(y=-75, linestyle='--', color='k', lw=.5)
        ax[i_a].axhline(0, color='k', lw=.5)

    df_trajs = df.copy()
    # TRAJECTORIES CONDITIONED ON PRIOR
    plots_trajs_conditioned(df=df_trajs.loc[df_trajs.special_trial == 2],
                            ax=ax_zt, data_folder=data_folder,
                            condition='choice_x_prior',
                            prior_limit=1, cmap='copper')
    # TRAJECTORIES CONDITIONED ON COH
    plots_trajs_conditioned(df=df_trajs, ax=ax_cohs,
                            data_folder=data_folder,
                            condition='choice_x_coh',
                            prior_limit=0.1,  # 10% quantile
                            cmap='coolwarm')
    # SPLITTING TIME EXAMPLE
    plot_trajs_splitting_example(df, ax=ax_top, rtbins=np.linspace(150, 300, 2))
    plot_trajs_splitting_example(df, ax=ax_bottom, rtbins=np.linspace(0, 15, 2),
                                 xlabel='Time from stimulus onset (ms)', show_legend=True)
    plot_trajs_splitting_example(df, ax=ax_middle, rtbins=np.linspace(45, 65, 2),
                                  ylabel='y-position (cm)')
    # TRAJECTORY SPLITTING STIMULUS
    split_data_all_s = trajs_splitting_stim(df=df, data_folder=data_folder, ax=ax[9],
                                            connect_points=True, p_val=0.01)
    corr_rt_time_stim(df=df, split_data_all_s=split_data_all_s,
                      ax=ax[10], data_folder=data_folder)
    corr_rt_time_prior(df=df, fig=f, ax=ax[11], data_folder=data_folder)
    pos = ax[11].get_position()
    ax[11].set_position([pos.x0, pos.y0+pos.height/12,
                         pos.width*0.9, pos.height*0.9])
    pos = ax[10].get_position()
    pos9 = ax[9].get_position()
    ax[9].set_position([pos9.x0, pos.y0, pos9.width, pos9.height])
    f.savefig(sv_folder+'/Fig2.png', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'/Fig2.svg', dpi=400, bbox_inches='tight')
