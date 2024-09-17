import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pylab as pl
import seaborn as sns
from scipy import interpolate
from scipy.stats import sem
import figures_paper as fp
import figure_2 as fig_2
import figure_3 as fig_3
import figure_1 as fig_1
import analyses_humans as ah


# ---GLOBAL VARIABLES
VAR_INC = fig_1.VAR_INC
VAR_CON = fig_1.VAR_CON
VAR_INC_SHORT = fig_1.VAR_INC_SHORT
VAR_CON_SHORT = fig_1.VAR_CON_SHORT
VAR_L = fig_1.VAR_L
VAR_R = fig_1.VAR_R


# ---FUNCTIONS
def plot_coms(df, ax, human=False):
    """
    Plots some reversal and non-reversal trajectories.
    """
    coms = df.CoM_sugg.values
    decision = df.R_response.values
    if human:
        ran_max = 600
        max_val = 600
    if not human:
        ran_max = 400
        max_val = 77
    for tr in reversed(range(ran_max)):
        if tr > (ran_max/2) and not coms[tr] and decision[tr] == 1:
            trial = df.iloc[tr]
            traj = trial['trajectory_y']
            if not human:
                time = df.time_trajs.values[tr]
                ax.plot(time, traj, color=fig_3.COLOR_NO_COM, lw=.5)
                ax.set_xlim(-100, 800)
            if human:
                time = np.array(trial['times'])
                if time[-1] < 0.3 and time[-1] > 0.1:
                    ax.plot(time*1e3, traj, color=fig_3.COLOR_NO_COM, lw=.5)
        elif tr < (ran_max/2-1) and coms[tr] and decision[tr] == 0:
            trial = df.iloc[tr]
            traj = trial['trajectory_y']
            if not human:
                time = df.time_trajs.values[tr]
                ax.plot(time, traj, color=fig_3.COLOR_COM, lw=2)
                ax.set_xlim(-100, 800)
            if human:
                time = np.array(trial['times'])
                if time[-1] < 0.3 and time[-1] > 0.2:
                    ax.plot(time*1e3, traj, color=fig_3.COLOR_COM, lw=2)
    fp.rm_top_right_lines(ax)
    if human:
        var = 'x'
    if not human:
        var = 'y'
    ax.set_ylabel('{} position (cm)'.format(var))
    ax.set_xlabel('Time from movement \n onset (ms)')
    ax.axhline(y=max_val, linestyle='--', color='Green', lw=1)
    ax.axhline(y=-max_val, linestyle='--', color='Purple', lw=1)
    ax.axhline(y=0, linestyle='--', color='k', lw=0.5)
    legendelements = [Line2D([0], [0], color=fig_3.COLOR_COM, lw=2,
                             label='reversal'),
                      Line2D([0], [0], color=fig_3.COLOR_NO_COM, lw=2,
                             label='No-reversal')]
    ax.legend(handles=legendelements, loc='upper left', borderpad=0.15,
              labelspacing=0.15, bbox_to_anchor=(0, 1.18), handlelength=1.5,
              frameon=False)
    if human:
        factor = 0.0096  # cm/px
        yticks = np.array([-6, -4, -2, 0, 2, 4, 6])/factor
        ax.set_yticks(yticks, np.int64(np.round(yticks*factor, 2)))


def com_statistics_humans(peak_com, time_com, ax, mean_mt):
    """
    Plots reversal point and reversal time distributions.
    """
    ax1, ax2 = ax
    fp.rm_top_right_lines(ax1)
    fp.rm_top_right_lines(ax2)
    ax1.hist(peak_com[peak_com != 0]/600*100, bins=67, range=(-100, -16.667),
             color=fig_3.COLOR_COM)
    ax1.hist(peak_com[peak_com != 0]/600*100, bins=14, range=(-16.667, 0),
             color=fig_3.COLOR_NO_COM)
    ax1.set_yscale('log')
    ax1.axvline(-100/6, linestyle=':', color='r')
    ax1.set_xlim(-100, 1)
    ax1.set_xlabel('Reversal point (%)')
    ax1.set_ylabel('# Trials')
    ax2.set_ylabel('# Trials')
    ax2.hist(time_com[time_com != -1]*1e3, bins=30, range=(0, 510),
             color=fig_3.COLOR_COM)
    ax2.set_xlabel('Reversal time (ms)')


def mean_com_traj_human(df_data, ax, max_mt=400):
    """
    Plots mean reversal and non-reversal trajectory across subjects.
    """
    # TRAJECTORIES
    fp.rm_top_right_lines(ax=ax)
    index1 = df_data.subjid > -1  # all subs
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    decision = df_data.R_response.values[index1]
    trajs = df_data.trajectory_y.values[index1]
    times = df_data.times.values[index1]
    com = df_data.CoM_sugg.values[index1]
    precision = 16
    mat_mean_trajs_subjs = np.empty((len(df_data.subjid.unique()), max_mt))
    mat_mean_trajs_subjs[:] = np.nan
    for i_s, subj in enumerate(df_data.subjid.unique()):
        index = com & (df_data.subjid.values[index1] == subj)
        all_trajs = np.empty((sum(index), max_mt))
        all_trajs[:] = np.nan
        for tr in range(sum(index)):
            vals = np.array(trajs[index][tr]) * (decision[index][tr]*2 - 1)
            ind_time = [True if t != '' else False for t in times[index][tr]]
            time = np.array(times[index][tr])[np.array(ind_time)].astype(float)
            max_time = max(time)*1e3
            if max_time > max_mt:
                continue
            all_trajs[tr, :len(vals)] = vals
            all_trajs[tr, len(vals):-1] = np.repeat(vals[-1],
                                                    int(max_mt-len(vals)-1))
        mean_traj = np.nanmean(all_trajs, axis=0)
        xvals = np.arange(len(mean_traj))*precision
        yvals = mean_traj
        ax.plot(xvals, yvals, color=fig_3.COLOR_COM, alpha=0.1)
        mat_mean_trajs_subjs[i_s, :] = yvals
    mean_traj_across_subjs = np.nanmean(mat_mean_trajs_subjs, axis=0)
    ax.plot(xvals, mean_traj_across_subjs, color=fig_3.COLOR_COM, linewidth=2)
    index = ~com
    all_trajs = np.empty((sum(index), max_mt))
    all_trajs[:] = np.nan
    for tr in range(sum(index)):
        vals = np.array(trajs[index][tr]) * (decision[index][tr]*2 - 1)
        ind_time = [True if t != '' else False for t in times[index][tr]]
        time = np.array(times[index][tr])[np.array(ind_time)].astype(float)
        max_time = max(time)*1e3
        if max_time > max_mt:
            continue
        all_trajs[tr, :len(vals)] = vals
        all_trajs[tr, len(vals):-1] = np.repeat(vals[-1],
                                                int(max_mt-len(vals)-1))
    mean_traj = np.nanmean(all_trajs, axis=0)
    xvals = np.arange(len(mean_traj))*precision
    yvals = mean_traj
    ax.plot(xvals, yvals, color=fig_3.COLOR_NO_COM, linewidth=2)
    ax.set_xlabel('Time from movement \n onset (ms)')
    ax.set_ylabel('x position (cm)')
    legendelements = [Line2D([0], [0], color=fig_3.COLOR_COM, lw=2,
                             label='reversal'),
                      Line2D([0], [0], color=fig_3.COLOR_NO_COM, lw=2,
                             label='No-reversal')]
    ax.axhline(-100, color='r', linestyle=':')
    ax.set_xlim(-5, 415)
    ax.text(150, -200, 'Detection threshold', color='r', fontsize=10.5)
    ax.legend(handles=legendelements, loc='upper left', borderpad=0.15,
              labelspacing=0.15, bbox_to_anchor=(0, 1.2), handlelength=1.5,
              frameon=False)
    factor = 0.0096  # cm/px
    yticks = np.array((-2, 0, 2, 4, 6))/0.0096
    ax.set_yticks(yticks, np.int64(np.round(yticks*factor, 2)))


def human_trajs_cond(congruent_coh, decision, trajs, prior, bins, times, ax,
                     n_subjects, sound_len, max_mt=400, max_px=800, rtlim=300,
                     condition='prior', interpolatespace=np.arange(500)):
    """
    Plots trajectories conditioned on stimulus and prior congruency.
    For condition = 'prior', it will plot conditioning on prior congruency.
    For any other value of condition, it will plot conditioning on stimulus.
    """
    
    if condition == 'prior':
       colormap = pl.cm.copper_r(np.linspace(0., 1, len(bins)-1))[::-1]
    else:
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["mediumblue","plum","firebrick"])
        colormap = colormap(np.linspace(0, 1, len(bins)))
        ev_vals = bins
        labels_stim = ['inc.', ' ', ' ', '0', ' ', ' ', 'cong.']
    mov_time_list = []
    for i_ev, ev in enumerate(bins):
        if condition == 'prior':
            if ev == 1:
                break
            index = (prior >= bins[i_ev])*(prior < bins[i_ev+1])

        else:
            index = (congruent_coh == ev) &\
                (np.abs(prior) <= np.quantile(np.abs(prior), 0.25)) &\
                (sound_len <= rtlim)
        all_trajs = np.empty((sum(index), max_mt))
        all_trajs[:] = np.nan
        for tr in range(sum(index)):
            vals = np.array(trajs[index][tr])*(decision[index][tr]*2-1)
            ind_time = [True if t != '' else False for t in times[index][tr]]
            time = np.array(times[index][tr])[np.array(ind_time)].astype(float)*1e3
            f = interpolate.interp1d(time, vals, bounds_error=False)
            vals_in = f(interpolatespace)
            vals_in = vals_in[~np.isnan(vals_in)]
            max_time = max(time)
            if max_time > max_mt:
                continue
            all_trajs[tr, :len(vals_in)] = vals_in
            all_trajs[tr, len(vals_in):-1] = np.repeat(vals[-1],
                                                       int(max_mt - len(vals_in)-1))
        mean_traj = np.nanmean(all_trajs, axis=0)
        mov_time = np.nanmean(np.array([float(t[-1]) for t in
                                        times[index]
                                        if t[-1] != '']))*1e3
        err_traj = np.nanstd(np.array([float(t[-1]) for t in
                                        times[index]
                                        if t[-1] != '']))*1e3 / np.sqrt(sum(index))
        mov_time_list.append(mov_time)
        x_val = i_ev if condition == 'prior' else ev
        if condition != 'prior':
            ax[1].errorbar(x_val, mov_time, err_traj, color=colormap[i_ev],
                           marker='o')
        if condition == 'prior':
            ax[1].errorbar(x_val, mov_time, err_traj, color=colormap[i_ev],
                           marker='o')
        xvals = np.arange(len(mean_traj))
        yvals = mean_traj
        if condition == 'prior':
            ax[0].plot(xvals[yvals <= max_px], mean_traj[yvals <= max_px],
                       color=colormap[i_ev])
        else:
            ax[0].plot(xvals[yvals <= max_px], mean_traj[yvals <= max_px],
                       color=colormap[i_ev], label='{}'.format(labels_stim[i_ev]))
    x_vals = np.arange(5) if condition == 'prior' else ev_vals
    ax[1].plot(x_vals, mov_time_list, ls='-', lw=0.5, color='k')
    ax[0].set_xlim(-0.1, 490)
    ax[0].set_ylim(-1, 620)
    ax[0].set_ylabel('x position (cm)')
    ax[0].set_xlabel('Time from movement \n onset (ms)')
    ax[1].set_xticks([])
    ax[1].set_title('MT (ms)', fontsize=10)
    if condition == 'prior':
        ax[1].set_xlabel('Prior')
        ax[1].set_xlim(-0.4, 4.4)
    else:
        ax[1].set_xlabel('Stimulus')
        ax[1].set_xlim(-1.2, 1.2)
    factor = 0.0096  # cm/px
    yticks = np.array((0, 2, 4, 6))/0.0096
    ax[0].set_yticks(yticks, np.int64(np.round(yticks*factor, 2)))


def human_trajs(df_data, ax, sv_folder, max_mt=400, max_px=800,
                interpolatespace=np.arange(500)):
    """
    Plots:
        - Human trajectories conditioned to stim and prior
        - Splitting time examples
        - Splitting time vs RT
    """
    # TRAJECTORIES
    index1 = (df_data.sound_len <= 300) &\
             (df_data.sound_len >= 0)
             # (df_data.subjid != 5) & (df_data.subjid != 6)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    coh = df_data.avtrapz.values[index1]
    decision = df_data.R_response.values[index1]
    trajs = df_data.trajectory_y.values[index1]
    times = df_data.times.values[index1]
    sound_len = df_data.sound_len.values[index1]
    prior_cong = df_data['norm_allpriors'][index1] * (decision*2 - 1)
    prior_cong = prior_cong.values
    ev_vals = np.sort(np.unique(np.round(coh, 2)))
    subjects = df_data.subjid.values[index1]
    ground_truth = (df_data.R_response.values*2-1) *\
        (df_data.hithistory.values*2-1)
    ground_truth = ground_truth[index1]
    congruent_coh = np.round(coh, 2) * (decision*2 - 1)
    # Trajs conditioned on stimulus congruency
    human_trajs_cond(congruent_coh=congruent_coh, decision=decision,
                     sound_len=sound_len,
                     trajs=trajs, prior=prior_cong, bins=ev_vals,
                     times=times, ax=ax[0:2],
                     n_subjects=len(df_data.subjid.unique()),
                     condition='stimulus', max_mt=max_mt)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["mediumblue","plum","firebrick"])
    colormap = colormap(np.linspace(0, 1, 7))[::-1]
    legendelements = [Line2D([0], [0], color=colormap[0], lw=1.3),
                      Line2D([0], [0], color=colormap[1], lw=1.3),
                      Line2D([0], [0], color=colormap[2], lw=1.3),
                      Line2D([0], [0], color=colormap[3], lw=1.3),
                      Line2D([0], [0], color=colormap[4], lw=1.3),
                      Line2D([0], [0], color=colormap[5], lw=1.3),
                      Line2D([0], [0], color=colormap[6], lw=1.3)]
    labs = ['cong', ' ', ' ', '0', ' ', ' ', 'inc']
    ax[0].legend(handles=legendelements, labels=labs, title='Stimulus', loc='center left',
                  labelspacing=0.001, bbox_to_anchor=(0.9, 1.22), fontsize=9,
                  handlelength=1.5, borderpad=0.15, handleheight=1.2,
                  handletextpad=0.2, frameon=False)
    bins = [-1, -0.5, -0.1, 0.1, 0.5, 1]
    # Trajs conditioned on prior congruency
    human_trajs_cond(congruent_coh=congruent_coh, decision=decision,
                     sound_len=sound_len,
                     trajs=trajs, prior=prior_cong, bins=bins,
                     times=times, ax=ax[2:4],
                     n_subjects=len(df_data.subjid.unique()),
                     condition='prior', max_mt=max_mt)
    colormap = pl.cm.copper_r(np.linspace(0., 1, len(bins)-1))[::-1]
    legendelements = [Line2D([0], [0], color=colormap[4], lw=1.3),
                      Line2D([0], [0], color=colormap[3], lw=1.3),
                      Line2D([0], [0], color=colormap[2], lw=1.3),
                      Line2D([0], [0], color=colormap[1], lw=1.3),
                      Line2D([0], [0], color=colormap[0], lw=1.3)]
    labs = ['cong', ' ', '0', ' ', 'inc']
    ax[2].legend(handles=legendelements, labels=labs,
                 title='Prior', loc='center left',
                 labelspacing=0.001, bbox_to_anchor=(0., 1.22), fontsize=9,
                 handlelength=1.5, borderpad=0.15, handleheight=1.2,
                 handletextpad=0.2, frameon=False)
    # extract splitting time
    out_data, rtbins = splitting_time_humans(sound_len=sound_len, coh=coh,
                                             trajs=trajs, times=times, subjects=subjects,
                                             ground_truth=ground_truth,
                                             interpolatespace=interpolatespace,
                                             max_mt=max_mt, n_rt_bins=5)
    # plot splitting time vs RT
    splitting_time_plot(sound_len=sound_len, out_data=out_data,
                        ax=ax[-1], subjects=subjects, n_rt_bins=5)
    rtbins = np.array((rtbins[0], rtbins[1], rtbins[2]))
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
    # plot splitting time examples
    splitting_time_example_human(rtbins=rtbins, ax=ax, sound_len=sound_len,
                                 ground_truth=ground_truth, coh=coh, trajs=trajs,
                                 times=times, max_mt=max_mt,
                                 interpolatespace=interpolatespace,
                                 colormap=colormap)
    # extract splitting time
    out_data, rtbins = splitting_time_humans(sound_len=sound_len, coh=coh,
                                              trajs=trajs, times=times,
                                              subjects=np.repeat(1, len(coh)),  # meta subject
                                              ground_truth=ground_truth,
                                              interpolatespace=interpolatespace,
                                              max_mt=max_mt, n_rt_bins=8)
    # plot splitting time vs RT
    splitting_time_plot(sound_len=sound_len, out_data=out_data,
                        ax=ax[-1], subjects=np.repeat(1, len(coh)),
                        color='b', n_rt_bins=8)
    legendelements = [Line2D([0], [0], color='firebrick', lw=2,
                             label='Across subjects'),
                      Line2D([0], [0], color='b', lw=2,
                             label='All subjects')]
    ax[-1].legend(handles=legendelements, loc='upper left', borderpad=0.15,
                  labelspacing=0.15, handlelength=1.5,
                  frameon=False, bbox_to_anchor=(0., 1.18))


def matrix_figure(df_data, humans, ax_tach, ax_pright, ax_mat, fig=None):
    """
    Function that plots tachometrics, p(right) matrix and p(reversal) matrices.
    """
    # plot tachometrics
    if humans:
        num = 8
        rtbins = np.linspace(0, 300, num=num)
        avtrapz_mod = np.round(df_data.avtrapz.values, 2)*5
        df_data['avtrapz_mod'] = avtrapz_mod 
        fp.tachometric(df_data, ax=ax_tach, fill_error=True, rtbins=rtbins,
                       cmap='gist_yarg', evidence='avtrapz_mod')
    else:
        fp.tachometric(df_data, ax=ax_tach, fill_error=True, cmap='gist_yarg')
    # tune tachometric panel
    ax_tach.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax_tach.set_xlabel('Reaction time (ms)')
    ax_tach.set_ylabel('Choice accuracy')
    ax_tach.set_ylim(0.3, 1.04)
    ax_tach.spines['right'].set_visible(False)
    ax_tach.spines['top'].set_visible(False)
    colormap = pl.cm.gist_gray_r(np.linspace(0.4, 1, 4))
    legendelements = [Line2D([0], [0], color=colormap[3], lw=1.5,
                             label='1'),
                      Line2D([0], [0], color=colormap[2], lw=1.5,
                             label='0.5'),
                      Line2D([0], [0], color=colormap[1], lw=1.5,
                             label='0.25'),
                      Line2D([0], [0], color=colormap[0], lw=1.5,
                             label='0')]
    ax_tach.legend(handles=legendelements, fontsize=8, labelspacing=0.01,
                   title='Stimulus strength')
    # plot Pcoms matrices
    nbins = 7
    # compute matrix for each side
    matrix_side_0 = fig_3.com_heatmap_paper_marginal_pcom_side(df=df_data, side=0)
    matrix_side_1 = fig_3.com_heatmap_paper_marginal_pcom_side(df=df_data, side=1)
    # tune panel
    vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    pcomlabel_1 = 'Left to right reversal'
    ax_mat[0].set_title(pcomlabel_1, fontsize=11.5)
    # plot matrix
    im = ax_mat[0].imshow(matrix_side_1, vmin=0, vmax=vmax, cmap='magma')
    # tune panel
    pcomlabel_0 = 'Right to left reversal'
    ax_mat[1].set_title(pcomlabel_0, fontsize=11.5)
    # plot matrix
    im = ax_mat[1].imshow(matrix_side_0, vmin=0, vmax=vmax, cmap='magma')
    # tune panel
    ax_mat[1].yaxis.set_ticks_position('none')
    if fig is not None:
        margin = 0.05
        pos = ax_mat[1].get_position()        
        cbar_ax = fig.add_axes([pos.x0+pos.width*1.09, pos.y0+margin/6,
                                pos.width/15, pos.height/1.5])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('p (reversal)', rotation=270, labelpad=14)    
    # pright matrix
    if humans:
        coh = df_data['avtrapz'].values
    else:
        coh = df_data['coh2'].values
    choice = df_data['R_response'].values
    prior = df_data['norm_allpriors'].values
    # compute matrix
    mat_pright, _ = fp.com_heatmap(prior, coh, choice, return_mat=True,
                                   annotate=False)
    mat_pright = np.flipud(mat_pright)
    # plot matrix
    im_2 = ax_pright.imshow(mat_pright, cmap='PRGn_r')
    # tune panel
    plt.sca(ax_pright)
    cbar_right = plt.colorbar(im_2, fraction=0.03, location='top')
    for t in cbar_right.ax.get_yticklabels():
        t.set_fontsize(7.5)
    cbar_right.ax.set_title('p (right response)', fontsize=10)
    cbar_right.ax.tick_params(rotation=45)
    for ax_i in [ax_pright, ax_mat[0], ax_mat[1]]:
        ax_i.set_xlabel('Prior evidence')
        ax_i.set_xticks([0, 3, 6])
        ax_i.set_xticklabels([VAR_L, '0', VAR_R])
    for ax_i in [ax_pright, ax_mat[0]]:
        ax_i.set_yticks([0, 3, 6])
        ax_i.set_yticklabels([VAR_R, '0', VAR_L])
        ax_i.set_ylabel('Stimulus evidence')
    ax_mat[1].set_yticklabels(['']*nbins)


def plot_xy(df_data, ax):
    """
    Plots raw trajectories in x-y
    """
    cont = 0
    subj_xy = 10
    index_sub = df_data.subjid == subj_xy
    ax.scatter(-500, 400, s=1100, color='grey', alpha=0.2)
    ax.scatter(500, 400, s=1100, color='grey', alpha=0.2)
    ax.scatter(0, -200, s=600, color='grey', alpha=0.8)
    for traj in range(800):
        # np.random.seed(1)
        tr_ind = np.random.randint(0, len(df_data['trajectory_y'][index_sub])-1)
        x_coord = df_data['trajectory_y'][tr_ind]
        y_coord = df_data['traj_y'][tr_ind]
        time_max = df_data['times'][tr_ind][-1]
        if time_max != '':
            if time_max < 0.3 and time_max > 0.1 and not df_data.CoM_sugg[tr_ind]:
                time = df_data['times'][tr_ind]
                ind_time = [True if t != '' else False for t in time]
                time = np.array(time)[np.array(ind_time)]
                ax.plot(x_coord, y_coord, color='grey', lw=.5, alpha=0.6)
                # ax[5].plot(time*1e3, x_coord, color='k', linewidth=0.5)
                cont += 1
        if cont == 50:
            break
    ax.set_xlabel('Position along x-axis (cm)')
    ax.set_ylabel('Position along y-axis (cm)')
    factor = 0.0096  # cm/px
    xticks = np.array([-5, 0, 5])/factor
    yticks = np.array([-2, 0, 2, 4])/factor
    ax.set_xticks(xticks, np.int64(np.round(xticks*factor, 2)))
    ax.set_yticks(yticks, np.int64(np.round(yticks*factor, 2)))


def splitting_time_plot(sound_len, out_data, ax, subjects, color='firebrick',
                        plot_sng=True, n_rt_bins=5):
    """
    Plots splitting time for all subjects and for a meta-subject.
    """
    rtbins = np.concatenate(([0], np.quantile(sound_len,
                                              [(i+1)/(n_rt_bins-1)
                                               for i in range(n_rt_bins-1)])))
    xvals = []
    for irtb, rtb in enumerate(rtbins[:-1]):
        sound_len_bin = sound_len[(sound_len >= rtb) &
                                  (sound_len < rtbins[irtb+1])]
        rtbins_window = np.median(sound_len_bin)
        xvals.append(rtbins_window)
    out_data = np.array(out_data).reshape(np.unique(subjects).size,
                                          rtbins.size-1, -1)
    out_data = np.swapaxes(out_data, 0, 1)
    out_data = out_data.astype(float)
    ax2 = ax
    if plot_sng:
        for i in range(len(np.unique(subjects))):
            for j in range(out_data.shape[2]):
                ax2.plot(xvals,
                         out_data[:, i, j],
                         marker='o', mfc=(.6, .6, .6, .3), mec=(.6, .6, .6, 1),
                         mew=1, color=(.6, .6, .6, .3))
    error_kws = dict(ecolor=color, capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color=color, marker='o', label='mean & SEM')
    
    if color == 'firebrick':
        ax2.errorbar(xvals,
                     np.nanmedian(out_data.reshape(rtbins.size-1, -1),
                                  axis=1),
                     yerr=sem(out_data.reshape(rtbins.size-1, -1),
                     axis=1, nan_policy='omit'), **error_kws)
    else:
        ax2.plot(xvals, np.nanmedian(out_data.reshape(rtbins.size-1, -1),
                                     axis=1),
                 color=color, marker='o', mfc=(1, 1, 1, 0), mec='k')
    ax2.set_xlabel('Reaction time (ms)')
    ax2.plot([0, 310], [0, 310], color='k')
    ax2.fill_between([0, 310], [0, 310], [0, 0],
                     color='grey', alpha=0.6)
    ax2.set_ylabel('Splitting time (ms)')
    fp.rm_top_right_lines(ax2)
    ax3 = ax2.twinx()
    sns.kdeplot(sound_len, color='k', ax=ax3, bw_adjust=5, fill=True)
    ax3.set_ylim([0, 0.02])
    ax3.set_xlim([0, 300])
    ax3.set_ylabel('')
    ax3.set_yticks([])
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax2.set_ylim([0, 401])


def splitting_time_example_human(rtbins, ax, sound_len, ground_truth, coh, trajs,
                                 times, max_mt, interpolatespace, colormap):
    """
    Plots splitting time examples for short (1st tercile) and long RTs (last tercile).
    """
    ev_vals = np.array([0, 0.25, 0.5, 1])
    rtbins = np.concatenate(([0], np.quantile(sound_len, [.3333])))
    labs = ['Short RT', 'Long RT']
    for i in range((rtbins.size-1)*2):
        if i >= rtbins.size-1:
            rtbins = np.concatenate(([0],
                                     np.quantile(sound_len, [.6666, 1.])))
        ax1 = ax[-3+i]
        for i_ev, ev in enumerate(ev_vals):
            index = (sound_len < rtbins[i+1]) & (sound_len >= rtbins[i]) &\
                    (np.abs(np.round(coh, 2)) == ev)
            all_trajs = np.empty((sum(index), int(max_mt+300)))
            all_trajs[:] = np.nan
            for tr in range(sum(index)):
                vals = np.array(trajs[index][tr]) * (ground_truth[index][tr])
                ind_time = [True if t != '' else False
                            for t in times[index][tr]]
                time = np.array(times[index][tr])[
                    np.array(ind_time)].astype(float)*1e3
                f = interpolate.interp1d(time, vals, bounds_error=False)
                vals_in = f(interpolatespace)
                vals_in = vals_in[~np.isnan(vals_in)]
                vals_in -= vals_in[0]
                vals_in = np.concatenate((np.zeros((int(sound_len[index][tr]))),
                                          vals_in))
                max_time = max(time)
                if max_time > max_mt:
                    continue
                all_trajs[tr, :len(vals_in)] = vals_in  # - vals[0]
                all_trajs[tr, len(vals_in):-1] =\
                    np.repeat(vals[-1], int(max_mt + 300 - len(vals_in)-1))
            if ev == 0:
                ev_mat = np.repeat(0, sum(index))
                traj_mat = all_trajs
            else:
                ev_mat = np.concatenate((ev_mat, np.repeat(ev, sum(index))))
                traj_mat = np.concatenate((traj_mat, all_trajs))
            ax1.plot(np.arange(len(np.nanmean(all_trajs, axis=0))),
                     np.nanmean(all_trajs, axis=0),
                     color=colormap[i_ev])
        ax1.set_xlim(-5, 405)
        ax1.set_ylim(-2, 400)
        ax1.set_title(labs[i], fontsize=11.5)
        ind = fig_2.get_split_ind_corr(traj_mat, ev_mat, startfrom=0,
                                       max_MT=max_mt+300, pval=0.01)
        ax1.set_xlabel('Time from stimulus \n onset (ms)')
        ax1.set_ylabel('x position (cm)')
        if i == 0:
            ax1.arrow(ind, 130, 0, -30, color='k', width=1.5, head_width=15,
                      head_length=15)
            ax1.text(ind-170, 140, 'Splitting Time', fontsize=10)
            labels = ['0', '0.25', '0.5', '1']
            legendelements = []
            for i_l, lab in enumerate(reversed(labels)):
                legendelements.append(Line2D([0], [0], color=colormap[::-1][i_l], lw=2,
                                      label=lab))
            ax1.legend(handles=legendelements, fontsize=9, loc='upper left',
                       title='Stimulus strength',
                       labelspacing=0.01, handlelength=1.5, frameon=False)
        else:
            if np.isnan(ind):
                ind = rtbins[i]
            ax1.arrow(ind, 110, 0, -65, color='k', width=1.5, head_width=15,
                      head_length=15)
            ax1.text(ind-150, 140, 'Splitting Time', fontsize=10)
        factor = 0.0096  # cm/px
        yticks = np.array((0, 1, 2, 3)) / factor
        ax1.set_yticks(yticks, np.int64(np.round(yticks*factor, 2)))


def splitting_time_humans(sound_len, coh, trajs, times, subjects, ground_truth,
                          interpolatespace, max_mt, n_rt_bins=5):
    """
    Computes splitting time for humans.
    """
    rtbins = np.concatenate(([0], np.quantile(sound_len,
                                              [(i+1)/(n_rt_bins-1)
                                               for i in range(n_rt_bins-1)])))
    split_ind = []
    ev_vals = [0, 0.25, 0.5, 1]
    for subj in np.unique(subjects):
        for i in range(rtbins.size-1):
            # fig, ax1 = plt.subplots(1)
            for i_ev, ev in enumerate(ev_vals):
                index = (sound_len < rtbins[i+1]) & (sound_len >= rtbins[i]) &\
                        (np.abs(np.round(coh, 2)) == ev) &\
                        (subjects == subj)  # & (prior <= 0.3)
                all_trajs = np.empty((sum(index), int(max_mt)))
                all_trajs[:] = np.nan
                for tr in range(sum(index)):
                    vals = np.array(trajs[index][tr]) * (ground_truth[index][tr])
                    ind_time = [True if t != '' else False
                                for t in times[index][tr]]
                    time = np.array(times[index][tr])[
                        np.array(ind_time)].astype(float)*1e3
                    f = interpolate.interp1d(time, vals, bounds_error=False)
                    vals_in = f(interpolatespace)
                    vals_in = vals_in[~np.isnan(vals_in)]
                    vals_in = vals_in - vals_in[0]
                    vals_in = np.concatenate((np.zeros((int(sound_len[index][tr]))),
                                              vals_in))
                    max_time = max(time)
                    if max_time > max_mt:
                        continue
                    all_trajs[tr, :len(vals_in)] = vals_in  # - vals[0]
                    all_trajs[tr, len(vals_in):-1] =\
                        np.repeat(vals[-1], int(max_mt - len(vals_in)-1))
                if ev == 0:
                    ev_mat = np.repeat(0, sum(index))
                    traj_mat = all_trajs
                else:
                    ev_mat = np.concatenate((ev_mat, np.repeat(ev, sum(index))))
                    traj_mat = np.concatenate((traj_mat, all_trajs))
            ind = fig_2.get_split_ind_corr(traj_mat, ev_mat, startfrom=0,
                                           max_MT=max_mt, pval=0.01)+5
            if ind < 410 and ind > rtbins[i]:
                split_ind.append(ind)
            else:
                split_ind.append(np.nan)
    out_data = np.array(split_ind)
    return out_data, rtbins


def acc_filt(df_data, acc_min=0.5, mt_max=300):
    """
    Filters subjects by accuracy and median MT.
    """
    subjects = df_data.subjid.unique()
    subs = subjects[(df_data.groupby('subjid').mean('hithistory')['hithistory']
                     > acc_min)]
    df_data = df_data.loc[df_data.subjid.isin(subs)]
    mvmt_time = np.array(fp.get_human_mt(df_data))
    df_data_mt = df_data.copy()
    df_data_mt['resp_len'] = mvmt_time
    subjects = df_data.subjid.unique()
    subs = subjects[(df_data_mt.groupby('subjid').median('resp_len')['resp_len']
                     < mt_max)]
    df_data = df_data.loc[df_data.subjid.isin(subs)]
    return df_data


def fig_4_humans(folder, human_task_img, sv_folder, nm='300',
                 max_mt=400, inset_sz=.06, marginx=0.004, marginy=0.025,
                 fgsz=(11, 13.5)):
    """
    Altogether, plots figure 4.
    Data from human subjects should be saved in a folder called human_data,
    inside the general folder.
    """
    subj = ['human_data']
    steps = [None]
    humans = True
    # retrieve data
    df_data = ah.traj_analysis(data_folder=folder,
                               subjects=subj, steps=steps, name=nm,
                               sv_folder=sv_folder)
    subs = df_data.subjid.unique()
    norm_allpriors = np.empty((0,))
    for subj in subs:
        df_1 = df_data.loc[df_data.subjid == subj]
        zt_tmp = df_1.norm_allpriors.values
        norm_allpriors = np.concatenate((norm_allpriors,
                                         zt_tmp/np.nanmax(abs(zt_tmp))))
    df_data['norm_allpriors'] = norm_allpriors
    minimum_accuracy = 0.7  # 70%
    max_median_mt = 400  # ms
    df_data = acc_filt(df_data, acc_min=minimum_accuracy, mt_max=max_median_mt)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    df_data.loc[df_data.subjid == 9, 'subjid'] = 8  # one subject did two sessions of 1h
    # create figure
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=fgsz)
    ax = ax.flatten()
    plt.subplots_adjust(top=0.95, bottom=0.09, left=0.09, right=0.95,
                        hspace=0.6, wspace=0.5)
    labs = ['', '', '',  'b', 'c', 'd', 'e', 'f', 'g', '', 'h', 'i', 'j', 'k',
            'l', '']
    for n, ax_1 in enumerate(ax):
        fp.rm_top_right_lines(ax_1)
        if n == 13:
            ax_1.text(-0.1, 3, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')
        elif n == 0:
            ax_1.text(-0.1, 1.15, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')
        elif n == 4:
            ax_1.text(-0.1, 1.2, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')     
        elif n == 14:
            ax_1.text(-0.1, 1.4, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')            
        else:
            ax_1.text(-0.1, 1.2, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')
    for i in [0, 1, 2]:
        ax[i].axis('off')
    # TASK PANEL
    ax_task = ax[0]
    pos_ax_0 = ax_task.get_position()
    # setting ax0 a bit bigger
    ax_task.set_position([pos_ax_0.x0-0.015, pos_ax_0.y0-0.02,
                          pos_ax_0.width*3.5, pos_ax_0.height+0.024])
    
    pos = ax_task.get_position()
    ax_task.set_position([pos.x0, pos.y0, pos.width, pos.height])
    task = plt.imread(human_task_img)
    ax_task.imshow(task)
    ax_task.text(0.01, 1.3, 'a', transform=ax_task.transAxes, fontsize=16,
                 fontweight='bold', va='top', ha='right')

    # changing ax x-y plot width
    pos_ax_1 = ax[2].get_position()
    ax[3].set_position([pos_ax_1.x0 + pos_ax_1.width, pos_ax_1.y0,
                        pos_ax_1.width+pos_ax_1.width/3, pos_ax_1.height])
    # plotting x-y trajectories
    plot_xy(df_data=df_data, ax=ax[3])
    # tachs and pright
    mvmt_time = np.array(fp.get_human_mt(df_data))
    df_data = df_data.loc[mvmt_time <= max_mt]
    ax_tach = ax[5]
    pos = ax_tach.get_position()
    ax_pright = ax[4]
    ax_mat = [ax[14], ax[15]]
    pos_com_0 = ax_mat[0].get_position()
    ax_mat[0].set_position([pos_com_0.x0 + pos_com_0.width*0.2, pos_com_0.y0,
                            pos_com_0.width, pos_com_0.height])
    ax_mat[1].set_position([pos_com_0.x0 + pos_com_0.width*1.4, pos_com_0.y0,
                            pos_com_0.width, pos_com_0.height])
    matrix_figure(df_data=df_data, ax_tach=ax_tach, ax_pright=ax_pright,
                  ax_mat=ax_mat, humans=humans, fig=fig)
    ax_tach.set_xticks([0, 100, 200])
    pos_com_0 = ax_mat[0].get_position()
    pos_com_1 = ax_mat[1].get_position()
    ax_mat[0].set_position([pos_com_0.x0, pos_com_1.y0,
                            pos_com_1.width, pos_com_1.height])
    # plots CoM trajectory examples
    ax_examples_com = ax[11]
    plot_coms(df=df_data, ax=ax_examples_com, human=humans)
    # prepare data for CoM peak/time distros plot
    peak_com = -df_data.com_peak.values
    time_com = df_data.time_com.values
    ax_com_stat = ax[13]
    pos = ax_com_stat.get_position()
    ax_com_stat.set_position([pos.x0, pos.y0, pos.width,
                              pos.height*2/5])
    ax_inset = plt.axes([pos.x0, pos.y0+pos.height*3.35/5, pos.width,
                         pos.height*2/5])
    ax_coms = [ax_com_stat, ax_inset]
    # CoM peak/time distributions
    mean_mt = np.median(fp.get_human_mt(df_data.loc[~df_data.CoM_sugg]))
    com_statistics_humans(peak_com=peak_com, time_com=time_com, ax=ax_coms,
                          mean_mt=mean_mt)
    # mean CoM trajectories
    mean_com_traj_human(df_data=df_data, ax=ax[12])
    # prepare axis for trajs conditioned on stim and prior
    ax_cohs = ax[7]
    ax_zt = ax[6]
    # trajs. conditioned on coh
    ax_inset = fp.add_inset(ax=ax_cohs, inset_sz=inset_sz, fgsz=(1, 1),
                         marginx=marginx, marginy=marginy, right=True)
    ax_cohs = np.insert(ax_cohs, 0, ax_inset)
    # trajs. conditioned on zt
    ax_inset = fp.add_inset(ax=ax_zt, inset_sz=inset_sz, fgsz=(1, 1),
                         marginx=marginx, marginy=marginy, right=True)
    ax_zt = np.insert(ax_zt, 0, ax_inset)
    axes_trajs = [ax_cohs[1], ax_cohs[0], ax_zt[1], ax_zt[0], ax[8],
                  ax[9], ax[10]]
    # trajectories conditioned on stim/prior, splitting time (vs RT and example)
    human_trajs(df_data, sv_folder=sv_folder, ax=axes_trajs, max_mt=max_mt)
    fig.savefig(sv_folder+'fig4.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'fig4.png', dpi=400, bbox_inches='tight')


def supp_pcom_rt(folder, sv_folder, ax, nm='300'):
    """
    Plots p(reversal) against RT.
    """
    subj = ['human_data']
    steps = [None]
    # retrieve data
    df_data = ah.traj_analysis(data_folder=folder,
                               subjects=subj, steps=steps, name=nm,
                               sv_folder=sv_folder)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    norm_allpriors = np.empty((0,))
    subs = df_data.subjid.unique()
    for subj in subs:
        df_1 = df_data.loc[df_data.subjid == subj]
        zt_tmp = df_1.norm_allpriors.values
        norm_allpriors = np.concatenate((norm_allpriors,
                                         zt_tmp/np.nanmax(abs(zt_tmp))))
    df_data['norm_allpriors'] = norm_allpriors
    minimum_accuracy = 0.7  # 70%
    max_median_mt = 400  # ms
    df_data = acc_filt(df_data, acc_min=minimum_accuracy, mt_max=max_median_mt)
    df_data.loc[df_data.subjid == 9, 'subjid'] = 8
    com_list = df_data.CoM_sugg
    reaction_time = df_data.sound_len
    ev = df_data.avtrapz
    df_plot = pd.DataFrame({'sound_len': reaction_time, 'CoM': com_list,
                            'ev': ev})
    bins = np.linspace(50, 300, 11)  # rt bins
    xpos = np.diff(bins)[0]  # rt binss
    fp.rm_top_right_lines(ax)
    fp.binned_curve(df_plot, 'CoM', 'sound_len', bins=bins,
                    xoffset=min(bins), xpos=xpos, ax=ax,
                    errorbar_kw={'marker': 'o', 'color': 'k'})
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylabel('p(reversal)')
    ax.set_ylim(0, 0.16)
    ax.get_legend().remove()
    return df_data


def inset_express(df_data, ax_inset=None, rtlim=50):
    """
    Plots p(correct) against stimulus evidence in RT <= 50 ms.
    """
    if ax_inset is None:
        fig, ax_inset = plt.subplots(1)
    fp.rm_top_right_lines(ax_inset)
    df_filt = df_data.loc[df_data.sound_len <= rtlim]
    ev_vals = np.array([0, 0.05, 0.1, 0.2])*5
    subjects = df_data.subjid.unique()
    acc_mat = np.empty((len(ev_vals), len(subjects)))
    for i_s, subj in enumerate(subjects):
        df_sub = df_filt.loc[df_filt.subjid == subj]
        for i_ev, ev in enumerate(ev_vals):
            hits = df_sub.loc[np.round(df_sub.avtrapz.abs(), 2) == ev]
            acc_mat[i_ev, i_s] = np.nanmean(hits['hithistory'])
    vals_acc = np.nanmean(acc_mat, axis=1)
    err_acc = np.nanstd(acc_mat, axis=1) / np.sqrt(len(subjects))
    ev_vals = np.array([0, 0.05, 0.1, 0.2])*5
    ax_inset.errorbar(ev_vals, vals_acc, err_acc, color='k', marker='o')
    ax_inset.set_xlabel('Stimulus strength')
    ax_inset.set_ylabel('Accuracy')


def get_opts_krnls(plot_opts, tag):
    """
    Plotting options.
    """
    opts = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
    opts['color'] = plot_opts['color'+tag]
    opts['linestyle'] = plot_opts['lstyle'+tag]
    return opts


def plot_trans_lat_weights(decays_all_ac, decays_all_ae, ax):
    """
    Plots transition kernels for all subjects, after correct and after error.
    """
    naranja = np.array((255, 127, 0))/255
    plot_opts = {'lw': 3,  'label': '', 'alpha': 0.1, 'color_ac': naranja,
                 'fntsz': 10, 'color_ae': (0, 0, 0), 'lstyle_ac': '-',
                 'lstyle_ae': '-', 'marker': ''}
    # plot_opts.update(kwargs)
    decays_ac = np.delete(decays_all_ac[0], 9, axis=0)
    decays_ac_lat = np.delete(decays_all_ac[1], 9, axis=0)
    decays_ae = np.delete(decays_all_ae[0], 9, axis=0)
    decays_ae_lat = np.delete(decays_all_ae[1], 9, axis=0)
    fntsz = plot_opts['fntsz']
    del plot_opts['fntsz']
    ax[0].set_ylabel('T++ weight', fontsize=fntsz)
    ax[0].set_xlabel('Trial lag', fontsize=fntsz)
    ax[1].set_ylabel('L+ weight', fontsize=fntsz)
    ax[1].set_xlabel('Trial lag', fontsize=fntsz)
    # After correct
    opts = get_opts_krnls(plot_opts=plot_opts, tag='_ac')
    for decay in decays_ac:
        ax[0].plot(decay, **opts)
    for decay in decays_ac_lat:
        ax[1].plot(decay, **opts)
    decays_ac = np.array(decays_ac)
    decays_ac_lat = np.array(decays_ac_lat)
    mean_decay_ac = np.mean(decays_ac, axis=0)
    mean_decay_ac_lat = np.mean(decays_ac_lat, axis=0)
    s_ac = np.std(decays_ac, axis=0)
    s_ac_lat = np.std(decays_ac_lat, axis=0)
    ax[0].errorbar(x=np.arange(6), y=mean_decay_ac, yerr=s_ac, linewidth=3,
                   color=naranja, marker='.', alpha=1)
    ax[1].errorbar(x=np.arange(6), y=mean_decay_ac_lat, yerr=s_ac_lat,
                   linewidth=3, color=naranja, marker='.', alpha=1)
    # After error
    opts = get_opts_krnls(plot_opts=plot_opts, tag='_ae')
    for decay in decays_ae:
        ax[0].plot(decay, **opts)
    for decay in decays_ae_lat:
        ax[1].plot(decay, **opts)
    decays_ae = np.array(decays_ae)
    decays_ae_lat = np.array(decays_ae_lat)
    mean_decay_ae = np.mean(decays_ae, axis=0)
    mean_decay_ae_lat = np.mean(decays_ae_lat, axis=0)
    s_ae = np.std(decays_ae, axis=0)
    s_ae_lat = np.std(decays_ae_lat, axis=0)
    ax[0].errorbar(x=np.arange(5), y=mean_decay_ae, yerr=s_ae, linewidth=3,
                   color=(0, 0, 0), marker='.', alpha=1)
    ax[1].errorbar(x=np.arange(5), y=mean_decay_ae_lat, yerr=s_ae_lat,
                   linewidth=3, color=(0, 0, 0), marker='.', alpha=1)
    for a in ax:
        a.set_xticks([0, 1, 2, 3, 4, 5])
        a.set_xticklabels(['6-10', '5', '4', '3', '2', '1'])
        a.set_xlim([-0.5, 5.5])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        ytckslbls = ['-1', '0', '1']
        a.axhline(y=0, linestyle='--', color='k', lw=0.5)
        ylims = a.get_ylim()
        ylims = [-np.max(np.abs(ylims)), np.max(np.abs(ylims))]
        a.set_ylim(ylims)
        yticks = [ylims[0], 0, ylims[1]]
        a.set_yticks(yticks)
        a.set_yticklabels(ytckslbls)


def supp_plot_weights_linear_reg(folder, sv_folder, ax, nm='300'):
    """
    Plots violins of the weights of the linear regression of MT with prior ev.,
    stimulus ev. and trial index.
    """
    subj = ['human_data']
    steps = [None]
    # retrieve data
    df_data = ah.traj_analysis(data_folder=folder,
                               subjects=subj, steps=steps, name=nm,
                               sv_folder=sv_folder)
    subs = df_data.subjid.unique()
    norm_allpriors = np.empty((0,))
    for subj in subs:
        df_1 = df_data.loc[df_data.subjid == subj]
        zt_tmp = df_1.norm_allpriors.values
        norm_allpriors = np.concatenate((norm_allpriors,
                                         zt_tmp/np.nanmax(abs(zt_tmp))))
    df_data['norm_allpriors'] = norm_allpriors
    minimum_accuracy = 0.7  # 70%
    max_median_mt = 400  # ms
    df_data = acc_filt(df_data, acc_min=minimum_accuracy, mt_max=max_median_mt)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    df_data.loc[df_data.subjid == 9, 'subjid'] = 8
    df_data_2 = df_data.copy()
    df_data_2['resp_len'] = fp.get_human_mt(df_data_2)
    df_data_2['coh2'] = df_data_2.avtrapz
    df_data_2['allpriors'] = df_data_2.norm_allpriors
    len_task = [len(df_data_2.loc[df_data_2.subjid == subject]) for subject in subs]
    trial_index = np.empty((0))
    for j in range(len(len_task)):
        trial_index = np.concatenate((trial_index, np.arange(len_task[j])+1))
    df_data_2['origidx'] = trial_index
    fig_1.mt_weights(df_data_2, ax=ax, plot=True, means_errs=False, mt=True, t_index_w=True)


def supp_human_behavior(folder, sv_folder):
    """
    Plots supplementary figure 3.
    From analyses_humans.py, one should run the function
    standard_glm(ev, choice_12, perf, tau), and either fix tau=1 or cross-validate
    the values of tau. Then, save the weights for the transition regressor
    (decays_ac.npy, decays_ae.npy) and the lateral regressor (decays_ac_lat.npy,
                                                              decays_ae_lat.npy).
    """
    decays_ac = np.load('...decays_ac.npy')
    decays_ac_lat = np.load('...decays_ac_lat.npy')
    decays_ae = np.load('...decays_ae.npy')
    decays_ae_lat = np.load('...decays_ae_lat.npy')
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    ax = ax.flatten()
    ax[-1].axis('off')
    plt.subplots_adjust(top=0.95, bottom=0.09, left=0.09, right=0.95,
                        hspace=0.6, wspace=0.5)
    letters = ['a', 'b', '', 'c', 'd', '']
    for n, ax_1 in enumerate(ax):
        fp.add_text(ax=ax_1, letter=letters[n], x=-0.12, y=1.12)
        fp.rm_top_right_lines(ax_1)
    plot_trans_lat_weights([decays_ac, decays_ac_lat],
                           [decays_ae, decays_ae_lat],
                           [ax[1], ax[2]])
    df_data = supp_pcom_rt(folder=folder, sv_folder=sv_folder, ax=ax[3])
    df_data.loc[df_data.subjid == 9, 'subjid'] = 8
    ah.psycho_curves_rep_alt(df_data, ax[0])
    supp_plot_weights_linear_reg(folder, sv_folder, ax=ax[4], nm='300')
    fig.savefig(sv_folder+'supp_human.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'supp_human.png', dpi=400, bbox_inches='tight')
