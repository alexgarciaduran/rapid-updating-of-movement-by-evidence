import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
from scipy.stats import zscore, ttest_1samp, ttest_rel
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import pandas as pd
import matplotlib

import figures_paper as fp
import figure_2 as fig2

# ---GLOBAL VARIABLES
VAR_INC = '-1'
VAR_CON = '+1'
VAR_INC_SHORT = 'inc.'
VAR_CON_SHORT = 'cong.'
VAR_L = 'L'
VAR_R = 'R'


# ---FUNCTIONS
def plot_rt_cohs_with_fb(df, ax, subj='LE46'):
    """
    Function that plots the RT line histogram (for a single rat, LE46)
    conditioning on stimulus evidence.
    """
    colormap = pl.cm.gist_yarg(np.linspace(0.3, 1, 4))
    df_1 = df[df.subjid == subj]
    coh_vec = df_1.coh2.values
    # extract fixation breaks (FB) from the fb column
    for ifb, fb in enumerate(df_1.fb):
        for j in range(len(fb)):
            # concatenate stimulus values
            coh_vec = np.append(coh_vec, [df_1.coh2.values[ifb]])
    for iev, ev in enumerate([0, 0.25, 0.5, 1]):  # for each stim. strength
        index = np.abs(coh_vec) == ev
        # take all RT, including FB
        fix_breaks =\
            np.vstack(np.concatenate([df_1.sound_len/1000,
                                      np.concatenate(df_1.fb.values)-0.3]))
        fix_breaks = fix_breaks[index]
        # compute histogram
        counts_coh, bins = np.histogram(fix_breaks*1000,
                                        bins=30, range=(-100, 300))
        # normalize
        norm_counts = counts_coh/sum(counts_coh)
        # plot
        ax.plot(bins[:-1]+(bins[1]-bins[0])/2, norm_counts,
                color=colormap[iev], label=ev)
    # tune panel
    ax.set_ylabel('Density')
    ax.set_xlabel('Reaction time (ms)')
    
    ax.fill_between([0, 50], [-0.01, -0.01], [0.15, 0.15], color='gray',
                    alpha=0.2, linewidth=0)
    rec1 = plt.Rectangle((-101, 0), 101, 0.15, facecolor="none", 
                         edgecolor="grey", hatch=r"\\", alpha=0.4)
    ax.add_patch(rec1)
    ax.text(-80, 0.09, 'fixation\nbreaks', rotation='vertical',
            fontsize=9.5, style='italic')
    ax.text(12, 0.09, 'express', rotation='vertical',
            fontsize=9.5, style='italic')
    legend = ax.legend(title='Stimulus strength', borderpad=0.3, fontsize=8,
                       loc='center left', bbox_to_anchor=(0.51, 0.95),
                       labelspacing=0.1, frameon=False)
    legend.get_title().set_fontsize('8') #legend 'Title' fontsize


def plot_mt_vs_evidence(df, ax, condition='choice_x_coh', prior_limit=0.25,
                        rt_lim=50, after_correct_only=True, rtmin=0, alpha=1,
                        write_arrows=True, num_bins_prior=5, correlation=False):
    """
    Function that plots the MT against prior or stimulus evidence. 
    If condition == choice_x_prior, then it plots MT vs prior congruency.
    If condition == choice_x_coh, then it plots MT vs stimulus congruency.
    """
    subjects = df['subjid'].unique()
    # vertical line at 0
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.6)
    # get where prior is NaN
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1) == 2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan
    if condition == 'choice_x_prior':  # prior congruency
        df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    # define bins (equipopulated for prior)
    bins, _, indx_trajs, _, colormap =\
          fp.get_bin_info(df=df, condition=condition, prior_limit=prior_limit,
                          after_correct_only=after_correct_only,
                          rt_lim=rt_lim, rtmin=rtmin, num_bins_prior=num_bins_prior)
    df = df.loc[indx_trajs]
    # MT is in s, ms conversion:
    df.resp_len *= 1e3
    if condition == 'choice_x_coh':  # stimulus congruency
        # compute median MT for each subject and each stim strength
        df['choice_x_coh'] = (df.R_response*2-1) * df.coh2
        mt_time = df.groupby(['choice_x_coh', 'subjid']).resp_len.median()
        # unstack to have a matrix with rows for subjects and columns for bins
        mt_time = mt_time.unstack(fill_value=np.nan).values.T
        plot_bins = sorted(df.coh2.unique())
        ax.set_xlabel('Stimulus evidence towards response')
        if write_arrows:  # display congruent/incongruent arrows
            ax.text(-0.45, 294.5, r'$\longleftarrow $', fontsize=10)
            ax.text(-1.05, 300, r'$\it{incongruent}$', fontsize=7.5)
            ax.text(0.09, 294.5, r'$\longrightarrow $', fontsize=10)
            ax.text(0.09, 300, r'$\it{congruent}$', fontsize=7.5)
            ax.set_title('Effect of stimulus', fontsize=10, pad=5)
            ax.text(-1.07, 242, "low prior\ntrials", fontsize=8)
        ax.set_ylim(238, 303)
    elif condition == 'choice_x_prior':  # prior congruency
        mt_time = fp.binning_mt_prior(df, bins)  # get mt binned by prior bins
        plot_bins = bins[:-1] + np.diff(bins)/2
        ax.set_xlabel('Prior evidence towards response')
        if write_arrows:  # display congruent/incongruent arrows
            ax.text(-0.45, 299, r'$\longleftarrow $', fontsize=10)
            ax.text(-1.05, 306, r'$\it{incongruent}$', fontsize=7.5)
            ax.text(0.09, 299, r'$\longrightarrow $', fontsize=10)
            ax.text(0.09, 306, r'$\it{congruent}$', fontsize=7.5)
            ax.set_title('Effect of prior', fontsize=10, pad=5)
            ax.text(-1.07, 225, "silent trials", fontsize=8)
        ax.set_ylim(219, 312)
    ax.set_xlim(-1.2, 1.2)
    # get S.E.M.
    mt_time_err = np.nanstd(mt_time, axis=0) / np.sqrt(len(subjects))
    for i_tr, bin in enumerate(plot_bins):  # for each bin
        c = colormap[i_tr]  
        mean_mt = np.mean(mt_time[:, i_tr])
        # plot errorbars with round marker
        ax.errorbar(bin,mean_mt, yerr=mt_time_err[i_tr],
                        color=c, marker='o', alpha=alpha)
    if correlation:  # compute correlation, default: False
        y_vals = mt_time.flatten()
        x_vals = np.empty((0))
        for i in range(mt_time.shape[0]):
            x_vals = np.concatenate((x_vals, plot_bins))
        pval = pearsonr(x_vals, y_vals).pvalue
        lab = f'p={pval:.2e}'
        y_vals_neg = np.array([mt_time[:, i].flatten() for i in range(4)]).flatten()
        y_vals_pos = np.array([mt_time[:, i].flatten() for i in range(3, 7)]).flatten()
        x_vals = np.repeat(plot_bins, mt_time.shape[0])
        x_vals_pos = x_vals[mt_time.shape[0]*3:]
        x_vals_neg = x_vals[:mt_time.shape[0]*(3+1)]
        if condition == 'choice_x_coh':
            pval = pearsonr(x_vals_pos, y_vals_pos).pvalue
            lab = f'p={pval:.2e}'
            print('congruent correlation:')
            print(lab)
            pval = pearsonr(x_vals_neg, y_vals_neg).pvalue
            lab = f'p={pval:.2e}'
            print('incongruent correlation:')
            print(lab)
    # tune panel
    ax.set_ylabel('Movement time (ms)')
    ax.set_xticks([-1, 0, 1], [VAR_INC, '0', VAR_CON])
    # plot line
    ax.plot(plot_bins, np.mean(mt_time, axis=0), color='k', ls='-', lw=0.5,
            alpha=alpha)


def linear_fun(x, a, b, c, d):
    """
    Just a linear function. y ~ a + b*x_1 + c*x_2 + d*x_3
    Will be used for linear regression.
    """
    return a + b*x[0] + c*x[1] + d*x[2]


def mt_linear_reg(mt, coh, trial_index, com, prior, plot=False):
    """
    Linear regression of MT with prior ev., trial index and stimulus ev..
    Inputs should be z-scored.
    Parameters
    ----------
    mt : array
        Movement time array.
    coh : array
        stimulus evidence.
    trial_index : array
        trial index values.
    prior : array
        congruent prior with final decision.

    Returns
    -------
    popt : list
        fitted parameters of the linear regression.

    """
    trial_index = trial_index.astype(float)[~com.astype(bool)]
    # filter reversals
    xdata = np.array([[coh[~com.astype(bool)]],
                      [trial_index],
                      [prior[~com.astype(bool)]]]).reshape(3, sum(~com))
    ydata = np.array(mt[~com.astype(bool)])
    # perform regression
    popt, pcov = curve_fit(f=linear_fun, xdata=xdata, ydata=ydata)
    if plot:
        df = pd.DataFrame({'coh': coh/max(coh), 'prior': prior/max(prior),
                           'MT': mt,
                           'trial_index': trial_index/max(trial_index)})
        plt.figure()
        sns.pointplot(data=df, x='coh', y='MT', label='coh')
        sns.pointplot(data=df, x='prior', y='MT', label='prior')
        sns.pointplot(data=df, x='trial_index', y='MT', label='trial_index')
        plt.ylabel('MT (ms)')
        plt.xlabel('normalized variables')
        plt.legend()
    return popt, pcov


def plot_mt_weights_bars(means, errors, ax, f5=False, means_model=None,
                         errors_model=None, width=0.35):
    # plot MT regression weights as barplots
    labels = ['Prior', '\n Stimulus']
    if not f5:
        ax.bar(x=labels, height=means, yerr=errors, capsize=3, color='gray',
               ecolor='blue')
        ax.set_ylabel('Impact on movement time')
    if f5:  # it was used in figure 6
        x = np.arange(len(labels))
        ax.bar(x=x-width/2, height=means, yerr=errors, width=width,
               capsize=3, color='k', label='Data', ecolor='blue')
        ax.bar(x=x+width/2, height=means_model, yerr=errors_model, width=width,
               capsize=3, color='red', label='Model')
        ax.set_ylabel('Impact on MT (weights, a.u)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=8)

def plot_mt_weights_violins(w_coh, w_t_i, w_zt, ax, mt=True,
                            t_index_w=False, ttest=True):
    """
    Plot the linear regression weights as violin plots.
    """
    if t_index_w:  # include trial index
        labels = ['Prior', 'Stimulus', 'Trial index']
        arr_weights = np.concatenate((w_zt, w_coh, w_t_i))
        palette = ['darkorange', 'red', 'steelblue']
    else:  # do not include
        labels = ['Prior', 'Stimulus']
        arr_weights = np.concatenate((w_zt, w_coh))
        palette = ['darkorange', 'red']
    label_1 = []
    for j in range(len(labels)):
        for i in range(len(w_coh)):
            label_1.append(labels[j])
    df_weights = pd.DataFrame({' ': label_1, 'weight': arr_weights})
    # plot violin
    violin = sns.violinplot(data=df_weights, x=" ", y="weight", ax=ax,
                            palette=palette, linewidth=0.1)
    for plot in violin.collections[::2]:
        plot.set_alpha(0.7)  # increase transparency
    if t_index_w:
        if ttest:  # perform t-test
            pval_zt = ttest_1samp(w_zt, popmean=0).pvalue
            pval_coh = ttest_1samp(w_coh, popmean=0).pvalue
            pval_t_i = ttest_1samp(w_t_i, popmean=0).pvalue
            pval_array = np.array((pval_zt, pval_coh, pval_t_i))
            formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pval_array]
            formatted_pvalues = [num_ast(decimal)*'*' for decimal in pval_array]
        arr_weights = np.array((w_zt, w_coh, w_t_i))  # weights for each regressor
    else:
        if ttest:  # perform t-test
            pval_zt = ttest_1samp(w_zt, popmean=0).pvalue
            pval_coh = ttest_1samp(w_coh, popmean=0).pvalue
            pval_array = np.array((pval_zt, pval_coh))
            formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pval_array]
            formatted_pvalues = [num_ast(decimal)*'*' for decimal in pval_array]
        arr_weights = np.array((w_zt, w_coh))  # weights for each regressor
    x, y = [0.11, 0.11, 0.11], [0.2, 0.2, 0.52]
    for i in range(len(labels)):  # plot indibidual points
        ax.plot(np.repeat(i, len(arr_weights[i])) +
                0.1*np.random.randn(len(arr_weights[i])),
                arr_weights[i], color='k', marker='o', linestyle='',
                markersize=2)
        if ttest:  # display asterisks for statistical significance
            ax.text(i-x[i], y[i], formatted_pvalues[i])
        if len(w_coh) > 1:
            ax.collections[0].set_edgecolor('k')
    # tune panel
    if t_index_w:
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticklabels([labels[0], labels[1], labels[2]], fontsize=9)
    else:
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticklabels([labels[0], labels[1]], fontsize=9)
    if mt:
        ax.set_ylabel('Impact on movement time')
    else:
        ax.set_ylabel('Impact on RT')
    ax.axhline(y=0, linestyle='--', color='k', alpha=.4)


def mt_weights(df, ax, plot=False, means_errs=True, mt=True, t_index_w=False):
    """
    Computes the weights from the linear regression for each subject, and plots them if plot=True.
    """
    # initialize weight lists for coh (stim), t_i (trial index), and zt (prior)
    w_coh = []
    w_t_i = []
    w_zt = []
    if ax is None and plot:
        fig, ax = plt.subplots(1)
    for subject in df.subjid.unique():  # for each subject
        df_1 = df.loc[df.subjid == subject]
        if mt:  # regression on MT
            var = zscore(np.array(df_1.resp_len))  #  z-score
        else:  # regression on RT
            var = zscore(np.array(df_1.sound_len))
        decision = np.array(df_1.R_response)*2 - 1
        coh = np.array(df_1.coh2)
        trial_index = np.array(df_1.origidx)
        com = df_1.CoM_sugg.values
        zt = df_1.allpriors.values
        # filter by reversal and z-score:
        params, pcov = mt_linear_reg(mt=var[~np.isnan(zt)],
                                     coh=zscore(coh[~np.isnan(zt)] *
                                                decision[~np.isnan(zt)]),
                                     trial_index=zscore(trial_index[~np.isnan(zt)].
                                                        astype(float)),
                                     prior=zscore(zt[~np.isnan(zt)] *
                                                  decision[~np.isnan(zt)]),
                                     plot=False,
                                     com=com[~np.isnan(zt)])
        w_coh.append(params[1])
        w_t_i.append(params[2])
        w_zt.append(params[3])
        # append result
    mean_2 = np.nanmean(w_coh)
    mean_1 = np.nanmean(w_zt)
    std_2 = np.nanstd(w_coh)/np.sqrt(len(w_coh))
    std_1 = np.nanstd(w_zt)/np.sqrt(len(w_zt))
    errors = [std_1, std_2]
    means = [mean_1, mean_2]
    if t_index_w:  # include trial_index
        errors = [std_1, std_2, np.nanstd(w_t_i)/np.sqrt(len(w_t_i))]
        means = [mean_1, mean_2, np.nanmean(w_t_i)]
    if plot:
        if means_errs:  # barplot
            plot_mt_weights_bars(means=means, errors=errors, ax=ax)
            fp.rm_top_right_lines(ax=ax)
        else:  # plot violins
            plot_mt_weights_violins(w_coh=w_coh, w_t_i=w_t_i, w_zt=w_zt,
                                    ax=ax, mt=mt, t_index_w=t_index_w)
    if means_errs:  # return weights
        return means, errors
    else:
        return w_coh, w_t_i, w_zt


def num_ast(decimal):
    """
    Returns the number of asterisks (*) for a given p-value (decimal).
    If p-value >= 0.05, 0 asterisks.
    If 0.01 <= p-value < 0.05, 1 asterisk.
    If 0.001 <= p-value < 0.01, 2 asterisks.
    If p-value < 0.001, 3 asterisks.
    """
    if decimal >= 0.05:
        return 0
    if decimal < 0.05 and decimal >= 0.01:
        return 1
    if decimal < 0.01 and decimal >= 0.001:
        return 2
    if decimal < 0.001:
        return 3


def plot_mt_vs_stim(df, ax, prior_min=0.1, rt_max=50, human=False, sim=False):
    """
    Plots MT vs stimulus evidence in low prior and short RT.
    """
    # plots vertical line at x=0
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.6)
    if not human:  # select subjects with silent trials so the comparison can be done
        subjects = df.loc[df.special_trial == 2, 'subjid'].unique()
    else:
        subjects = df['subjid'].unique()
    subjid = df.subjid
    # transform variables into congruent/incongruent with final response
    zt_cong = df.norm_allpriors.values * (df.R_response*2-1)
    coh_cong = np.round(df.coh2.values * (df.R_response*2-1), 2)
    rt = df.sound_len.values
    spec_trial = df.special_trial.values
    mt = df.resp_len.values
    mt_mat = np.empty((len(subjects), len(np.unique(coh_cong))))
    sil = []
    for i_s, subject in enumerate(subjects):  # for each subject
        # silent
        prior_min_quan = np.quantile(zt_cong[(subjid == subject) &
                                             (~np.isnan(zt_cong))], prior_min)
        # mean MT in silent trials
        sil.append(np.nanmean(mt[(zt_cong >= prior_min_quan) & (rt <= rt_max) &
                                 (spec_trial == 2) & (subjid == subject)])*1e3)
        for iev, ev in enumerate(np.unique(coh_cong)):  # for each stimulus strength
            index = (subjid == subject) & (zt_cong >= prior_min_quan) &\
                (rt <= rt_max) & (spec_trial == 0) & (coh_cong == ev)
            mt_ind = np.nanmean(mt[index])*1e3
            mt_mat[i_s, iev] = mt_ind
    pv = []
    mean_silent = np.nanmean(sil)
    for iev, ev in enumerate(np.unique(coh_cong)):  # for each stimulus strength
        mt_all_subs = mt_mat[:, iev]
        pv.append(ttest_rel(mt_all_subs, sil).pvalue)  # t-test comparing with the silent trials
    formatted_pvalues = [num_ast(pvalue)*'*' if ~np.isnan(pvalue) else '' for pvalue in pv]
    mean_mt_vs_coh = np.nanmean(mt_mat, axis=0)  # mean MT across subjects for each stim. strength
    sd_mt_vs_coh = np.nanstd(mt_mat, axis=0)/np.sqrt(len(subjects))  # S.E.M.
    # plot
    ax.axhline(y=mean_silent, color='k', alpha=0.6)
    coh_unq = np.unique(coh_cong)
    ax.plot(coh_unq, mean_mt_vs_coh, color='k', ls='-', lw=0.5)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["mediumblue","plum","firebrick"])
    colormap = colormap(np.linspace(0, 1, len(coh_unq)))
    i = 0
    for x, y, e, color in zip(coh_unq, mean_mt_vs_coh, sd_mt_vs_coh, colormap):
        ax.plot(x, y, 'o', color=color)
        ax.errorbar(x, y, e, color=color)
        ax.text(x-len(formatted_pvalues[i])*0.03, y*(1.03), formatted_pvalues[i])
        i += 1
    # tune panel
    ax.set_xticks([-1, 0, 1], [VAR_INC, '0', VAR_CON])
    ax.text(0.65, np.nanmean(sil)+5, 'silent', color='k', fontsize=10.5, alpha=0.7)
    ax.set_xlabel('Stimulus evidence towards response')
    ax.set_ylabel('Movement time (ms)')
    ax.text(-0.35, 293, r'$\longleftarrow $', fontsize=10)
    ax.text(-0.85, 300, r'$\it{incongruent}$', fontsize=8)
    ax.text(0.07, 293, r'$\longrightarrow $', fontsize=10)
    ax.text(0.07, 300, r'$\it{congruent}$', fontsize=8)
    ax.set_title('Net effect of stimulus', fontsize=10, pad=10)
    ax.set_xlim(-1.1, 1.1)
    if not sim:
        ax.set_ylim(218, 302)
        ax.text(-1.05, 224, "strong\ncongruent prior", fontsize=8)
    else:
        ax.set_ylim(195, 307)



def mt_matrix_ev_vs_zt(df, ax, f, silent_comparison=False, rt_bin=None,
                       collapse_sides=False, margin=.05):
    """
    Plots the MT matrix vs stimulus and prior evidence.
    collapse_sides = False: separate between right/left responses
    """
    coh_raw = df.coh2.copy()
    norm_allp_raw = df.norm_allpriors.copy()
    # transform variables into congruent/incongruent with final response
    df['norm_allpriors'] *= df.R_response.values*2-1
    df['coh2'] *= df.R_response.values*2-1
    if not collapse_sides:
        ax0, ax1 = ax
    if rt_bin is not None:
        if rt_bin > 100:
            df_fil = df.loc[df.sound_len > rt_bin]
        if rt_bin < 100:
            df_fil = df.loc[df.sound_len < rt_bin]
        if not collapse_sides:
            df_0 = df_fil.loc[(df_fil.R_response == 0)]
            df_1 = df_fil.loc[(df_fil.R_response == 1)]
        else:
            df_s = df_fil
    else:
        if not collapse_sides:
            df_0 = df.loc[(df.R_response == 0)]
            df_1 = df.loc[(df.R_response == 1)]
        else:
            df_s = df.copy()
    # equipopulated bins for absolute value
    bins_zt = [1.01]
    for i_p, perc in enumerate([0.75, 0.5, 0.25, 0.25, 0.5, 0.75]):
        if i_p < 3:
            bins_zt.append(df.norm_allpriors.abs().quantile(perc))
        else:
            bins_zt.append(-df.norm_allpriors.abs().quantile(perc))
    bins_zt.append(-1.01)
    coh_vals = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    nsubs = len(df.subjid.unique())
    if not collapse_sides:
        mat_0 = np.zeros((len(bins_zt)-1, 7, nsubs))
        mat_1 = np.zeros((len(bins_zt)-1, 7, nsubs))
        silent_mat_per_rows_0 = np.zeros((len(bins_zt)-1, 7))
        silent_mat_per_rows_1 = np.zeros((len(bins_zt)-1, 7))
    else:
        mat_s = np.zeros((len(bins_zt)-1, 7, nsubs))
        silent_mat_per_rows = np.zeros((len(bins_zt)-1, 7))
    # reference MT computation (silent)
    for i_zt, zt in enumerate(bins_zt[:-1]):
        if not collapse_sides:
            mt_sil_per_sub_0 = []
            mt_sil_per_sub_1 = []
        else:
            mt_sil_per_sub = []
        for subj in df.subjid.unique():  # for each subject
            if not collapse_sides:  # for each response
                mt_silent_0 = df_0.loc[(df_0.special_trial == 2) &
                                       (df_0.norm_allpriors < zt) &
                                       (df_0.subjid == subj) &
                                       (df_0.norm_allpriors > bins_zt[i_zt+1]),
                                       'resp_len']
                mt_silent_1 = df_1.loc[(df_1.special_trial == 2) &
                                       (df_1.norm_allpriors < zt) &
                                       (df_1.subjid == subj) &
                                       (df_1.norm_allpriors > bins_zt[i_zt+1]),
                                       'resp_len']
                mt_sil_per_sub_0.append(np.nanmean(mt_silent_0))
                mt_sil_per_sub_1.append(np.nanmean(mt_silent_1))
            else:  # group responses
                mt_silent = df_s.loc[(df_s.special_trial == 2) &
                                     (df_s.norm_allpriors < zt) &
                                     (df_s.subjid == subj) &
                                     (df_s.norm_allpriors > bins_zt[i_zt+1]),
                                     'resp_len']
                mt_sil_per_sub.append(np.nanmean(mt_silent))
        if not collapse_sides:  # for each response
            silent_mat_per_rows_0[i_zt, :] = np.nanmean(mt_sil_per_sub_0)*1e3
            silent_mat_per_rows_1[i_zt, :] = np.nanmean(mt_sil_per_sub_1)*1e3
        else:  # combine responses
            silent_mat_per_rows[i_zt, :] = np.nanmean(mt_sil_per_sub)*1e3
    # MT matrix computation, reference MT (silent) can be substracted if desired (silent_comparison=True)
    for i_s, subj in enumerate(df.subjid.unique()):  # for each subject
        if not collapse_sides:
            df_sub_0 = df_0.loc[(df_0.subjid == subj)]
            df_sub_1 = df_1.loc[(df_1.subjid == subj)]
        else:
            df_sub = df_s.loc[df_s.subjid == subj]
        for i_ev, ev in enumerate(coh_vals):  # for each stimulus bin
            for i_zt, zt in enumerate(bins_zt[:-1]):  # for each prior bin
                if not collapse_sides:  # for each response
                    mt_vals_0 = df_sub_0.loc[(df_sub_0.coh2 == ev) &
                                             (df_sub_0.norm_allpriors < zt)
                                             & (df_sub_0.norm_allpriors >=
                                                bins_zt[i_zt+1]),
                                             'resp_len']
                    mt_vals_1 = df_sub_1.loc[(df_sub_1.coh2 == ev) &
                                             (df_sub_1.norm_allpriors < zt)
                                             & (df_sub_1.norm_allpriors >=
                                                bins_zt[i_zt+1]),
                                             'resp_len']
                    mat_0[i_zt, i_ev, i_s] = np.nanmean(mt_vals_0)*1e3
                    mat_1[i_zt, i_ev, i_s] = np.nanmean(mt_vals_1)*1e3
                else:  # combine responses
                    mt_vals = df_sub.loc[(df_sub.coh2 == ev) &
                                         (df_sub.norm_allpriors < zt)
                                         & (df_sub.norm_allpriors >=
                                            bins_zt[i_zt+1]),
                                         'resp_len']
                    mat_s[i_zt, i_ev, i_s] = np.nanmean(mt_vals)*1e3
    if not collapse_sides:
        mat_0 = np.nanmean(mat_0, axis=2)
        mat_1 = np.nanmean(mat_1, axis=2)
    else:
        mat_s = np.nanmean(mat_s, axis=2)
    if silent_comparison:  # We substract reference MT (in silent trials) at each row
        if not collapse_sides:
            mat_0 -= silent_mat_per_rows_0
            mat_1 -= silent_mat_per_rows_1
        else:
            mat_s -= silent_mat_per_rows
    df['norm_allpriors'] = norm_allp_raw
    df['coh2'] = coh_raw
    if not collapse_sides:
        # SIDE 0
        im_0 = ax0.imshow(mat_0, cmap='RdGy', vmin=np.nanmin((mat_1, mat_0)),
                          vmax=np.nanmax((mat_1, mat_0)))
        plt.sca(ax0)
        cbar_0 = plt.colorbar(im_0, fraction=0.04)
        cbar_0.remove()
        ax0.set_xlabel('Evidence')
        ax0.set_ylabel('Prior')
        if rt_bin is not None:
            if rt_bin > 100:
                ax0.set_title('Left, RT > ' + str(rt_bin) + ' ms')
            else:
                ax0.set_title('Left, RT < ' + str(rt_bin) + ' ms')
        else:
            ax0.set_title('Left')
        ax0.set_yticks([0, 3, 6], [VAR_R, '0', VAR_L])
        ax0.set_xticks([0, 3, 6], [VAR_L, '0', VAR_R])
        # SIDE 1
        ax1.set_title('Right')
        im_1 = ax1.imshow(mat_1, cmap='RdGy', vmin=np.nanmin((mat_1, mat_0)),
                          vmax=np.nanmax((mat_1, mat_0)))
        plt.sca(ax1)
        ax1.set_xlabel('Evidence')
        cbar_1 = plt.colorbar(im_1, fraction=0.04, pad=-0.05)
        if silent_comparison:
            cbar_1.set_label(r'$MT \; - MT_{silent}(ms)$')
        else:
            cbar_1.set_label(r'$MT \;(ms)$')
        ax1.set_yticks([0, 3, 6], [' ', ' ', ' '])
        ax1.set_xticks([0, 3, 6], [VAR_L, '0', VAR_R])
        ax0pos = ax0.get_position()
        ax1pos = ax1.get_position()
        ax0.set_position([ax0pos.x0, ax1pos.y0, ax1pos.width, ax1pos.height])
        ax1.set_position([ax1pos.x0-ax1pos.width*0.2,
                          ax1pos.y0, ax1pos.width, ax1pos.height])
    else:
        # plot matrix
        im_s = ax.imshow(np.flip(mat_s).T, cmap='RdGy')
        # tune panel
        plt.sca(ax)
        ax.set_yticks([0, 3, 6])
        ax.set_xticks([0, 3, 6])
        ax.set_yticklabels([VAR_CON_SHORT, '0', VAR_INC_SHORT])
        ax.set_xticklabels([VAR_INC_SHORT, '0', VAR_CON_SHORT])
        ax.set_ylim([-0.5, 6.5])
        ax.set_xlim([-0.5, 6.5])
        ax.set_xlabel('Prior evidence')
        ax.set_ylabel('Stimulus evidence')
        pos = ax.get_position()
        cbar_ax = f.add_axes([pos.x0+pos.width+margin/2, pos.y0+margin/6,
                              pos.width/12, pos.height/1.4])
        plt.colorbar(im_s, cax=cbar_ax)
        cbar_ax.set_title(r'  $MT \;(ms)$', fontsize=8)


def fig_1_rats_behav(df_data, task_img, repalt_img, sv_folder,
                     figsize=(10, 8), margin=.06, max_rt=300):
    """
    Plots figure 1 of the paper.
    """
    f, ax = plt.subplots(nrows=3, ncols=4, figsize=figsize)
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.98,
                        hspace=0.45, wspace=0.6)
    ax = ax.flatten()
    # TUNE PANELS
    # all panels
    letters = ['', '',  '', '', 'c', 'd', 'e', 'f', 'g', 'h', 'i', '']
    for n, ax_1 in enumerate(ax):
        fp.add_text(ax=ax_1, letter=letters[n], x=-0.12, y=1.25)
        if n not in [4, 11]:
            fp.rm_top_right_lines(ax_1)

    for i in [0, 1, 2, 11]:
        ax[i].axis('off')
    # task panel
    ax_task = ax[0]
    pos_task = ax_task.get_position()
    factor = 2.3
    ax_task.set_position([pos_task.x0, pos_task.y0-0.04,
                          pos_task.width*factor, pos_task.height*factor])
    fp.add_text(ax=ax_task, letter='a', x=-0.05, y=.95)
    # rep-alt img
    ax_repalt = ax[2]
    pos_repalt = ax_repalt.get_position()
    factor = 1.3
    ax_repalt.set_position([pos_repalt.x0-pos_repalt.width/1.5,
                            pos_repalt.y0+0.035,
                            pos_repalt.width*factor, pos_repalt.height*factor])
    # rt panel
    ax_rts = ax[4]
    fp.rm_top_right_lines(ax=ax_rts)
    ax_rts.set_xlabel('Reaction Time (ms)')
    ax_rts.set_ylabel('Density')
    ax_rts.set_xlim(-101, max_rt+1)
    ax_rts.set_ylim(0, 0.1495)
    ax_rts.set_xticks(np.arange(0, max_rt+1, 100))
    fp.add_text(ax=ax_rts, letter='rat LE46', x=0.35, y=1.08, fontsize=8)
    # pright panel
    ax_pright = ax[3]
    pos_pright = ax_pright.get_position()
    factor = 1.1
    ax_pright.set_position([pos_pright.x0,
                            pos_pright.y0 + 0.05,
                            pos_pright.width*factor,
                            pos_pright.height*factor])
    pos_pright = ax_pright.get_position()
    ax_pright.set_yticks([0, 3, 6])
    ax_pright.set_ylim([-0.5, 6.5])
    ax_pright.set_yticklabels([VAR_L, '0', VAR_R])
    ax_pright.set_xticks([0, 3, 6])
    ax_pright.set_xlim([-0.5, 6.5])
    ax_pright.set_xticklabels([VAR_L, '0', VAR_R])
    ax_pright.set_xlabel('Prior evidence')
    ax_pright.set_ylabel('Stimulus evidence')
    fp.add_text(ax=ax_pright, letter='b', x=-0.17, y=1.12)
    # tachometric panel
    ax_tach = ax[5]
    ax_tach.set_xlabel('Reaction time (ms)')
    ax_tach.set_ylabel('Accuracy')
    ax_tach.set_ylim(0.5, 1.02)
    ax_tach.set_xlim(-101, max_rt+1)
    ax_tach.set_xticks(np.arange(0, max_rt+1, 100))
    ax_tach.axvline(x=0, linestyle='--', color='k', lw=0.5)
    ax_tach.set_yticks([0.5, 0.75, 1])
    ax_tach.set_yticklabels(['0.5', '0.75', '1'])
    ax_tach.fill_between([0, 50], [0, 0], [1.04, 1.04], color='gray', alpha=0.2,
                         linewidth=0)
    rec1 = plt.Rectangle((-101, 0), 101, 1.04, facecolor="none", 
                         edgecolor="grey", hatch=r"\\", alpha=0.4)
    ax_tach.add_patch(rec1)
    fp.rm_top_right_lines(ax_tach)
    pos = ax_tach.get_position()
    ax_tach.set_position([pos.x0, pos.y0, pos.width, pos.height])
    fp.add_text(ax=ax_tach, letter='rat LE46', x=0.35, y=1.08, fontsize=8)
    # mt versus evidence panels
    # move axis 6 to the right
    shift = 0.12
    factor = 0.8
    ax_mt_coh = ax[6]
    pos = ax_mt_coh.get_position()
    # move axis 7 to the right
    ax_mt_zt = ax[7]
    # regression weights panel
    # make axis 8 bigger and move it to the right
    factor = 1.2
    pos = ax[8].get_position()
    ax[8].set_position([pos.x0, pos.y0-margin, pos.width*factor, pos.height])
    # mt matrix panel
    # make axis 9 bigger and move it to the right
    factor = 1.2
    pos = ax[9].get_position()
    ax[9].set_position([pos.x0+shift/1.8, pos.y0-margin, pos.width*factor, pos.height])
    # mt versus silent panel
    # make axis 11 bigger
    pos = ax[10].get_position()
    ax[10].set_position([pos.x0+shift*1.6, pos.y0-margin, pos.width*factor, pos.height])

    # RTs
    df_rts = df_data.copy()
    plot_rt_cohs_with_fb(df=df_rts, ax=ax_rts, subj='LE46')
    del df_rts
    ax_rts.axvline(x=0, linestyle='--', color='k', lw=0.5)

    # TASK PANEL
    task = plt.imread(task_img)
    ax_task.imshow(task)

    # REPALT PANEL
    task = plt.imread(repalt_img)
    ax_repalt.imshow(task)

    # P(RIGHT) MATRIX
    mat_pright_all = np.zeros((7, 7))
    for subject in df_data.subjid.unique():
        df_sbj = df_data.loc[(df_data.special_trial == 0) &
                             (df_data.subjid == subject)]
        choice = df_sbj['R_response'].values
        coh = df_sbj['coh2'].values
        prior = df_sbj['norm_allpriors'].values
        indx = ~np.isnan(prior)
        mat_pright, _ = fp.com_heatmap(prior[indx], coh[indx], choice[indx],
                                       return_mat=True, annotate=False)
        mat_pright_all += mat_pright
    mat_pright = mat_pright_all / len(df_data.subjid.unique())

    im_2 = ax_pright.imshow(mat_pright, cmap='PRGn_r')
    cbar = f.colorbar(im_2, ax=ax_pright, location='right',
                      label='p (right response)', shrink=0.7, aspect=10)
    im = ax_pright.images
    cb = im[-1].colorbar
    pos_cb = cb.ax.get_position()
    pos_pright = ax_pright.get_position()
    factor = 1.3
    ax_pright.set_position([pos_pright.x0-pos_pright.width/1.8,
                            pos_pright.y0 + 0.07,
                            pos_pright.width*factor,
                            pos_pright.height*factor])
    cb.ax.set_position([pos_cb.x0-pos_pright.width/1.8+0.045,
                        pos_cb.y0+0.1, pos_cb.width, pos_cb.height])
    df_data = df_data.loc[df_data.soundrfail == 0]
    # TACHOMETRICS
    bin_size = 5
    rtbins_1 = np.arange(0, 100, bin_size)
    rtbins_2 = np.arange(100, 301, bin_size*3)
    rtbins = np.concatenate((rtbins_1, rtbins_2))
    labels = ['0', '0.25', '0.5', '1']
    df_tachos = df_data.copy().loc[df_data.subjid == 'LE46']
    fp.tachometric(df=df_tachos, ax=ax_tach, fill_error=True, cmap='gist_yarg',
                   labels=labels, rtbins=rtbins,
                   evidence='coh2')
    del df_tachos
    ax_tach.axvline(x=0, linestyle='--', color='k', lw=0.5)

    # MT VS PRIOR
    df_mt = df_data.copy()
    plot_mt_vs_evidence(df=df_mt.loc[df_mt.special_trial == 2], ax=ax_mt_coh,
                        condition='choice_x_prior', prior_limit=1,
                        rt_lim=200)
    del df_mt
    # MT VS COH
    df_mt = df_data.copy()
    plot_mt_vs_evidence(df=df_mt, ax=ax_mt_zt, prior_limit=0.1,  # 10% quantile
                        condition='choice_x_coh', rt_lim=50)
    del df_mt
    # REGRESSION WEIGHTS
    df_wghts = df_data.copy()
    mt_weights(df=df_wghts, ax=ax[8], plot=True, means_errs=False)
    del df_wghts
    # MT MATRIX
    df_mtx = df_data.copy()
    mt_matrix_ev_vs_zt(df=df_mtx, ax=ax[9], f=f, silent_comparison=False,
                          rt_bin=60, collapse_sides=True)
    del df_mtx
    # SLOWING
    df_slow = df_data.copy()
    plot_mt_vs_stim(df=df_slow, ax=ax[10], prior_min=0.8, rt_max=50)
    del df_slow

    f.savefig(sv_folder+'fig1.svg', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'fig1.png', dpi=400, bbox_inches='tight')


def plot_mt_weights_rt_bins(df, rtbins=np.linspace(0, 150, 16), ax=None):
    """
    Performs linear regression of MT vs stimulus ev., prior ev. and trial index in bins of RT,
    and plots the weights vs RT afterwards.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1)
    fp.rm_top_right_lines(ax)
    # initialize lists
    coh_weights = []
    zt_weights = []
    ti_weights = []
    coh_err = []
    zt_err = []
    ti_err = []
    for irt, rtbin in enumerate(rtbins[:-1]):  # for each rt bin
        # filter
        df_rt = df.loc[(df.sound_len >= rtbin) &
                       (df.sound_len < rtbins[irt+1])]
        # compute weights
        mean_sub, err_sub = mt_weights(df_rt, ax=None, plot=False,
                                       means_errs=True, mt=True, t_index_w=True)
        # append weights and SEMs
        coh_weights.append(mean_sub[1])
        zt_weights.append(mean_sub[0])
        ti_weights.append(mean_sub[2])
        coh_err.append(err_sub[1])
        zt_err.append(err_sub[0])
        ti_err.append(err_sub[2])
    # plotting
    error_kws = dict(ecolor='goldenrod', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='goldenrod', marker='o', label='Prior')
    ax.errorbar(rtbins[:-1]+(rtbins[1]-rtbins[0])/2, zt_weights, zt_err,
                **error_kws)
    error_kws = dict(ecolor='firebrick', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='firebrick', marker='o', label='Stimulus')
    ax.errorbar(rtbins[:-1]+(rtbins[1]-rtbins[0])/2, coh_weights, coh_err,
                **error_kws)
    error_kws = dict(ecolor='steelblue', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='steelblue', marker='o', label='Trial index')
    ax.errorbar(rtbins[:-1]+(rtbins[1]-rtbins[0])/2, ti_weights, ti_err,
                **error_kws)
    # tune panel
    ax.set_ylabel('Impact on movement time')
    ax.set_xlabel('Reaction time (ms)')
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.8)


def supp_trajs_cond_trial_index(df, data_folder, ax):
    """
    Plots the trajectories depending on trial index.
    """
    fig, ax2 = plt.subplots(2)
    ax2 = ax2.flatten()
    for a in ax:
        fp.rm_top_right_lines(a)
    ax = [ax[0], ax2[0], ax[1], ax2[1]]
    fig2.plots_trajs_conditioned(df, ax, data_folder, condition='origidx',
                                 cmap='viridis',
                                 prior_limit=0.25, rt_lim=50,
                                 after_correct_only=True,
                                 trajectory="trajectory_y",
                                 velocity=("traj_d1", 1))
    plt.close(fig)


def supp_trial_index_analysis(df, data_folder, sv_folder):
    """
    Plots supplementary figure 1.
    """
    fig, ax = plt.subplots(ncols=2, nrows=2)
    # tune panels
    fig.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.35)
    ax = ax.flatten()
    labs = ['a', 'c', 'b', 'd']
    for i_ax, a in enumerate(ax):
        fp.rm_top_right_lines(a)
        a.text(-0.11, 1.12, labs[i_ax], transform=a.transAxes, fontsize=16,
               fontweight='bold', va='top', ha='right')
    # trajs/velocities conditioned on trial index
    supp_trajs_cond_trial_index(df=df, data_folder=data_folder,
                                ax=[ax[0], ax[2]])
    ax[2].set_ylim(-0.05, 0.455)
    # MT regression weights vs RT
    plot_mt_weights_rt_bins(df=df, ax=ax[3], rtbins=np.linspace(0, 150, 16))
    # overall MT regression weights
    mt_weights(df=df, ax=ax[1], plot=True, t_index_w=True, means_errs=False)
    # save figure
    fig.savefig(sv_folder+'supp_fig_1.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'supp_fig_1.png', dpi=400, bbox_inches='tight')

