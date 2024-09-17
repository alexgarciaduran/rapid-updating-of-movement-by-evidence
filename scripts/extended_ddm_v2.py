# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:17:47 2022
@author: Alex Garcia-Duran
2 Readout model
"""

import numpy as np
import pandas as pd
import glob

# SV_FOLDER = '/archive/molano/CoMs/'  # Cluster Manuel
# SV_FOLDER = '/home/garciaduran/'  # Cluster Alex
# SV_FOLDER = '/home/molano/Dropbox/project_Barna/ChangesOfMind/'  # Manuel
SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
# SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
# DATA_FOLDER = '/archive/molano/CoMs/data/'  # Cluster Manuel
# DATA_FOLDER = '/home/garciaduran/data/'  # Cluster Alex
# DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
# DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
BINS = np.linspace(1, 301, 11)


def v_(t):
    return t.reshape(-1, 1) ** np.arange(6)


def get_Mt0te(t0, te):
    """

    Parameters
    ----------
    t0 : TYPE
        DESCRIPTION.
    te : TYPE
        DESCRIPTION.

    Returns
    -------
    Mt0te : TYPE
        DESCRIPTION.

    """
    Mt0te = np.array(
        [
            [1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
            [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
            [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
            [1, te, te ** 2, te ** 3, te ** 4, te ** 5],
            [0, 1, 2 * te, 3 * te ** 2, 4 * te ** 3, 5 * te ** 4],
            [0, 0, 2, 6 * te, 12 * te ** 2, 20 * te ** 3],
        ]
    )
    return Mt0te


def compute_traj(jerk_lock_ms, mu, resp_len):
    """

    Parameters
    ----------
    jerk_lock_ms : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    resp_len : TYPE
        DESCRIPTION.

    Returns
    -------
    traj : TYPE
        DESCRIPTION.

    """
    resp_len = float(resp_len)
    t_arr = np.arange(jerk_lock_ms, resp_len)
    m_sup = np.concatenate((get_m1_inv(), np.zeros((3, 3))), axis=1)
    m_inf = np.concatenate((get_m3_inv(resp_len), get_m4_inv(resp_len)), axis=1)
    M_1 = np.concatenate((m_sup, m_inf))
    vt = v_(t_arr)
    N = np.matmul(vt, M_1)
    traj = np.matmul(N, mu).ravel()
    traj = np.concatenate([[0]*jerk_lock_ms, traj])
    return traj


def get_m4_inv(t):
    m4_inv = np.array([[10/t**3, -4/t**2, 1/(2*t)],
                       [-15/t**4, 7/t**3, -1/t**2],
                       [6/t**5, -3/t**4, 1/(2*t**3)]])
    return m4_inv


def get_m1_inv():
    m1_inv = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1/2]])
    return m1_inv


def get_m3_inv(t):
    m3_inv = np.array([[-10/t**3, -6/t**2, -3/(2*t)],
                       [15/t**4, 8/t**3, 3/(2*t**2)],
                       [-6/t**5, -3/t**4, -1/(2*t**3)]])
    return m3_inv


def trial_ev_vectorized(zt, stim, coh, trial_index, p_MT_slope, p_MT_intercept, p_w_zt,
                        p_w_stim, p_e_bound, p_com_bound, p_t_eff, p_t_aff,
                        p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
                        p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
                        num_tr, stim_res, human=False,
                        compute_trajectories=False, num_trials_per_session=600,
                        all_trajs=True, num_computed_traj=int(3e4),
                        fixation_ms=300):
    """
    Generate stim and time integration and trajectories

    Parameters
    ----------
    zt : array
        priors for each trial (transition bias + lateral (CWJ) 1xnum-trials).
    stim : array
        stim sequence for each trial 20xnum-trials.
    MT_slope : float
        slope corresponding to motor time and trial index linear relation (0.15).
    MT_intercep : float
        intercept corresponding to motor-time and trial index relation (110).
    p_w_zt : float
        fitting parameter: gain for prior (zt).
    p_w_stim : float
        fitting parameter: gain for stim (stim).
    p_e_bound : float
        fitting parameter: bounds for the evidence integrator.
    p_com_bound : float
        fitting parameter: change-of-mind bound (will have opposite sign of
        first choice).
    p_t_eff : float
        fitting parameter: efferent latency to initiate movement.
    p_t_aff : float
        fitting parameter: afferent latency to integrate stimulus.
    p_t_a : float
        fitting parameter: latency for action integration.
    p_w_a_intercept : float
        fitting parameter: drift of action noise.
    p_a_bound : float
        fitting parameter: bounds for the action integrator.
    p_1st_readout : float
        fitting parameter: slope of the linear realtion with time and evidence
        for trajectory update.
    num_tr : int
        number of trials.
    trajectories : boolean, optional
        Whether trajectories are computed or not. The default is False.

    Returns
    -------
    E : array
        evidence integration matrix (num_tr x stim.shape[0]).
    A : array
        action integration matrix (num_tr x stim.shape[0]).
    com : boolean array
        whether each trial is or not a change-of-mind (num_tr x 1).
    first_ind : list
        first choice indexes (num_tr x 1).
    second_ind : list
        second choice indexes (num_tr x 1).
    resp_first : list
        first choice (-1 if left and 1 if right, num_tr x 1).
    resp_fin : list
        second (final) choice (-1 if left and 1 if right, num_tr x 1).
    pro_vs_re : boolean array
        whether each trial is reactive or not (proactive) ( num_tr x 1).
    total_traj: tuple
        total trajectory of the rat, containing the update (num_tr x 1).
    init_trajs: tuple
        pre-planned trajectory of the rat.
    final_trajs: tuple
        trajectory after the update.

    """
    if human:
        px_final = 600
    if not human:
        px_final = 75
    bound = p_e_bound
    bound_a = p_a_bound
    dt = stim_res*1e-3
    p_e_noise = np.sqrt(dt)
    p_a_noise = np.sqrt(dt)
    fixation = int(fixation_ms / stim_res)  # ms/stim_resolution
    prior = zt*p_w_zt + np.random.choice([-1 ,1], stim.shape[1])*1e-6
    # instantaneous evidence
    Ve = np.concatenate((np.zeros((p_t_aff + fixation, num_tr)), stim*p_w_stim))
    max_integration_time = Ve.shape[0]-1
    N = Ve.shape[0]
    zero_or_noise_evidence = np.concatenate((np.zeros((fixation, num_tr)),
                                             np.random.randn(N - fixation, num_tr)))
    # add noise
    dW = zero_or_noise_evidence*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+p_w_a_intercept +\
        p_w_a_slope*trial_index
    # zeros before p_t_a
    dA[:p_t_a, :] = 0
    # accumulate
    A = np.cumsum(dA, axis=0)
    dW[0, :] = prior
    E = np.copy(dW)
    # adding leak
    E[:fixation, :] = np.cumsum(E[:fixation, :], axis=0)
    for i in range(fixation, N):
        E[i, :] += E[i-1, :]*(1-p_leak)
    com = False
    # check docstring for definitions
    first_ind = []
    second_ind = []
    pro_vs_re = []
    resp_first = np.ones(E.shape[1])
    resp_fin = np.ones(E.shape[1])
    # evidences at 1st/2nd readout
    first_ev = []
    second_ev = []
    for i_t in range(E.shape[1]):
        # search where evidence bound is reached
        indx_hit_bound = np.abs(E[:, i_t]) >= bound
        hit_bound = max_integration_time
        if (indx_hit_bound).any():
            hit_bound = np.where(indx_hit_bound)[0][0]
        # search where action bound is reached
        indx_hit_action = A[:, i_t] >= bound_a
        hit_action = max_integration_time
        if (indx_hit_action).any():
            hit_action = np.where(indx_hit_action)[0][0]
        # set first readout as the minimum
        hit_dec = min(hit_bound, hit_action)
        pro_vs_re.append(np.argmin([hit_action, hit_bound]))
        # store first readout index
        first_ind.append(hit_dec)
        # store first readout evidence
        first_ev.append(E[hit_dec, i_t])
        # first categorical response
        resp_first[i_t] *= (-1)**(E[hit_dec, i_t] < 0)
        # CoM bound with sign depending on first response
        com_bound_signed = (-resp_first[i_t])*p_com_bound
        # second response
        indx_final_ch = hit_dec+p_t_eff+p_t_aff
        indx_final_ch = min(indx_final_ch, max_integration_time)
        # get post decision accumulated evidence with respect to CoM bound
        post_dec_integration = E[hit_dec:indx_final_ch, i_t]-com_bound_signed
        # get CoMs indexes
        indx_com =\
            np.where(np.sign(E[hit_dec, i_t]) != np.sign(post_dec_integration))[0]
        # get CoM effective index
        indx_update_ch = indx_final_ch if len(indx_com) == 0\
            else indx_com[0] + hit_dec
        # get final decision
        resp_fin[i_t] = resp_first[i_t] if len(indx_com) == 0 else -resp_first[i_t]
        second_ind.append(indx_update_ch)
        second_ev.append(E[indx_update_ch, i_t])
    com = resp_first != resp_fin
    first_ind = np.array(first_ind)
    pro_vs_re = np.array(pro_vs_re)
    rt_vals, rt_bins = np.histogram((first_ind-fixation+p_t_eff)*stim_res,
                                    bins=np.linspace(-100, 300, 81))
   
    matrix = None
    if compute_trajectories:  # in case trajectories are to be computed
        # Trajectories
        RLresp = resp_fin
        prechoice = resp_first
        jerk_lock_ms = 0
        # initial positions, speed and acc; final position, speed and acc
        initial_mu = np.array([0, 0, 0, px_final, 0, 0]).reshape(-1, 1)
        indx_trajs = np.arange(len(first_ind)) if all_trajs\
            else np.random.choice(len(first_ind), num_computed_traj)
        # check docstring for definitions
        init_trajs = []
        final_trajs = []
        total_traj = []
        # first trajectory motor time w.r.t. first readout
        frst_traj_motor_time = []
        # x value of trajectory at second readout update time
        x_val_at_updt = []
        for i_t in indx_trajs:
            # pre-planned Motor Time
            MT = p_MT_slope*trial_index[i_t] + p_MT_intercept +\
                p_mt_noise*np.random.gumbel()
            first_resp_len = float(MT-p_1st_readout*np.abs(first_ev[i_t]))
            # first_resp_len: evidence influence on MT. The larger the ev,
            # the smaller the motor time
            initial_mu_side = initial_mu * prechoice[i_t]
            prior0 = compute_traj(jerk_lock_ms, mu=initial_mu_side,
                                  resp_len=first_resp_len)
            init_trajs.append(prior0)
            # TRAJ. UPDATE
            try:
                velocities = np.gradient(prior0)
                accelerations = np.gradient(velocities)  # acceleration
                t_updt = int(second_ind[i_t] - first_ind[i_t])  # time indx
                t_updt = int(np.min((t_updt*stim_res, len(velocities)-1)))
                frst_traj_motor_time.append(t_updt)
                vel = velocities[t_updt]  # velocity at the timepoint
                acc = accelerations[t_updt]
                pos = prior0[t_updt]  # position
                mu_update = np.array([pos, vel, acc, px_final*RLresp[i_t],
                                      0, 0]).reshape(-1, 1)
                # new mu, considering new position/speed/acceleration
                remaining_m_time = first_resp_len-t_updt
                sign_ = resp_first[i_t]
                # this sets the maximum updating evidence equal to the ev bound
                updt_ev = np.clip(second_ev[i_t], a_min=-bound, a_max=bound)
                # second_response_len: motor time update influenced by difference
                # between the evidence second and first read-outs
                difference = (updt_ev - first_ev[i_t])*sign_
                # updated trajectory time
                second_response_len =\
                    float(remaining_m_time -
                          p_2nd_readout*(difference))
                # updated trajectory
                traj_fin = compute_traj(jerk_lock_ms, mu=mu_update,
                                        resp_len=second_response_len)
                # joined trajectories
                traj_before_uptd = prior0[0:t_updt]
                traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
                if com[i_t]:
                    opp_side_values = traj_updt.copy()
                    opp_side_values[np.sign(traj_updt) == resp_fin[i_t]] = 0
                    max_val_towards_opposite = np.max(np.abs(opp_side_values))
                    x_val_at_updt.append(max_val_towards_opposite)
                else:
                    x_val_at_updt.append(0)
            except Exception:
                traj_fin = [np.nan]
                traj_updt = np.concatenate((prior0, traj_fin))
                x_val_at_updt.append(0)
            total_traj.append(traj_updt)
            final_trajs.append(traj_fin)
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, frst_traj_motor_time,\
            x_val_at_updt
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None, None


def get_trajs_time(resp_len, traj_stamps, fix_onset, com, sound_len,
                   com_cond=False):
    time = []
    if com_cond:
        traj_st_com = traj_stamps[com.astype(bool)]
        fix_onset_com = fix_onset[com.astype(bool)]
        sound_len_com = sound_len[com.astype(bool)]
    else:
        traj_st_com = traj_stamps
        fix_onset_com = fix_onset
        sound_len_com = sound_len
    for j in range(len(traj_st_com)):
        t = traj_st_com[j] - fix_onset_com[j]
        t = (t.astype(int) / 1000_000 - 300 - sound_len_com[j])
        time.append(t)
    return np.array(time, dtype='object')


def data_augmentation(stim, daf, sigma=0):
    """

    Parameters
    ----------
    stim : array
        stim sequence for each trial 20xnum-trials.
    daf : float
        data augmentation factor (quantity of precision augmentation)
    sigma : float, optional
        Weight of the random part of the augmented stimulus. The default is 0.

    Returns
    -------
    augm_stim : array
        Augmented stimulus, from precision of 50ms to precision of 50/daf ms.

    """
    augm_stim = np.zeros((daf*stim.shape[0], stim.shape[1]))
    for tmstp in range(stim.shape[0]):
        augm_stim[daf*tmstp:daf*(tmstp+1), :] =\
            np.random.randn()*sigma+stim[tmstp, :]
    return augm_stim


def get_data_and_matrix(dfpath='C:/Users/Alexandre/Desktop/CRM/Alex/paper/',
                        num_tr_per_rat=int(1e4), after_correct=True, silent=False,
                        all_trials=False, return_df=False, srfail=False):
    """
    Import data for 1 rat.
    """
    print('Loading data')
    files = glob.glob(dfpath+'*.pkl')
    for f in files:
        df = pd.read_pickle(f)
        if return_df:
            if after_correct:
                if not silent:
                    if srfail:
                        return df.query(
                               "sound_len <= 400 and\
                                   resp_len <=1 and R_response>= 0\
                                       and hithistory >= 0 and special_trial == 0\
                                           and aftererror==0")
                    else:
                        return df.query(
                                "sound_len <= 400 and soundrfail ==\
                                    False and resp_len <=1 and R_response>= 0\
                                        and hithistory >= 0 and special_trial == 0\
                                            and aftererror==0")
                if silent:
                    if srfail:
                        return df.query(
                                "sound_len <= 400 and \
                                    resp_len <=1 and R_response>= 0\
                                        and hithistory >= 0\
                                            and aftererror==0")
                    else:
                        return df.query(
                                "sound_len <= 400 and soundrfail ==\
                                    False and resp_len <=1 and R_response>= 0\
                                        and hithistory >= 0\
                                            and aftererror==0")
            else:
                if not silent:
                    if srfail:
                        return df.query(
                            "sound_len <= 400 and\
                            resp_len <=1 and R_response>= 0\
                            and hithistory >= 0 and special_trial == 0")
                    else:
                        return df.query(
                            "sound_len <= 400 and soundrfail ==\
                                False and resp_len <=1 and R_response>= 0\
                                    and hithistory >= 0 and special_trial == 0")
                if silent:
                    if srfail:
                        return df.query(
                         "sound_len <= 400 and\
                         resp_len <=1 and R_response>= 0\
                         and hithistory >= 0")
                    else:
                        return df.query(
                            "sound_len <= 400 and soundrfail ==\
                            False and resp_len <=1 and R_response>= 0\
                            and hithistory >= 0")
