# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:29:19 2022

@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sbi.inference import MNLE
from sbi.utils import MultipleIndependent
import torch
from torch.distributions import Uniform
import pickle
from pybads import BADS
from scipy.spatial import distance as dist
import os

from extended_ddm_v2 import trial_ev_vectorized, data_augmentation
from skimage.transform import resize
import figures_paper as fp
import analyses_humans as ah
import different_models as model_variations


# ---GLOBAL VARIABLES
DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
# DATA_FOLDER = '/home/garciaduran/data/'  # Cluster Alex
# DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM

SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
# SV_FOLDER = '/home/garciaduran/opt_results/'  # Cluster Alex
# SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/opt_results/' # Jordi
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM

BINS = np.arange(1, 320, 20)
CTE = 1/2 * 1/600 * 1/995  # contaminants
CTE_FB = 1/600

plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})
plt.rcParams.update({'font.size': 14})


# ---FUNCTIONS
def simulation(stim, zt, coh, trial_index, p_w_zt,
               p_w_stim, p_e_bound, p_com_bound, p_t_aff, p_t_eff, p_t_a,
               p_w_a_intercept, p_w_a_slope, p_a_bound, p_1st_readout,
               p_2nd_readout, p_leak, p_mt_noise, p_MT_intercept, p_MT_slope,
               num_times_tr=int(1e3), mnle=True, extra_label=''):
    """
    Simulates the model with a parameter set and experimental conditions
    and returns a tensor with 3 columns and N trials rows.
    Columns correspond to:
        - movement time
        - reaction time
        - choice
    """
    if extra_label == '':
        model = trial_ev_vectorized
    if 'only_prior' in extra_label:
        model = model_variations.trial_ev_vectorized_only_prior_1st_choice
    data_augment_factor = 10
    if isinstance(coh, np.ndarray):
        num_tr = stim.shape[1]
        stim = stim[:, :int(num_tr)]
        zt = zt[:int(num_tr)]
        coh = coh[:int(num_tr)]
        trial_index = trial_index[:int(num_tr)]
        stim = data_augmentation(stim=stim, daf=data_augment_factor)
        stim_temp = np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                                    stim.shape[1]))))
    else:
        augm_stim = np.zeros((data_augment_factor*len(stim), 1))
        for tmstp in range(len(stim)):
            augm_stim[data_augment_factor*tmstp:data_augment_factor*(tmstp+1)] =\
                stim[tmstp]
        stim = augm_stim
        stim_temp = np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff), 1))))
        num_tr = 1
        stim_temp = np.array(stim_temp)
    compute_trajectories = True
    all_trajs = True

    stim_res = 50/data_augment_factor
    mt = torch.tensor(())
    rt = torch.tensor(())
    choice = torch.tensor(())
    for i in range(num_times_tr):
        _, _, com_model, first_ind, _, _, resp_fin,\
            _, _, total_traj, _, _,\
            _, x_val_at_updt =\
            model(zt=zt, stim=stim_temp, coh=coh,
                  trial_index=trial_index,
                  p_w_zt=p_w_zt, p_w_stim=p_w_stim,
                  p_e_bound=p_e_bound, p_com_bound=p_com_bound,
                  p_t_aff=p_t_aff, p_t_eff=p_t_eff, p_t_a=p_t_a,
                  num_tr=num_tr, p_w_a_intercept=p_w_a_intercept,
                  p_w_a_slope=p_w_a_slope,
                  p_a_bound=p_a_bound,
                  p_1st_readout=p_1st_readout,
                  p_2nd_readout=p_2nd_readout,
                  p_leak=p_leak, p_mt_noise=p_mt_noise,
                  p_MT_intercept=p_MT_intercept,
                  p_MT_slope=p_MT_slope,
                  compute_trajectories=compute_trajectories,
                  stim_res=stim_res, all_trajs=all_trajs)
        reaction_time = (first_ind-int(300/stim_res) + p_t_eff)*stim_res
        motor_time = np.array([len(t) for t in total_traj])
        first_ind = []
        total_traj = []
        mt = torch.cat((mt, torch.tensor(motor_time)))
        rt = torch.cat((rt, torch.tensor(reaction_time)))
        choice = torch.cat((choice, torch.tensor((resp_fin+1)/2)))
    x = torch.column_stack((mt, rt+300, choice))  # add fixation
    if mnle:
        return x
    if not mnle:
        return x.detach().numpy()


def build_prior_sample_theta(num_simulations):
    # 1. Parameters' prior distro definition
    prior =\
        MultipleIndependent([
            Uniform(torch.tensor([1e-3]),
                    torch.tensor([1.])),  # prior weight
            Uniform(torch.tensor([1e-3]),
                    torch.tensor([0.8])),  # stim weight
            Uniform(torch.tensor([1e-2]),
                    torch.tensor([4.])),  # evidence integrator bound
            Uniform(torch.tensor([1e-8]),
                    torch.tensor([1.])),  # CoM bound
            Uniform(torch.tensor([3.]),
                    torch.tensor([12.])),  # afferent time
            Uniform(torch.tensor([3.]),
                    torch.tensor([12.])),  # efferent time
            Uniform(torch.tensor([4.]),
                    torch.tensor([24.])),  # time offset action
            Uniform(torch.tensor([1e-2]),
                    torch.tensor([0.1])),  # intercept trial index for action drift
            Uniform(torch.tensor([1e-6]),
                    torch.tensor([5e-5])),  # slope trial index for action drift
            Uniform(torch.tensor([1.]),
                    torch.tensor([4.])),  # bound for action integrator
            Uniform(torch.tensor([1.]),
                    torch.tensor([500.])),  # weight of evidence at first readout (for MT reduction)
            Uniform(torch.tensor([1.]),
                    torch.tensor([500.])),  # weight of evidence at second readout
            Uniform(torch.tensor([1e-6]),
                    torch.tensor([0.9])),  # leak
            Uniform(torch.tensor([5.]),
                    torch.tensor([60.])),  # std of the MT noise
            Uniform(torch.tensor([120.]),
                    torch.tensor([400.])),  # MT offset
            Uniform(torch.tensor([0.01]),
                    torch.tensor([0.5]))],  # MT slope with trial index
            validate_args=False)

    # 2. define all theta space with samples from prior
    theta_all = prior.sample((num_simulations,))
    return prior, theta_all


def closest(lst, K):
    # returns index of closest value of K in a list lst
    return min(range(len(lst)), key=lambda i: abs(lst[i]-K))


def prob_rt_fb_action(t, v_a, t_a, bound_a):  # inverse gaussian
    # returns p(RT | theta) for RT < 0
    return (bound_a / np.sqrt(2*np.pi*(t - t_a)**3)) *\
        np.exp(- ((v_a**2)*((t-t_a) - bound_a/v_a)**2)/(2*(t-t_a)))


def get_log_likelihood_fb_psiam(rt_fb, theta_fb, eps, dt=5e-3):
    # returns -LLH ( RT | theta ) for RT < 0
    v_a = -theta_fb[:, 8]*theta_fb[:, -1] + theta_fb[:, 7]  # v_a = b_o - b_1*t_index
    v_a = v_a.detach().numpy()/dt
    bound_a = theta_fb[:, 9].detach().numpy()
    t_a = dt*(theta_fb[:, 6] + theta_fb[:, 5]).detach().numpy()
    t = rt_fb*1e-3
    prob = prob_rt_fb_action(t=t, v_a=v_a, t_a=t_a, bound_a=bound_a)
    prob[np.isnan(prob)] = 0
    # apply contaminants, log and sum to return - \sum LLH ( RT | theta )
    return -np.nansum(np.log(prob*(1-eps) + eps*CTE_FB))


def fun_theta(theta, data, estimator, n_trials, eps=1e-3, weight_LLH_fb=1):
    """
    Returns -sum(log likelihood (MT, RT, choice | theta)).
    """
    zt = data[:, 0]
    coh = data[:, 1]
    trial_index = data[:, 2]
    x_o = data[:, 3::]
    theta = torch.reshape(torch.tensor(theta),
                          (1, len(theta))).to(torch.float32)
    theta = theta.repeat(n_trials, 1)
    theta[:, 0] *= torch.tensor(zt[:n_trials])
    theta[:, 1] *= torch.tensor(coh[:n_trials])
    t_i = torch.tensor(trial_index[:n_trials]).to(torch.float32)
    theta = torch.column_stack((theta, t_i))
    x_o = x_o[:n_trials].detach().numpy()
    # trials with RT >= 0
    # we have to pass the same parameters as for the training (14 columns)
    x_o_no_fb = torch.tensor(
        x_o[np.isnan(x_o).sum(axis=1) == 0, :]).to(torch.float32)
    theta_no_fb = torch.tensor(
        theta.detach().numpy()[np.isnan(x_o).sum(axis=1) == 0, :]).to(torch.float32)
    theta_no_fb[:, 14] += theta_no_fb[:, 15]*theta_no_fb[:, -1]
    theta_no_fb[:, 7] -= theta_no_fb[:, 8]*theta_no_fb[:, -1]
    theta_no_fb = torch.column_stack((theta_no_fb[:, :8],
                                      theta_no_fb[:, 9:15]))
    # take log prob from MNLE
    x_o = x_o.reshape((1, x_o.shape[0], x_o.shape[1]))
    log_liks = estimator.log_prob(x_o_no_fb, condition=theta_no_fb).detach().numpy()
    log_liks = np.exp(log_liks)*(1-eps) + eps*CTE
    log_liks = np.log(log_liks)
    log_liks_no_fb = -np.nansum(log_liks)  # -LLH (data | theta) for RT > 0
    # trials with RT < 0
    # we use the analytical computation of p(RT | parameters) for FB
    x_o_with_fb = x_o[np.isnan(x_o).sum(axis=1) > 0, :]
    theta_fb = theta[np.isnan(x_o).sum(axis=1) > 0, :]
    log_liks_fb = get_log_likelihood_fb_psiam(rt_fb=x_o_with_fb[:, 1],
                                              theta_fb=theta_fb, eps=eps)
    # returns -LLH (data (RT > 0) | theta) + -LLH (data (RT < 0) | theta)
    return log_liks_fb*weight_LLH_fb + log_liks_no_fb  # *(1-weight_LLH_fb)


def simulations_for_mnle(theta_all, stim, zt, coh, trial_index,
                         simulate=False, extra_label=''):
    # run simulations
    x = torch.tensor(())
    simul_data = SV_FOLDER+'/network/NN_simulations'+str(len(zt))+'.npy'
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(simul_data), exist_ok=True)
    if os.path.exists(simul_data) and not simulate:
        print('Loading Simulated Data')
        x = np.load(simul_data, allow_pickle=True)
        x = torch.tensor(x).to(torch.float32)
    else:
        print('Starting simulation')
        time_start = time.time()
        for i_t, theta in enumerate(theta_all):  # simulation trial by trial
            if (i_t+1) % 100000 == 0 and i_t != 0:
                print('Simulation number: ' + str(i_t+1))
                print('Time elapsed: ' + str((time.time()-time_start)/60) +
                      ' mins')
            # get the parameters
            p_w_zt = float(theta[0])
            p_w_stim = float(theta[1])
            p_e_bound = float(theta[2])
            p_com_bound = float(theta[3])*p_e_bound
            p_t_aff = int(np.round(theta[4]))
            p_t_eff = int(np.round(theta[5]))
            p_t_a = int(np.round(theta[6]))
            p_w_a_intercept = float(theta[7])
            p_w_a_slope = -float(theta[8])
            p_a_bound = float(theta[9])
            p_1st_readout = float(theta[10])
            p_2nd_readout = float(theta[11])
            p_leak = float(theta[12])
            p_mt_noise = float(theta[13])
            p_mt_intercept = float(theta[14])
            p_mt_slope = float(theta[15])
            try:
                # run simulation
                x_temp = simulation(stim[i_t, :], zt[i_t], coh[i_t],
                                    np.array([trial_index[i_t]]),
                                    p_w_zt, p_w_stim, p_e_bound, p_com_bound,
                                    p_t_aff, p_t_eff, p_t_a, p_w_a_intercept,
                                    p_w_a_slope, p_a_bound, p_1st_readout,
                                    p_2nd_readout, p_leak, p_mt_noise,
                                    p_mt_intercept, p_mt_slope,
                                    num_times_tr=1, mnle=True,
                                    extra_label=extra_label)
            except ValueError:  # in case there is a problem with the parameter combination
                x_temp = torch.tensor([[np.nan, np.nan, np.nan]])
            x = torch.cat((x, x_temp))
        x = x.to(torch.float32)
        np.save(simul_data, x.detach().numpy())
    return x


def get_x0():
    # Returns an initial configuration of parameters
    p_t_aff = 6
    p_t_eff = 6
    p_t_a = 12
    p_w_zt = 0.5
    p_w_stim = 0.12
    p_e_bound = 2.
    p_com_bound = 0.1
    p_w_a_intercept = 0.05
    p_w_a_slope = 2e-5
    p_a_bound = 2.6
    p_1st_readout = 200
    p_2nd_readout = 170
    p_leak = 0.05
    p_mt_noise = 9
    p_MT_intercept = 320
    p_MT_slope = 0.07
    return [p_w_zt, p_w_stim, p_e_bound, p_com_bound, p_t_aff,
            p_t_eff, p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
            p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
            p_MT_intercept, p_MT_slope]


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
    lb_w_zt = 1e-4
    lb_w_st = 1e-4
    lb_e_bound = 0.01
    lb_com_bound = 0
    lb_w_intercept = 0.01
    lb_w_slope = 1e-6
    lb_a_bound = 1
    lb_1st_r = 75
    lb_2nd_r = 50
    lb_leak = 1e-6
    lb_mt_n = 5
    lb_mt_int = 120
    lb_mt_slope = 0.01
    return [lb_w_zt, lb_w_st, lb_e_bound, lb_com_bound, lb_aff,
            lb_eff, lb_t_a, lb_w_intercept, lb_w_slope, lb_a_bound,
            lb_1st_r, lb_2nd_r, lb_leak, lb_mt_n,
            lb_mt_int, lb_mt_slope]


def get_lb_human():
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
    lb_w_zt = 0
    lb_w_st = 0
    lb_e_bound = 0.3
    lb_com_bound = 0
    lb_w_intercept = 0.01
    lb_w_slope = 1e-6
    lb_a_bound = 0.1
    lb_1st_r = 75
    lb_2nd_r = 75
    lb_leak = 0.
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
    ub_t_a = 24
    ub_w_zt = 1
    ub_w_st = 0.8
    ub_e_bound = 4
    ub_com_bound = 1
    ub_w_intercept = 0.1
    ub_w_slope = 5e-5
    ub_a_bound = 4
    ub_1st_r = 500
    ub_2nd_r = 500
    ub_leak = 0.08
    ub_mt_n = 10
    ub_mt_int = 400
    ub_mt_slope = 0.5
    return [ub_w_zt, ub_w_st, ub_e_bound, ub_com_bound, ub_aff,
            ub_eff, ub_t_a, ub_w_intercept, ub_w_slope, ub_a_bound,
            ub_1st_r, ub_2nd_r, ub_leak, ub_mt_n,
            ub_mt_int, ub_mt_slope]


def get_ub_human():
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
    ub_w_st = 0.2
    ub_e_bound = 4
    ub_com_bound = 1
    ub_w_intercept = 0.12
    ub_w_slope = 1e-3
    ub_a_bound = 4
    ub_1st_r = 500
    ub_2nd_r = 500
    ub_leak = 0.15
    ub_mt_n = 20
    ub_mt_int = 370
    ub_mt_slope = 0.6
    return [ub_w_zt, ub_w_st, ub_e_bound, ub_com_bound, ub_aff,
            ub_eff, ub_t_a, ub_w_intercept, ub_w_slope, ub_a_bound,
            ub_1st_r, ub_2nd_r, ub_leak, ub_mt_n,
            ub_mt_int, ub_mt_slope]


def get_pub():
    """
    Returns list with plausible upper bounds (PUB) for BADS optimization.

    Returns
    -------
    list
        List with plausible upper bounds.

    """
    pub_aff = 10
    pub_eff = 10
    pub_t_a = 18
    pub_w_zt = 0.8
    pub_w_st = 0.5
    pub_e_bound = 3.5
    pub_com_bound = 0.15
    pub_w_intercept = 0.08
    pub_w_slope = 1e-4
    pub_a_bound = 3
    pub_1st_r = 400
    pub_2nd_r = 400
    pub_leak = 0.08
    pub_mt_n = 20
    pub_mt_int = 320
    pub_mt_slope = 0.12
    return [pub_w_zt, pub_w_st, pub_e_bound, pub_com_bound, pub_aff,
            pub_eff, pub_t_a, pub_w_intercept, pub_w_slope, pub_a_bound,
            pub_1st_r, pub_2nd_r, pub_leak, pub_mt_n,
            pub_mt_int, pub_mt_slope]


def get_plb():
    """
    Returns list with plausible lower bounds (PLB) for BADS optimization.

    Returns
    -------
    list
        List with plausible lower bounds.

    """
    plb_aff = 5
    plb_eff = 5
    plb_t_a = 7
    plb_w_zt = 0.2
    plb_w_st = 0.04
    plb_e_bound = 1.6
    plb_com_bound = 1e-2
    plb_w_intercept = 0.03
    plb_w_slope = 1.5e-5
    plb_a_bound = 2.2
    plb_1st_r = 90
    plb_2nd_r = 90
    plb_leak = 0.05
    plb_mt_n = 8
    plb_mt_int = 260
    plb_mt_slope = 0.04
    return [plb_w_zt, plb_w_st, plb_e_bound, plb_com_bound, plb_aff,
            plb_eff, plb_t_a, plb_w_intercept, plb_w_slope, plb_a_bound,
            plb_1st_r, plb_2nd_r, plb_leak, plb_mt_n,
            plb_mt_int, plb_mt_slope]


def nonbox_constraints_bads(x):
    """
    Constraints for BADS: 40 ms < t_aff + t_eff < 80 ms.
    """
    x_1 = np.atleast_2d(x)
    cond6 = np.int32(x_1[:, 4]) + np.int32(x_1[:, 5]) < 8  # aff + eff < 40 ms
    cond7 = np.int32(x_1[:, 4]) + np.int32(x_1[:, 5]) > 16  # aff + eff > 80 ms
    return np.bool_(cond6 + cond7)


def prepare_fb_data(df):
    """
    Function that extracts FB trials from the data frame.
    Returns a tensor with three columns (movement and reaction time, and choice).
    If there is a fixation break (RT < 0 ms), movement time and choice will be NaNs.
    """
    print('Preparing FB data')
    coh_vec = df.coh2.values
    dwl_vec = df.dW_lat.values
    dwt_vec = df.dW_trans.values
    mt_vec = df.resp_len.values
    ch_vec = df.R_response.values
    tr_in_vec = df.origidx.values
    for ifb, fb in enumerate(df.fb):
        for j in range(len(fb)):
            coh_vec = np.append(coh_vec, [df.coh2.values[ifb]])
            dwl_vec = np.append(dwl_vec, [df.dW_lat.values[ifb]])
            dwt_vec = np.append(dwt_vec, [df.dW_trans.values[ifb]])
            mt_vec = np.append(mt_vec, [np.nan])
            ch_vec = np.append(ch_vec, [np.nan])
            tr_in_vec = np.append(tr_in_vec, [df.origidx.values[ifb]])
    rt_vec =\
        np.vstack(np.concatenate([df.sound_len,
                                  1e3*(np.concatenate(
                                      df.fb.values)-0.3)])).reshape(-1)+300
    zt_vec = np.nansum(np.column_stack((dwl_vec, dwt_vec)), axis=1)
    x_o = torch.column_stack((torch.tensor(mt_vec*1e3),
                              torch.tensor(rt_vec),
                              torch.tensor(ch_vec)))
    data = torch.column_stack((torch.tensor(zt_vec), torch.tensor(coh_vec),
                               torch.tensor(tr_in_vec.astype(float)),
                               x_o))
    data = data[np.round(rt_vec) > 50, :]
    return data


def opt_mnle(df, num_simulations, bads=True, training=False, extra_label=""):
    """
    MNLE network training and BADS optimization.
    """
    if training:
        # 1st: loading data
        zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
        stim = np.array([stim for stim in df.res_sound])
        coh = np.array(df.coh2)
        trial_index = np.array(df.origidx)
        stim[df.soundrfail, :] = 0
        # Prepare data:
        coh = np.resize(coh, num_simulations)
        # np.random.shuffle(coh)
        zt = np.resize(zt, num_simulations)
        # np.random.shuffle(zt)
        trial_index = np.resize(trial_index, num_simulations)
        # np.random.shuffle(trial_index)
        stim = np.resize(stim, (num_simulations, 20))
        # np.random.shuffle(stim)
        if not bads:
            # motor time: in seconds (must be multiplied then by 1e3)
            mt = df.resp_len.values
            choice = df.R_response.values
            sound_len = np.array(df.sound_len)
            mt = np.resize(mt, num_simulations)
            choice = np.resize(choice, num_simulations)
            # com = np.resize(com, num_simulations)
            # choice_and_com = com + choice*2
            rt = np.resize(sound_len + 300, num_simulations)
            # w.r.t fixation onset
            x_o = torch.column_stack((torch.tensor(mt*1e3),  # MT in ms
                                      torch.tensor(rt),
                                      torch.tensor(choice)))
            x_o = x_o.to(torch.float32)
            # to save some memory
            choice = []
            rt = []
            mt = []
        print('Data preprocessed, building prior distros')
        # build prior: ALL PARAMETERS ASSUMED POSITIVE
        df = []  # ONLY FOR TRAINING
        prior, theta_all = build_prior_sample_theta(num_simulations=num_simulations)
        # add zt, coh, trial index
        theta_all_inp = theta_all.clone().detach()
        theta_all_inp[:, 0] *= torch.tensor(zt[:num_simulations]).to(torch.float32)
        theta_all_inp[:, 1] *= torch.tensor(coh[:num_simulations]).to(torch.float32)
        theta_all_inp = torch.column_stack((
            theta_all_inp, torch.tensor(
                trial_index[:num_simulations].astype(float)).to(torch.float32)))
        theta_all_inp = theta_all_inp.to(torch.float32)
        # SIMULATION
        x = simulations_for_mnle(theta_all, stim, zt, coh, trial_index, simulate=True,
                                 extra_label=extra_label)
        # now we have a matrix of (num_simulations x 3):
        # MT, RT, CHOICE for each simulation

        # NETWORK TRAINING
        # transform parameters related to trial index. 14 params instead of 17
        # MT_in = MT_0 + MT_1*trial_index
        theta_all_inp[:, 14] += theta_all_inp[:, 15]*theta_all_inp[:, -1]
        # V_A = vA_0 - vA_1*trial_index
        theta_all_inp[:, 7] -= theta_all_inp[:, 8]*theta_all_inp[:, -1]
        theta_all_inp = torch.column_stack((theta_all_inp[:, :8],
                                            theta_all_inp[:, 9:15]))
        coh = []
        zt = []
        trial_index = []
        stim = []
        nan_mask = torch.sum(torch.isnan(x), axis=1).to(torch.bool)
        # define network MNLE
        trainer = MNLE(prior=prior)
        time_start = time.time()
        print('Starting network training')
        # network training
        trainer = trainer.append_simulations(theta_all_inp[~nan_mask, :],
                                             x[~nan_mask, :])
        estimator = trainer.train(show_train_summary=True)
        # save the network
        with open(SV_FOLDER + f"/mnle_n{num_simulations}_no_noise" + extra_label + ".p",
                  "wb") as fh:
            pickle.dump(dict(estimator=estimator,
                             num_simulations=num_simulations), fh)
        with open(SV_FOLDER + f"/trainer_n{num_simulations}_no_noise" + extra_label + ".p",
                  "wb") as fh:
            pickle.dump(dict(trainer=trainer,
                             num_simulations=num_simulations), fh)
        print('For a batch of ' + str(num_simulations) +
              ' simulations, it took ' + str(int(time.time() - time_start)/60)
              + ' mins')
    else:  # network is already trained
        x_o = []
        with open(SV_FOLDER + f"/mnle_n{num_simulations}_no_noise" + extra_label + ".p",
                  'rb') as f:
            estimator = pickle.load(f)
        if not bads:
            with open(SV_FOLDER + f"/trainer_n{num_simulations}_no_noise" + extra_label + ".p",
                      'rb') as f:
                trainer = pickle.load(f)
            trainer = estimator['trainer']
        estimator = estimator['estimator']
    if bads:
        # starting point
        x0 = get_x0()
        print('Initial guess is: ' + str(x0))
        time_start = time.time()
        # define upper/lower bounds
        lb = np.array(get_lb())
        ub = np.array(get_ub())
        plb = lb + (-lb+ub)/10
        pub = ub - (-lb+ub)/10
        # get fixation break (FB) data
        data = prepare_fb_data(df=df)
        print('Optimizing')
        n_trials = len(data)
        # define fun_target as function to optimize
        # returns -LLH( data | parameters )
        fun_target = lambda x: fun_theta(x, data, estimator, n_trials)  # f(theta | data, MNLE)
        # define optimizer (BADS)
        bads = BADS(fun_target, x0, lb, ub, plb, pub,
                    non_box_cons=nonbox_constraints_bads)
        # optimization
        optimize_result = bads.optimize()
        print(optimize_result.total_time)
        return optimize_result.x  # return parameters


def create_parameters_prt(sv_folder=SV_FOLDER, n_sims=50):
    """
    Function that saves n_sims parameter configurations to perform parameter
    recovery.
    """
    ub = np.array(get_ub())
    lb = np.array(get_lb())
    plb = lb + (-lb+ub)/10
    pub = ub - (-lb+ub)/10
    for i in range(n_sims):
        pars = []
        for j in range(len(lb)):
            par = np.random.uniform(low=plb[j], high=pub[j])
            pars.append(par)
        pars = np.array(pars)
        path = sv_folder + '/virt_params/parameters_MNLE_BADS_prt_n50_prt_' + str(i) + '.npy'
        np.save(path, pars)


def parameter_recovery_test_data_frames(df, subjects, extra_label=''):
    """
    Extracts (or simulates) simulated data from the 50 generated parameter configurations.
    """
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
    coh = np.array(df.coh2)
    stim = np.array([stim for stim in df.res_sound])
    if stim.shape[0] != 20:
        stim = stim.T
    gt = np.array(df.rewside) * 2 - 1
    subjects = ['Virtual_rat_random_params']
    subjid = np.array(subjects*len(coh))
    trial_index = np.array(df.origidx)
    # run model
    hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
        _, trajs, x_val_at_updt =\
        fp.run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                          trial_index=trial_index, num_tr=len(df),
                                          subject_list=subjects, subjid=subjid,
                                          simulate=False,
                                          extra_label=extra_label)
    MT = [len(t) for t in trajs]
    # returns df with MT, RT and choice
    df['sound_len'] = reaction_time
    df['resp_len'] = np.array(MT)*1e-3
    df['R_response'] = (resp_fin+1)/2
    return df


def matrix_probs(x, bins_rt=np.arange(200, 600, 13),
                 bins_mt=np.arange(100, 600, 26)):
    """
    Compute 2D histogram of MT and RT for simulated trials.
    """
    mt = np.array(x[:, 0])
    rt = np.array(x[:, 1])
    counts = np.histogram2d(mt, rt, bins=[bins_mt, bins_rt])[0]
    counts /= np.sum(counts)
    return counts


def plot_network_model_comparison(df, ax, sv_folder=SV_FOLDER, num_simulations=int(5e5),
                                  n_list=[4000000], cohval=0.5, ztval=0.5, tival=10,
                                  plot_nn=False, simulate=False, plot_model=True,
                                  plot_nn_alone=False, xt=False, eps=1e-5, n_trials_sim=100):
    """
    Plots model likelihood with the network approximate likelihood on top.
    """
    # to simulate
    if simulate:
        # we will simulate 100 randomly selected trials
        coh = df.coh2.values
        zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
        trial_index = df.origidx.values
        idxs = np.random.choice(np.arange(len(coh)), size=100)
        cohvals = coh[idxs]
        coh = []
        ztvals = np.round(zt[idxs], 2)
        zt = []
        tivals = trial_index[idxs]
        trial_index = []
        stims = np.array(
            [stim for stim in df.res_sound])[idxs]
        # save experimental conditions of 100 randomly selected trials
        np.save(sv_folder + '/10M/cohvals.npy', cohvals)
        np.save(sv_folder + '/10M/ztvals.npy', ztvals)
        np.save(sv_folder + '/10M/tivals.npy', tivals)
        np.save(sv_folder + '/10M/stims.npy', stims)
        np.save(sv_folder + '/10M/idxs.npy', idxs)
        i = 0
        for cohval, ztval, tival in zip(cohvals, ztvals, tivals): # for each combination
            stim = stims[i]
            theta = get_x0()  # select set of parameters defined by get_x0()
            theta = torch.reshape(torch.tensor(theta),
                                  (1, len(theta))).to(torch.float32)
            theta = theta.repeat(num_simulations, 1)
            stim = np.array(
                [np.concatenate((stim, stim)) for i in range(len(theta))])
            trial_index = np.repeat(tival, len(theta))
            # simulate
            x = simulations_for_mnle(theta_all=np.array(theta), stim=stim,
                                     zt=np.repeat(ztval, len(theta)),
                                     coh=np.repeat(cohval, len(theta)),
                                     trial_index=trial_index, simulate=True)
            np.save(sv_folder + '/10M/coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival), x)
            # let's compute prob for each bin (2d histogram of MT, RT)
            mat_0 = matrix_probs(x[x[:, 2] == 0])  # choice = 0
            mat_1 = matrix_probs(x[x[:, 2] == 1])  # choice = 1
            np.save(sv_folder + '/10M/mat0_coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival), mat_0)
            np.save(sv_folder + '/10M/mat1_coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival), mat_1)
            x = []
            mat_0 = []
            mat_1 = []
            i += 1
    else:
        if not plot_nn_alone:
            mat_0 = np.load(SV_FOLDER + '/10M/mat0_coh{}_zt{}_ti{}.npy'
                            .format(cohval, ztval, tival))
            mat_1 = np.load(SV_FOLDER + '/10M/mat1_coh{}_zt{}_ti{}.npy'
                            .format(cohval, ztval, tival))
            x = np.load(SV_FOLDER + '/10M/coh{}_zt{}_ti{}.npy'
                        .format(cohval, ztval, tival))
        trial_index = np.repeat(tival, num_simulations)
    # grid of MT, RT on which approximate likelihood of the network will be computed
    grid_rt = np.arange(-100, 300, 1) + 300
    grid_mt = np.arange(100, 600, 1)
    all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    mat_0_nn = np.empty((len(grid_mt), len(grid_rt)))
    mat_1_nn = np.copy(mat_0_nn)
    if plot_nn:
        for n_sim_train in n_list:
            # we load estimator
            with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_sim_train),
                      'rb') as f:
                estimator = pickle.load(f)
            estimator = estimator['estimator']
            # load parameter configuration
            theta = get_x0()
            theta = torch.reshape(torch.tensor(theta),
                                  (1, len(theta))).to(torch.float32)
            theta = theta.repeat(len(x_o), 1)
            # prepare parameters considering experimental conditions
            theta[:, 0] *= torch.tensor(ztval)
            theta[:, 1] *= torch.tensor(cohval)
            theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                                torch.tensor(trial_index[
                                                    :len(x_o)]).to(torch.float32)))
            theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
            theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
            theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                                theta_tri_ind[:, 9:15]))
            # compute log likelihood of the network
            x_o = x_o.reshape((1, x_o.shape[0], x_o.shape[1]))
            lprobs = estimator.log_prob(x_o, theta_tri_ind)
            lprobs = torch.exp(lprobs)
            mat_0_nn = lprobs[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                      len(grid_rt)).detach().numpy()
            mat_1_nn = lprobs[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                      len(grid_rt)).detach().numpy()
            if plot_nn_alone:  # to plot the network alone
                fig, ax1 = plt.subplots(ncols=2)
                fig.suptitle('Network + {}'.format(n_sim_train))
                cte_nn1 = np.sum(mat_0_nn + mat_1_nn)
                mat_0_nn /= cte_nn1
                mat_1_nn /= cte_nn1
                ax1[0].imshow(mat_0_nn, vmin=0, vmax=np.max((mat_0_nn, mat_1_nn)))
                ax1[0].set_title('Choice 0')
                ax1[0].set_yticks(np.arange(0, len(grid_mt), 50), grid_mt[::50])
                ax1[0].set_ylabel('MT (ms)')
                ax1[0].set_xticks(np.arange(0, len(grid_rt), 50), grid_rt[::50]-300)
                ax1[0].set_xlabel('RT (ms)')
                im1 = ax1[1].imshow(mat_1_nn, vmin=0, vmax=np.max((mat_0_nn, mat_1_nn)))
                ax1[1].set_title('Choice 1')
                ax1[1].set_yticks([])
                ax1[1].set_xticks(np.arange(0, len(grid_rt), 50), grid_rt[::50]-300)
                ax1[1].set_xlabel('RT (ms)')
                plt.colorbar(im1)
                return
            # plot and tune panels
            ax[0].imshow(resize(mat_0, mat_0_nn.shape), vmin=0, cmap='Blues')
            ax[0].contour(mat_0_nn, cmap='Reds', linewidths=0.8)
            ax[0].set_yticks(np.arange(0, len(grid_mt), 100), grid_mt[::100])
            ax[0].set_ylabel('MT (ms)')
            if xt:
                ax[0].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
                                 rotation=45)
                ax[0].set_xlabel('RT (ms)')
            else:
                ax[0].set_xticks([])
            im1 = ax[1].imshow(resize(mat_1, mat_1_nn.shape), vmin=0, cmap='Blues')
            plt.sca(ax[1])
            ax[1].contour(mat_1_nn, cmap='Reds', linewidths=0.8)
            ax[1].set_yticks([])
            if xt:
                ax[1].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
                                 rotation=45)
                ax[1].set_xlabel('RT (ms)')
            else:
                ax[1].set_xticks([])
            # compute p(right) and p(FB) for model and network
            p_ch0_model = np.nanmean(x[:,2] == 0)
            p_fb_model = np.nanmean(x[:, 1][x[:, 1] > 200] < 300)
            p_ch1_model = 1-p_ch0_model
            p_ch1_nn = np.nansum(mat_1_nn)/np.nansum(mat_0_nn+mat_1_nn)
            idx = grid_rt < 300
            p_fb_nn = (np.nansum(mat_0_nn[:, idx]) + np.nansum(mat_1_nn[:, idx])) /\
                np.nansum(mat_0_nn + mat_1_nn)
            # tune and plot
            ax[2].set_ylim(-0.05, 1.05)
            ax[2].bar(['Model', 'NN'],
                      [p_ch1_model, p_ch1_nn],
                      color=['cornflowerblue', 'firebrick'])
            ax[2].set_ylabel('p(Right)')
            ax[3].set_ylim(-0.02, 0.15)
            ax[3].bar(['Model', 'NN'],
                      [p_fb_model, p_fb_nn],
                      color=['cornflowerblue', 'firebrick'])
            ax[3].set_ylabel('p(FB)')


def plot_nn_to_nn_comparison(n_trials=10000000, sv_folder=SV_FOLDER):
    """
    Plots supplementary figure 15. 
    Comparison between two networks trained on 10M different simulations.
    """
    # create figure and tune panels
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    ax = ax.flatten()
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.12, right=0.95,
                        hspace=0.4, wspace=0.4)
    ax[1].set_title('Right choice')
    ax[0].set_title('Left choice')
    ax[3].set_title('Right choice')
    ax[2].set_title('Left choice')
    # create MT, RT grid to evaluate network's approximated likelihood
    grid_rt = np.arange(-100, 300, 1) + 300
    grid_mt = np.arange(100, 600, 1)
    all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    # we load estimator
    with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_trials),
              'rb') as f:
        estimator_1 = pickle.load(f)
    estimator_1 = estimator_1['estimator']
    with open(SV_FOLDER + "/mnle_n{}_no_noise_v2.p".format(n_trials),
              'rb') as f:
        estimator_2 = pickle.load(f)
    estimator_2 = estimator_2['estimator']
    # for these experimental conditions (different trials)
    ztvals = [1.5, 0.05, 1.5, -1.5, .5, .5]
    cohvals = [0, 1, 0.5, 0.5, 0.25, 0.25]
    tivals = [400, 400, 400, 400, 10, 800]
    p = 0
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g' , ' ']
    for ztval, cohval, tival in zip(ztvals, cohvals, tivals):  # for each trial
        if p <= 12:
            ax[p].text(-0.16, 1.2, letters[p // 2], transform=ax[p].transAxes,
                       fontsize=16, fontweight='bold', va='top', ha='right')
        pos_ax_1 = ax[p+1].get_position()
        ax[p+1].set_position([pos_ax_1.x0 - pos_ax_1.width/9,
                              pos_ax_1.y0, pos_ax_1.width,
                              pos_ax_1.height])
        # load parameter configuration
        theta = get_x0()
        theta = torch.reshape(torch.tensor(theta),
                              (1, len(theta))).to(torch.float32)
        theta = theta.repeat(len(x_o), 1)
        trial_index = np.repeat(tival, len(theta))
        # prepare parameters considering experimental conditions
        theta[:, 0] *= torch.tensor(ztval)
        theta[:, 1] *= torch.tensor(cohval)
        theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                            torch.tensor(trial_index[
                                                :len(x_o)]).to(torch.float32)))
        theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
        theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
        theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                            theta_tri_ind[:, 9:15]))
        x_o = x_o.reshape((1, x_o.shape[0], x_o.shape[1]))
        # compute log likelihood of network 1
        lprobs1 = estimator_1.log_prob(x_o, theta_tri_ind)
        lprobs1 = torch.exp(lprobs1)
        mat_0_nn1 = lprobs1[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        mat_1_nn1 = lprobs1[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        # compute log likelihood of network 2
        lprobs2 = estimator_2.log_prob(x_o, theta_tri_ind)
        lprobs2 = torch.exp(lprobs2)
        mat_0_nn2 = lprobs2[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        mat_1_nn2 = lprobs2[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        # plot contours of both networks
        ax[p].contour(mat_0_nn1, cmap='hot', linewidths=1.2)
        ax[p].contour(mat_0_nn2, cmap='cool', linewidths=1.2)
        ax[p+1].contour(mat_1_nn1, cmap='hot', linewidths=1.2)
        ax[p+1].contour(mat_1_nn2, cmap='cool', linewidths=1.2)
        # tune panels
        if p % 4 == 0:
            ax[p].set_ylabel('MT (ms)')
            ax[p].set_yticks(np.arange(0, len(grid_mt), 100), grid_mt[::100])
            ax[p+1].set_yticks([])
        else:
            ax[p].set_yticks([])
            ax[p+1].set_yticks([])
        if p >= 8:
            ax[p].set_xlabel('RT (ms)')
            ax[p+1].set_xlabel('RT (ms)')
            ax[p].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
                             rotation=45)
            ax[p+1].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
                               rotation=45)
        else:
            ax[p].set_xticks([])
            ax[p+1].set_xticks([])
        pos_ax_12 = ax[p].get_position()
        ax[p].set_position([pos_ax_12.x0 + pos_ax_12.width/12,
                             pos_ax_12.y0, pos_ax_12.width, pos_ax_12.height])
        pos_ax_12 = ax[p+1].get_position()
        ax[p+1].set_position([pos_ax_12.x0 - pos_ax_12.width/7,
                             pos_ax_12.y0, pos_ax_12.width, pos_ax_12.height])
        p += 2
    # tune panels
    ax[13].axis('off')
    ax[15].axis('off')
    pos_ax_12 = ax[12].get_position()
    ax[12].set_position([pos_ax_12.x0 + pos_ax_12.width/3,
                         pos_ax_12.y0-pos_ax_12.height/5, pos_ax_12.width*1.8,
                         pos_ax_12.height])
    pos_ax_12 = ax[14].get_position()
    ax[14].set_position([pos_ax_12.x0 + pos_ax_12.width/4,
                         pos_ax_12.y0-pos_ax_12.height/5, pos_ax_12.width*1.8,
                         pos_ax_12.height])
    # plot distance vs N of trials
    supp_plot_dist_vs_n(ax=[ax[12], ax[14]])
    ax[12].text(-0.15, 1.2, letters[-2], transform=ax[12].transAxes,
               fontsize=16, fontweight='bold', va='top', ha='right')
    fig.savefig(sv_folder+'supp_fig_5.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'supp_fig_5.png', dpi=400, bbox_inches='tight')
    

def rm_top_right_lines(ax, right=True):
    """
    Same as figures_paper.py
    Function to remove top and right (or left) lines from panels.
    """
    if right:
        ax.spines['right'].set_visible(False)
    else:
        ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)


def supp_plot_lh_model_network(df, sv_folder=SV_FOLDER, n_trials=10000000):
    """
    Plots supplementary figure 14. 
    Comparison between two networks trained on 10M different simulations.
    """
    fig, ax = plt.subplots(6, 4, figsize=(8, 12))
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95,
                        hspace=0.4, wspace=0.4)
    col_labs = ['a', '', 'b', 'c']
    ax = ax.flatten()
    row_labs = ['i', 'ii', 'iii', 'iv', 'v', 'vi']
    for i in range(4):
        ax[i].text(-0.65, 1.45, col_labs[i], transform=ax[i].transAxes,
                   fontsize=12, fontweight='bold', va='top', ha='right')
    for j in range(6):  # for each trial, tune panels
        pos_ax_1 = ax[4*j].get_position()
        ax[4*j].set_position([pos_ax_1.x0,
                              pos_ax_1.y0, pos_ax_1.width*1.1, pos_ax_1.height])
        pos_ax_1 = ax[4*j+1].get_position()
        ax[4*j+1].set_position([pos_ax_1.x0 - pos_ax_1.width/4,
                            pos_ax_1.y0, pos_ax_1.width*1.1, pos_ax_1.height])
        pos_ax_1 = ax[4*j+2].get_position()
        ax[4*j+2].set_position([pos_ax_1.x0,
                                pos_ax_1.y0, pos_ax_1.width*0.7,
                                pos_ax_1.height])
        rm_top_right_lines(ax[4*j+2])
        pos_ax_1 = ax[4*j+3].get_position()
        ax[4*j+3].set_position([pos_ax_1.x0,
                                pos_ax_1.y0, pos_ax_1.width*0.7,
                                pos_ax_1.height])
        rm_top_right_lines(ax[4*j+3])
        ax[4*j].text(-0.4, 1.3, row_labs[j], transform=ax[4*j].transAxes,
                     fontsize=12, fontweight='bold', va='top', ha='right')
    i = 0
    xt = False
    ax[1].set_title('Right choice', pad=14, fontsize=11)
    ax[0].set_title('Left choice', pad=14, fontsize=11)
    for cohval, ztval, tival in zip([0, 1, 0.5, 0.25, 0.5, 0.25],
                                    [1.5, 0.05, -1.5, .5, 1.5, 0.5],
                                    [400, 400, 400, 10, 400, 800]):  # for each trial
        if i == 5:
            xt = True
        # plot model - network comparison
        plot_network_model_comparison(df, ax[4*i:4*(i+1)],
                                      sv_folder=SV_FOLDER, num_simulations=int(5e5),
                                      n_list=[n_trials], cohval=cohval,
                                      ztval=ztval, tival=tival,
                                      plot_nn=True, simulate=False, plot_model=False,
                                      plot_nn_alone=False, xt=xt)
        i += 1
    # save figure
    fig.savefig(sv_folder+'supp_fig_4.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'supp_fig_4.png', dpi=400, bbox_inches='tight')


def bhatt_dist(p, q):
    """
    Computes the Bhattacharyya distance between p and q.
    """
    return -np.log(np.sum(np.sqrt(p*q)))


def supp_plot_dist_vs_n(ax, n_list=[1000, 10000, 100000, 500000, 1000000, 2000000,
                                    4000000, 10000000]):
    """
    Plot distance between model and networks trained on increasing number of trials.
    """
    # retrieve simulated data on 100 random selected trials
    cohvals = np.load(SV_FOLDER + '/10M/100_sims/cohvals.npy', allow_pickle=True)
    ztvals = np.load(SV_FOLDER + '/10M/100_sims/ztvals.npy', allow_pickle=True)
    tivals = np.load(SV_FOLDER + '/10M/100_sims/tivals.npy', allow_pickle=True)
    bhat_mat = np.zeros((len(cohvals), len(n_list)))
    js_mat = np.zeros((len(cohvals), len(n_list)))
    for i_n, n_trial in enumerate(n_list):  # for each network
        i = 0
        for cohval, ztval, tival in zip(cohvals, ztvals, tivals):
            # compute two distances
            bhat, jens_shan = dist_lh_model_nn(n_trial, cohval, ztval,
                                               tival, num_simulations=int(5e5))
            bhat_mat[i, i_n] = bhat
            js_mat[i, i_n] = jens_shan
            i += 1
    for a in ax:  # tune panels
        a.set_xscale('log')
        rm_top_right_lines(a)
        a.set_xlabel('N trials for training')
    # average across trials
    mean_bhat = np.nanmean(bhat_mat, axis=0)
    mean_js = np.nanmean(js_mat, axis=0)
    err_bhat = np.nanstd(bhat_mat, axis=0)/10  # standard error (std / sqrt(N)), with N = 100
    err_js = np.nanstd(js_mat, axis=0)/10
    # plot
    ax[0].plot(n_list, mean_bhat, linewidth=2, color='r')
    ax[1].plot(n_list, mean_js, linewidth=2, color='r')
    ax[0].fill_between(n_list, mean_bhat-err_bhat, mean_bhat+err_bhat, color='r',
                       alpha=0.2)
    ax[1].fill_between(n_list, mean_js-err_js, mean_js+err_js, color='r',
                       alpha=0.2)
    # tune
    ax[0].set_ylabel('Bhattacharyya \n distance')
    ax[1].set_ylabel('Jensen-Shannon \n distance')


def dist_lh_model_nn(n_sim_train, cohval, ztval, tival, num_simulations=int(5e5)):
    """
    Computes Bhattacharyya and Jensen-Shannon distances between network and
    model simulations.
    """
    # we define a grid of MT/RT on which we will compute the approximate likelihoods
    grid_rt = np.arange(200, 600, 13)
    grid_rt = grid_rt[:-1] + np.diff(grid_rt)[0]/2
    grid_mt = np.arange(100, 600, 26)
    grid_mt = grid_mt[:-1] + np.diff(grid_mt)[0]/2
    all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    # for choice=0
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    # for choice=1
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    trial_index = np.repeat(tival, num_simulations)
    mat_0 = np.load(SV_FOLDER + '/10M/100_sims/mat0_coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival))
    mat_1 = np.load(SV_FOLDER + '/10M/100_sims/mat1_coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival))
    # we load estimator
    with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_sim_train),
              'rb') as f:
        estimator = pickle.load(f)
    estimator = estimator['estimator']
    theta = get_x0()  # parameters used to simulate
    theta = torch.reshape(torch.tensor(theta),
                          (1, len(theta))).to(torch.float32)
    theta = theta.repeat(len(x_o), 1)
    theta[:, 0] *= torch.tensor(ztval)
    theta[:, 1] *= torch.tensor(cohval)
    theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                        torch.tensor(trial_index[
                                            :len(x_o)]).to(torch.float32)))
    theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
    theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
    theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                        theta_tri_ind[:, 9:15]))
    # compute approximate likelihood of the network given parameters
    x_o = x_o.reshape((1, x_o.shape[0], x_o.shape[1]))
    lprobs = estimator.log_prob(x_o, theta_tri_ind)
    lprobs = torch.exp(lprobs)
    mat_0_nn = lprobs[x_o[:, 2] == 0].reshape(len(grid_mt),
                                              len(grid_rt)).detach().numpy()
    mat_1_nn = lprobs[x_o[:, 2] == 1].reshape(len(grid_mt),
                                              len(grid_rt)).detach().numpy()
    cte_nn = np.sum(mat_0_nn + mat_1_nn)  # normalize
    mat_0_nn /= cte_nn
    mat_1_nn /= cte_nn
    cte_mod = np.sum(mat_0 + mat_1)  # normalize
    mat_0 /= cte_mod
    mat_1 /= cte_mod
    mat_model = np.array((((mat_0), (mat_1))))
    mat_nn = np.array((((mat_0_nn), (mat_1_nn))))
    estimator = []
    return bhatt_dist(mat_model, mat_nn), np.nansum(dist.jensenshannon(mat_model, mat_nn))


def get_human_data(user_id, sv_folder=SV_FOLDER, nm='300'):
    """
    Function to retrieve human data.
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
    subj = ['general_traj_all']
    steps = [None]
    # retrieve data
    df = ah.traj_analysis(data_folder=folder,
                          subjects=subj, steps=steps, name=nm,
                          sv_folder=sv_folder)
    return df


def human_fitting(df, subject, sv_folder=SV_FOLDER,  num_simulations=int(10e6)):
    """
    Function to fit human data. Same procedure as for rats.
    """
    df_data = df.loc[df.subjid == subject]
    reac_time = df_data.sound_len.values
    reaction_time = []
    for rt in reac_time:
        if rt > 500:
            rt = 500
        reaction_time.append(rt+300)
    choice = df_data.R_response.values
    coh = df_data.avtrapz.values*5
    zt = df_data.norm_allpriors.values*3
    times = df_data.times.values
    trial_index = df_data.origidx.values
    motor_time = []
    for tr in range(len(choice)):
        ind_time = [True if t != '' else False for t in times[tr]]
        time_tr = np.array(times[tr])[np.array(ind_time)].astype(float)
        mt = time_tr[-1]
        if mt > 1:
            mt = 1
        motor_time.append(mt*1e3)
    x_o = torch.column_stack((torch.tensor(motor_time),
                              torch.tensor(reaction_time),
                              torch.tensor(choice)))
    data = torch.column_stack((torch.tensor(zt), torch.tensor(coh),
                               torch.tensor(trial_index.astype(float)),
                               x_o))
    # load network
    with open(SV_FOLDER + f"/mnle_n{num_simulations}_no_noise.p", 'rb') as f:
        estimator = pickle.load(f)
    estimator = estimator['estimator']
    x0 = get_x0()  # get initial parameter configuration
    print('Initial guess is: ' + str(x0))
    # get hard and plausible upper/lower bounds
    lb = np.array(get_lb_human())
    ub = np.array(get_ub_human())
    plb = lb + (-lb+ub)/10
    pub = ub - (-lb+ub)/10
    print('Optimizing')
    n_trials = len(data)
    # define fun_target as function to optimize
    # returns -LLH( data | parameters )
    fun_target = lambda x: fun_theta(x, data, estimator, n_trials)
    # define optimizer (BADS)
    bads = BADS(fun_target, x0, lb, ub, plb, pub)
    # optimization
    optimize_result = bads.optimize()
    print(optimize_result.total_time)
    return optimize_result.x  # returns optimized parameters

