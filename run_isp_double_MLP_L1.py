import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from Myloss import L_TV, L_TV_L1, L_TV_frac, L_TV_reg
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from run_nerf_helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

DEBUG = False
eta_0 = 120 * np.pi
c = 3e8
eps_0 = 8.85e-12

# bool_plane = 0
# Coef = torch.complex(0.0, k_0 * eta_0)
N_rec = 32  # Nb. of Receiver
N_inc = 16  # Nb. of Incidence

# i = torch.complex(0.0, 1)


MAX = 1
Mx = 64
step_size = 2 * MAX / (Mx - 1)
cell_area = step_size ** 2  # the area of the sub-domain


def run_network(inputs, fn, embed_fn):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    outputs_flat = fn(embedded)

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render(freq, H, W, N_cell, E_inc, Phi_mat, R_mat, input,input_J,  network_fn, network_fn_J, network_query_fn):
    re = {}
    lam_0 = c / (freq * 1e9)
    # lam_0 = 0.75
    k_0 = 2 * np.pi / lam_0
    omega = k_0 * c
    epsilon = network_query_fn(input, network_fn)
    epsilon = epsilon.squeeze(-1)
    # epsilon = epsilon_gt
    # input_numpy = input.cpu().numpy()
    # input_numpy_1d = torch.reshape(input.transpose(0, 1), [-1, 2]).cpu().numpy()
    re['epsilon'] = epsilon

    J = network_query_fn(input_J, network_fn_J)
    J = torch.complex(J[..., 0], J[..., 1])
    # J = J_gt
    re['J'] = J
    J_ = J.detach()
    re['J_'] = J_
    re['R_mat_J'] = R_mat @ J
    # epsilon_numpy = epsilon.cpu().numpy()
    xi_all = torch.complex(torch.Tensor([0.0]), -omega * (epsilon - 1) * eps_0 * cell_area)
    xi_forward = torch.reshape(xi_all.t(), [-1, 1])
    xi_forward_mat = torch.diag_embed(xi_forward.squeeze(-1))
    xi_E_inc = xi_forward_mat @ E_inc
    re['J_state'] = xi_E_inc + xi_forward_mat @ Phi_mat @ J

    re['norm_xi_E_inc'] = torch.mean(xi_E_inc.real ** 2 + xi_E_inc.imag ** 2)
    # xi_forward_numpy = xi_forward.cpu().numpy()
    # aa = torch.eye(N_cell)
    # bb = torch.diag_embed(xi_forward.squeeze(-1))
    # E_tot = torch.linalg.inv(torch.eye(N_cell) - (Phi_mat @ torch.diag_embed(xi_forward.squeeze(-1)))) @ E_inc
    # E_s = R_mat @ torch.diag_embed(xi_forward.squeeze(-1)) @ E_tot
    # re['E_s'] = E_s
    return re

def create_isp_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    output_ch = 1
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips, tanh=True).to(device)
    grad_vars = list(model.parameters())

    model_J = NeRF_J(D=args.netdepth, W=args.netwidth,
                   input_ch=input_ch * 2, output_ch=2, skips=skips, tanh=True).to(device)
    grad_vars += list(model_J.parameters())


    network_query_fn = lambda inputs, network_fn: run_network(inputs, network_fn,
                                                              embed_fn=embed_fn, )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        model_J.load_state_dict(ckpt['network_fn_J_state_dict'])
    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'network_fn': model,
        'network_fn_J': model_J,
    }

    # NDC only good for LLFF-style forward facing data

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')

    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=1,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=100,
                        help='frequency of testset saving')

    parser.add_argument("--noise_ratio", type=float, default=0.05,
                        help='noise_ratio')
    parser.add_argument("--sample_perturb", type=float, default=0.005,
                        help='random sample perturbation')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    # loss_TV = L_TV_reg(TVLoss_weight=1)
    loss_TV = L_TV_L1(TVLoss_weight=1)


    x_dom = np.load(os.path.join(args.datadir, 'x_dom.npy'))
    y_dom = np.load(os.path.join(args.datadir, 'y_dom.npy'))
    x_dom = torch.Tensor(x_dom).to(device)
    y_dom = torch.Tensor(y_dom).to(device)
    xy_dom = torch.stack([x_dom, y_dom], -1)

    xy_t = np.load(os.path.join(args.datadir, 'xy_t.npy'))
    xy_t = torch.Tensor(xy_t).to(device)



    N_cell = Mx * Mx

    # coords_inc = torch.cat(
    #     (torch.reshape(xy_dom.transpose(0, 1), [-1, 2]).unsqueeze(-2).repeat([1, N_inc, 1]),
    #      xy_t.unsqueeze(0).repeat([N_cell, 1, 1])), -1)

    # Load data
    freq = 0.4

    gt_real = np.load(os.path.join(args.datadir, 'E_s_real.npy'))
    gt_imag = np.load(os.path.join(args.datadir, 'E_s_imag.npy'))
    if args.noise_ratio != 0:
        energe = np.sqrt(np.mean((gt_real ** 2 + gt_imag ** 2))) * (1 / np.sqrt(2))
        gt_real = gt_real + energe*args.noise_ratio*np.random.randn(N_rec, N_inc)
        gt_imag = gt_imag + energe * args.noise_ratio * np.random.randn(N_rec, N_inc)
    gt_real = torch.Tensor(gt_real).to(device)
    gt_imag = torch.Tensor(gt_imag).to(device)


    Phi_mat_real = np.load(os.path.join(args.datadir, 'Phi_mat_real.npy'))
    Phi_mat_imag = np.load(os.path.join(args.datadir, 'Phi_mat_imag.npy'))
    Phi_mat = torch.complex(torch.Tensor(Phi_mat_real), torch.Tensor(Phi_mat_imag)).to(device)

    R_mat_real = np.load(os.path.join(args.datadir, 'R_mat_real.npy'))
    R_mat_imag = np.load(os.path.join(args.datadir, 'R_mat_imag.npy'))
    R_mat = torch.complex(torch.Tensor(R_mat_real), torch.Tensor(R_mat_imag)).to(device)

    E_inc_real = np.load(os.path.join(args.datadir, 'E_inc_real.npy'))
    E_inc_imag = np.load(os.path.join(args.datadir, 'E_inc_imag.npy'))


    E_inc = torch.complex(torch.Tensor(E_inc_real), torch.Tensor(E_inc_imag)).to(device)


    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    writer = SummaryWriter(os.path.join(basedir, expname, 'tb'))
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_isp_nerf(args)
    global_step = start


    H = Mx
    W = Mx

    if args.render_only:
        testsavedir = os.path.join(basedir, expname,
                                   'renderonly_{}_{:06d}.npy'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            fn_test = render_kwargs_test['network_query_fn']
            output = fn_test(xy_dom, render_kwargs_test['network_fn'])
            np.save(testsavedir, output.squeeze(-1).numpy().cpu())
        print('Saved test set')


    testsavedir = os.path.join(basedir, expname, 'testset_000000.npy')
    # os.makedirs(testsavedir, exist_ok=True)
    with torch.no_grad():
        fn_test = render_kwargs_test['network_query_fn']
        output = fn_test(xy_dom, render_kwargs_test['network_fn'])
        np.save(testsavedir, output.squeeze(-1).cpu().numpy())
    print('Saved test set')

    N_iters = 50000 + 1
    print('Begin')

    start = start + 1


    for i in trange(start, N_iters):
        xy_dom_random = xy_dom + torch.randn_like(xy_dom) * args.sample_perturb
        # xy_dom_random = xy_dom
        coords_inc = torch.cat(
            (torch.reshape(xy_dom_random.transpose(0, 1), [-1, 2]).unsqueeze(-2).repeat([1, N_inc, 1]),
             xy_t.unsqueeze(0).repeat([N_cell, 1, 1])), -1)

        #####  Core optimization loop  #####
        re = render(freq, H, W, N_cell=N_cell, E_inc=E_inc, Phi_mat=Phi_mat, R_mat=R_mat, input=xy_dom, input_J=coords_inc, **render_kwargs_train)

        optimizer.zero_grad()
        # tt = re.real
        img_loss_data = (img2mse(re['R_mat_J'].real, gt_real) + img2mse(re['R_mat_J'].imag, gt_imag))/torch.mean(gt_real **2 + gt_imag **2)
        img_loss_state = (img2mse(re['J_state'].real, re['J'].real) + img2mse(re['J_state'].imag, re['J'].imag))/(re['norm_xi_E_inc'])

        TV_loss = loss_TV(re['epsilon'])
        # loss = img_loss
        if global_step <= 2000:
            loss = (img_loss_data + img_loss_state)
        else:
            loss = (img_loss_data + img_loss_state + 0.01*TV_loss)
        loss.backward()
        optimizer.step()
        # beta = min(beta * kappa, betamax)
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################


        # Rest is logging
        writer.add_scalar("loss_data", img_loss_data, i)
        writer.add_scalar("loss_state", img_loss_state, i)
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fn_J_state_dict': render_kwargs_train['network_fn_J'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}.npy'.format(i))
            testsavedir_img = os.path.join(basedir, expname, 'testset_{:06d}.png'.format(i))
            # os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                fn_test = render_kwargs_test['network_query_fn']
                output = fn_test(xy_dom, render_kwargs_test['network_fn'])
                # print('epsilon_loss: ', epsilon_loss.item())
                output = output.squeeze(-1).cpu().numpy()
                np.save(testsavedir, output)
                sc = plt.imshow(output)
                sc.set_cmap('hot')
                plt.colorbar()
                plt.savefig(testsavedir_img)
                plt.close()
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} freq: {freq} img_loss_data: {img_loss_data.item()} img_loss_state: {img_loss_state.item()}  TV_loss: {TV_loss.item()}")
        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
