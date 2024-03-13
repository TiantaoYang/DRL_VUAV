import os
import csv
import argparse
import numpy as np
import math as M
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def get_args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-Quadrotor")
    parser.add_argument("--max_train_steps", type=int, default=1e7, help="Maximum number of training steps")#3e6
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/PPO/')

    parser.add_argument("--state_dim", type=int, default=12, help="Dimension of state space")
    parser.add_argument("--action_dim", type=int, default=4, help="Dimension of action space")
    parser.add_argument("--max_action",type=float, default=1000, help="Maximum value of action")
    parser.add_argument("--simu_duration",type=int, default=20, help="Number of seconds for simulation")
    parser.add_argument("--simu_step", type=int, default=1e3, help="Number of simulation steps per second")
    parser.add_argument("--train_step", type=int, default=10, help="Number of train steps per second")

    parser.add_argument("--c_1", type=float, default=4e-1, help="Coefficient of err_")
    parser.add_argument("--c_2", type=float, default=2e-2, help="Coefficient of angular_norm2")
    parser.add_argument("--c_3", type=float, default=3e-2, help="Coefficient of angularspeed_norm2")
    parser.add_argument("--c_4", type=float, default=5e-2, help="Coefficient of velocity_norm2")
    parser.add_argument("--c_5", type=float, default=1e-4, help="Coefficient of motor_norm2")#1e-4
    parser.add_argument("--bound_train",type=float, default=5, help="Boundary of training")
    parser.add_argument("--bound_init",type=float, default=1, help="Boundary of initialize")
    args = parser.parse_args()

    args.max_episode_steps = args.train_step * args.simu_duration # Maximum number of steps per episode
    args.L0 = [0.15, 0.15, 0.15, 0.15]
    args.m, args.Ixx, args.Iyy, args.Izz = calc_inertia(args.L0)
    # print(args.m, args.Ixx, args.Iyy, args.Izz)

    args.ddx_sd = 0
    args.ddy_sd = 0
    args.ddz_sd = 0
    args.ddPhi_sd = 0
    args.ddtheta_sd = 0
    args.ddPsi_sd = 0

    return args


def calc_inertia(l_tar):
    l1 = l_tar[0]
    l2 = l_tar[1]
    l3 = l_tar[2]
    l4 = l_tar[3]
    mbody = 1.06
    marm = 0.068
    mrot = 0.1
    g = 9.8
    lbody = 0.13
    hbody = 0.02
    warm = 0.015
    harm = 0.015
    rmot = 0.08
    hmot = 0.003
    rrot = 0.03
    hrot = 0.02
    kf, km = 3.03e-5, 5.5e-7
    gama = 0.08
    m = mbody+4*(marm+mrot)
    ix = mbody/12*(hbody**2+lbody**2)+marm*((4*harm**2+2*warm**2+l2**2+l4**2)/12+(M.sqrt(2)*lbody+l2/2)**2+(M.sqrt(2)*lbody+l4/2)**2)+mrot*(rrot**2+hrot**2/3+(M.sqrt(2)*lbody+l2+rrot)**2+(M.sqrt(2)*lbody+l4+rrot)**2)
    iy = mbody/12*(hbody**2+lbody**2)+marm*((4*harm**2+2*warm**2+l1**2+l3**2)/12+(M.sqrt(2)*lbody+l1/2)**2+(M.sqrt(2)*lbody+l3/2)**2)+mrot*(rrot**2+hrot**2/3+(M.sqrt(2)*lbody+l1+rrot)**2+(M.sqrt(2)*lbody+l3+rrot)**2)
    iz = mbody*lbody**2/6+marm*(warm**2/3+(l1**2+l2**2+l3**2+l4**2)/12+(M.sqrt(2)*lbody+l1/2)**2+(M.sqrt(2)*lbody+l2/2)**2+(M.sqrt(2)*lbody+l3/2)**2+(M.sqrt(2)*lbody+l4/2)**2)+mrot*(2*rrot**2+(M.sqrt(2)*lbody+l1+rrot)**2+(M.sqrt(2)*lbody+l2+rrot)**2+(M.sqrt(2)*lbody+l3+rrot)**2+(M.sqrt(2)*lbody+l4+rrot)**2)
    return m, ix, iy, iz


class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


def create_directory(path: str, sub_paths: list):
    for sub_path in sub_paths:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)
            # print('Create path: {} successfully'.format(path+sub_path))
        else:
            # print('Path: {} is already existence'.format(path+sub_path))
            pass


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure(figsize=(8, 2))
    plt.plot(episodes, records, color='r', linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(figure_file)


def plot_train_position_curve(time, refer, positions, title, figure_file):
    plt.figure(figsize=(8, 2))
    plt.plot(time, refer[:len(time)], linestyle='-', color='b')
    plt.plot(time, positions, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('time/s')
    plt.ylabel('position/m')
    plt.savefig(figure_file)
    plt.close()


def plot_test_position_curve(time, positions, references, title, figure_file):
    plt.figure(figsize=(8, 2))
    plt.plot(time, positions, linestyle='-', color='r')
    plt.plot(time, references, linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('time/s')
    plt.ylabel('position/m')
    plt.savefig(figure_file)
    plt.close()


def plot_err_curve(time, positions, title, figure_file):
    plt.figure(figsize=(8, 2))
    plt.plot(time, positions, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('time/s')
    plt.ylabel('position/m')
    plt.savefig(figure_file)
    plt.close()


def plot_orientation_curve(time, orientations, title, figure_file):
    plt.figure(figsize=(8, 2))
    plt.plot(time, orientations, linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('time/s')
    plt.ylabel('angle/rad')
    plt.savefig(figure_file)
    plt.close()


def plot_motorspeed_curve(time, ns, title, figure_file):
    plt.figure(figsize=(8, 2))
    plt.plot(time, ns, linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('time/s')
    plt.ylabel('speed/(r/min)')
    plt.savefig(figure_file)
    plt.close()


def plot_PID_curve(episodes, KP1, KP2, KI2, KD2, figure_file):
    plt.figure(figsize=(8, 2))
    plt.plot(episodes, KP1, linestyle='-', color='c')
    plt.plot(episodes, KP2, linestyle='-', color='m')
    plt.plot(episodes, KI2, linestyle='-', color='y')
    plt.plot(episodes, KD2, linestyle='-', color='k')
    plt.title('PID')
    plt.xlabel('time/s')
    plt.savefig(figure_file)
    plt.close()


def plot_trace_3D(arrx, arry, arrz, figure_file):
    fig = plt.figure(figsize=(8, 8))
    trace3D = fig.gca(projection='3d')
    trace3D.set_title("trace3D")
    trace3D.set_xlim([-1, 1])
    trace3D.set_ylim([-1, 1])
    trace3D.set_zlim([-1, 1])
    trace3D.set_xlabel("x")
    trace3D.set_ylabel("y")
    trace3D.set_zlabel("z")
    trace3D.plot(arrx, arry, arrz, c='r')
    plt.savefig(figure_file)
    plt.close()


def plot_trace_2D(arrx, arry, figure_file):
    plt.figure(figsize=(8, 8))
    plt.plot(arrx, arry)
    plt.xlabel('x/m')
    plt.ylabel('z/m')
    plt.savefig(figure_file)
    plt.close()


def train_save_ep_result(refer_x, refer_y, refer_z, arrx, arry, arrz, arrerr, arrPhi, arrtheta, arrPsi, arrdx, arrdy, arrdz, arrdPhi, arrdtheta, arrdPsi,
                         arrn1, arrn2, arrn3, arrn4, simu_step, train_step, path):
    create_directory(path, sub_paths=['Position', 'Orientation', 'Motorspeed'])
    plot_train_position_curve(np.arange(0, (len(arrx)-1)/simu_step, 1/simu_step), refer_x[:-1], arrx[:-1], 'x',
                            path + 'Position/x.pdf')
    plot_train_position_curve(np.arange(0, (len(arry)-1)/simu_step, 1/simu_step), refer_y[:-1], arry[:-1], 'y',
                            path + 'Position/y.pdf')
    plot_train_position_curve(np.arange(0, (len(arrz)-1)/simu_step, 1/simu_step), refer_z[:-1], arrz[:-1], 'z',
                            path + 'Position/z.pdf')
    plot_err_curve(np.arange(0, (len(arrerr)-1)/simu_step, 1/simu_step), arrerr[:-1], 'err',
                            path + 'err.pdf')
    plot_orientation_curve(np.arange(0, (len(arrPhi)-1)/simu_step, 1/simu_step), arrPhi[:-1], 'Phi',
                            path + 'Orientation/Phi.pdf')
    plot_orientation_curve(np.arange(0, (len(arrtheta)-1)/simu_step, 1/simu_step), arrtheta[:-1], 'theta',
                            path + 'Orientation/theta.pdf')
    plot_orientation_curve(np.arange(0, (len(arrPsi)-1)/simu_step, 1/simu_step), arrPsi[:-1], 'Psi',
                            path + 'Orientation/Psi.pdf')
    plot_motorspeed_curve(np.arange(0, (len(arrn1)-1)/train_step, 1/train_step), arrn1[:-1], 'n1',
                            path + 'Motorspeed/n1.pdf')
    plot_motorspeed_curve(np.arange(0, (len(arrn2)-1)/train_step, 1/train_step), arrn2[:-1], 'n2',
                            path + 'Motorspeed/n2.png')
    plot_motorspeed_curve(np.arange(0, (len(arrn3)-1)/train_step, 1/train_step), arrn3[:-1], 'n3',
                            path + 'Motorspeed/n3.png')
    plot_motorspeed_curve(np.arange(0, (len(arrn4)-1)/train_step, 1/train_step), arrn4[:-1], 'n4',
                            path + 'Motorspeed/n4.png')
    plot_trace_3D(arrx, arry, arrz, path + 'trace3D.png')
    arrstates = np.concatenate(([arrx], [arry], [arrz], [arrPhi], [arrtheta], [arrPsi],
                                [arrdx], [arrdy], [arrdz], [arrdPhi], [arrdtheta], [arrdPsi]), axis = 0)
    arractions = np.concatenate(([arrn1], [arrn2], [arrn3], [arrn4]), axis=0)
    statesheader = ['x', 'y', 'z', 'Phi', 'theta', 'Psi', 'dx', 'dy', 'dz', 'dPhi', 'dtheta', 'dPsi']
    actionsheader = ['n1', 'n2', 'n3', 'n4']
    with open(path + 'states.csv', 'w', encoding='utf8', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(statesheader)
        writer.writerows(arrstates.T)
    with open(path + 'actions.csv', 'w', encoding='utf8', newline='') as f2:
        writer = csv.writer(f2)
        writer.writerow(actionsheader)
        writer.writerows(arractions.T)


def test_save_ep_result(arrxr, arryr, arrzr, arrx, arry, arrz, arrerr, arrPhi, arrtheta, arrPsi, arrdx, arrdy, arrdz,
                   arrdPhi, arrdtheta, arrdPsi, arrn1, arrn2, arrn3, arrn4, simu_step, train_step, path):
    create_directory(path, sub_paths=['Position', 'Orientation', 'Motorspeed'])
    plot_test_position_curve(np.arange(0, (len(arrx)-1)/simu_step, 1/simu_step), arrx[:-1], arrxr, 'x',
                            path + 'Position/x.pdf')
    plot_test_position_curve(np.arange(0, (len(arry)-1)/simu_step, 1/simu_step), arry[:-1], arryr, 'y',
                            path + 'Position/y.pdf')
    plot_test_position_curve(np.arange(0, (len(arrz)-1)/simu_step, 1/simu_step), arrz[:-1], arrzr, 'z',
                            path + 'Position/z.pdf')
    plot_err_curve(np.arange(0, (len(arrerr)-1)/simu_step, 1/simu_step), arrerr[:-1], 'err',
                            path + 'err.pdf')
    plot_orientation_curve(np.arange(0, (len(arrPhi)-1)/simu_step, 1/simu_step), arrPhi[:-1], 'Phi',
                            path + 'Orientation/Phi.pdf')
    plot_orientation_curve(np.arange(0, (len(arrtheta)-1)/simu_step, 1/simu_step), arrtheta[:-1], 'theta',
                            path + 'Orientation/theta.pdf')
    plot_orientation_curve(np.arange(0, (len(arrPsi)-1)/simu_step, 1/simu_step), arrPsi[:-1], 'Psi',
                            path + 'Orientation/Psi.pdf')
    plot_motorspeed_curve(np.arange(0, (len(arrn1)-1)/train_step, 1/train_step), arrn1[:-1], 'n1',
                            path + 'Motorspeed/n1.pdf')
    plot_motorspeed_curve(np.arange(0, (len(arrn2)-1)/train_step, 1/train_step), arrn2[:-1], 'n2',
                            path + 'Motorspeed/n2.pdf')
    plot_motorspeed_curve(np.arange(0, (len(arrn3)-1)/train_step, 1/train_step), arrn3[:-1], 'n3',
                            path + 'Motorspeed/n3.pdf')
    plot_motorspeed_curve(np.arange(0, (len(arrn4)-1)/train_step, 1/train_step), arrn4[:-1], 'n4',
                            path + 'Motorspeed/n4.pdf')
    plot_trace_3D(arrx, arry, arrz, path + 'trace3D.pdf')
    arrstates = np.concatenate(([arrx], [arry], [arrz], [arrPhi], [arrtheta], [arrPsi],
                                [arrdx], [arrdy], [arrdz], [arrdPhi], [arrdtheta], [arrdPsi]), axis = 0)
    arractions = np.concatenate(([arrn1], [arrn2], [arrn3], [arrn4]), axis=0)
    statesheader = ['x', 'y', 'z', 'Phi', 'theta', 'Psi', 'dx', 'dy', 'dz', 'dPhi', 'dtheta', 'dPsi']
    actionsheader = ['n1', 'n2', 'n3', 'n4']
    with open(path + 'states.csv', 'w', encoding='utf8', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(statesheader)
        writer.writerows(arrstates.T)
    with open(path + 'actions.csv', 'w', encoding='utf8', newline='') as f2:
        writer = csv.writer(f2)
        writer.writerow(actionsheader)
        writer.writerows(arractions.T)


def scale_action(action, high, low):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_


def z_score(arrx):
    arry = (arrx-np.average(arrx, axis=0))/(np.std(arrx, axis=0)+1e-30)
    return arry


def test_sample(bound = 1, number = 5):
    samples = []
    dots = np.linspace(-bound, bound, number)
    for i in range(len(dots)):
        for j in range(len(dots)):
            for k in range(len(dots)):
                sample = []
                sample.append(dots[i])
                sample.append(dots[j])
                sample.append(dots[k])
                samples.append(sample)
    return samples


def refer_trace():
    arr_x = []
    arr_y = []
    arr_z = []
    arr_v_x = []
    arr_v_y = []
    arr_v_z = []
    r = 0.5
    for i in range(20001):
        t = 0.001 * i
        x = 2*r * M.cos(2*M.pi/10 * t)
        y = r * M.sin(2*M.pi/10 * 2*t)
        z = 0
        v_x = - 2*r * 2*M.pi/10 * M.sin(2*M.pi/10 * t)
        v_y = r * 2*M.pi/10 * 2 * M.cos(2*M.pi/10 * 2*t)
        v_z = 0
        arr_x.append(x)
        arr_y.append(y)
        arr_z.append(z)
        arr_v_x.append(v_x)
        arr_v_y.append(v_y)
        arr_v_z.append(v_z)

    return np.array([arr_x, arr_y, arr_z]), np.array([arr_v_x, arr_v_y, arr_v_z])


def get_coef(l_tar):
    e = 1e-6

    fun = lambda x: -np.linalg.norm(x)
    cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] - 1},
            {'type': 'eq', 'fun': lambda x: 0.15*x[0] + 0.15*x[1] + 0.15*x[2] + 0.15*x[3] + 0.15*x[4] + 0.15*x[5] + 0.15*x[6] + 0.15*x[7] + \
                                            0.25*x[8] + 0.25*x[9] + 0.25*x[10] + 0.25*x[11] + 0.25*x[12] + 0.25*x[13] + 0.25*x[14] + 0.25*x[15] - l_tar[0]},

            {'type': 'eq', 'fun': lambda x: 0.15*x[0] + 0.15*x[1] + 0.15*x[2] + 0.15*x[3] + 0.25*x[4] + 0.25*x[5] + 0.25*x[6] + 0.25*x[7] + \
                                            0.15*x[8] + 0.15*x[9] + 0.15*x[10] + 0.15*x[11] + 0.25*x[12] + 0.25*x[13] + 0.25*x[14] + 0.25*x[15] - l_tar[1]},

            {'type': 'eq', 'fun': lambda x: 0.15*x[0] + 0.15*x[1] + 0.25*x[2] + 0.25*x[3] + 0.15*x[4] + 0.15*x[5] + 0.25*x[6] + 0.25*x[7] + \
                                            0.15*x[8] + 0.15*x[9] + 0.25*x[10] + 0.25*x[11] + 0.15*x[12] + 0.15*x[13] + 0.25*x[14] + 0.25*x[15] - l_tar[2]},

            {'type': 'eq', 'fun': lambda x: 0.15*x[0] + 0.25*x[1] + 0.15*x[2] + 0.25*x[3] + 0.15*x[4] + 0.25*x[5] + 0.15*x[6] + 0.25*x[7] + \
                                            0.15*x[8] + 0.25*x[9] + 0.15*x[10] + 0.25*x[11] + 0.15*x[12] + 0.25*x[13] + 0.15*x[14] + 0.25*x[15] - l_tar[3]},
            {'type': 'ineq', 'fun': lambda x: x[0] - e},
            {'type': 'ineq', 'fun': lambda x: x[1] - e},
            {'type': 'ineq', 'fun': lambda x: x[2] - e},
            {'type': 'ineq', 'fun': lambda x: x[3] - e},
            {'type': 'ineq', 'fun': lambda x: x[4] - e},
            {'type': 'ineq', 'fun': lambda x: x[5] - e},
            {'type': 'ineq', 'fun': lambda x: x[6] - e},
            {'type': 'ineq', 'fun': lambda x: x[7] - e},
            {'type': 'ineq', 'fun': lambda x: x[8] - e},
            {'type': 'ineq', 'fun': lambda x: x[9] - e},
            {'type': 'ineq', 'fun': lambda x: x[10] - e},
            {'type': 'ineq', 'fun': lambda x: x[11] - e},
            {'type': 'ineq', 'fun': lambda x: x[12] - e},
            {'type': 'ineq', 'fun': lambda x: x[13] - e},
            {'type': 'ineq', 'fun': lambda x: x[14] - e},
            {'type': 'ineq', 'fun': lambda x: x[15] - e}
        )

    num_model = 16
    while num_model > 5:
        x0 = np.random.random(16)
        x0 = x0/sum(x0)
        res = minimize(fun, x0, method='SLSQP', constraints=cons, tol=1e-5)

        coef = np.around(res.x, 3)
        num_model = len(coef.nonzero()[0])

    return coef


def get_l_refer_sym(t):
    if t < 10:
        l_refer = [0.15+0.01*t, 0.15+0.01*t, 0.15+0.01*t, 0.15+0.01*t]
    elif t < 20:
        l_refer = [0.25, 0.25, 0.25, 0.25]
    elif t < 30:
        l_refer = [0.25-0.01*(t-20), 0.25-0.01*(t-20), 0.25-0.01*(t-20), 0.25-0.01*(t-20)]
    elif t < 45:
        l_refer = [0.15, 0.15, 0.15, 0.15]
    elif t < 55:
        l_refer = [0.15+0.01*(t-45), 0.15+0.01*(t-45), 0.15+0.01*(t-45), 0.15+0.01*(t-45)]
    elif t < 70:
        l_refer = [0.25, 0.25, 0.25, 0.25]
    elif t < 80:
        l_refer = [0.25-0.01*(t-70), 0.25-0.01*(t-70), 0.25-0.01*(t-70), 0.25-0.01*(t-70)]
    elif t < 95:
        l_refer = [0.15, 0.15, 0.15, 0.15]
    elif t < 105:
        l_refer = [0.15+0.01*(t-95), 0.15+0.01*(t-95), 0.15+0.01*(t-95), 0.15+0.01*(t-95)] 
    elif t < 120:
        l_refer = [0.25, 0.25, 0.25, 0.25]
    elif t < 130:
        l_refer = [0.25-0.01*(t-120), 0.25-0.01*(t-120), 0.25-0.01*(t-120), 0.25-0.01*(t-120)]
    elif t < 145:
        l_refer = [0.15, 0.15, 0.15, 0.15]
    elif t < 155:
        l_refer = [0.15+0.01*(t-145), 0.15+0.01*(t-145), 0.15+0.01*(t-145), 0.15+0.01*(t-145)]
    elif t < 170:
        l_refer = [0.25, 0.25, 0.25, 0.25]
    elif t < 180:
        l_refer = [0.25-0.01*(t-170), 0.25-0.01*(t-170), 0.25-0.01*(t-170), 0.25-0.01*(t-170)]
    else:
        l_refer = [0.15, 0.15, 0.15, 0.15]

    return l_refer


def get_l_refer_asym_channel(t):
    if t < 10:
        l_refer = [0.15+0.01*t, 0.15+0.01*t, 0.15+0.01*t, 0.15+0.01*t]
    elif t < 20:
        l_refer = [0.25, 0.25, 0.25, 0.25]
    elif t < 30:
        l_refer = [0.25, 0.25, 0.25-0.01*(t-20), 0.25-0.01*(t-20)]
    elif t < 45:
        l_refer = [0.25, 0.25, 0.15, 0.15]
    elif t < 55:
        l_refer = [0.25, 0.25, 0.15+0.01*(t-45), 0.15+0.01*(t-45)]
    elif t < 70:
        l_refer = [0.25, 0.25, 0.25, 0.25]
    elif t < 80:
        l_refer = [0.25-0.01*(t-70), 0.25, 0.25, 0.25-0.01*(t-70)]
    elif t < 95:
        l_refer = [0.15, 0.25, 0.25, 0.15]
    elif t < 105:
        l_refer = [0.15+0.01*(t-95), 0.25, 0.25, 0.15+0.01*(t-95)] 
    elif t < 120:
        l_refer = [0.25, 0.25, 0.25, 0.25]
    elif t < 130:
        l_refer = [0.25-0.01*(t-120), 0.25-0.01*(t-120), 0.25, 0.25]
    elif t < 145:
        l_refer = [0.15, 0.15, 0.25, 0.25]
    elif t < 155:
        l_refer = [0.15+0.01*(t-145), 0.15+0.01*(t-145), 0.25, 0.25]
    elif t < 170:
        l_refer = [0.25, 0.25, 0.25, 0.25]
    elif t < 180:
        l_refer = [0.25, 0.25-0.01*(t-170), 0.25-0.01*(t-170), 0.25]
    else:
        l_refer = [0.25, 0.15, 0.15, 0.25]

    return l_refer


def get_l_refer_asym(t):
    if t < 10:
        l_refer = [0.15+0.003*t, 0.15+0.009*t, 0.15+0.005*t, 0.15]
    elif t < 50:
        l_refer = [0.18, 0.24, 0.20, 0.15]
    elif t < 60:
        l_refer = [0.18+0.003*(t-50), 0.24-0.009*(t-50), 0.20+0.005*(t-50), 0.15+0.008*(t-50)]
    elif t < 100:
        l_refer = [0.21, 0.15, 0.25, 0.23]
    elif t < 110:
        l_refer = [0.21-0.004*(t-100), 0.15+0.008*(t-100), 0.25-0.009*(t-100), 0.23-0.004*(t-100)]
    elif t < 150:
        l_refer = [0.17, 0.23, 0.16, 0.19]
    elif t < 160:
        l_refer = [0.17+0.007*(t-150), 0.23-0.007*(t-150), 0.16+0.006*(t-150), 0.19+0.006*(t-150)] 
    else:
        l_refer = [0.24, 0.16, 0.22, 0.25]
    

    return l_refer


def l_curve(l_refers, test_mode):
    os.makedirs('test_results/' + test_mode + '/Armlength/', exist_ok=True)
    
    plt.rcParams['font.size']=15
    plt.rcParams['font.family']='Times New Roman'
    for i in range(4):
        plt.figure(figsize=(8, 2.5))
        plt.plot(np.array(range(len(l_refers)))/10, [l[i] for l in l_refers], linestyle='-')
        plt.xlabel('time/s')
        plt.ylabel('L{}'.format(i+1) + '/m')
        plt.savefig('test_results/' + test_mode + '/Armlength/l{}.pdf'.format(i+1))
        plt.close()

    plt.figure(figsize=(8, 2.5))
    plt.plot(np.array(range(len(l_refers)))/10, l_refers, linestyle='-')
    plt.title('L')
    plt.xlabel('time/s')
    plt.ylabel('length/m')
    plt.savefig('test_results/' + test_mode + '/Armlength/ls.pdf')
    plt.close()

    lsheader = ['L1', 'L2', 'L3', 'L4']
    with open('test_results/' + test_mode + '/Armlength/ls.csv', 'w', encoding='utf8', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(lsheader)
        writer.writerows(np.array(l_refers))
