import numpy as np
import math as M
import random
from ctypes import *
from utils import *


class env_Quad():
    def __init__(self, args):
        self.args = args
        self.Quadrotor = cdll.LoadLibrary('./Quadrotormodel/Quadrotor.dll')
        self.MotorModel = cdll.LoadLibrary('./Quadrotormodel/MotorModel.dll')

        self.quad_init = self.Quadrotor.Quadrotor_initialize
        self.quad_step = self.Quadrotor.Quadrotor_step
        self.quad_term = self.Quadrotor.Quadrotor_terminate
        self.quad_input = self.Quadrotor.motor
        self.arm = self.Quadrotor.arm
        self.inertia = self.Quadrotor.inertia
        self.states_init = self.Quadrotor.initializestates
        self.noiseinputs = self.Quadrotor.noiseinputs

        self.motor_init = self.MotorModel.MotorModel_initialize
        self.motor_step = self.MotorModel.MotorModel_step
        self.motor_term = self.MotorModel.MotorModel_terminate
        self.motorinputs = self.MotorModel.Motorinputs

        self.quad_step.restype = c_float
        self.quad_input.argtypes = (c_float, c_float, c_float, c_float)
        self.arm.argtypes = (c_float, c_float, c_float, c_float)
        self.inertia.argtypes = (c_float, c_float, c_float, c_float)
        self.states_init.argtypes = (c_float, c_float, c_float, c_float, c_float, c_float,
                                     c_float, c_float, c_float, c_float, c_float, c_float)
        self.motor_step.restype = c_float
        self.motorinputs.argtypes = (c_float, c_float, c_float, c_float)

        self.observaation_dim = 12
        self.action_dim = 4


    def reset_track(self, trajectory_type, rank):
        if trajectory_type == 'figure_8':
            self.gener_refer_figure_8()
        elif trajectory_type == 'quadrifolium':
            self.gener_refer_quadrifolium()
        self.done = False

        self.r_x = self.refer_trace[rank[0]]
        self.r_y = self.refer_trace[rank[1]]
        self.r_z = self.refer_trace[rank[2]]
        self.r_v_x = self.refer_speed[rank[0]]
        self.r_v_y = self.refer_speed[rank[1]]
        self.r_v_z = self.refer_speed[rank[2]]

        self.quad_init()
        self.quad_step()
        self.quad_term()
        self.states_init(self.r_x[0], self.r_y[0], self.r_z[0], 0, 0, 0,
                         self.r_v_x[0], self.r_v_y[0], self.r_v_z[0], 0, 0, 0)
        self.quad_init()

        self.motor_init()
        self.motor_step()
        self.motor_term()
        self.motor_init()

        self.arm(self.args.L0[0], self.args.L0[1], self.args.L0[2], self.args.L0[3])
        self.inertia(self.args.m, self.args.Ixx, self.args.Iyy, self.args.Izz)

        self.episode_step = 0

        self.arr_x = [self.r_x[0]]
        self.arr_y = [self.r_y[0]]
        self.arr_z = [self.r_z[0]]
        self.arr_Phi = [0]
        self.arr_theta = [0]
        self.arr_Psi = [0]
        self.arr_dx = [self.r_v_x[0]]
        self.arr_dy = [self.r_v_y[0]]
        self.arr_dz = [self.r_v_z[0]]
        self.arr_dPhi = [0]
        self.arr_dtheta = [0]
        self.arr_dPsi = [0]
        self.arr_n1 = []
        self.arr_n2 = []
        self.arr_n3 = []
        self.arr_n4 = []
        self.arr_err = [0]

        return np.array([0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0])


    def reset_regulate(self, positionbound=1, attitudebound=1, velocitybound=1, angularvelocitybound=1):
        self.done = False
        self.r_x = np.zeros(20001)
        self.r_y = np.zeros(20001)
        self.r_z = np.zeros(20001)
        self.r_v_x = np.zeros(20001)
        self.r_v_y = np.zeros(20001)
        self.r_v_z = np.zeros(20001)
        x_0 = random.uniform(-positionbound, positionbound)
        y_0 = random.uniform(-positionbound, positionbound)
        z_0 = random.uniform(-positionbound, positionbound)
        Phi_0 = random.uniform(-attitudebound, attitudebound)
        theta_0 = random.uniform(-attitudebound, attitudebound)
        Psi_0 = random.uniform(-attitudebound, attitudebound)
        v_x_0 = random.uniform(-velocitybound, velocitybound)
        v_y_0 = random.uniform(-velocitybound, velocitybound)
        v_z_0 = random.uniform(-velocitybound, velocitybound)
        v_Phi_0 = random.uniform(-angularvelocitybound, angularvelocitybound)
        v_theta_0 = random.uniform(-angularvelocitybound, angularvelocitybound)
        v_Psi_0 = random.uniform(-angularvelocitybound, angularvelocitybound)

        self.quad_init()
        self.quad_step()
        self.quad_term()
        self.states_init(x_0, y_0, z_0, Phi_0, theta_0, Psi_0,
                         v_x_0, v_y_0, v_z_0, v_Phi_0, v_theta_0, v_Psi_0)
        self.quad_init()

        self.motor_init()
        self.motor_step()
        self.motor_term()
        self.motor_init()

        self.arm(self.args.L0[0], self.args.L0[1], self.args.L0[2], self.args.L0[3])
        self.inertia(self.args.m, self.args.Ixx, self.args.Iyy, self.args.Izz)

        self.episode_step = 0

        self.arr_x = [x_0]
        self.arr_y = [y_0]
        self.arr_z = [z_0]
        self.arr_Phi = [Phi_0]
        self.arr_theta = [theta_0]
        self.arr_Psi = [Psi_0]
        self.arr_dx = [v_x_0]
        self.arr_dy = [v_y_0]
        self.arr_dz = [v_z_0]
        self.arr_dPhi = [v_Phi_0]
        self.arr_dtheta = [v_theta_0]
        self.arr_dPsi = [v_Psi_0]
        self.arr_n1 = []
        self.arr_n2 = []
        self.arr_n3 = []
        self.arr_n4 = []
        self.arr_err = [M.sqrt(pow(x_0, 2) + pow(y_0, 2) + pow(z_0, 2))]
        
        return np.array([x_0, y_0, z_0, Phi_0, theta_0, Psi_0,
                         v_x_0, v_y_0, v_z_0, v_Phi_0, v_theta_0, v_Psi_0])


    def morph(self, l_tar):
        self.arm(l_tar[0], l_tar[1], l_tar[2], l_tar[3])
        m_tar, Ixx_tar, Iyy_tar, Izz_tar = calc_inertia(l_tar)
        self.inertia(m_tar, Ixx_tar, Iyy_tar, Izz_tar)


    def env_step(self, action):
        self.motorinputs(action[0], action[1], action[2], action[3])
        for step in range(int(self.args.simu_step/self.args.train_step)):
            r_x = self.r_x[self.episode_step*int(self.args.simu_step/self.args.train_step)+step+1]
            r_y = self.r_y[self.episode_step*int(self.args.simu_step/self.args.train_step)+step+1]
            r_z = self.r_z[self.episode_step*int(self.args.simu_step/self.args.train_step)+step+1]
            r_v_x = self.r_v_x[self.episode_step*int(self.args.simu_step/self.args.train_step)+step+1]
            r_v_y = self.r_v_y[self.episode_step*int(self.args.simu_step/self.args.train_step)+step+1]
            r_v_z = self.r_v_z[self.episode_step*int(self.args.simu_step/self.args.train_step)+step+1]
            self.motor_step()
            n1 = c_float.in_dll(self.MotorModel, "n1_o").value
            n2 = c_float.in_dll(self.MotorModel, "n2_o").value
            n3 = c_float.in_dll(self.MotorModel, "n3_o").value
            n4 = c_float.in_dll(self.MotorModel, "n4_o").value
            self.quad_input(n1, n2, n3, n4)
            self.quad_step()
            x_ = c_float.in_dll(self.Quadrotor, "x_o").value
            y_ = c_float.in_dll(self.Quadrotor, "y_o").value
            z_ = c_float.in_dll(self.Quadrotor, "z_o").value
            Phi_ = c_float.in_dll(self.Quadrotor, "Phi_o").value
            theta_ = c_float.in_dll(self.Quadrotor, "theta_o").value
            Psi_ = c_float.in_dll(self.Quadrotor, "Psi_o").value
            dx_ = c_float.in_dll(self.Quadrotor, "dx_o").value
            dy_ = c_float.in_dll(self.Quadrotor, "dy_o").value
            dz_ = c_float.in_dll(self.Quadrotor, "dz_o").value
            dPhi_ = c_float.in_dll(self.Quadrotor, "dPhi_o").value
            dtheta_ = c_float.in_dll(self.Quadrotor, "dtheta_o").value
            dPsi_ = c_float.in_dll(self.Quadrotor, "dPsi_o").value
            err_ = M.sqrt(pow(x_-r_x, 2) + pow(y_-r_y, 2) + pow(z_-r_z, 2))
            self.arr_x.append(x_)
            self.arr_y.append(y_)
            self.arr_z.append(z_)
            self.arr_Phi.append(Phi_)
            self.arr_theta.append(theta_)
            self.arr_Psi.append(Psi_)
            self.arr_dx.append(dx_)
            self.arr_dy.append(dy_)
            self.arr_dz.append(dz_)
            self.arr_dPhi.append(dPhi_)
            self.arr_dtheta.append(dtheta_)
            self.arr_dPsi.append(dPsi_)
            self.arr_err.append(err_)
            self.arr_n1.append(n1)
            self.arr_n2.append(n2)
            self.arr_n3.append(n3)
            self.arr_n4.append(n4)
            
        s_ = [x_-r_x, y_-r_y, z_-r_z, Phi_, theta_, Psi_, dx_-r_v_x, dy_-r_v_y, dz_-r_v_z, dPhi_, dtheta_, dPsi_]
        angular_norm2 = M.sqrt(pow(Phi_, 2) + pow(theta_, 2) + pow(Psi_, 2))
        velocity_norm2 = M.sqrt(pow(dx_-r_v_x, 2) + pow(dy_-r_v_y, 2) + pow(dz_-r_v_z, 2))
        angularspeed_norm2 = M.sqrt(pow(dPhi_, 2) + pow(dtheta_, 2) + pow(dPsi_, 2))
        motor_norm2 = M.sqrt(pow(action[0], 2) + pow(action[0], 2) + pow(action[0], 2) + pow(action[0], 2))
        r = - (self.args.c_1 * err_ + self.args.c_2 * angular_norm2 + self.args.c_3 * angularspeed_norm2 +
               self.args.c_4 * velocity_norm2 + self.args.c_5 * motor_norm2) + 1

        self.episode_step += 1

        if err_ > self.args.bound_train:
            self.done = True
            r += -150
            self.env_term()

        if self.episode_step == self.args.max_episode_steps:
            r += -10*err_
            self.done = True
            self.env_term()
        
        return s_, r, self.done
    
    def env_term(self):
        self.quad_term()
        self.motor_term()

    def gener_refer_figure_8(self):
        r_x = []
        r_y = []
        r_z = []
        r_v_x = []
        r_v_y = []
        r_v_z = []
        r = 0.5
        for i in range(20001):
            t = 0.001 * i
            x = 2*r * M.cos(2*M.pi/10 * t)
            y = r * M.sin(2*M.pi/10 * 2*t)
            z = 0
            v_x = - 2*r * 2*M.pi/10 * M.sin(2*M.pi/10 * t)
            v_y = r * 2*M.pi/10 * 2 * M.cos(2*M.pi/10 * 2*t)
            v_z = 0
            r_x.append(x)
            r_y.append(y)
            r_z.append(z)
            r_v_x.append(v_x)
            r_v_y.append(v_y)
            r_v_z.append(v_z)

        self.refer_trace = np.array([r_x, r_y, r_z])
        self.refer_speed = np.array([r_v_x, r_v_y, r_v_z])
        
        
    def gener_refer_quadrifolium(self):
        r_x = []
        r_y = []
        r_z = []
        r_v_x = []
        r_v_y = []
        r_v_z = []
        r = 1
        for i in range(20001):
            t = 0.001 * i
            x = r * M.cos(2*M.pi/10 * t) * M.cos(M.pi/10 * t)
            y = r * M.cos(2*M.pi/10 * t) * M.sin(M.pi/10 * t)
            # x = r * M.cos(2*M.pi/20 * t) * M.cos(M.pi/20 * t)
            # y = r * M.cos(2*M.pi/20 * t) * M.sin(M.pi/20 * t)
            z = 0
            v_x = - r * 2*M.pi/10*M.sin(2*M.pi/10 * t) * M.cos(M.pi/10 * t) - r * M.pi/10*M.cos(2*M.pi/10 * t) * M.sin(M.pi/10 * t)
            v_y = - r * 2*M.pi/10*M.sin(2*M.pi/10 * t) * M.cos(M.pi/10 * t) + r * M.pi/10*M.cos(2*M.pi/10 * t) * M.cos(M.pi/10 * t)
            # v_x = - r * 2*M.pi/20*M.sin(2*M.pi/20 * t) * M.cos(M.pi/20 * t) - r * M.pi/20*M.cos(2*M.pi/20 * t) * M.sin(M.pi/20 * t)
            # v_y = - r * 2*M.pi/20*M.sin(2*M.pi/20 * t) * M.cos(M.pi/20 * t) + r * M.pi/20*M.cos(2*M.pi/20 * t) * M.cos(M.pi/20 * t)
            v_z = 0
            r_x.append(x)
            r_y.append(y)
            r_z.append(z)
            r_v_x.append(v_x)
            r_v_y.append(v_y)
            r_v_z.append(v_z)

        self.refer_trace = np.array([r_x, r_y, r_z])
        self.refer_speed = np.array([r_v_x, r_v_y, r_v_z])


    def save_ep_result(self, train_or_test, evaluate_num=0, test_mode=''):
        if train_or_test == 'train':
            path='./episodes/EP_{}/'.format(evaluate_num)
        elif train_or_test == 'test':
            path='./test_results/' + test_mode + '/'
        create_directory(path, sub_paths=['Position', 'Orientation', 'Motorspeed'])
        plot_train_position_curve(np.arange(0, (len(self.arr_x)-1)/self.args.simu_step, 1/self.args.simu_step), self.r_x[:-1], self.arr_x[:-1], 'x',
                                path + 'Position/x.pdf')
        plot_train_position_curve(np.arange(0, (len(self.arr_y)-1)/self.args.simu_step, 1/self.args.simu_step), self.r_y[:-1], self.arr_y[:-1], 'y',
                                path + 'Position/y.pdf')
        plot_train_position_curve(np.arange(0, (len(self.arr_z)-1)/self.args.simu_step, 1/self.args.simu_step),self.r_z[:-1], self.arr_z[:-1], 'z',
                                path + 'Position/z.pdf')
        plot_err_curve(np.arange(0, (len(self.arr_err)-1)/self.args.simu_step, 1/self.args.simu_step), self.arr_err[:-1], 'err',
                                path + 'err.pdf')
        plot_orientation_curve(np.arange(0, (len(self.arr_Phi)-1)/self.args.simu_step, 1/self.args.simu_step), self.arr_Phi[:-1], 'Phi',
                                path + 'Orientation/Phi.pdf')
        plot_orientation_curve(np.arange(0, (len(self.arr_theta)-1)/self.args.simu_step, 1/self.args.simu_step), self.arr_theta[:-1], 'theta',
                                path + 'Orientation/theta.pdf')
        plot_orientation_curve(np.arange(0, (len(self.arr_Psi)-1)/self.args.simu_step, 1/self.args.simu_step), self.arr_Psi[:-1], 'Psi',
                                path + 'Orientation/Psi.pdf')
        plot_motorspeed_curve(np.arange(0, (len(self.arr_n1)-1)/self.args.simu_step, 1/self.args.simu_step), self.arr_n1[:-1], 'n1',
                                path + 'Motorspeed/n1.pdf')
        plot_motorspeed_curve(np.arange(0, (len(self.arr_n2)-1)/self.args.simu_step, 1/self.args.simu_step), self.arr_n2[:-1], 'n2',
                                path + 'Motorspeed/n2.pdf')
        plot_motorspeed_curve(np.arange(0, (len(self.arr_n3)-1)/self.args.simu_step, 1/self.args.simu_step), self.arr_n3[:-1], 'n3',
                                path + 'Motorspeed/n3.pdf')
        plot_motorspeed_curve(np.arange(0, (len(self.arr_n4)-1)/self.args.simu_step, 1/self.args.simu_step), self.arr_n4[:-1], 'n4',
                                path + 'Motorspeed/n4.pdf')
        plot_trace_3D(self.arr_x, self.arr_y, self.arr_z, path + 'trace3D.pdf')
        plot_trace_2D(self.arr_x, self.arr_z, path + 'trace2D.pdf')
        arrstates = np.concatenate(([self.arr_x], [self.arr_y], [self.arr_z], [self.arr_Phi], [self.arr_theta], [self.arr_Psi],
                                    [self.arr_dx], [self.arr_dy], [self.arr_dz], [self.arr_dPhi], [self.arr_dtheta], [self.arr_dPsi]), axis = 0)
        arractions = np.concatenate(([self.arr_n1], [self.arr_n2], [self.arr_n3], [self.arr_n4]), axis=0)
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
            