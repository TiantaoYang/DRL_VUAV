import numpy as np
from torch.utils.tensorboard import SummaryWriter
from normalization import RewardScaling
from replaybuffer import ReplayBuffer
from ppo import PPO
from utils import *
from env_Quadrotor import *
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def evaluate_policy(args, agent, evaluate_num):
    evaluate_reward = 0
    env_eval = env_Quad(args)
    
    done = False
    episode_reward = 0
    # s = env_eval.reset_track(trajectory_type='figure_8', rank = [0, 2, 1])
    s = env_eval.reset_regulate()
    while not done:
        a = agent.evaluate(s)
        action = a * args.max_action
        s_, r, done = env_eval.env_step(action)
        episode_reward += r
        s = s_
    evaluate_reward += episode_reward
    env_eval.save_ep_result('train', evaluate_num)
    
    return evaluate_reward


def main(args): 
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))
    create_directory(args.checkpoint_dir, sub_paths=['Actor', 'Critic'])

    evaluate_num = 0
    max_reward = 0
    evaluate_rewards = []
    total_steps = 0

    replay_buffer = ReplayBuffer(args)
    agent = PPO(args)
    
    writer = SummaryWriter(log_dir='runs/PPO_Quadrotor')
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    
    env_train = env_Quad(args)

    while total_steps < args.max_train_steps:
        # s = env_train.reset_track(trajectory_type='figure_8', rank = [0, 2, 1])
        s = env_train.reset_regulate()

        reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            action = a * args.max_action
            s_, r, done = env_train.env_step(action)
            r = reward_scaling(r)

            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, agent, evaluate_num)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{:2f}".format(evaluate_num, evaluate_reward))
                if evaluate_num == 1 or evaluate_reward > max_reward:
                    max_reward = evaluate_reward
                    agent.save_models(evaluate_num)
                writer.add_scalar('step_rewards_Quadrotor', evaluate_rewards[-1], global_step=total_steps)
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_Quadrotor.npy', np.array(evaluate_rewards))
                    agent.save_models(evaluate_num)
            

if __name__ == '__main__':
    args = get_args()
    main(args)
