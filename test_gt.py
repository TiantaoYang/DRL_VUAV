from ppo import PPO
from ctypes import *
from utils import *
from env_Quadrotor import *
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def test(args, agent):
    env_test = env_Quad(args)

    done = False
    episode_reward = 0
    s = env_test.reset(trajectory_type='figure_8', rank = [0, 2, 1])
    
    l_tar = [0.19, 0.17, 0.23, 0.21]
    env_test.morph(l_tar)

    while not done:
        a = agent.evaluate(s)
        action = a * args.max_action
        s_, r, done = env_test.env_step(action)
        episode_reward += r
        s = s_
    env_test.save_ep_result(train_or_test='test', test_mode='gt')
    print(episode_reward)
    

if __name__ == '__main__':
    args = get_args()
    agent = PPO(args)
    agent.load_pretrain_models('test')
    test(args, agent)
