from abc import ABC

import gym
from gym import spaces
import numpy as np

from fltenv.scene import ConflictScene
from fltenv.cmd import CmdCount, reward_for_cmd
from fltsim.load import load_and_split_data


def calc_reward(has_conflict, delta, cmd_info):
    rew = reward_for_cmd(cmd_info)
    if has_conflict:  # failed
        reward = -1.0
    elif delta >= 300:   # solved
        reward = 0.5+min(rew, 0)
    else:             # undone
        reward = rew
    # print(has_conflict, delta, reward)
    print('({:<2})'.format(reward), end='\t')
    return reward, has_conflict or delta >= 300


class ConflictEnv(gym.Env, ABC):
    def __init__(self, limit=30):
        self.limit = limit
        self.train, self.test = load_and_split_data('scenarios_gail_final', size=6000, split_ratio=0.8)
        # self.test = self.train[:]

        self.action_space = spaces.Discrete(CmdCount*2)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(350, ), dtype=np.float64)

        print('----------env----------')
        print('    train size: {:>6}'.format(len(self.train)))
        print(' validate size: {:>6}'.format(len(self.test)))
        print('  action shape: {}'.format((self.action_space.n,)))
        print('   state shape: {}'.format(self.observation_space.shape))
        print('-----------------------')
        self.scene = None

    def test_over(self):
        return len(self.test) <= 0

    def shuffle_data(self):
        np.random.shuffle(self.train)

    def reset(self, test=False):
        if not test:
            info = self.train.pop(0)
            self.scene = ConflictScene(info, limit=self.limit)
            self.train.append(info)
        else:
            info = self.test.pop(0)
            self.scene = ConflictScene(info, limit=self.limit)
        return self.scene.get_states()

    def step(self, action, scene=None):
        if scene is None:
            scene = self.scene

        cmd_info = scene.assign_cmd_idx(action)
        has_conflict, delta = scene.do_step()
        rewards, done = calc_reward(has_conflict, delta, cmd_info)
        states = scene.get_states()

        return states, rewards, done, {'result': not has_conflict and delta >= 300}