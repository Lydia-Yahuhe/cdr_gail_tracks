from abc import ABC

import gym
from gym import spaces

from fltenv.scene import ConflictScene
from fltenv.cmd import CmdCount, reward_for_cmd

from fltsim.load import load_and_split_data
from fltsim.utils import build_rt_index_with_list, make_bbox, distance
from fltsim.visual import *


def calc_reward(solved, done, cmd_info):
    if not solved and done:  # failed
        reward = -5.0
        print('({:<2})'.format(round(reward, 2)), end='\t')
        return reward

    rew = reward_for_cmd(cmd_info)
    if solved:   # solved
        reward = 0.5+min(rew, 0)
    else:        # undone
        reward = rew
    print('({:<2})'.format(round(reward, 2)), end='\t')
    return reward


class ConflictEnv(gym.Env, ABC):
    def __init__(self, limit=30, act='Discrete'):
        self.limit = limit
        self.train, self.test = load_and_split_data('scenarios_gail_final', split_ratio=0.8)

        if act == 'Discrete':
            self.action_space = spaces.Discrete(CmdCount)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float64)

        self.picture_size = (width, height, channel) = (670, 450, 3)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf,
                                            shape=(height, width, channel),
                                            dtype=np.float64)
        print('----------env----------')
        print('    train size: {:>6}'.format(len(self.train)))
        print(' validate size: {:>6}'.format(len(self.test)))
        if act == 'Discrete':
            print('  action shape: {}'.format((self.action_space.n,)))
        else:
            print('  action shape: {}'.format(self.action_space.shape))
        print('   state shape: {}'.format(self.observation_space.shape))
        print('-----------------------')

        self.scene = None
        self.video_out = None
        self.result = None

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

        return self.scene.get_states(*self.picture_size)

    def step(self, action, scene=None):
        if scene is None:
            scene = self.scene

        solved, done, cmd_info = scene.do_step(action)
        rewards = calc_reward(solved, done, cmd_info)
        states = scene.get_states(*self.picture_size)
        self.result = solved
        return states, rewards, done, {'result': solved}

    def evaluate(self, act, save_path='policy', **kwargs):
        obs_array = []
        act_array = []
        rew_array = []
        n_obs_array = []
        indexes = []

        size = len(self.test)

        episode = 0
        while not self.test_over():
            print(episode, size)
            obs_collected = {'obs': [], 'act': [], 'rew': [], 'n_obs': []}

            obs, done = self.reset(test=True), False
            result = {'result': True}
            count = 0
            while not done:
                if 'gail' in save_path:
                    action, _ = act(kwargs['stochastic'], obs)
                    print(action)
                    action = int(action[0]*54+54)
                    print(action)
                else:
                    action = act(np.array(obs)[None])[0]
                    print(action)
                next_obs, rew, done, result = self.step(action)

                obs_collected['obs'].append(obs)
                obs_collected['act'].append(action)
                obs_collected['rew'].append(rew)
                obs_collected['n_obs'].append(next_obs)
                obs = next_obs
                count += 1

            if result['result']:
                obs_array += obs_collected['obs']
                act_array += obs_collected['act']
                rew_array += obs_collected['rew']
                n_obs_array += obs_collected['n_obs']
                indexes.append(count)

            episode += 1

        obs_array = np.array(obs_array, dtype=np.float64)
        act_array = np.array(act_array, dtype=np.float64)
        rew_array = np.array(rew_array, dtype=np.float64)
        n_obs_array = np.array(n_obs_array, dtype=np.float64)
        indexes = np.array(indexes, dtype=np.int8)

        print('Success Rate is {}%'.format(len(indexes) * 100.0 / size))
        print(obs_array.shape, act_array.shape, rew_array.shape, n_obs_array.shape)
        np.savez(save_path+'.npz', obs=obs_array, acs=act_array, rews=rew_array, n_obs=n_obs_array, indexes=indexes)

    def render(self, mode='human'):
        picture_size = (670, 450)
        wait = 1000
        if self.video_out is None:
            self.video_out = cv2.VideoWriter('env_train.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, picture_size)

        # kwargs = dict(border=[108, 118, 28, 35], scale=200)
        kwargs = dict(border=[109.3, 116, 29, 33.5], scale=100)

        play_speed = int(8000 / wait)

        info = self.scene.info
        agent_set = self.scene.agentSet
        cmd_info_dict = self.scene.cmd_info
        conflict_ac, clock = info.conflict_ac, info.time

        if self.result:
            agent_set.do_step(duration=clock+300, basic=True)

        # 轨迹点
        points_dict = {}
        for key, agent in agent_set.agents.items():
            for clock_, track in agent.tracks.items():
                if clock_ in points_dict.keys():
                    points_dict[clock_].append([key, key in conflict_ac] + track)
                else:
                    points_dict[clock_] = [[key, key in conflict_ac]+track]

        # 指令信息整理（class → str）
        cmd_display_dict = {}
        for key, cmd_info in cmd_info_dict.items():
            cmd_dict = {'Time': '{}({}, {})'.format(key, cmd_info['hold'], clock-key), 'Agent': cmd_info['agent']}
            for cmd in cmd_info['cmd']:
                tmp = cmd.to_dict()
                tmp['time'] = cmd.assignTime
                cmd_dict.update(tmp)

            cmd_display_dict[key] = cmd_dict

        points, cmd_display = [], [{">>> Instructions: {}, Result".format(len(cmd_info_dict)): self.result}]
        for t in range(clock-300+self.limit, clock+300):
            # 武汉扇区的底图（有航路）
            base_img = cv2.imread('dataset/wuhan_base.jpg', cv2.IMREAD_COLOR)

            # 全局信息
            global_info = {'>>> Global Information': '',
                           'Time': '{}({}), ac_en: {}, speed: x{}'.format(t, clock-t, len(points), play_speed)}
            add_texts_on_base_map(global_info, base_img, start=(700, 30), color=(255, 255, 0))

            # 冲突信息
            conflict_info = {'>>> Conflict Information': ''}
            conflict_info.update(info.to_dict())
            add_texts_on_base_map(conflict_info, base_img, start=(700, 80), color=(180, 238, 180))

            # 指令信息
            if t in cmd_display_dict.keys():
                tmp = cmd_display_dict[t]
                # print(t, tmp)
                cmd_display.append(tmp)
            for j, cmd_dict in enumerate(cmd_display):
                if j == 0:
                    start = (50, 30)
                else:
                    start = (-100+j*150, 50)
                add_texts_on_base_map(cmd_dict, base_img, start=start, color=(255, 0, 255))

            if t not in points_dict.keys():
                continue

            # print(t, points)
            points = points_dict[t]

            # 将当前时刻所有航空器的位置和状态放在图上
            _, points_just_coord = add_points_on_base_map(points, base_img, display=True, **kwargs)

            idx = build_rt_index_with_list(points_just_coord)
            lines = []
            # 增加连线
            for i, [_, is_c_ac, *point] in enumerate(points):
                if not is_c_ac:  # 只加冲突航空器与其它航空器之间的连线
                    continue

                pos0 = point[:3]
                for j in idx.intersection(make_bbox(pos0, ext=(0.2, 0.2, 450))):
                    if i == j:
                        continue

                    pos1 = points_just_coord[j]
                    h_dist = distance(pos0, pos1) / 1000
                    v_dist = abs(pos0[-1] - pos1[-1])
                    has_conflict = h_dist <= 10 and v_dist < 300
                    lines.append([pos0, pos1, h_dist, v_dist, has_conflict])

            frame = add_lines_on_base_map(lines, base_img, color=(0, 0, 255), **kwargs)
            frame = cv2.resize(frame, picture_size)
            cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('video', frame)
            cv2.waitKey(wait)
            self.video_out.write(frame)

    def close(self):
        if self.video_out is not None:
            self.video_out.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
