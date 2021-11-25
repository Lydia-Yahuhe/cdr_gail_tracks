import numpy as np

from fltenv import ConflictScene
from fltenv.cmd import CmdCount
from fltenv.env import ConflictEnv

from fltsim.visual import *


def generate_random_policy():
    env = ConflictEnv(limit=0)
    size = len(env.train)
    action_list = list(range(CmdCount))

    obs_array = []
    act_array = []
    num_array = []
    rew_array = []
    count = 0
    for info in env.train:
        num = info.id
        print('>>>', num)

        if num == '15':
            continue

        np.random.shuffle(action_list)

        for i, action in enumerate(action_list):
            scene = ConflictScene(info, limit=0, read=True)

            obs, done, result = None, False, {}
            while not done:
                next_obs, rew, done, result = env.step(action, scene=scene)
                obs = next_obs

            if result['result']:
                # cv2.imwrite("No.{}_{}_{}.jpg".format(num, action, scene.now()), obs)
                obs_array.append(obs)
                act_array.append([action, scene.now()])
                num_array.append(num)
                rew_array.append(rew)
                count += 1

                print('\t', i, count/size*100)
                break
            else:
                print('\t', i)

    obs_array = np.array(obs_array, dtype=np.float)
    act_array = np.array(act_array)
    num_array = np.array(num_array)
    rew_array = np.array(rew_array, dtype=np.float)
    print('Success Rate is {}%'.format(count * 100.0 / size))

    np.savez('dqn_policy_with_tracks.npz', obs=obs_array, acs=act_array, num=num_array, rews=rew_array)


generate_random_policy()
