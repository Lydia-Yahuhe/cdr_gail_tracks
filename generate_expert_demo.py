from fltenv import ConflictScene
from fltenv.cmd import CmdCount
from fltenv.env import ConflictEnv

from fltsim.visual import *


def generate_random_policy():
    env = ConflictEnv(limit=0)
    size = len(env.train)
    action_list = list(range(CmdCount))

    count = 0
    for info in env.train:
        num = info.id
        print('>>>', num)

        np.random.shuffle(action_list)

        for i, action in enumerate(action_list):
            scene = ConflictScene(info, limit=0, read=False)

            obs, done, result = None, False, {}
            while not done:
                next_obs, rew, done, result = env.step(action, scene=scene)
                obs = next_obs

            if result['result']:
                cv2.imwrite("No.{}_{}_{}.jpg".format(num, action, scene.now()), obs)
                count += 1

                print('\t', i, count/size*100)
                break
            else:
                print('\t', i)

    print('Success Rate is {}%'.format(count * 100.0 / size))


generate_random_policy()
