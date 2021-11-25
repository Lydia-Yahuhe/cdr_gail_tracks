from baselines import deepq
from baselines.common import models
from baselines.gail.adversary import TransitionClassifier
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset

from fltenv.env import ConflictEnv

root = ".\\dataset\\my_model"


def train(test=False, path='dqn_policy'):
    env = ConflictEnv(limit=0)
    # dataset = Mujoco_Dset(expert_path='C:\\Users\\lydia\\Desktop\\Workspace\\data set\\Videos',
    #                       picture_size=env.picture_size)
    dataset = Mujoco_Dset(expert_path='dataset\\dqn_policy_with_tracks.npz',
                          picture_size=env.picture_size)
    reward_giver = TransitionClassifier(env, hidden_size=64, entcoeff=1e-3)

    network = models.mlp(num_hidden=64, num_layers=2, layer_norm=True)
    if not test:
        act = deepq.learn(
            env,
            network=network,  # 隐藏节点，隐藏层数
            lr=1e-4,
            batch_size=16,
            total_timesteps=100000,
            buffer_size=1000,

            learning_starts=100,

            reward_giver=reward_giver,
            expert_dataset=dataset,

            prioritized_replay=True
        )
        print('Save model to my_model.pkl')
        act.save(root+'.pkl')
    else:
        act = deepq.learn(env,
                          network=network,
                          total_timesteps=0,
                          load_path=root+"_{}.pkl".format(10000))
        env.evaluate(act, save_path=path)
    env.close()


if __name__ == '__main__':
    train()
    # train(test=True, path='random_policy')
    # for key, value in np.load('.\\dataset\\dqn_policy.npz').items():
    #     print(key, value.shape, value)
