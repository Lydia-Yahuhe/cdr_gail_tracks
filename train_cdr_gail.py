"""
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
"""

import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset import generate_dataset
from baselines.gail.adversary import TransitionClassifier
from fltenv.env import ConflictEnv


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--video_path', type=str, default='dataset/env_train.avi')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='evaluate')
    # for evaluation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    # Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=1e-3)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=2)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=int(5e6))
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def main():
    args = argsparser()

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = ConflictEnv(limit=0, act='Box')

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)

    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = "gail.seed_{}.BC_{}.iters_{}".format(args.seed, args.BC_max_iter, args.num_timesteps)
    checkpoint_dir = osp.abspath(args.checkpoint_dir)
    save_dir = osp.join(checkpoint_dir, task_name)

    if args.task == 'train':
        dataset = generate_dataset(args.video_path, env.picture_size)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)

        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              save_dir,
              task_name)
    elif args.task == 'evaluate':
        output_gail_policy(env,
                           policy_fn,
                           save_dir,
                           stochastic_policy=args.stochastic_policy)
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, task_name=None):
    from baselines.gail import trpo_mpi
    # Set up for MPI seed
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env.seed(workerseed)
    trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                   g_step=g_step, d_step=d_step,
                   entcoeff=policy_entcoeff,
                   max_timesteps=num_timesteps,
                   ckpt_dir=checkpoint_dir,
                   save_per_iter=save_per_iter,
                   timesteps_per_batch=32,
                   max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   gamma=0.995, lam=0.97,
                   vf_iters=5, vf_stepsize=1e-3,
                   task_name=task_name)


def output_gail_policy(env, policy_func, load_model_path, stochastic_policy=False, reuse=False):
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    print(load_model_path)
    U.load_variables(load_model_path, variables=pi.get_variables())
    env.evaluate(pi.act, save_path='gail_policy', stochastic=stochastic_policy)


if __name__ == '__main__':
    main()
