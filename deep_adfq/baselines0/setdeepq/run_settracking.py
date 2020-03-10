"""
This code was slightly modified from the baselines0/baselines0/deepq/train_cartpole.py in order to use
a different evaluation method. In order to run, simply replace the original code with this code
in the original directory.
"""
import argparse
import tensorflow as tf
import datetime, json, os, argparse, time, pickle
import numpy as np

from baselines0.common import set_global_seeds
from baselines0 import setdeepq

import envs
from baselines0.setdeepq.logger import Logger, batch_plot

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='setTracking-v3')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--prioritized', type=int, default=0)
parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
parser.add_argument('--double_q', type=int, default=1)  #not used: always clipped double q
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dueling', type=int, default=0)
parser.add_argument('--nb_train_steps', type=int, default=1000000)
parser.add_argument('--buffer_size', type=int, default=1e6)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--nb_warmup_steps', type=int, default=20000)
parser.add_argument('--nb_epoch_steps', type=int, default=25000)
parser.add_argument('--checkpoint_freq', type=int, default=5000)
parser.add_argument('--target_update_freq', type=float, default=0.005) #tau in polyak averaging
parser.add_argument('--nb_test_steps',type=int, default=None)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--learning_rate_decay_factor', type=float, default=1.0)
parser.add_argument('--learning_rate_growth_factor', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--hiddens', type=str, default='64:128:64')
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--log_fname', type=str, default='model.pkl')
parser.add_argument('--eps_fraction', type=float, default=0.1)
parser.add_argument('--eps_min', type=float, default=.02)
parser.add_argument('--test_eps', type=float, default=.05)
parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--record',type=int, default=0)
parser.add_argument('--render', type=int, default=0)
parser.add_argument('--gpu_memory',type=float, default=0.2)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--scope',type=str, default='setdeepq')
parser.add_argument('--ros', type=int, default=0)
parser.add_argument('--ros_log', type=int, default=0)
parser.add_argument('--map', type=str, default="emptyMed")
parser.add_argument('--nb_agents', type=int, default=4)
parser.add_argument('--nb_targets', type=int, default=4)
parser.add_argument('--eval_type', choices=['random', 'random_zone', 
                                            'fixed_nb', 'fixed'], default='fixed_nb')
parser.add_argument('--init_file_path', type=str, default=".", help="use eval_type: fixed_pose")

args = parser.parse_args()

def train(seed, save_dir):
    set_global_seeds(seed)
    save_dir_0 = os.path.join(save_dir, 'seed_%d'%seed)
    os.makedirs(save_dir_0)

    env = envs.make(args.env,
                    'ma_target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=save_dir_0,
                    ros=bool(args.ros),
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=False,
                    )

    with tf.device(args.device):
        with tf.compat.v1.variable_scope('seed_%d'%seed):
            hiddens = args.hiddens.split(':')
            hiddens = [int(h) for h in hiddens]
            model = setdeepq.models.SetTransformer()
            act = setdeepq.learn(
                env,
                q_func=model,
                lr=args.learning_rate,
                lr_decay_factor=args.learning_rate_decay_factor,
                lr_growth_factor=args.learning_rate_growth_factor,
                max_timesteps=args.nb_train_steps,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                exploration_fraction=args.eps_fraction,
                exploration_final_eps=args.eps_min,
                target_network_update_freq=args.target_update_freq,
                print_freq=10,
                checkpoint_freq=args.checkpoint_freq,
                learning_starts=args.nb_warmup_steps,
                gamma = args.gamma,
                prioritized_replay=bool(args.prioritized),
                prioritized_replay_alpha=args.prioritized_replay_alpha,
                callback=None,#callback,
                double_q = args.double_q,
                scope=args.scope,
                epoch_steps = args.nb_epoch_steps,
                eval_logger=Logger(args.env,
                                env_type='ma_target_tracking',
                                save_dir=save_dir_0,
                                render=bool(args.render),
                                figID=1,
                                ros=bool(args.ros),
                                map_name=args.map,
                                num_agents=args.nb_agents,
                                num_targets=args.nb_targets,
                                eval_type=args.eval_type,
                                init_file_path=args.init_file_path,
                                ),
                save_dir=save_dir_0,
                test_eps = args.test_eps,
                gpu_memory=args.gpu_memory,
                render = (bool(args.render) or bool(args.ros)),
            )
            print("Saving model to model.pkl")
            act.save(os.path.join(save_dir_0,"model.pkl"))
    if args.record == 1:
        env.moviewriter.finish()

def test(seed):
    learning_prop = json.load(open(os.path.join(args.log_dir, 'learning_prop.json'),'r'))
    env = envs.make(args.env,
                    'ma_target_tracking',
                    render=bool(args.render),
                    record=bool(args.record),
                    directory=args.log_dir,
                    ros=bool(args.ros),
                    map_name=args.map,
                    num_agents=args.nb_agents, #learning_prop['nb_agents'],
                    num_targets=args.nb_targets, #learning_prop['nb_targets'],
                    is_training=False,
                    )

    act_params = {'scope': "seed_%d"%learning_prop['seed']+"/"+learning_prop['scope'], 'eps': args.test_eps}
    act = setdeepq.load(os.path.join(args.log_dir, args.log_fname), act_params)

    from baselines0.evaluation import Test
    Eval = Test()
    Eval.test(args, env, act)

if __name__ == '__main__':
    if args.mode == 'train':
        save_dir = os.path.join(args.log_dir, '_'.join([args.env, datetime.datetime.now().strftime("%m%d%H%M")]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            ValueError("The directory already exists...", save_dir)
        json.dump(vars(args), open(os.path.join(save_dir, 'learning_prop.json'), 'w'))

        notes = input("Any notes for this experiment? : ")
        f = open(os.path.join(save_dir, "notes.txt"), 'w')
        f.write(notes)
        f.close()

        seed = args.seed
        list_records = []
        for _ in range(args.repeat):
            print("===== TRAIN A TARGET TRACKING RL AGENT : SEED %d ====="%seed)
            results = train(seed, save_dir)
            list_records.append(pickle.load(open(os.path.join(save_dir, "seed_%d"%seed, "records.pkl"), "rb")))
            seed += 1

        batch_plot(list_records, save_dir, args.nb_train_steps,
            args.nb_epoch_steps, is_target_tracking=True)

    elif args.mode =='test':
        test(args.seed)
