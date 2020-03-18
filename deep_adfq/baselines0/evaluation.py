import datetime, json, os, argparse, time
import pickle, tabulate
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from baselines0.common import set_global_seeds

class Test:
    def __init__(self):
        pass

    def test(self, args, env, act):
        seed = args.seed
        env.seed(seed)
        set_global_seeds(seed)

        if args.eval_type == 'random':
            params_set = [{}]
        elif args.eval_type == 'fixed_nb':
            if args.env == 'setTracking-v1':
                params_set = [{}]
            elif args.env == 'setTracking-v2':
                params_set = SET_EVAL_v4
            elif args.env == 'setTracking-v3':
                params_set = SET_EVAL_v3
            elif args.env == 'setTracking-v4':
                params_set = SET_EVAL_v4
            elif args.env == 'setTracking-v5':
                params_set = SET_EVAL_v4
            else:
                raise ValueError("Eval set not created for this env.")
        else:
            raise ValueError("Wrong evaluation type for ttenv.")

        timelimit_env = env
        while( not hasattr(timelimit_env, '_elapsed_steps')):
            timelimit_env = timelimit_env.env

        if args.ros_log:
            from envs.target_tracking.ros_wrapper import RosLog
            ros_log = RosLog(num_targets=args.nb_targets, wrapped_num=args.ros + args.render + args.record + 1)


        total_nlogdetcov = []
        for params in params_set:
            ep = 0
            ep_nlogdetcov = [] #'Episode nLogDetCov'
            time_elapsed = ['Elapsed Time (sec)']

            while(ep < args.nb_test_steps): # test episode
                ep += 1
                episode_rew, nlogdetcov = 0, 0
                done = {}
                obs = env.reset(**params)

                s_time = time.time()

                action_dict = {}
                while type(done) is dict:
                    if args.render:
                        env.render()
                    if args.ros_log:
                        ros_log.log(env)
                    for agent_id, a_obs in obs.items():
                        action_dict[agent_id] = act(np.array(a_obs)[None])[0]
                    obs, rew, done, info = env.step(action_dict)
                    episode_rew += rew['__all__']
                    nlogdetcov += info['mean_nlogdetcov']

                time_elapsed.append(time.time() - s_time)
                ep_nlogdetcov.append(nlogdetcov)
                if ep % 50 == 0:
                    print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f"%(ep, episode_rew, nlogdetcov))

            if args.record :
                env.moviewriter.finish()
            if args.ros_log :
                ros_log.save(args.log_dir)

            # Stats
            meanofeps = np.mean(ep_nlogdetcov)
            total_nlogdetcov.append(meanofeps)
            # Eval plots and saves
            eval_dir = os.path.join(args.log_dir, 'eval_seed%d_'%(seed)+args.map)
            model_seed = os.path.split(args.log_fname)[0]
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            matplotlib.use('Agg')
            f0, ax0 = plt.subplots()
            _ = ax0.plot(ep_nlogdetcov, '.')
            _ = ax0.set_title(args.env)
            _ = ax0.set_xlabel('episode number')
            _ = ax0.set_ylabel('mean nlogdetcov')
            _ = ax0.axhline(y=meanofeps, color='r', linestyle='-', label='mean over episodes: %.2f'%(meanofeps))
            _ = ax0.legend()
            _ = ax0.grid()
            _ = f0.savefig(os.path.join(eval_dir, "%da%dt_%d_eval_"%(env.nb_agents, env.nb_targets, args.nb_test_steps)
                                                    +model_seed+".png"))
            plt.close()
            pickle.dump(ep_nlogdetcov, open(os.path.join(eval_dir,"%da%dt_%d_eval_"%(env.nb_agents, env.nb_targets, args.nb_test_steps))
                                                                    +model_seed+".pkl", 'wb'))
        #Plot over all example episode sets
        f1, ax1 = plt.subplots()
        _ = ax1.plot(total_nlogdetcov, '.')
        _ = ax1.set_title(args.env)
        _ = ax1.set_xlabel('example episode set number')
        _ = ax1.set_ylabel('mean nlogdetcov over episodes')
        _ = ax1.grid()
        _ = f1.savefig(os.path.join(eval_dir,'all_%d_eval'%(args.nb_test_steps)+model_seed+'.png'))
        plt.close()        
        pickle.dump(total_nlogdetcov, open(os.path.join(eval_dir,'all_%d_eval'%(args.nb_test_steps))+model_seed+'.pkl', 'wb'))


SET_EVAL_v3 = [{
        'nb_agents': 1,
        'nb_targets': 1
        },
        {
        'nb_agents': 1,
        'nb_targets': 2
        },
        {
        'nb_agents': 2,
        'nb_targets': 2
        },
        {
        'nb_agents': 2,
        'nb_targets': 3
        },
        {
        'nb_agents': 3,
        'nb_targets': 3,
        },
        {
        'nb_agents': 3,
        'nb_targets': 4
        },
        {
        'nb_agents': 4,
        'nb_targets': 4
        }
]

SET_EVAL_v4 = [{
        'nb_agents': 2,
        'nb_targets': 1
        },
        {
        'nb_agents': 2,
        'nb_targets': 2
        },
        {
        'nb_agents': 2,
        'nb_targets': 3
        },
        {
        'nb_agents': 2,
        'nb_targets': 4
        }
]