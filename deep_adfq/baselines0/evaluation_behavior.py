import datetime, json, os, argparse, time
import pickle, tabulate
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from baselines0.common import set_global_seeds
import tensorflow as tf


def get_init_pose_list(nb_test_steps, eval_type):
    init_pose_list = []
    if eval_type == 'fixed_2':
        for ii in range(nb_test_steps):
            left = np.random.uniform(20,30)
            Lyaxis = np.random.uniform(25,35)
            right = np.random.uniform(30,40)
            Ryaxis = np.random.uniform(35,45)
            init_pose_list.append({'agents':[[24.5, 15.5, 1.57], [26.5, 15.5, 1.57]],
                            'targets':[[left, Lyaxis, 0, 0],[right, Ryaxis, 0, 0]],
                            'belief_targets':[[left, Lyaxis, 0, 0], [right, Ryaxis, 0, 0]]})
    else:
        for ii in range(nb_test_steps):
            xone = np.random.uniform(20,30)
            yone = np.random.uniform(15,25)
            xtwo = np.random.uniform(30,40)
            ytwo = np.random.uniform(25,35)
            xthree = np.random.uniform(20,30)
            ythree = np.random.uniform(35,45)
            xfour = np.random.uniform(10,20)
            yfour = np.random.uniform(35,45)
            init_pose_list.append({'agents':[[24.5, 10, 1.57], [26.5, 10, 1.57], 
                                            [22.5, 10, 1.57], [28.5, 10, 1.57]],
                            'targets':[[xone, yone, 0, 0],[xtwo, ytwo, 0, 0],
                                        [xthree, ythree, 0, 0],[xfour, yfour, 0, 0]],
                            'belief_targets':[[xone, yone, 0, 0], [xtwo, ytwo, 0, 0],
                                            [xthree, ythree, 0, 0], [xfour, yfour, 0, 0]]}) 

    return init_pose_list

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
                # params_set = SET_EVAL_8a
            elif args.env == 'setTracking-v4':
                params_set = SET_EVAL_v4
            elif args.env == 'setTracking-v5':
                params_set = SET_EVAL_v4
            elif args.env == 'maTracking-v4':
                params_set = MA_EVAL
            elif args.env == 'setTracking-v6':
                params_set = SET_EVAL_v3
            elif args.env == 'setTracking-v7':
                params_set = SET_EVAL_v3
                # params_set = SET_EVAL_8a
            else:
                raise ValueError("Eval set not created for this env.")
        elif args.eval_type == 'fixed_2':
            params_set = EVAL_BEHAVIOR_2
            tot_eplen = 60
        elif args.eval_type == 'fixed_4':
            params_set = EVAL_BEHAVIOR_4
            tot_eplen = 100

        else:
            raise ValueError("Wrong evaluation type for ttenv.")

        timelimit_env = env
        while( not hasattr(timelimit_env, '_elapsed_steps')):
            timelimit_env = timelimit_env.env

        if args.ros_log:
            from envs.target_tracking.ros_wrapper import RosLog
            ros_log = RosLog(num_targets=args.nb_targets, wrapped_num=args.ros + args.render + args.record + 1)

        init_pose_list = get_init_pose_list(args.nb_test_steps, args.eval_type)
        total_nlogdetcov = []
        for params in params_set:
            ep = 0
            ep_nlogdetcov = [] #'Episode nLogDetCov'
            time_elapsed = ['Elapsed Time (sec)']
            test_observations = np.zeros(args.nb_test_steps)

            while(ep < args.nb_test_steps): # test episode
                ep += 1
                episode_rew, nlogdetcov, ep_len = 0, 0, 0
                done = {}
                obs = env.reset(init_pose_list=init_pose_list, **params)

                s_time = time.time()

                all_observations = np.zeros(env.nb_targets, dtype=bool)
                action_dict = {}
                bigq0 = []
                bigq1 = []
                # while type(done) is dict:
                while ep_len < tot_eplen:
                    if args.render:
                        env.render()
                    if args.ros_log:
                        ros_log.log(env)
                    for agent_id, a_obs in obs.items():
                        action_dict[agent_id] = act(np.array(a_obs)[None])[0]
                        # record target observations
                        observed = np.zeros(env.nb_targets, dtype=bool)
                        all_observations = np.logical_or(all_observations, a_obs[:,5].astype(bool))  
                    obs, rew, done, info = env.step(action_dict)
                    episode_rew += rew['__all__']
                    nlogdetcov += info['mean_nlogdetcov']

                    rearrange = [0,3,6,9,1,4,7,10,2,5,8,11]
                    qs0 = np.zeros((12))
                    qs1 = np.zeros((12))
                    q0 = np.zeros((12))
                    q1 = np.zeros((12))
                    qs0[action_dict['agent-0']] = 1
                    qs1[action_dict['agent-1']] = 1
                    for ii, val in enumerate(rearrange):
                        q0[ii] = qs0[val]
                        q1[ii] = qs1[val] 

                    bigq0.append(q0)
                    bigq1.append(q1)
                    ep_len += 1
                bigq0 = np.asarray(bigq0)
                bigq1 = np.asarray(bigq1)

                time_elapsed.append(time.time() - s_time)
                ep_nlogdetcov.append(nlogdetcov)
                if args.render:
                    print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f"%(ep, episode_rew, nlogdetcov))
                if ep % 50 == 0:
                    print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f"%(ep, episode_rew, nlogdetcov))

            if args.record :
                env.moviewriter.finish()
            if args.ros_log :
                ros_log.save(args.log_dir)

            # Stats
            # meanofeps = np.mean(ep_nlogdetcov)
            # total_nlogdetcov.append(meanofeps)
            # # Eval plots and saves
            # if args.env == 'setTracking-v7':
            #     eval_dir = os.path.join(os.path.split(args.log_dir)[0], 'v7_eval_seed%d_'%(seed)+args.map)
            # else:
            #     eval_dir = os.path.join(os.path.split(args.log_dir)[0], 'eval_seed%d_'%(seed)+args.map)
            # model_seed = os.path.split(args.log_dir)[-1]           
            # # eval_dir = os.path.join(args.log_dir, 'eval_seed%d_'%(seed)+args.map)
            # # model_seed = os.path.split(args.log_fname)[0]
            # if not os.path.exists(eval_dir):
            #     os.makedirs(eval_dir)
            # # matplotlib.use('Agg')
            # f0, ax0 = plt.subplots()
            # _ = ax0.plot(ep_nlogdetcov, '.')
            # _ = ax0.set_title(args.env)
            # _ = ax0.set_xlabel('episode number')
            # _ = ax0.set_ylabel('mean nlogdetcov')
            # _ = ax0.axhline(y=meanofeps, color='r', linestyle='-', label='mean over episodes: %.2f'%(meanofeps))
            # _ = ax0.legend()
            # _ = ax0.grid()
            # _ = f0.savefig(os.path.join(eval_dir, "%da%dt_%d_eval_"%(env.nb_agents, env.nb_targets, args.nb_test_steps)
            #                                         +model_seed+".png"))
            # plt.close()
            # pickle.dump(ep_nlogdetcov, open(os.path.join(eval_dir,"%da%dt_%d_eval_"%(env.nb_agents, env.nb_targets, args.nb_test_steps))
            #                                                         +model_seed+".pkl", 'wb'))

            f2 = plt.figure()
            ax2 = f2.add_subplot(121,projection='3d')
            ax3 = f2.add_subplot(122,projection='3d')

            lx = len(bigq0[0])
            ly = len(bigq0[:,0])
            xpos = np.arange(0,lx,1)
            ypos = np.arange(0,ly,1)
            xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)

            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros(lx*ly)

            dx = 0.5 *np.ones_like(zpos)
            dy = dx.copy()
            dz0 = bigq0.flatten()
            dz1 = bigq1.flatten()
            
            cs = ['r', 'r', 'r', 'r', 'g', 'g', 'g', 'g','b','b','b','b'] * ly

            ax2.bar3d(xpos,ypos,zpos, dx, dy, dz0, color=cs)
            ax3.bar3d(xpos,ypos,zpos, dx, dy, dz1, color=cs)

            print(test_observations)
            print("Cooperation ratio over total evals: %.2f"%(np.sum(test_observations)/args.nb_test_steps))
            plt.show()

        #Plot over all example episode sets
        # f1, ax1 = plt.subplots()
        # _ = ax1.plot(total_nlogdetcov, '.')
        # _ = ax1.set_title(args.env)
        # _ = ax1.set_xlabel('example episode set number')
        # _ = ax1.set_ylabel('mean nlogdetcov over episodes')
        # _ = ax1.grid()
        # _ = f1.savefig(os.path.join(eval_dir,'all_%d_eval'%(args.nb_test_steps)+model_seed+'.png'))
        # plt.close()        
        # pickle.dump(total_nlogdetcov, open(os.path.join(eval_dir,'all_%d_eval'%(args.nb_test_steps))+model_seed+'%da%dt'%(args.nb_agents,args.nb_targets)+'.pkl', 'wb'))


# init_pose_list = [{'agents':[[24.5, 15.5, 1.57], [26.5, 15.5, 1.57]],
#                   'targets':[[20, 35, 0, 0],[40, 35, 0, 0]],
#                   'belief_targets':[[20, 35, 0, 0], [40, 35, 0, 0]]}]

EVAL_BEHAVIOR_2 = [
        {'nb_agents': 2, 'nb_targets': 2},
]
EVAL_BEHAVIOR_4 = [
        {'nb_agents': 4, 'nb_targets': 4},
]


SET_EVAL_8a = [
        {'nb_agents': 4, 'nb_targets': 4},
        {'nb_agents': 6, 'nb_targets': 6},      
        {'nb_agents': 8, 'nb_targets': 4},
        {'nb_agents': 8, 'nb_targets': 6},
        {'nb_agents': 8, 'nb_targets': 8},
]


SET_EVAL_v3 = [
        {'nb_agents': 2, 'nb_targets': 2},
        # {'nb_agents': 2, 'nb_targets': 1},
        # {'nb_agents': 3, 'nb_targets': 1},
        # {'nb_agents': 4, 'nb_targets': 1},
        # {'nb_agents': 1, 'nb_targets': 2},
        # {'nb_agents': 2, 'nb_targets': 2},
        # {'nb_agents': 3, 'nb_targets': 2},
        # {'nb_agents': 4, 'nb_targets': 2},
        # {'nb_agents': 1, 'nb_targets': 3},
        # {'nb_agents': 2, 'nb_targets': 3},
        # {'nb_agents': 3, 'nb_targets': 3},
        # {'nb_agents': 4, 'nb_targets': 3},
        # {'nb_agents': 1, 'nb_targets': 4},
        # {'nb_agents': 2, 'nb_targets': 4},
        # {'nb_agents': 3, 'nb_targets': 4},
        # {'nb_agents': 4, 'nb_targets': 4},
]

SET_EVAL_v4 = [
        {'nb_agents': 2,'nb_targets': 1},
        {'nb_agents': 2,'nb_targets': 2},
        {'nb_agents': 2,'nb_targets': 3},
        {'nb_agents': 2,'nb_targets': 4}
]

MA_EVAL = [
        {'nb_agents': 4},
        # {'nb_agents': 2},
        # {'nb_agents': 3},
        # {'nb_agents': 4}
]