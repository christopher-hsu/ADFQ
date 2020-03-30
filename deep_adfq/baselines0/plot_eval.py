import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle, tabulate, os, argparse
from baselines0.setdeepq import logger

def contour():
    setv3 = []
    eval_seed = 'seed5'
    save_dir = args.log_dir
    seed = 0
    for ii in range(args.repeat):
        setv3.append(pickle.load(open(os.path.join(save_dir, 'eval_'+eval_seed+"_"+args.map, 'all_'+args.nb_test_steps+"_evalseed_%d"%seed+'.pkl'),'rb')))
        seed +=1
    mean = np.mean(setv3,axis=0)
    mean = np.reshape(np.asarray(mean),(4,4))


    # setup the figure and axes
    fig = plt.figure(figsize=(8, 7))
    # ax1 = fig.add_subplot(projection='3d')

    x = np.arange(1,5)
    y = np.arange(1,5)

    plt.contourf(x, y, mean, 50, cmap="viridis")

    plt.ylabel('number of targets')
    plt.xlabel('number of agents')
    plt.colorbar()
    plt.show()

def batch():
    list_records = []
    seed = args.seed
    save_dir = args.log_dir
    for ii in range(args.repeat):
        list_records.append(pickle.load(open(os.path.join(save_dir, "seed_%d"%seed, "records.pkl"), "rb")))
        seed += 1
    logger.batch_plot(list_records, save_dir, args.nb_train_steps, args.nb_epoch_steps, is_target_tracking=True)

def eval():
    eval_seed = 'seed5'
    list_records = []
    seed = args.seed
    save_dir = args.log_dir
    for ii in range(args.repeat):
        list_records.append(pickle.load(open(os.path.join(save_dir, 'eval_'+eval_seed+"_"+args.map,'all_'+args.nb_test_steps+"_evalseed_%d"%seed+'.pkl'),'rb')))
        seed +=1
    mean = np.mean(list_records,axis=0)
    std = np.std(list_records,axis=0)
    upstd = mean + 0.5*std
    downstd = mean - 0.5*std

    f0, ax0 = plt.subplots(figsize=(20,10))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    _ = ax0.plot(mean)
    _ = ax0.fill_between(range(len(mean)), upstd, downstd,
                    facecolor='k', alpha=0.2)
    _ = ax0.grid()
    _ = ax0.table(np.vstack((mean.round(4),std.round(4))))
    _ = ax0.axes.get_xaxis().set_visible(False)
    _ = f0.savefig(os.path.join(save_dir, 'eval_'+eval_seed+"_"+args.map, '0allseedsEval'))

def v7eval():
    eval_seed = 'seed5'
    list_records = []
    seed = args.seed
    save_dir = args.log_dir
    for ii in range(args.repeat):
        list_records.append(pickle.load(open(os.path.join(save_dir, 'v7_eval_'+eval_seed+"_"+args.map,'all_'+args.nb_test_steps+"_evalseed_%d"%seed+'.pkl'),'rb')))
        seed +=1
    mean = np.mean(list_records,axis=0)
    std = np.std(list_records,axis=0)
    upstd = mean + 0.5*std
    downstd = mean - 0.5*std

    f0, ax0 = plt.subplots(figsize=(20,10))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    _ = ax0.plot(mean)
    _ = ax0.fill_between(range(len(mean)), upstd, downstd,
                    facecolor='k', alpha=0.2)
    _ = ax0.grid()
    _ = ax0.table(np.vstack((mean.round(4),std.round(4))))
    _ = ax0.axes.get_xaxis().set_visible(False)
    _ = f0.savefig(os.path.join(save_dir, 'v7_eval_'+eval_seed+"_"+args.map, '0allseedsEval'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--repeat', help='number of seeds increment', type=int, default=3)
    parser.add_argument('--nb_train_steps', type=int, default=150000)
    parser.add_argument('--nb_epoch_steps', type=int, default=5000)
    parser.add_argument('--nb_test_steps', type=str, default='50')
    parser.add_argument('--batch', type=bool, default=0)
    parser.add_argument('--eval', type=bool, default=0)
    parser.add_argument('--v7', type=bool, default=0)
    parser.add_argument('--map', type=str, default='emptyMed')
    args = parser.parse_args()
    if args.batch:
        batch()
    elif args.eval:
        eval()
    elif args.v7:
        v7eval()
    else:
        contour()