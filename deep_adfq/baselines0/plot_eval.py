import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle, tabulate, os, argparse
from baselines0.setdeepq import logger

def contour():

    data1 = np.array([2176.30, 825.09, 710.07, 454.25])
    data2 = np.array([2842.65, 1808.21, 1134.55, 690.46])
    data3 = np.array([2908.38, 2102.65, 1392.37, 1143.18])
    data4 = np.array([3133.18, 2159.98, 1575.53, 1286.16])

    data = np.array([data1,data2,data3,data4]).T

    # setup the figure and axes
    fig = plt.figure(figsize=(8, 7))
    # ax1 = fig.add_subplot(projection='3d')

    x = np.arange(1,5)
    y = np.arange(1,5)

    plt.contourf(x, y, data, 50, cmap="viridis")

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
    eval_seed = 'seed0'
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

    f0, ax0 = plt.subplots()
    plt.subplots_adjust(left=0.2, bottom=0.2)
    _ = ax0.plot(mean)
    _ = ax0.fill_between(range(len(mean)), upstd, downstd,
                    facecolor='k', alpha=0.2)
    _ = ax0.grid()
    _ = ax0.table(np.vstack((mean,std)))
    _ = ax0.axes.get_xaxis().set_visible(False)
    _ = f0.savefig(os.path.join(save_dir, 'eval_'+eval_seed+"_"+args.map, 'allseedsEval'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--repeat', help='number of seeds increment', type=int, default=3)
    parser.add_argument('--nb_train_steps', type=int, default=150000)
    parser.add_argument('--nb_epoch_steps', type=int, default=5000)
    parser.add_argument('--nb_test_steps', type=str, default='100')
    parser.add_argument('--batch', type=bool, default=0)
    parser.add_argument('--eval', type=bool, default=1)
    parser.add_argument('--map', type=str, default='emptyMed')
    args = parser.parse_args()
    if args.batch:
        batch()
    elif args.eval:
        eval()
    else:
        contour()