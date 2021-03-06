from baselines0.setdeepq import models  # noqa
from baselines0.setdeepq.build_graph import build_act, build_train, build_act_greedy  # noqa
from baselines0.setdeepq.setdeepq import learn, load
from baselines0.setdeepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines0.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)
