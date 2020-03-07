"""
This code was slightly modified from the baselines/baselines/deepq/build_graph.py in order to use
a different evaluation method. In order to run, simply replace the original code with this code
in the original directory.
"""
"""Clipped Double Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= act (in case of parameter noise) ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)
    reset_ph: bool
        reset the perturbed policy by sampling a new perturbation
    update_param_noise_threshold_ph: float
        the desired threshold for the difference between non-perturbed and perturbed policy
    update_param_noise_scale_ph: bool
        whether or not to update the scale of the noise for the next time it is re-perturbed

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
import math
import baselines0.common.tf_util as U


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string
    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.
    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.compat.v1.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name


def default_param_noise_filter(var):
    if var not in tf.compat.v1.trainable_variables():
        # We never perturb non-trainable vars.
        return False
    if "fully_connected" in var.name:
        # We perturb fully-connected layers.
        return True

    # The remaining layers are likely conv or layer norm layers, which we do not wish to
    # perturb (in the former case because they only extract features, in the latter case because
    # we use them for normalization purposes). If you change your network, you will likely want
    # to re-consider which layers to perturb and which to keep untouched.
    return False


def build_act(make_obs_ph, q_func, num_actions, scope="setdeepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.compat.v1.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.compat.v1.placeholder(tf.float32, (), name="update_eps")

        eps = tf.compat.v1.get_variable("eps", (), initializer=tf.compat.v1.constant_initializer(0))
        # Clipped Double q
        q1_values = q_func.forward(observations_ph.get(), num_actions, scope="q1_func", reuse=reuse)
        q2_values = q_func.forward(observations_ph.get(), num_actions, scope="q2_func", reuse=reuse)
        # Sum over q1 and q2 and find the action with argmax
        deterministic_actions = tf.argmax(input=q1_values+q2_values, axis=1)

        batch_size = tf.shape(input=observations_ph.get())[0]
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.compat.v1.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(pred=stochastic_ph, true_fn=lambda: stochastic_actions, false_fn=lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(pred=update_eps_ph >= 0, true_fn=lambda: update_eps_ph, false_fn=lambda: eps))
        _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        def act(ob, stochastic=True, update_eps=-1):
            return _act(ob, stochastic, update_eps)
        return act

def build_act_greedy(make_obs_ph, q_func, num_actions, scope="setdeepq", reuse=True, eps=0.0):
    """Creates the act function for a simple fixed epsilon greedy
       Added by HJ
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.compat.v1.placeholder(tf.bool, (), name="stochastic")
        # Clipped Double q
        q1_values = q_func.forward(observations_ph.get(), num_actions, scope="q1_func", reuse=reuse)
        q2_values = q_func.forward(observations_ph.get(), num_actions, scope="q2_func", reuse=reuse)
        # Sum over q1 and q2 and find the action with argmax
        deterministic_actions = tf.argmax(input=q1_values+q2_values, axis=1)

        batch_size = tf.shape(input=observations_ph.get())[0]
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.compat.v1.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(pred=stochastic_ph, true_fn=lambda: stochastic_actions, false_fn=lambda: deterministic_actions)
        _act = U.function(inputs=[observations_ph, stochastic_ph],
                         outputs=output_actions)
        def act(ob, stochastic=True):
            return _act(ob, stochastic)
        return act


def build_train(make_obs_ph, q_func, num_actions, optimizer_f,
    grad_norm_clipping=None, gamma=1.0, double_q=False, scope="setdeepq",
    reuse=None, param_noise=False, param_noise_filter_func=None, test_eps=0.05,
    lr_init = 0.001, lr_decay_factor=0.99, lr_growth_factor=1.001, tau=0.05):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.
    lr_init : float
        initial learning rate
    lr_decay_factor : float
        learning rate decay factor. It should be equal to or smaller than 1.0.
    lr_growth_factor : float
        learning rate growth factor. It should be equal to or larger than 1.0.
    tau : float
        parameter for the soft target network update. tau <= 1.0 and 1.0 for
        the hard update.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    # Build action graphs
    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    act_greedy = build_act_greedy(make_obs_ph, q_func, num_actions, scope=scope, reuse=True, eps=test_eps)

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.compat.v1.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.compat.v1.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.compat.v1.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.compat.v1.placeholder(tf.float32, [None], name="weight")
        iteration = tf.compat.v1.placeholder(tf.float32, name="iteration")

        # Cosine learning rate adjustment
        lr = tf.Variable(float(lr_init), trainable=False, dtype = tf.float32, name='lr')
        lr = tf.clip_by_value(0.0005*tf.math.cos(math.pi*iteration/500000)+0.00051, 1e-5, 1e-3)
        optimizer = optimizer_f(learning_rate = lr)

        # q network evaluation
        q1_t = q_func.forward(obs_t_input.get(), num_actions, scope="q1_func", reuse=True)  # reuse q1 parameters from act
        q1_func_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=tf.compat.v1.get_variable_scope().name + "/q1_func")
        q2_t = q_func.forward(obs_t_input.get(), num_actions, scope="q2_func", reuse=True)  # reuse q2 parameters from act
        q2_func_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=tf.compat.v1.get_variable_scope().name + "/q2_func")

        # target q network evalution
        q1_tp1 = q_func.forward(obs_tp1_input.get(), num_actions, scope="target_q1_func", reuse=False)
        target_q1_func_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=tf.compat.v1.get_variable_scope().name + "/target_q1_func")
        q2_tp1 = q_func.forward(obs_tp1_input.get(), num_actions, scope="target_q2_func", reuse=False)
        target_q2_func_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=tf.compat.v1.get_variable_scope().name + "/target_q2_func")

        # q scores for actions which we know were selected in the given state.
        q1_t_selected = tf.reduce_sum(input_tensor=q1_t * tf.one_hot(act_t_ph, num_actions), axis=1)
        q2_t_selected = tf.reduce_sum(input_tensor=q2_t * tf.one_hot(act_t_ph, num_actions), axis=1)

        # Actions selected with current q funcs at state t+1.
        q1_tp1_using_online_net = q_func.forward(obs_tp1_input.get(), num_actions, scope="q1_func", reuse=True)
        q2_tp1_using_online_net = q_func.forward(obs_tp1_input.get(), num_actions, scope="q2_func", reuse=True)
        tp1_best_action_using_online_net = tf.argmax(input=q1_tp1_using_online_net+q2_tp1_using_online_net, axis=1)
        # Using action at t+1 find target value associated with the action
        q1_tp1_selected = tf.reduce_sum(input_tensor=q1_tp1 * tf.one_hot(tp1_best_action_using_online_net, num_actions), axis=1)
        q2_tp1_selected = tf.reduce_sum(input_tensor=q2_tp1 * tf.one_hot(tp1_best_action_using_online_net, num_actions), axis=1)
        # Min of target q values to be used bellman equation
        q_tp1_best = tf.minimum(q1_tp1_selected, q2_tp1_selected)

        # compute RHS of bellman equation
        q_tp1_selected_target = rew_t_ph + gamma * q_tp1_best

        # compute the error (potentially clipped)
        td_error1 = q1_t_selected - tf.stop_gradient(q_tp1_selected_target)
        td_error2 = q2_t_selected - tf.stop_gradient(q_tp1_selected_target)
        errors1 = U.huber_loss(td_error1)
        errors2 = U.huber_loss(td_error2)
        errors = errors1 + errors2
        weighted_error = tf.reduce_mean(input_tensor=importance_weights_ph * errors)

        #Print total number of params
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            # print("var params", variable_parameters)
            total_parameters += variable_parameters
        print("===============================================================")
        print("Total number of trainable params:", total_parameters)
        print("===============================================================")

        # Log for tensorboard
        tf.summary.scalar('q1_values', tf.math.reduce_mean(q1_t))
        tf.summary.scalar('q2_values', tf.math.reduce_mean(q2_t))
        tf.summary.scalar('td_1', tf.math.reduce_mean(td_error1))
        tf.summary.scalar('td_2', tf.math.reduce_mean(td_error2))
        tf.summary.scalar('weighted_loss', weighted_error)
        tf.summary.scalar('lr_schedule', lr)
        tf.summary.scalar('td_MSE_1', tf.math.reduce_mean(tf.math.square(td_error1)))
        tf.summary.scalar('td_MSE_2', tf.math.reduce_mean(tf.math.square(td_error2)))

        # combine variable scopes
        q_func_vars = q1_func_vars+q2_func_vars
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called every step to copy Q network to target Q network
        # target network is updated with polyak averaging
        update_target_expr1 = []
        for var, var_target in zip(sorted(q1_func_vars, key=lambda v: v.name),
                                   sorted(target_q1_func_vars, key=lambda v: v.name)):
            update_target_expr1.append(var_target.assign(tau*var + (1-tau)*var_target))
        update_target_expr1 = tf.group(*update_target_expr1)

        update_target_expr2 = []
        for var, var_target in zip(sorted(q2_func_vars, key=lambda v: v.name),
                                   sorted(target_q2_func_vars, key=lambda v: v.name)):
            update_target_expr2.append(var_target.assign(tau*var + (1-tau)*var_target))
        update_target_expr2 = tf.group(*update_target_expr2)

        merged_summary = tf.compat.v1.summary.merge_all(scope=tf.compat.v1.get_variable_scope().name)
        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph,
                iteration
            ],
            outputs=[td_error1, td_error2, tf.reduce_mean(input_tensor=errors), merged_summary],
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr1, update_target_expr2])

        update_lr = U.function(inputs=[iteration], outputs=[], updates=[lr])

        q_values = U.function(inputs=[obs_t_input], outputs=[q1_t, q2_t])

        return act_f, act_greedy, q_values, train, update_target, update_lr, {'q_values': q_values}
