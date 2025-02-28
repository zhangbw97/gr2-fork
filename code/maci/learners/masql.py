import numpy as np
import tensorflow as tf

from maci.misc import logger
from maci.misc.overrides import overrides

from maci.misc.kernel import adaptive_isotropic_gaussian_kernel
from maci.misc import tf_utils

from .base import MARLAlgorithm

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class MASQL(MARLAlgorithm):
    def __init__(
            self,
            base_kwargs,
            agent_id,
            env,
            pool,
            qf,
            target_qf,
            policy,
            tb_writer,
            plotter=None,
            policy_lr=1E-3,
            qf_lr=1E-3,
            tau=0.01,
            value_n_particles=16,
            td_target_update_interval=1,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            discount=0.99,
            reward_scale=.1,
            joint=False,
            use_saved_qf=False,
            use_saved_policy=False,
            save_full_state=False,
            train_qf=True,
            train_policy=True,
            joint_policy=True,
            opponent_action_range=None,
            opponent_action_range_normalize=True
    ):
        super(MASQL, self).__init__(**base_kwargs)


        self._env = env
        self._pool = pool
        self.qf = qf
        self.target_qf = target_qf
        self._policy = policy
        self.plotter = plotter
        self._tb_writer = tb_writer
        self.agent_id = agent_id

        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._tau = tau
        self._discount = discount
        self._reward_scale = reward_scale
        self.joint_policy = joint_policy
        self.opponent_action_range = opponent_action_range
        self.opponent_action_range_normalize = opponent_action_range_normalize

        self.joint = joint
        self._value_n_particles = value_n_particles
        self._qf_target_update_interval = td_target_update_interval

        self._kernel_fn = kernel_fn
        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio

        self._save_full_state = save_full_state
        self._train_qf = train_qf
        self._train_policy = train_policy

        self._observation_dim = self.env.observation_spaces[self.agent_id].flat_dim
        self._action_dim = self.env.action_spaces[self.agent_id].flat_dim
        # just for two agent case
        self._opponent_action_dim = self.env.action_spaces.opponent_flat_dim(self.agent_id)

        self._create_placeholders()

        self._training_ops = []
        self._target_ops = []

        self._create_td_update()
        self._create_svgd_update()
        self._create_target_ops()

        if use_saved_qf:
            saved_qf_params = qf.get_param_values()
        if use_saved_policy:
            saved_policy_params = policy.get_param_values()

        self._sess = tf_utils.get_default_session()
        self._sess.run(tf.global_variables_initializer())

        if use_saved_qf:
            self.qf.set_param_values(saved_qf_params)
        if use_saved_policy:
            self.policy.set_param_values(saved_policy_params)

    def _create_placeholders(self):
        """Create all necessary placeholders."""

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observations')

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='next_observations')

        self._actions_pl = tf.placeholder(
            tf.float32, shape=[None, self._action_dim],
            name='actions_agent_{}'.format(self.agent_id))

        self._opponent_actions_pl = tf.placeholder(
            tf.float32, shape=[None, self._opponent_action_dim],
            name='opponent_actions_agent_{}'.format(self.agent_id))
        # self._next_actions_ph = tf.placeholder(
        #     tf.float32, shape=[None, self._action_dim + self._opponent_action_dim],
        #     name='next_actions_agent_{}'.format(self._agent_id))

        self._rewards_pl = tf.placeholder(
            tf.float32, shape=[None],
            name='rewards_agent_{}'.format(self.agent_id))

        self._terminals_pl = tf.placeholder(
            tf.float32, shape=[None],
            name='terminals_agent_{}'.format(self.agent_id))

        self._annealing_pl = tf.placeholder(
            tf.float32, shape=[],
            name='annealing_agent_{}'.format(self.agent_id))

    def _create_td_update(self):
        """Create a minimization operation for Q-function update."""

        with tf.variable_scope('target_agent_{}'.format(self.agent_id), reuse=tf.AUTO_REUSE):
            # The value of the next state is approximated with uniform samples.
            # target_actions = tf.random_uniform(
            #     (1, self._value_n_particles, self._action_dim), *self._env.action_range)
            # opponent_target_actions = tf.random_uniform(
            #     (1, self._value_n_particles, self._opponent_action_dim), *self._env.action_range)

            if self.opponent_action_range is None:
                target_actions = tf.random_uniform(
                    (1, self._value_n_particles, self._action_dim), *self._env.action_range)
                opponent_target_actions = tf.random_uniform(
                    (1, self._value_n_particles, self._opponent_action_dim), *self._env.action_range)
            else:
                target_actions = tf.random_uniform(
                    (1, self._value_n_particles, self._action_dim), *self._env.action_range)
                opponent_target_actions = tf.random_uniform(
                    (1, self._value_n_particles, self._opponent_action_dim), *self._env.action_range)
                if self.opponent_action_range_normalize:
                    target_actions = tf.nn.softmax(target_actions, axis=-1)
                    opponent_target_actions = tf.nn.softmax(opponent_target_actions, axis=-1)

            target_actions = tf.concat([target_actions, opponent_target_actions], axis=2)

            q_value_targets = self.target_qf.output_for(
                observations=self._next_observations_ph[:, None, :],
                actions=target_actions)

            assert_shape(q_value_targets, [None, self._value_n_particles])

        joint_action = tf.concat([self._actions_pl, self._opponent_actions_pl], axis=1)
        self._q_values = self.qf.output_for(
            self._observations_ph, joint_action, reuse=True)
        assert_shape(self._q_values, [None])

        # Equation 10:

        next_value = self._annealing_pl * tf.reduce_logsumexp(q_value_targets / self._annealing_pl, axis=1)
        # next_value = tf.reduce_logsumexp(q_value_targets, axis=1)
        assert_shape(next_value, [None])


        # Importance weights add just a constant to the value.
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))
        next_value += (self._action_dim + self._opponent_action_dim) * np.log(2)

        # \hat Q in Equation 11:
        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl + (
            1 - self._terminals_pl) * self._discount * next_value)
        assert_shape(ys, [None])

        # Equation 11:
        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)
        with tf.variable_scope('target_agent_{}'.format(self.agent_id), reuse=tf.AUTO_REUSE):
            if self._train_qf:

                td_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
                    loss=bellman_residual, var_list=self.qf.get_params_internal())
                self._training_ops.append(td_train_op)

        self._bellman_residual = bellman_residual

    def _create_svgd_update(self):
        """Create a minimization operation for policy update (SVGD)."""
        # print('actions')
        actions = self.policy.actions_for(
            observations=self._observations_ph,
            n_action_samples=self._kernel_n_particles,
            reuse=True)
        assert_shape(actions,
                     [None, self._kernel_n_particles, self._action_dim + self._opponent_action_dim])

        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        n_updated_actions = int(
            self._kernel_n_particles * self._kernel_update_ratio)
        n_fixed_actions = self._kernel_n_particles - n_updated_actions

        fixed_actions, updated_actions = tf.split(
            actions, [n_fixed_actions, n_updated_actions], axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)
        assert_shape(fixed_actions, [None, n_fixed_actions, self._action_dim + self._opponent_action_dim])
        assert_shape(updated_actions,
                     [None, n_updated_actions, self._action_dim + self._opponent_action_dim])
        # print('target actions')
        svgd_target_values = self.qf.output_for(
            self._observations_ph[:, None, :], fixed_actions, reuse=True) / self._annealing_pl

        # Target log-density. Q_soft in Equation 13:
        squash_correction = tf.reduce_sum(
            tf.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction

        grad_log_p = tf.gradients(log_p, fixed_actions)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)
        assert_shape(grad_log_p, [None, n_fixed_actions, 1, self._action_dim + self._opponent_action_dim])

        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = tf.expand_dims(kernel_dict["output"], dim=3)
        assert_shape(kappa, [None, n_fixed_actions, n_updated_actions, 1])

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(
            kappa * grad_log_p + kernel_dict["gradient"], reduction_indices=1)
        assert_shape(action_gradients,
                     [None, n_updated_actions, self._action_dim + self._opponent_action_dim])

        # Propagate the gradient through the policy network (Equation 14).
        gradients = tf.gradients(
            updated_actions,
            self.policy.get_params_internal(),
            grad_ys=action_gradients)

        surrogate_loss = tf.reduce_sum([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self.policy.get_params_internal(), gradients)
        ])
        with tf.variable_scope('policy_opt_agent_{}'.format(self.agent_id), reuse=tf.AUTO_REUSE):
            if self._train_policy:
                optimizer = tf.train.AdamOptimizer(self._policy_lr)
                svgd_training_op = optimizer.minimize(
                    loss=-surrogate_loss,
                    var_list=self.policy.get_params_internal())
                self._training_ops.append(svgd_training_op)

    def _create_target_ops(self):
        """Create tensorflow operation for updating the target Q-function."""
        if not self._train_qf:
            return

        source_params = self.qf.get_params_internal()
        target_params = self.target_qf.get_params_internal()

        self._target_ops = [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target_params, source_params)
        ]

    # TODO: do not pass, policy, and pool to `__init__` directly.
    def train(self):
        self._train(self.env, self.policy, self.pool)

    @overrides
    def _init_training(self):
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch, annealing=1.):
        """Run the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(batch, annealing)
        self._sess.run(self._training_ops, feed_dict)
        if iteration % self._qf_target_update_interval == 0 and self._train_qf:
            self._sess.run(self._target_ops)
        self.log_diagnostics(iteration,batch)  
    def _get_feed_dict(self, batch, annealing):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        feeds = {
            self._observations_ph: batch['observations'],
            self._actions_pl: batch['actions'],
            self._opponent_actions_pl: batch['opponent_actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_pl: batch['rewards'],
            self._terminals_pl: batch['terminals'],
            self._annealing_pl: annealing
        }

        return feeds

    @overrides
    def log_diagnostics(self,iteration, batch):
        """Record diagnostic information.
        Records the mean and standard deviation of Q-function and the
        squared Bellman residual of the  s (mean squared Bellman error)
        for a sample batch.
        Also call the `draw` method of the plotter, if plotter is defined.
        """

        feeds = self._get_feed_dict(batch)
        qf, bellman_residual = self._sess.run(
            [self._q_values, self._bellman_residual], feeds)
        self._tb_writer.add_scalars("bellman_residual",{"Agent" + str(self._agent_id): bellman_residual}, iteration)
        logger.record_tabular('qf-avg-agent-{}'.format(self.agent_id), np.mean(qf))
        logger.record_tabular('qf-std-agent-{}'.format(self.agent_id), np.std(qf))
        logger.record_tabular('mean-sq-bellman-error-agent-{}'.format(self.agent_id), bellman_residual)

        self.policy.log_diagnostics(batch)
        # if self.plotter:
        #     self.plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SQL algorithm.
        If `self._save_full_state == True`, returns snapshot including the
        replay buffer. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, and environment instances.
        """

        state = {
            'epoch_agent_{}'.format(self.agent_id): epoch,
            'policy_agent_{}'.format(self.agent_id): self.policy,
            'qf_agent_{}'.format(self.agent_id): self.qf,
            'env_agent_{}'.format(self.agent_id): self.env,
        }

        if self._save_full_state:
            state.update({'replay_buffer_agent_{}'.format(self.agent_id): self.pool})

        return state