import numpy as np
import tensorflow as tf
from maci.utils import compute_cvar, gaussian_likelihood
from maci.misc import logger
from maci.misc.overrides import overrides

from maci.misc.kernel import adaptive_isotropic_gaussian_kernel
from maci.misc import tf_utils

from .base import MARLAlgorithm

EPS = 1e-6
INITIAL_BETA = 0.0

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class MAVBAC(MARLAlgorithm):
    def __init__(
            self,
            base_kwargs,
            agent_id,
            env,
            pool,
            joint_qf,
            safe_joint_qf,
            safe_vf,
            target_joint_qf,
            centralized_v_fn,
            target_centralized_v_fn,
            target_safe_joint_qf,
            target_safe_vf,
            qf,
            policy,
            target_policy,
            conditional_policy,
            tb_writer,
            logging,
            plotter=None,
            policy_lr=1E-3,
            beta_lr=1E-3,
            qf_lr=1E-3,
            tau=0.01,
            value_n_particles=16,
            td_target_update_interval=1,
            beta_update_interval = 1,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            discount=0.99,
            safety_discount=0.99,
            reward_scale=1,
            safety_cost_scale=1,
            fixed_entropy_bonus = None,
            use_saved_qf=False,
            use_saved_policy=False,
            save_full_state=False,
            train_qf=True,
            train_safe_qf=True,
            train_safe_vf=True,
            train_policy=True,
            use_safe_qf=False,
            joint=False,
            joint_policy=False,
            opponent_action_range=None,
            opponent_action_range_normalize=True,
            safety_bound=None,
            risk_level=0.1,
            cost_std=0.05,
            max_episode_len=30,
            damp_scale=0,
            k=0,
            aux=True,
            lagrangian=False
    ):
        super(MAVBAC, self).__init__(**base_kwargs,logging=logging)

        self._env = env
        self._pool = pool
        self.qf = qf
        self.joint_qf = joint_qf
        self.safe_joint_qf = safe_joint_qf
        self.safe_vf = safe_vf
        self.target_joint_qf = target_joint_qf
        self.target_safe_joint_qf=target_safe_joint_qf
        self.target_safe_vf = target_safe_vf
        self.centralized_v_fn = centralized_v_fn
        self.target_centralized_v_fn = target_centralized_v_fn
        self._policy = policy
        self._target_policy = target_policy
        self._conditional_policy = conditional_policy
        self.plotter = plotter
        self._tb_writer = tb_writer
        self.joint = joint
        self.joint_policy = joint_policy
        self.opponent_action_range = opponent_action_range
        self.opponent_action_range_normalize = opponent_action_range_normalize
        self._k = k
        self._aux = aux
        self._lagrangian = lagrangian
        self._logging = logging
        self._agent_id = agent_id

        self._tau = tau
        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._beta_lr = beta_lr
        self._discount = discount
        self._safety_discount = safety_discount
        self._safety_bound = safety_bound
        self._risk_level = risk_level
        self._cost_std = cost_std
        self.max_episode_len=max_episode_len
        self._reward_scale = reward_scale
        self._safety_cost_scale = safety_cost_scale
        self.damp_scale = damp_scale
        self._fixed_entropy_bonus=fixed_entropy_bonus
        self._value_n_particles = value_n_particles
        self._qf_target_update_interval = td_target_update_interval
        self._beta_update_interval = beta_update_interval

        self._kernel_fn = kernel_fn
        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio

        self._save_full_state = save_full_state
        self._train_qf = train_qf
        self._train_safe_qf = train_safe_qf
        self._train_safe_vf = train_safe_vf
        self._train_policy = train_policy
        self._use_safe_qf = use_safe_qf 
        self._use_safe_vf = not use_safe_qf
        self._observation_dim = self.env.observation_spaces[self._agent_id].flat_dim
        self._action_dim = self.env.action_spaces[self._agent_id].flat_dim
        # just for two agent case
        self._opponent_action_dim = self.env.action_spaces.opponent_flat_dim(self._agent_id)

        self._create_placeholders()

        self._training_ops = []
        self._beta_training_ops = []
        self._target_ops = []
        # if self._fixed_entropy_bonus is None:
        #     with tf.variable_scope('entreg'):
        #         soft_alpha = tf.get_variable('soft_alpha',
        #                                  initializer=0.0,
        #                                  trainable=True,
        #                                  dtype=tf.float32)
        #         alpha = tf.nn.softplus(soft_alpha)
        # else:
        #     alpha = tf.constant(self._fixed_entropy_bonus)
            # Cost penalty
        if self._lagrangian is True:
            with tf.variable_scope('costpen_agent', reuse=tf.AUTO_REUSE):
                self.soft_beta = tf.get_variable('soft_beta',
                                            initializer=INITIAL_BETA,
                                            trainable=True,
                                            dtype=tf.float32,
                                            constraint=lambda z: tf.clip_by_value(z, clip_value_min = 0.0,clip_value_max=20.0))
            self.beta = tf.nn.softplus(self.soft_beta)

        else:
            self.beta = 0.0  # costs do not contribute to policy optimization
            print('Not using costs')
        
        self._create_q_update()
        self._create_q_cost_update()
        self._create_v_cost_update()
        self._create_conditional_policy_svgd_update()
        self._create_p_update()
        self._create_target_ops()
        self._create_coefficient_update()

        if use_saved_qf:
            saved_qf_params = qf.get_param_values()
        if use_saved_policy:
            saved_policy_params = policy.get_param_values()

        self._sess = tf_utils.get_default_session()

        # # set seed
        seed = 1
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self._sess.run(tf.global_variables_initializer())

        if use_saved_qf:
            self.qf.set_param_values(saved_qf_params)
        if use_saved_policy:
            self.policy.set_param_values(saved_policy_params)

    def _create_placeholders(self):
        """Create all necessary placeholders."""
        AGENT_NUM = 2
        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observations_agent_{}'.format(self._agent_id))

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None,self._observation_dim],
            name='next_observations_agent_{}'.format(self._agent_id))

        self._centralized_states_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim*AGENT_NUM],
            name='centralized_states')

        self._next_centralized_states_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim*AGENT_NUM],
            name='next_centralized_states')         

        self._actions_pl = tf.placeholder(
            tf.float32, shape=[None, self._action_dim],
            name='actions_agent_{}'.format(self._agent_id))
        self._next_actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim],
            name='next_actions_agent_{}'.format(self._agent_id))
        self._opponent_actions_pl = tf.placeholder(
                tf.float32, shape=[None, self._opponent_action_dim],
                name='opponent_actions_agent_{}'.format(self._agent_id))
        self._opponent_next_actions_ph = tf.placeholder(
                tf.float32, shape=[None, self._opponent_action_dim],
                name='opponent_next_actions_agent_{}'.format(self._agent_id))
        self._rewards_pl = tf.placeholder(
            tf.float32, shape=[None],
            name='rewards_agent_{}'.format(self._agent_id))
        self._safety_costs_pl = tf.placeholder(
            tf.float32, shape=[None],
            name='safety_costs_agent_{}'.format(self._agent_id))
        # self._episode_safety_costs_pl = tf.placeholder(
        #     tf.float32, shape=[],
        #     name='episode_safety_costs_agent_{}'.format(self._agent_id))
        self._terminals_pl = tf.placeholder(
            tf.float32, shape=[None],
            name='terminals_agent_{}'.format(self._agent_id))
        self._annealing_pl = tf.placeholder(
            tf.float32, shape=[],
            name='annealing_agent_{}'.format(self._agent_id))
        # self._noise_pl = noise_pl

    def _create_q_update(self):
        """Create a minimization operation for Q-function update."""
        with tf.variable_scope('target_joint_q_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self.opponent_action_range is None:
                opponent_target_actions = tf.random_uniform(
                    (1, self._value_n_particles, self._opponent_action_dim), *self._env.action_range)
            else:
                opponent_target_actions = tf.random_uniform(
                    (1, self._value_n_particles, self._opponent_action_dim), *self._env.action_range)
                if self.opponent_action_range_normalize:
                    opponent_target_actions = tf.nn.softmax(opponent_target_actions, axis=1)

            # next step Q（s',a'^i,a'^-i)
            q_value_targets = self.target_joint_qf.output_for(
                observations=self._next_observations_ph[:, None, :],
                actions=self._next_actions_ph[:, None, :],
                opponent_actions=opponent_target_actions)
            assert_shape(q_value_targets, [None, self._value_n_particles])
        self.q_value_targets=q_value_targets
        # Q(s,a^i,a^-i)
        self._q_values = self.joint_qf.output_for(
            self._observations_ph, self._actions_pl, self._opponent_actions_pl, reuse=True)
        assert_shape(self._q_values, [None])

        # Q_soft(s,a^i)=alpha*logsumexp(Q_soft(s,a^i,a^-i)/alpha); alpha:annealing; sum over opponent actions
        # next_value: Q_soft(s,a^i)
        next_value = self._annealing_pl * tf.reduce_logsumexp(q_value_targets / self._annealing_pl, axis=1)

        assert_shape(next_value, [None])
        # represent uniform opponent policy
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))  
        # assume each action dimension has pi(a^i|s) = 0.5 possibility of selecting target action
        # so uniform distribution between up and down, left and right        
        # next_value += (self._opponent_action_dim) * np.log(2) 

        # target Q function 
        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl + (
            1 - self._terminals_pl) * self._discount * next_value)
        assert_shape(ys, [None])

        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)


        with tf.variable_scope('target_joint_qf_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_qf:
                td_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
                    loss=bellman_residual, var_list=self.joint_qf.get_params_internal())
                self._training_ops.append(td_train_op)

        self._bellman_residual = bellman_residual

        
        # Q(s,a^i) individual Q fn?
        self._ind_q_values = self.qf.output_for(self._observations_ph, self._actions_pl, reuse=True)
        assert_shape(self._ind_q_values, [None])
        ind_bellman_residual = 0.5 * tf.reduce_mean((ys - self._ind_q_values) ** 2)
        with tf.variable_scope('target_qf_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_qf:
                ind_q_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
                    loss=ind_bellman_residual, var_list=self.qf.get_params_internal())
                self._training_ops.append(ind_q_train_op)

        self._ind_bellman_residual = ind_bellman_residual   
    
    def _create_q_cost_update(self):

        with tf.variable_scope('target_safe_joint_q_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self.opponent_action_range is None:
                opponent_target_actions = tf.random_uniform(
                    (1, self._value_n_particles, self._opponent_action_dim), *self._env.action_range)
            else:
                opponent_target_actions = tf.random_uniform(
                    (1, self._value_n_particles, self._opponent_action_dim), *self._env.action_range)
                if self.opponent_action_range_normalize:
                    opponent_target_actions = tf.nn.softmax(opponent_target_actions, axis=1)  
       
            safe_q_value_targets = self.target_safe_joint_qf.output_for(
                observations=self._next_observations_ph[:, None, :],
                actions=self._next_actions_ph[:, None, :],
                opponent_actions=opponent_target_actions)
            assert_shape(safe_q_value_targets, [None, self._value_n_particles])

        self._safe_q_values = self.safe_joint_qf.output_for(
            self._observations_ph, self._actions_pl, self._opponent_actions_pl, reuse=True)
        assert_shape(self._safe_q_values, [None]) 
        
        #safety Q update is different from Q, because safety Q is not a soft Q
        # next_safe_value =  tf.reduce_mean(safe_q_value_targets, axis=1)
        # assert_shape(next_safe_value, [None])
        # represent uniform opponent policy
        next_safe_value = self._annealing_pl * tf.reduce_logsumexp(safe_q_value_targets / self._annealing_pl, axis=1)

        assert_shape(next_safe_value, [None])
        next_safe_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))        
        #next_safe_value += (self._opponent_action_dim) * np.log(2) 

        safety_ys = tf.stop_gradient(self._safety_cost_scale * self._safety_costs_pl + (
            1 - self._terminals_pl) *self._safety_discount * next_safe_value)
        assert_shape(safety_ys, [None])
       
        safety_q_bellman_residual = 0.5 * tf.reduce_mean((safety_ys - self._safe_q_values)**2)


        with tf.variable_scope('target_safe_joint_qf_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_safe_qf:
                safe_q_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
                    loss=safety_q_bellman_residual, var_list=self.safe_joint_qf.get_params_internal())
                self._training_ops.append(safe_q_train_op)

        self._safety_q_bellman_residual = safety_q_bellman_residual

    def _create_v_cost_update(self):
        # with tf.variable_scope('target_safe_v_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):

        safe_v_value_targets = self.target_centralized_v_fn.output_for(
            centralized_states=self._next_centralized_states_ph)
        assert_shape(safe_v_value_targets, [None])

        self._safe_v_values = self.centralized_v_fn.output_for(
            self._centralized_states_ph, reuse=True)
        assert_shape(self._safe_v_values, [None]) 
        
        safety_ys = tf.stop_gradient(self._safety_cost_scale * self._safety_costs_pl + (
            1 - self._terminals_pl) *self._safety_discount * safe_v_value_targets)
        assert_shape(safety_ys, [None])
       
        safety_v_bellman_residual = 0.5 * tf.reduce_mean((safety_ys - self._safe_v_values)**2)


        with tf.variable_scope('target_safe_vf_opt', reuse=tf.AUTO_REUSE):
            if self._train_safe_vf:
                safe_v_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
                    loss=safety_v_bellman_residual, var_list=self.centralized_v_fn.get_params_internal())
                self._training_ops.append(safe_v_train_op)

        self._safety_v_bellman_residual = safety_v_bellman_residual
    def _create_coefficient_update(self):

        #self.cvar_safe_q = compute_cvar(self._safe_q_values,risk_level=self._risk_level,std=self._cost_std)
        # self._safety_bound * 0.8677
        safety_bound = self._safety_bound #* (1 - self._discount ** self.max_episode_len) / (1 - self._discount)/ self.max_episode_len
        #violation = self._episode_safety_costs_pl - self._safety_bound 
        if self._use_safe_qf:
            safety_values = self._safe_q_values
        elif self._use_safe_vf:
            safety_values = self._safe_v_values
        violation = safety_values - safety_bound 
        beta_loss = self.beta *tf.reduce_mean(-violation)  
        self._violation = violation
        with tf.variable_scope('beta_opt_agent', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self._beta_lr)
            beta_training_op = optimizer.minimize(
                loss=beta_loss,
                var_list=get_vars('costpen_agent'))
            self._beta_training_ops.append(beta_training_op)

    def _create_p_update(self):
        """Create a minimization operation for policy update """
        # with tf.variable_scope('target_p_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
        #     self_target_actions = self._target_policy.actions_for(
        #         observations=self._observations_ph,
        #         reuse=tf.AUTO_REUSE)
        if self._k <= 1:
            self_actions = self.policy.actions_for(
                observations=self._observations_ph,
                reuse=tf.AUTO_REUSE)
            assert_shape(self_actions, [None, self._action_dim])
        else:
            self_actions, all_actions = self.policy.actions_for(
                observations=self._observations_ph,
                reuse=tf.AUTO_REUSE, all_action=True)
            assert_shape(self_actions, [None, self._action_dim])

        # opponent_target_actions = tf.random_uniform(
        #     (1, self._value_n_particles, self._opponent_action_dim), -1, 1)

        opponent_target_actions = self._conditional_policy.actions_for(
            observations=self._observations_ph,
            actions=self._actions_pl,
            n_action_samples=self._value_n_particles,
            reuse=True)

        assert_shape(opponent_target_actions,
                     [None, self._value_n_particles, self._opponent_action_dim])
        # why next observation but current action?
        q_targets = self.joint_qf.output_for(
            observations=self._next_observations_ph[:, None, :],
            actions=self_actions[:, None, :],
            opponent_actions=opponent_target_actions)

        q_targets = self._annealing_pl * tf.reduce_logsumexp(q_targets / self._annealing_pl, axis=1)
        

        assert_shape(q_targets, [None])

        # Importance weights add just a constant to the value.
        q_targets -= tf.log(tf.cast(self._value_n_particles, tf.float32))
        q_targets += (self._opponent_action_dim) * np.log(2)
        pg_loss = -tf.reduce_mean(q_targets) # minimize -Q == maximize Q
        # auxiliary loss
        if self._aux:
            # only works for k = 2, 3
            if self._k > 1:
                q_k= self.joint_qf.output_for(
                    observations=self._next_observations_ph,
                    actions=all_actions[-1],
                    opponent_actions=all_actions[-2], reuse=tf.AUTO_REUSE)
                q_k_2 = self.joint_qf.output_for(
                    observations=self._next_observations_ph,
                    actions=all_actions[-3],
                    opponent_actions=all_actions[-2], reuse=tf.AUTO_REUSE)
                pg_loss += tf.reduce_mean(q_k_2-q_k)
            if self._k > 3:
                print(self._k , 'self._k ', 'self._k ')
                q_k = self.joint_qf.output_for(
                    observations=self._next_observations_ph,
                    actions=all_actions[-3],
                    opponent_actions=all_actions[-4], reuse=tf.AUTO_REUSE)
                q_k_2 = self.joint_qf.output_for(
                    observations=self._next_observations_ph,
                    actions=all_actions[-5],
                    opponent_actions=all_actions[-4], reuse=tf.AUTO_REUSE)
                pg_loss += tf.reduce_mean(q_k_2 - q_k)
        if self._lagrangian:
            #TODO check the usage of next observation and current actions
            if self._use_safe_qf:
                safe_target = self.safe_joint_qf.output_for(
                    self._next_observations_ph[:, None, :],
                    self_actions[:, None, :], 
                    opponent_target_actions, reuse=True)
                assert_shape(safe_target, [None,self._value_n_particles])
            elif self._use_safe_vf:
                safe_target = self.centralized_v_fn.output_for(
                    self._centralized_states_ph[:, :],reuse=True)
                assert_shape(safe_target, [None])
            # cvar_safe_q = compute_cvar(self._safe_q_values,risk_level=self._risk_level,std=self._cost_std)
            # #maximize L = f - lambda * g == minimize -L = -f + lambda * g
            # constraint_term = self._lambda * tf.reduce_mean(cvar_safe_q - self._safety_bound)
            safety_bound = self._safety_bound #* (1 - self._discount ** self.max_episode_len) / (1 - self._discount)/ self.max_episode_len
            damp = self.damp_scale * tf.reduce_mean(safety_bound - safe_target)
            constraint_term = ( self.beta - damp) * tf.reduce_mean(safe_target)
            pg_loss = pg_loss + constraint_term
            self.pg_loss = pg_loss
            self.constraint_term = constraint_term
        # todo add level k Q loss:
        with tf.variable_scope('policy_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_policy:
                optimizer = tf.train.AdamOptimizer(self._policy_lr)
                pg_training_op = optimizer.minimize(
                    loss=pg_loss,
                    var_list=self.policy.get_params_internal())
                self._training_ops.append(pg_training_op)

    def _create_conditional_policy_svgd_update(self):
        """Create a minimization operation for policy update (SVGD)."""
        # print('actions')
        actions = self._conditional_policy.actions_for(
            observations=self._observations_ph,
            actions=self._actions_pl,
            n_action_samples=self._kernel_n_particles,
            reuse=True)
        print(actions.shape.as_list(), [None, self._kernel_n_particles, self._opponent_action_dim])
        assert_shape(actions,
                     [None, self._kernel_n_particles, self._opponent_action_dim])


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
        assert_shape(fixed_actions, [None, n_fixed_actions, self._opponent_action_dim])
        assert_shape(updated_actions,
                     [None, n_updated_actions, self._opponent_action_dim])
        # print('target actions')
        svgd_target_values = self.joint_qf.output_for(
            self._observations_ph[:, None, :], self._actions_pl[:, None, :], fixed_actions, reuse=True)

        assert_shape(svgd_target_values, [None, n_fixed_actions])


        baseline_ind_q = self.qf.output_for(self._observations_ph, self._actions_pl, reuse=True)
        assert_shape(baseline_ind_q, [None])

        baseline_ind_q = tf.tile(tf.reshape(baseline_ind_q, [-1, 1]), [1, n_fixed_actions])
        # baseline_ind_q = tf.reshape(baseline_ind_q, [-1, 1])
        assert_shape(baseline_ind_q, [None, n_fixed_actions])
        # target_df_values = self.
        # Target log-density. Q_soft in Equation 13:
        svgd_target_values = (svgd_target_values - baseline_ind_q) / self._annealing_pl


        squash_correction = tf.reduce_sum(
            tf.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction

        grad_log_p = tf.gradients(log_p, fixed_actions)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)
        assert_shape(grad_log_p, [None, n_fixed_actions, 1, self._opponent_action_dim])

        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = tf.expand_dims(kernel_dict["output"], dim=3)
        assert_shape(kappa, [None, n_fixed_actions, n_updated_actions, 1])

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(
            kappa * grad_log_p + kernel_dict["gradient"], reduction_indices=1)
        assert_shape(action_gradients,
                     [None, n_updated_actions, self._opponent_action_dim])

        # Propagate the gradient through the policy network (Equation 14).
        gradients = tf.gradients(
            updated_actions,
            self._conditional_policy.get_params_internal(),
            grad_ys=action_gradients)

        surrogate_loss = tf.reduce_sum([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self._conditional_policy.get_params_internal(), gradients)
        ])
        with tf.variable_scope('conditional_policy_opt_agent_{}'.format(self._agent_id), reuse=tf.AUTO_REUSE):
            if self._train_policy:
                optimizer = tf.train.AdamOptimizer(self._policy_lr)
                svgd_training_op = optimizer.minimize(
                    loss=-surrogate_loss,
                    var_list=self._conditional_policy.get_params_internal())
                self._training_ops.append(svgd_training_op)

    def _create_target_ops(self):
        """Create tensorflow operation for updating the target Q-function."""
        # 1. update target network less frequently
        # 2. use tau<1 
        # aim: stablize training process
        if not self._train_qf:
            return

        source_q_params = self.joint_qf.get_params_internal()
        target_q_params = self.target_joint_qf.get_params_internal()
        source_sq_params = self.safe_joint_qf.get_params_internal()
        target_sq_params = self.target_safe_joint_qf.get_params_internal()
        # source_sv_params = self.safe_vf.get_params_internal()
        # target_sv_params = self.target_safe_vf.get_params_internal()
        source_sv_params = self.centralized_v_fn.get_params_internal()
        target_sv_params = self.target_centralized_v_fn.get_params_internal()
        source_p_params = self._policy.get_params_internal()
        target_p_params = self._target_policy.get_params_internal()
        target_p_op = [tf.assign(target, (1 - self._tau) * target + self._tau * source)
                               for target, source in zip(target_p_params, source_p_params)]
                           
        target_q_op = [tf.assign(target, (1 - self._tau) * target + self._tau * source)
                               for target, source in zip(target_q_params, source_q_params)]
        target_sq_op = [tf.assign(target, (1 - self._tau) * target + self._tau * source)
                               for target, source in zip(target_sq_params, source_sq_params)]
        target_sv_op =  [tf.assign(target, (1 - self._tau) * target + self._tau * source)
                               for target, source in zip(target_sv_params, source_sv_params)]                
        self._target_ops = target_q_op + target_sq_op + target_p_op + target_sv_op
        # [
        #                        tf.assign(target, (1 - self._tau) * target + self._tau * source)
        #                        for target, source in zip(target_q_params, source_q_params)
        #                    ] + [
        #                        tf.assign(target, (1 - self._tau) * target + self._tau * source)
        #                        for target, source in zip(target_p_params, source_p_params)
        #                    ]+ [
        #                        tf.assign(target, (1 - self._tau) * target + self._tau * source)
        #                        for target, source in zip(target_sq_params, source_sq_params)
        #                    ]

    # TODO: do not pass, policy, and pool to `__init__` directly.
    def train(self):
        self._train(self.env, self.policy, self.pool)

    @overrides
    def _init_training(self):
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch, annealing=1., **kwargs):
        """Run the operations for updating training and target ops."""
        for k,v in kwargs.items():
            if k=='safety_cost_episode':
                safety_cost_episode = v
        feed_dict = self._get_feed_dict(batch, annealing)
        self._sess.run(self._training_ops, feed_dict)
        # mark qf_target_update_interval, async update of actor and critic
        if iteration % self._qf_target_update_interval == 0 and self._train_qf:
            self._sess.run(self._target_ops)
        if iteration % self._beta_update_interval == 0 and self._lagrangian:
            self._sess.run(self._beta_training_ops, feed_dict)
        self.log_diagnostics(iteration,batch,annealing)  
    def _get_feed_dict(self, batch, annealing):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        feeds = {
            self._observations_ph: batch['observations'],
            self._centralized_states_ph: batch['centralized_states'],
            self._actions_pl: batch['actions'],
            self._opponent_actions_pl: batch['opponent_actions'],
            self._next_actions_ph: batch['next_actions'],
            #self._opponent_next_actions_ph: batch['opponent_next_actions'],
            self._next_observations_ph: batch['next_observations'],
            self._next_centralized_states_ph : batch['next_centralized_states'],
            self._rewards_pl: batch['rewards'],
            self._safety_costs_pl: batch['safety_costs'],
            self._terminals_pl: batch['terminals'],
            self._annealing_pl: annealing,
            #self._episode_safety_costs_pl:safety_cost_episode
        }

        return feeds

    @overrides
    def log_diagnostics(self,iteration, batch, annealing,**kwargs):
        """Record diagnostic information.
        Records the mean and standard deviation of Q-function and the
        squared Bellman residual of the  s (mean squared Bellman error)
        for a sample batch.
        Also call the `draw` method of the plotter, if plotter is defined.
        """
        for k,v in kwargs.items():
            if k=='safety_cost_episode':
                safety_cost_episode = v
        feeds = self._get_feed_dict(batch,annealing)
        q_values, bellman_residual,safe_q_values,safety_q_bellman_residual,beta,violation= self._sess.run(
            [self._q_values, self._bellman_residual, self._safe_q_values, self._safety_q_bellman_residual,self.beta,self._violation], feeds)
        safety_v_bellman_residual, safe_v_values, pg_loss, constraint_term= self._sess.run(
            [self._safety_v_bellman_residual,self._safe_v_values,self.pg_loss,self.constraint_term], feeds)
        if self._logging and self._tb_writer is not None:
        
            self._tb_writer.add_scalars("qf-avg-agent",{"Agent" + str(self._agent_id): np.mean(q_values)}, iteration)
            self._tb_writer.add_scalars("bellman_residual",{"Agent" + str(self._agent_id): bellman_residual}, iteration)
            self._tb_writer.add_scalars("safe-qf-avg",{ "Agent" + str(self._agent_id): np.mean(safe_q_values)}, iteration)
            self._tb_writer.add_scalars("safety_q_bellman_residual",{"Agent" + str(self._agent_id): safety_q_bellman_residual}, iteration)
            # self._tb_writer.add_scalars("safe-vf-avg",{ "Agent" + str(self._agent_id): np.mean(safe_v_values)}, iteration)
            # self._tb_writer.add_scalars("safety_v_bellman_residual",{"Agent" + str(self._agent_id): safety_v_bellman_residual}, iteration)
            self._tb_writer.add_scalars("safe-vf-avg",{ "centralized-safe-vf-avg ": np.mean(safe_v_values)}, iteration)
            self._tb_writer.add_scalars("safety_v_bellman_residual",{"centralized-safety_v_bellman_residual": safety_v_bellman_residual}, iteration)
            self._tb_writer.add_scalars("beta", { "Agent" + str(self._agent_id): beta}, iteration)
            self._tb_writer.add_scalars("violation/beta_gradient", { "Agent" + str(self._agent_id): np.mean(violation)}, iteration)
            #self._tb_writer.add_scalars("cvar-safe-qf-avg", {"cvar-safe-qf-avg": np.mean(cvar_safe_q)}, iteration)
        logger.record_tabular('qf-avg-agent-{}'.format(self._agent_id), np.mean(q_values))
        logger.record_tabular('qf-std-agent-{}'.format(self._agent_id), np.std(q_values))
        logger.record_tabular('safe-qf-avg-agent-{}'.format(self._agent_id), np.mean(safe_q_values))
        logger.record_tabular('safe-qf-std-agent-{}'.format(self._agent_id), np.std(safe_q_values))
        logger.record_tabular('beta-agent-{}'.format(self._agent_id),beta)
        logger.record_tabular('violation-agent-{}'.format(self._agent_id),np.mean(violation))
        logger.record_tabular('mean-sq-bellman-error-agent-{}'.format(self._agent_id), bellman_residual)
        logger.record_tabular('mean-sq-safety-bellman-error-agent-{}'.format(self._agent_id), safety_q_bellman_residual)
        logger.record_tabular('iteration',iteration)
        self.policy.log_diagnostics(batch)
        if self.plotter:
            self.plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SQL algorithm.
        If `self._save_full_state == True`, returns snapshot including the
        replay buffer. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, and environment instances.
        """

        state = {
            'epoch_agent_{}'.format(self._agent_id): epoch,
            'policy_agent_{}'.format(self._agent_id): self.policy,
            'qf_agent_{}'.format(self._agent_id): self.qf,
            'env_agent_{}'.format(self._agent_id): self.env,
        }

        if self._save_full_state:
            state.update({'replay_buffer_agent_{}'.format(self._agent_id): self.pool})

        return state