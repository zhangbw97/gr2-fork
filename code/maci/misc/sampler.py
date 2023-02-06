import numpy as np
import time
#import wandb
from maci.misc import logger
from copy import deepcopy
import tensorflow as tf
import wandb
def rollout(env, policy, path_length, render=False, speedup=None):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    safety_costs = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0
    for t in range(path_length):

        action, agent_info = policy.get_action(observation)
        next_obs, reward, safety_cost, terminal, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        actions[t] = action
        terminals[t] = terminal
        rewards[t] = reward
        safety_costs[t] = safety_cost
        observations[t] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'safety_costs':safety_costs[:t + 1],
        'terminals': terminals[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = [
        rollout(env, policy, path_length)
        for i in range(n_paths)
    ]

    return paths


class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size, render_enabled=True):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None
        self._render_enabled = render_enabled
    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def set_policy(self, policy):
        self.policy = policy

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('pool-size', self.pool.size)


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)
        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        
    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action, _ = self.policy.get_action(self._current_observation)
        next_observation, reward, safety_cost, terminal, info = self.env.step(action)
        self._path_length += 1
        # separate reward and safety cost, but consider return together
        self._path_return += reward - safety_cost
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            safety_cost=safety_cost,
            terminal=terminal,
            next_observation=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)


class MASampler(SimpleSampler):
    def __init__(self, agent_num, joint,**kwargs):
        super(SimpleSampler, self).__init__(**kwargs)
        self.agent_num = agent_num
        self.joint = joint
        self._path_length = 0
        self._path_return = np.array([0.] * self.agent_num, dtype=np.float32)
        self._last_path_return = np.array([0.] * self.agent_num, dtype=np.float32)
        self._max_path_return = np.array([-np.inf] * self.agent_num, dtype=np.float32)
        self._path_collision_num = np.zeros(self.agent_num)
        self._mean_path_collision_num = np.zeros(self.agent_num)
        self._total_collision_num = np.zeros(self.agent_num)
        self._mean_training_run_collision_num = np.zeros(self.agent_num)
        self._n_episodes = 0
        self._total_samples = 0
        self._current_observation_n = None
        self.env = None
        self.agents = None
        #self._val_at_risk = -1.6 # VaR_0.05 for standard normal distribution
    def set_policy(self, policies):
        for agent, policy in zip(self.agents, policies):
            agent.policy = policy

    def batch_ready(self):
        enough_samples = self.agents[0].pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self, i):
        return self.agents[i].pool.random_batch(self._batch_size)

    def initialize(self, env, agents):
        self._current_observation_n = None
        self.env = env
        self.agents = agents
    def record_video(self):
        ...
    def sample(self) -> bool:
        """
        1. get action and step forward
        2. add transition into replay buffer
        """
        if self._current_observation_n is None:
            self._current_observation_n = self.env.reset()
        action_n = []
        self.episode_finish = False
        for agent, current_observation in zip(self.agents, self._current_observation_n):
            action, _ = agent.policy.get_action(current_observation)
            if agent.joint_policy:
                action_n.append(np.array(action)[0:agent._action_dim])
            else:
                action_n.append(np.array(action))
        # for i, (agent, current_observation) in enumerate(zip(self.agents, self._current_observation_n)):
        #     safety_q = agent.safe_joint_qf.eval(np.array(current_observation,dtype=np.float32).reshape(1,-1),np.array(action_n[i]).reshape(1,-1),np.array(action_n[1-i]).reshape(1,-1))
        #     safety_q_n.append(np.array(safety_q))
        next_observation_n, reward_n, safety_cost_n, done_n, info = self.env.step(action_n)
        self._path_length += 1
        self._path_return += np.array(reward_n, dtype=np.float32) #- np.array(safety_cost_n, dtype=np.float32) 
        for i, agent in enumerate(self.agents):
            # assert collision -> cost 1
            if safety_cost_n[i] == 1:
                self._path_collision_num[i] += 1
                self._total_collision_num[i] += 1
        #self._mean_reward = (self._mean_reward * self._total_samples + np.sum(reward_n)) / (self._total_samples + len(reward_n))
        self._total_samples += 1
        if self._render_enabled:
            self.env.render()
        for i, agent in enumerate(self.agents):
            action = deepcopy(action_n[i])
            # noise_var = 1.0
            # noise = np.random.normal(0.0,noise_var)

            # reward_n[i] += noise
            #if safety_q_n[i] > self._safety_bound:
                #reward_n[i] -= 1000
                #continue
            if agent.pool.joint:
                opponent_action = deepcopy(action_n)
                del opponent_action[i]
                opponent_action = np.array(opponent_action).flatten()
                agent.pool.add_sample(observation=self._current_observation_n[i],
                                    action=action,
                                    reward=reward_n[i],
                                    safety_cost=safety_cost_n[i],
                                    terminal=done_n[i],
                                    next_observation=next_observation_n[i],
                                    opponent_action=opponent_action)
            else:
                agent.pool.add_sample(observation=self._current_observation_n[i],
                                    action=action,
                                    reward=reward_n[i],
                                    safety_cost=safety_cost_n[i],
                                    terminal=done_n[i],
                                    next_observation=next_observation_n[i])

        if np.all(done_n) or self._path_length >= self._max_path_length:
            self._current_observation_n = self.env.reset()
            self._max_path_return = np.maximum(self._max_path_return, self._path_return)
            self._mean_path_return = self._path_return / self._path_length
            self._last_path_return = self._path_return
            for i in range(self.agent_num):
                self._mean_path_collision_num[i] = self._path_collision_num[i] / self._path_length
                self._mean_training_run_collision_num[i] = self._total_collision_num[i] / self._total_samples

            self._path_length = 0

            self._path_return = np.array([0.] * self.agent_num, dtype=np.float32)
            self.episode_finish = True
            self._path_collision_num = np.zeros(self.agent_num)
            self._n_episodes += 1
            # if self._n_episodes > 1000 / self._max_path_length + 1:
            #     self.log_diagnostics()
            #     logger.dump_tabular(with_prefix=False)

        else:
            self._current_observation_n = next_observation_n

    def log_diagnostics(self, log_dict):
        
        for i in range(self.agent_num):
            
        #     self._tb_writer.add_scalars("max-path-return_agent_" + str(i),{"Agent" + str(i): self._max_path_return[i]}, self._total_samples)
        #     self._tb_writer.add_scalars("mean-path-return_agent_" + str(i),{"Agent" + str(i): self._mean_path_return[i]}, self._total_samples)
        #     self._tb_writer.add_scalars("mean-path-collision_num" ,{"mean-path-collision_num" : self._mean_path_collision_num}, self._total_samples)
        #     self._tb_writer.add_scalars("mean-training-run-collision_num" ,{"mean-training-run-collision_num" : self._mean_training_run_collision_num}, self._total_samples)
        # logger.record_tabular('max-path-return_agent_{}'.format(i), self._max_path_return[i])
        # logger.record_tabular('mean-path-return_agent_{}'.format(i), self._mean_path_return[i])
        # logger.record_tabular('last-path-return_agent_{}'.format(i), self._last_path_return[i])
        # logger.record_tabular('mean-path-collision_num', self._mean_path_collision_num)
        # logger.record_tabular('mean-training-run-collision_num', self._mean_training_run_collision_num)

            log_dict.update({'max-path-return_agent_{}'.format(i): self._max_path_return[i]})
            log_dict.update({'mean-path-return_agent_{}'.format(i): self._mean_path_return[i]})
            log_dict.update({'mean-path-collision_num_{}'.format(i): self._mean_path_collision_num[i]})
            log_dict.update({'mean-training-run-collision_num_{}'.format(i): self._mean_training_run_collision_num[i]})
        log_dict.update({'episodes': self._n_episodes})
        log_dict.update({'total-samples': self._total_samples})
        return log_dict        
        # logger.record_tabular('episodes', self._n_episodes)
        # logger.record_tabular('total-samples', self._total_samples)
        # log_dict = {}
        # for i in range(self.agent_num):
        #     log_dict.update({'max-path-return_agent_{}'.format(i): self._max_path_return[i]})
        #     log_dict.update({'mean-path-return_agent_{}'.format(i): self._mean_path_return[i]})
        #     log_dict.update({'last-path-return_agent_{}'.format(i): self._last_path_return[i]})
        # log_dict.update({'episodes': self._n_episodes})
        # log_dict.update({'total-samples': self._total_samples})
        #wandb.log(log_dict)
        #wandb.tensorflow.log(tf.summary.merge_all())

class DummySampler(Sampler):
    def __init__(self, batch_size, max_path_length):
        super(DummySampler, self).__init__(
            max_path_length=max_path_length,
            min_pool_size=0,
            batch_size=batch_size)

    def sample(self):
        pass
