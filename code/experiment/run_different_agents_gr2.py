import numpy as np
import argparse
from maci.learners import MAVBAC, MASQL
from maci.misc.sampler import MASampler
from maci.environments import PBeautyGame, MatrixGame
from maci.environments import make_particle_env
from maci.misc import logger
import random
import gtimer as gt
import datetime
from copy import deepcopy
from maci.get_agents import ddpg_agent, masql_agent, pr2ac_agent

import maci.misc.tf_utils as U
import os
from tensorboardX import SummaryWriter
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)


def get_particle_game(particle_game_name, arglist):
    env = make_particle_env(game_name=particle_game_name)
    print(env.action_space, env.observation_space)
    agent_num = env.n
    adv_agent_num = 0
    if particle_game_name == 'simple_push' or particle_game_name == 'simple_adversary':
        adv_agent_num = 1
    elif particle_game_name == 'simple_tag':
        adv_agent_num = 3
    model_names_setting = arglist.model_names_setting.split('_')
    model_name = '_'.join(model_names_setting)
    model_names = [model_names_setting[1]] * adv_agent_num + [model_names_setting[0]] * (agent_num - adv_agent_num)
    return env, agent_num, model_name, model_names

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # ['particle-simple_spread', 'particle-simple_adversary', 'particle-simple_tag', 'particle-simple_push']
    # matrix-prison , matrix-prison
    # pbeauty
    parser.add_argument('-g', "--game_name", type=str, default="particle-simple_spread", help="name of the game")
    parser.add_argument('-p', "--p", type=float, default=1.1, help="p")
    parser.add_argument('-mu', "--mu", type=float, default=1.5, help="mu")
    parser.add_argument('-seed', "--seed", type=float, default=1, help="seed")
    parser.add_argument('-r', "--reward_type", type=str, default="abs", help="reward type")
    parser.add_argument('-mp', "--max_path_length", type=int, default=30, help="reward type")
    parser.add_argument('-ms', "--max_steps", type=int, default=100000, help="reward type")
    parser.add_argument('-me', "--memory", type=int, default=0, help="reward type")
    parser.add_argument('-n', "--n", type=int, default=2, help="name of the game")
    parser.add_argument('-bs', "--batch_size", type=int, default=64, help="name of the game")
    parser.add_argument('-hm', "--hidden_size", type=int, default=100, help="name of the game")
    parser.add_argument('-re', "--repeat", type=bool, default=False, help="name of the game")
    parser.add_argument('-a', "--aux", type=bool, default=True, help="name of the game")
    parser.add_argument('-m', "--model_names_setting", type=str, default='PR2AC4_PR2AC4', help="models setting agent vs adv")
    parser.add_argument("--logging-enabled", action="store_true", help="enable logging")
    return parser.parse_args()

def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # # prng was removed in latest gym version
    # if hasattr(gym.spaces, 'prng'):
    #     gym.spaces.prng.seed(seed)
def main(arglist):

    set_global_seeds(arglist.seed)
    game_name = arglist.game_name
    # 'abs', 'one'
    reward_type = arglist.reward_type
    p = arglist.p
    agent_num = arglist.n
    u_range = 1.
    k = 0
    print(arglist.aux, 'arglist.aux')
    model_names_setting = arglist.model_names_setting.split('_')
    model_names = [model_names_setting[0]] + [model_names_setting[1]] * (agent_num - 1)
    model_name = '_'.join(model_names)
    path_prefix = game_name
    if game_name == 'pbeauty':
        env = PBeautyGame(agent_num=agent_num, reward_type=reward_type, p=p)
        path_prefix  = game_name + '-' + reward_type + '-' + str(p)
    elif 'matrix' in game_name:
        matrix_game_name = game_name.split('-')[-1]
        repeated = arglist.repeat
        max_step = arglist.max_path_length
        memory = arglist.memory
        env = MatrixGame(game=matrix_game_name, agent_num=agent_num,
                         action_num=2, repeated=repeated,
                         max_step=max_step, memory=memory,
                         discrete_action=False, tuple_obs=False)
        path_prefix = '{}-{}-{}-{}'.format(game_name, repeated, max_step, memory)

    elif 'particle' in game_name:
        particle_game_name = game_name.split('-')[-1]
        env, agent_num, model_name, model_names = get_particle_game(particle_game_name, arglist)

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
    if 'CG' in model_name:
        model_name = model_name + '-{}'.format(arglist.mu)
    if not arglist.aux:
        model_name = model_name + '-{}'.format(arglist.aux)
    
    suffix = 'lagrangian/{}/{}/{}/{}'.format(path_prefix, agent_num, model_name, timestamp)
    tb_suffix = 'lagrangian/{}_{}_{}_{}'.format(path_prefix, agent_num, model_name, timestamp)
    print(suffix)
    logger.add_tabular_output('./log/no_reward_noise/stay_in_sight/{}.csv'.format(suffix))
    snapshot_dir = './snapshot/{}'.format(suffix)
    policy_dir = './policy/{}'.format(suffix)
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)
    logger.set_snapshot_dir(snapshot_dir)
    tb_writer = None
    if arglist.logging_enabled:
        tb_writer = SummaryWriter('./log/no_reward_noise/stay_in_sight/tb_{}'.format(tb_suffix)) # NOTE added

    agents = []
    M = arglist.hidden_size
    batch_size = arglist.batch_size
    sampler = MASampler(tb_writer=tb_writer ,agent_num=agent_num, joint=True, max_path_length=arglist.max_path_length, min_pool_size=100, batch_size=batch_size, logging = arglist.logging_enabled)

    base_kwargs = {
        'sampler': sampler,
        'epoch_length': arglist.max_path_length,
        'n_epochs': arglist.max_steps,
        'n_train_repeat': 1,
        'eval_render': True,
        'eval_n_episodes': 10
    }
    # centralized_v_fn = CentralizedNNVFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='centralized_vf',agent_num=len(agents))
    # target_centralized_v_fn = CentralizedNNVFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_centralized_vf',agent_num=len(agents))
    with U.single_threaded_session():
        for i, model_name in enumerate(model_names):
            if 'PR2AC' in model_name:
                k = int(model_name[-1])
                g = False
                mu = arglist.mu
                if 'G' in model_name:
                    g = True
                agent = pr2ac_agent(tb_writer,model_name, i, env, M, u_range, base_kwargs, k=k, g=g, mu=mu, game_name=game_name, aux=arglist.aux,logging=arglist.logging_enabled,lagrangian=True)
            elif model_name == 'MASQL':
                agent = masql_agent(tb_writer,model_name, i, env, M, u_range, base_kwargs, game_name=game_name)
            else:
                if model_name == 'DDPG':
                    joint = False
                    opponent_modelling = False
                elif model_name == 'MADDPG':
                    joint = True
                    opponent_modelling = False
                elif model_name == 'DDPG-OM' or model_name == 'DDPG-ToM':
                    joint = True
                    opponent_modelling = True
                agent = ddpg_agent(tb_writer ,joint, opponent_modelling, model_names, i, env, M, u_range, base_kwargs, game_name=game_name)

            agents.append(agent)

        sampler.initialize(env, agents)
        for agent in agents:
            agent._init_training()
        gt.rename_root('MARLAlgorithm')
        gt.reset()
        gt.set_def_unique(False)
        initial_exploration_done = False
        # noise = .1
        noise = 0.1
        alpha = .5


        for agent in agents:
            try:
                agent.policy.set_noise_level(noise)
            except:
                pass

        for epoch in gt.timed_for(range(base_kwargs['n_epochs'] + 1)):
            logger.push_prefix('Epoch #%d | ' % epoch)
            if epoch % 1 == 0:
                print(suffix)
            for t in range(base_kwargs['epoch_length']):
                # TODO.code consolidation: Add control interval to sampler
                if not initial_exploration_done:
                    if epoch >= 1000/arglist.max_path_length: # NOTE: change?
                        initial_exploration_done = True
                sampler.sample()
                # print('Sampling')
                if not initial_exploration_done:
                    continue
                gt.stamp('sample')
                # print('Sample Done')
                if epoch == base_kwargs['n_epochs']:
                    noise = 0.1

                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                    # alpha = .1
                if epoch > base_kwargs['n_epochs'] / 10:
                    noise = 0.1
                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                    # alpha = .1
                if epoch > base_kwargs['n_epochs'] / 5:
                    noise = 0.05
                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                if epoch > base_kwargs['n_epochs'] / 6:
                    noise = 0.01
                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass

                for j in range(base_kwargs['n_train_repeat']):
                    batch_n = []
                    recent_batch_n = []
                    indices = None
                    recent_indices = None
                    # extract batch of transitions from replay buffer
                    for i, agent in enumerate(agents):
                        if i == 0:
                            batch = agent.pool.random_batch(batch_size)
                            indices = agent.pool.indices
                            recent_indices = list(range(agent.pool._top-batch_size, agent.pool._top))

                        batch_n.append(agent.pool.random_batch_by_indices(indices))
                        recent_batch_n.append(agent.pool.random_batch_by_indices(recent_indices))

                    # print(len(batch_n))
                    # extend batch of transitions with next actions by applying current policy
                    target_next_actions_n = []
                    try:
                        for agent, batch in zip(agents, batch_n):
                            target_next_actions_n.append(agent._target_policy.get_actions(batch['next_observations']))
                    except:
                        pass


                    opponent_actions_n = np.array([batch['actions'] for batch in batch_n])
                    recent_opponent_actions_n = np.array([batch['actions'] for batch in recent_batch_n])

                    ####### figure out
                    recent_opponent_observations_n = []
                    for batch in recent_batch_n:
                        recent_opponent_observations_n.append(batch['observations'])


                    current_actions = [agents[i]._policy.get_actions(batch_n[i]['next_observations'])[0][0] for i in range(agent_num)]
                    all_actions_k = []
                    for i, agent in enumerate(agents):
                        if isinstance(agent, MAVBAC):
                            if agent._k > 0:
                                batch_actions_k = agent._policy.get_all_actions(batch_n[i]['next_observations'])
                                actions_k = [a[0][0] for a in batch_actions_k]
                                all_actions_k.append(';'.join(list(map(str, actions_k))))
                    if len(all_actions_k) > 0:
                        with open('{}/all_actions.csv'.format(policy_dir), 'a') as f:
                            f.write(','.join(list(map(str, all_actions_k))) + '\n')
                    with open('{}/policy.csv'.format(policy_dir), 'a') as f:
                        f.write(','.join(list(map(str, current_actions)))+'\n')
                    # print('============')
                    ##################### training ####################
                    for i, agent in enumerate(agents):
                        try:
                            batch_n[i]['next_actions'] = deepcopy(target_next_actions_n[i])
                        except:
                            pass
                        batch_n[i]['opponent_actions'] = np.reshape(np.delete(deepcopy(opponent_actions_n), i, 0), (-1, agent._opponent_action_dim))
                        # joint means centeralized action space
                        if agent.joint:
                            if agent.opponent_modelling:
                                batch_n[i]['recent_opponent_observations'] = recent_opponent_observations_n[i]
                                batch_n[i]['recent_opponent_actions'] = np.reshape(np.delete(deepcopy(recent_opponent_actions_n), i, 0), (-1, agent._opponent_action_dim))
                                batch_n[i]['opponent_next_actions'] = agent.opponent_policy.get_actions(batch_n[i]['next_observations'])
                            else:
                                batch_n[i]['opponent_next_actions'] = np.reshape(np.delete(deepcopy(target_next_actions_n), i, 0), (-1, agent._opponent_action_dim))
                        if isinstance(agent, MAVBAC) or isinstance(agent, MASQL):
                            # safety_cost_episode = base_kwargs['epoch_length'] + 1.0# hacky way to define an impossible number
                            # if sampler.episode_finish:
                            #     safety_cost_episode = sampler.safety_cost_episode
                            
                            agent._do_training(iteration=t + epoch * agent._epoch_length, batch=batch_n[i], annealing=alpha)
                        else:
                            agent._do_training(iteration=t + epoch * agent._epoch_length, batch=batch_n[i])
                gt.stamp('train')

            # self._evaluate(epoch)

            # for agent in agents:
            #     params = agent.get_snapshot(epoch)
            #     logger.save_itr_params(epoch, params)
            # times_itrs = gt.get_times().stamps.itrs
            #
            # eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
            # total_time = gt.get_times().total
            # logger.record_tabular('time-train', times_itrs['train'][-1])
            # logger.record_tabular('time-eval', eval_time)
            # logger.record_tabular('time-sample', times_itrs['sample'][-1])
            # logger.record_tabular('time-total', total_time)
            # logger.record_tabular('epoch', epoch)

            # sampler.log_diagnostics()
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            sampler.terminate()



if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)