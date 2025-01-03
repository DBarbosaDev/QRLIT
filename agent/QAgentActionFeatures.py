import random
import csv
import signal
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

from db_env.DatabaseEnvironment import DatabaseEnvironment
from shared_utils.consts import PROJECT_DIR
from shared_utils.utils import create_logger

from quantum.GroverSearchAlgorithm import GroverSearchAlgorithm
from db_env.tpch.config import SCALE_FACTOR

# K_FACTOR = 0.00037 -> for 200mb
K_FACTOR = 0.00017
SCALE_FACTOR_BASE_UNIT_MB = 1000 # SCALE FACTOR = 1 for 1000mb

AGENT_CSV_FILE = f'{PROJECT_DIR}/data/hybrid_agent_history/quantum_agent_history_sf{SCALE_FACTOR_BASE_UNIT_MB * SCALE_FACTOR}_k{K_FACTOR}_25epi.csv'
WEIGHTS_FILE = f'{PROJECT_DIR}/data/hybrid_agent_history/weights_sf{SCALE_FACTOR_BASE_UNIT_MB * SCALE_FACTOR}_k{K_FACTOR}_25epi.csv'


class Agent:
    def __init__(self, env: DatabaseEnvironment):
        random.seed(2)
        np.random.seed(2)
        self._log = create_logger('agent')
        self._env = env
        self.exploration_probability = 0.9
        self.exploration_probability_discount = 0.9
        self.learning_rate = 0.001
        self.discount_factor = 0.8
        self.k = K_FACTOR
        self._grover_algorithm = GroverSearchAlgorithm(self._env.action_space.n)

        self._experience_memory_max_size = np.inf
        self._experience_replay_count = 30

        self._weights = np.random.rand(1 + self._env.action_space.n + self._env.observation_space.n)
        """Estimated weights of features with bias for every action."""
        self._experience_memory: List[Tuple[List[int], int, float, List[int]]] = []
        self.dict_info = {
            'episode': int,
            'step': int,
            'state': List[bool],
            'action': int,
            'is_max_a': bool,
            'grover_iterations': int,
            'reward': float,
            'next_state': List[bool],
            'q': float,
            'max_a': int,
            'max_q': float,
            'td_target': float,
            'td_error': float,
            'total_reward': float,
            'initial_state_reward': float,

            'step_execution_time_seconds': float,
            'power_queries_exec_time_sum_sec': float,
            'power_refresh_function_exec_time_sum_sec': float,
            'throughput_total_exec_time_sec': float,
            'benchmark_metrics_raw_data': dict
        }

        self._pause_request = False

        def signal_handler(sig, frame):
            self._log.info('CTRL+C pressed - pausing training requested')
            self._pause_request = True

        signal.signal(signal.SIGINT, signal_handler)

    def train(self, episode_count: int, steps_per_episode: int):
        with open(AGENT_CSV_FILE, 'w', newline='') as file:
            wr = csv.writer(file)
            wr.writerow(self.dict_info.keys())
        with open(WEIGHTS_FILE, 'w', newline='') as file:
            wr = csv.writer(file)
            wr.writerow(self._weights)
            
        for episode in range(episode_count):
            state = self._env.reset()
            total_reward = 0.0
            self._build_grovers_algorithm(0, state)

            for step in range(steps_per_episode):
                start_time = time.time()
                
                self._log.info(f'EPISODE {episode} - STEP {step} '
                               f'({(episode_count - episode) * steps_per_episode - step - 1} more steps to go)')

                action = self._run_grover_algorithm(state)
                next_state, reward, _, info = self._env.step(action)
                total_reward += reward
                previous_weights = self._weights.copy()
                self._build_grovers_algorithm(reward, next_state)
                self._update_weights(state, action, reward, next_state, previous_weights)
                self._save_agent_information(episode, step, state, next_state, action, reward, total_reward, info, start_time)
                self._experience_replay(previous_weights)
                self._experience_append(state, action, reward, next_state)
                self._save_agent_weights()
                state = next_state

                if self._pause_request:
                    return

            self._reduce_exploration_probability()

    def _get_features(self, state, action):
        # bias + each action has own feature + state has n columns - n features
        # return np.array([0] + [action == a for a in range(self._env.action_space.n)] + state)
        l = np.array([0] + [action == a for a in range(self._env.action_space.n)] + state)
        return l

    def _experience_append(self, state, action, reward, next_state):
        self._experience_memory.append((state, action, reward, next_state))

        if len(self._experience_memory) > self._experience_memory_max_size:
            self._experience_memory.pop(0)

    def _experience_replay(self, previous_weights):
        samples_count = min(len(self._experience_memory), self._experience_replay_count)
        samples = random.sample(self._experience_memory, k=samples_count)

        for state, action, reward, next_state in samples:
            self._update_weights(state, action, reward, next_state, previous_weights)

    def _update_weights(self, state, action, reward, next_state, weights):
        features = self._get_features(state, action)
        max_action, max_q = self._get_max_action(next_state)

        approx_q = weights @ features
        td_target = reward + self.discount_factor * max_q
        td_error = td_target - approx_q

        # w = w + a(r + y(max q) - w^T * F(s)) * F(s)
        self._weights += self.learning_rate * td_error * features

        self.dict_info['q'] = approx_q
        self.dict_info['max_a'] = max_action
        self.dict_info['max_q'] = max_q
        self.dict_info['td_target'] = td_target
        self.dict_info['td_error'] = td_error

    def _get_max_action(self, state):
        max_q = float('-inf')
        max_action = None

        for action in self._possible_actions(state):
            q = self._calculate_q_value(state, action)
            if max_q < q:
                max_q = q
                max_action = action
        return max_action, max_q

    def _calculate_q_value(self, state, action):
        """:return dot product of weights and features"""
        return self._weights @ self._get_features(state, action)

    def _possible_actions(self, state) -> List[int]:
        return [i * 2 + (not is_indexed) for i, is_indexed in enumerate(state)]

    def _choose_action(self, state):
        """
        :param state: current environment state
        :return: random action with probability epsilon otherwise best action with probability 1-epsilon
        """

        self.dict_info['random_action'] = False
        self.dict_info['exploration_probability'] = self.exploration_probability

        if random.random() < self.exploration_probability:
            self.dict_info['random_action'] = True
            return random.choice(self._possible_actions(state))

        max_action, _ = self._get_max_action(state)
        return max_action

    def _reduce_exploration_probability(self):
        self.exploration_probability = self.exploration_probability_discount * self.exploration_probability

    def _save_agent_information(self, episode, step, state, next_state, action, reward, total_reward, info, start_time):
        self.dict_info['episode'] = episode
        self.dict_info['step'] = step
        self.dict_info['state'] = state
        self.dict_info['next_state'] = next_state
        self.dict_info['action'] = action
        self.dict_info['reward'] = reward
        self.dict_info['total_reward'] = total_reward
        self.dict_info['initial_state_reward'] = info['initial_state_reward']

        self.dict_info['step_execution_time_seconds'] = time.time() - start_time
        self.dict_info['power_queries_exec_time_sum_sec'] = info['power_queries_exec_time_sum_sec']
        self.dict_info['power_refresh_function_exec_time_sum_sec'] = info['power_refresh_function_exec_time_sum_sec']
        self.dict_info['throughput_total_exec_time_sec'] = info['throughput_total_exec_time_sec']
        self.dict_info['benchmark_metrics_raw_data'] = info['benchmark_metrics_raw_data']

        with open(AGENT_CSV_FILE, 'a', newline='') as file:
            wr = csv.writer(file)
            wr.writerow(self.dict_info.values())

    def _save_agent_weights(self):
        with open(WEIGHTS_FILE, 'a', newline='') as file:
            wr = csv.writer(file)
            wr.writerow(self._weights)
        # pd.DataFrame(self._weights).to_csv(WEIGHTS_FILE, index=False)

    def _get_grover_iterations(self, reward, new_state):
        _, max_q = self._get_max_action(new_state)
        steps_num = int(self.k * (reward + max_q))

        return min(steps_num, self._grover_algorithm.get_optimal_iterations())
    
    def _build_grovers_algorithm(self, reward, new_state):
        max_a, _ = self._get_max_action(new_state)
        iterations = self._get_grover_iterations(reward, new_state)

        self.dict_info["grover_iterations"] = iterations

        return self._grover_algorithm.build(iterations, max_a)

    def _run_grover_algorithm(self, state):
        available_actions = self._possible_actions(state)
        max_a, _ = self._get_max_action(state)
        
        # Runs until it finds an available action
        while(True):
            action_bin = self._grover_algorithm.run()

            action = int(action_bin, 2)

            if (action in available_actions):
                self.dict_info["is_max_a"] = max_a == action
                return action