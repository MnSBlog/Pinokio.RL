import copy
import os
import gymnasium as gym
import numpy as np
import torch
from envs import REGISTRY as ENV_REGISTRY
from runners.general_runner import GeneralRunner


class EpisodeRunner(GeneralRunner):
    def __init__(self, config: dict, env: gym.Env):
        super(EpisodeRunner, self).__init__(config, env)

    def run(self):
        steps = 0
        state = self._env_init(reset_env=True)
        for ep in range(1, self._config['runner']['max_iteration_num'] + 1):
            # 에피소드 초기화
            while not self.done:
                steps += 1
                action = self._select_action(state)
                state = self._interaction(action)
                if self._update_agent(next_state=state, steps=steps):
                    steps = 0
            print('Episode: ', ep, 'Steps: ', self.count, 'Reward: ', self.batch_reward)
            self.save_batch_reward.append(self.batch_reward)
            self._sweep_cycle(ep)
            # 아잇 시발 자꾸 죽으니 100단위로 저장하고 self.batch_reward는 매 에피소드마다 append하는걸로
            temp = open("reward_history.txt", "a")
            temp.write("%.4f\n" % self.batch_reward)
            temp.close()
            self._save_agent()
            # **** gym auto done(auto reset) 모드랑 아닐때랑 OHTRouting 이 세개가 모두 다름..
            self.done = False
            self.count = 0
            self.batch_reward = 0

        self._save_reward_log()


class StepRunner(GeneralRunner):
    def __init__(self, config: dict, env: gym.Env):
        super(StepRunner, self).__init__(config, env)

    def run(self):
        state = self._env_init(reset_env=True)
        for update in range(1, self._config['runner']['max_iteration_num'] + 1):
            self._env_init()
            steps = 0
            while self._update_agent(next_state=state, steps=steps) is False:
                steps += 1
                action = self._select_action(state)
                state = self._interaction(action)
            print('Update: ', update, 'Steps: ', self.count, 'Reward: ', self.batch_reward)
            self._sweep_cycle(update)

    def _load_pretrain_network(self, prefix_option=True):
        try:
            if os.listdir(self._config['runner']['history_path']):
                name_list = os.listdir(self._config['runner']['history_path'])
                if prefix_option:
                    prefix = self.network_type + "-mem_len-" \
                             + str(self.memory_len) \
                             + "-layer_len-" + str(self.layer_len)
                    name_list = [file for file in name_list if prefix in file]

                if len(name_list) > 0:
                    full_list = [os.path.join(self._config['runner']['history_path'], name) for name in name_list]
                    time_sorted_list = sorted(full_list, key=os.path.getmtime)
                    last_file = time_sorted_list[-1]
                    self._agent.load(checkpoint_path=last_file)
        finally:
            pass


class IntrinsicParallelRunner(GeneralRunner):
    def __init__(self, config: dict, env: gym.Env):
        super(IntrinsicParallelRunner, self).__init__(config, env)

        if os.path.exists(os.path.join('./figures', config['env_name'])) is False:
            os.mkdir(os.path.join('./figures', config['env_name']))

    def run(self):
        runner_config = self._config["runner"]
        max_step_num = runner_config['max_step_num']
        steps = 0
        updates = 1
        update_reward = 0
        update_fit_reward = 0

        init_state = self._env.reset()
        for _ in range(self.memory_len):
            self._insert_q(init_state)
        state = self._update_memory()
        # 1, 5, 180
        # 1, 5, 8, 30, 30
        while max_step_num >= 0:
            steps += 1
            max_step_num -= 1
            map_count = state['matrix'].shape[0]
            actions: torch.tensor = None
            for idx in range(map_count):
                temp_matrix = state['matrix'][idx]
                temp_vector = state['vector'][idx]
                temp_action_mask = None
                if state['action_mask'] is not None:
                    temp_action_mask = state['action_mask'][idx]
                temp_state = {'matrix': temp_matrix,
                              'vector': temp_vector,
                              'action_mask': temp_action_mask}
                temp = self._agent.select_action(temp_state)
                action = torch.stack([temp[0], temp[1], temp[2]], dim=1)
                action = self.__add_index(idx, action)
                if actions is None:
                    actions = action.unsqueeze(dim=0)
                else:
                    actions = torch.cat([actions, action.unsqueeze(dim=0)], dim=0)
            next_state, reward, done, _ = self._env.step(actions)
            state = self._update_memory(next_state)

            train_reward = self._fit_reward(reward)
            for idx in range(map_count):
                self._agent.batch_reward.append(train_reward[idx])
                self._agent.batch_done.append(done[idx])
            update_fit_reward += train_reward
            update_reward += reward

            # Step runner는 self-play 아직 적용안함
            if steps >= runner_config['update_step']:
                # 업데이트마다 결과 보상값 출력
                print('Update: ', updates, 'Steps: ', runner_config['max_step_num'] - max_step_num,
                      'Reward: ', update_reward, 'fit_Reward: ', update_fit_reward)
                if max(update_reward.shape) > 1:
                    mem_reward = torch.mean(update_reward)
                else:
                    mem_reward = update_reward.item()
                self.reward_info['memory'].append(mem_reward)
                self.save_batch_reward.append(mem_reward)
                self._agent.update()
                if runner_config['self_play']:
                    # agent_name = random.choice(self.config['agents'])
                    # config = config_loader.config_copy(
                    #     config_loader.get_config(filenames='agents//' + agent_name))
                    # agent_config = config['agent']
                    raise NotImplementedError

                # 업데이트 특정값 신경망 파라미터를 파일에 저장
                if updates % 20 == 0:
                    self._save_agent()
                    self._draw_reward_plot(now_ep=updates, y_lim=500)

                update_reward = 0
                update_fit_reward = 0
                steps = 0
                updates += 1

        self._save_reward_log()

    def __add_index(self, env_index, action: torch.tensor):
        temp_numpy = action.numpy()
        agent_index = list(range(action.size(0)))
        temp_numpy = np.c_[agent_index, temp_numpy]
        env_index = [env_index for _ in range(action.size(0))]
        temp_numpy = np.c_[env_index, temp_numpy]

        return torch.tensor(temp_numpy, dtype=torch.float)

    def _fit_reward(self, rew):
        if True in (self.rew_min > rew[:]):
            print('reward min is updated: ', rew)
            self.rew_min = rew.min().item()
            self.rew_gap = (self.rew_max - self.rew_min) / 2
            self.rew_numerator = (self.rew_max + self.rew_min) / 2
        elif True in (self.rew_max < rew[:]):
            print('reward max is updated: ', rew)
            self.rew_max = rew.max().item()
            self.rew_gap = (self.rew_max - self.rew_min) / 2
            self.rew_numerator = (self.rew_max + self.rew_min) / 2
        # 학습용 보상 [-1, 1]로 fitting
        train_reward = (rew - self.rew_numerator) / self.rew_gap

        return train_reward


class ParallelStepRunner:
    def __init__(self, config: dict):
        env_names = self.get_element(config['env_name'])
        q_cases = self.get_element(config['network']['actor']['memory_q_len'])
        layer_cases = self.get_element(config['network']['actor']['use_memory_layer'])
        self.worker_count = len(env_names) * len(q_cases) * len(layer_cases)
        self.env_names = env_names
        self.q_cases = q_cases
        self.layer_cases = layer_cases
        self.config = config

    def run(self):
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(self.worker_count)
        args = []
        for env_name in self.env_names:
            sub_config = copy.deepcopy(self.config)
            sub_config['runner']['history_path'] = os.path.join("./history", env_name)
            if os.path.exists(sub_config['runner']['history_path']) is False:
                os.mkdir(sub_config['runner']['history_path'])

            sub_config['runner']['history_path'] = os.path.join(sub_config['runner']['history_path'],
                                                                sub_config['agent_name'])
            if os.path.exists(sub_config['runner']['history_path']) is False:
                os.mkdir(sub_config['runner']['history_path'])

            for q_len in self.q_cases:
                for layer_len in self.layer_cases:
                    sub_config['env_name'] = env_name
                    sub_config['network_name'] = env_name
                    sub_config = self.update_config(config=sub_config, key='envs', name=env_name)
                    sub_config = self.update_config(config=sub_config, key='network', name=env_name)
                    sub_config['network']['actor']['memory_q_len'] = q_len
                    sub_config['network']['actor']['use_memory_layer'] = layer_len
                    args.append(sub_config)

        pool.map(self.sub_runner_start, args)

    @staticmethod
    def update_config(config, key, name):
        from utils.yaml_config import YamlConfig
        if key == 'network':
            path_key = 'networks'
        else:
            path_key = key
        root = os.path.join("./config/yaml/", path_key)
        name = name + '.yaml'
        sub_dict = YamlConfig.get_dict(os.path.join(root, name))
        config[key] = sub_dict[key]
        return copy.deepcopy(config)

    @staticmethod
    def sub_runner_start(config: dict):
        env = ENV_REGISTRY[config['env_name']](**config['envs'])
        sub_runner = StepRunner(config=config, env=env)
        sub_runner.run()

    @staticmethod
    def get_element(target):
        rtn_list = []
        if isinstance(target, list):
            return target
        else:
            rtn_list.append(target)
            return rtn_list
