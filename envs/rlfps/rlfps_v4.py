import time
import gym
import torch
import numpy as np
from typing import Optional, Union, Tuple

from gym.core import ObsType
from matplotlib import pyplot as plt
from utils.comm_manger import CommunicationManager


class RLFPSv4(gym.Env):
    def __init__(self, env_config):
        self.__version = 4
        self.__envConfig = env_config
        self.__pause_mode = self.__envConfig['mode'] == "Pause"
        self.__period = self.__envConfig['period']
        self.__communication_manager = CommunicationManager(self.__envConfig['port'])
        self.__self_play = self.__envConfig['self_play']
        self.__debug = self.__envConfig['debug']
        self.__visual_imgs = [[] * self.__envConfig['spatial_dim']] * self.__envConfig['agent_count']
        self.__map_capacity = self.__envConfig['map_capacity']
        self.__agents_of_map = self.__envConfig['agent_count']
        if self.__envConfig['self_play']:
            self.__agents_of_map += self.__envConfig['enemy_count']
        self.__total_agent_count = self.__agents_of_map * self.__map_capacity

        if self.__debug:
            self.__initialize_feature_display(self.__total_agent_count, self.__envConfig['spatial_dim'])

        self.initialize()

    def initialize(self):
        self.__communication_manager.send_info(
            'Initialize:MapType"' + str(','.join(str(e) for e in self.__envConfig['map_type']))
            + '"MapCapa"' + str(int(self.__envConfig['map_capacity']))
            + '"Pause"' + str(self.__pause_mode)
            + '"Period"' + str(self.__envConfig['period'])
            + '"Acceleration"' + str(self.__envConfig['acceleration'])
            + '"')

    def step(self, action: Optional[list] = None):
        # Send action list
        begin = time.time()
        action_buffer = self.__communication_manager.serialize_action_list(action.cpu(), self.__agents_of_map)
        self.__communication_manager.send_info("Action")
        print("Action serializing: ", (time.time() - begin) * 1000, "ms")

        begin = time.time()
        self.__communication_manager.send_info(action_buffer)
        print("Action interation: ", (time.time() - begin) * 1000, "ms")
        return self.__get_observation()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:

        self_play = self.__envConfig['self_play']
        n_of_current_agent = self.__envConfig['agent_count']
        n_of_current_enemy = self.__envConfig['enemy_count']
        msg = 'Rebuild:"Selfplay"' + str(self_play) + \
              '"MaxStep"' + str(self.__envConfig['max_steps']) + \
              '"AgentType"' + str(','.join(str(e) for e in self.__envConfig['agent_type'])) + \
              '"FirstWeaponType"' + str(','.join(str(e) for e in self.__envConfig['first_weapon_type'])) + \
              '"SecondWeaponType"' + str(','.join(str(e) for e in self.__envConfig['second_weapon_type'])) + \
              '"NoA"' + str(n_of_current_agent) + \
              '"NoE"' + str(n_of_current_enemy) + \
              '"GlRe"' + str(self.__envConfig['global_resolution']) + \
              '"LcRe"' + str(self.__envConfig['local_resolution']) + \
              '"InitHeight"' + str(self.__envConfig['init_height']) + '"'
        self.__communication_manager.send_info(msg)
        # To gathering spatial-feature
        state, _, _, _ = self.__get_observation()
        return state

    def __get_observation(self):
        begin = time.time()
        character_info_list = self.__communication_manager.send_info(
            'CharacterInfo: "' + str(self.__envConfig['non_spatial_dim']) + '"')
        field_info_list = self.__communication_manager.send_info(
            'FieldInfo: "' + str(self.__envConfig['spatial_dim']) + '"')
        step_results = self.__communication_manager.send_info('Step: "' + str(self.__envConfig['step_info_dim']) + '"')
        print("Getting information: ", (time.time() - begin) * 1000, "ms")
        total_agent_num = self.__total_agent_count
        non_spatial_batch_num = self.__envConfig['non_spatial_dim']
        spatial_batch_num = self.__envConfig['spatial_dim']
        step_batch_num = self.__envConfig['step_info_dim']
        agent_num = self.__agents_of_map
        begin = time.time()
        character_info_shape = (total_agent_num, non_spatial_batch_num)
        action_mask_shape = (total_agent_num, self.__envConfig['move_dim'] + self.__envConfig['attack_dim'] + self.__envConfig['view_dim'])
        field_info_shape = (
            total_agent_num, spatial_batch_num, self.__envConfig['local_resolution'],
            self.__envConfig['local_resolution'])
        step_info_shape = (total_agent_num, step_batch_num)
        deserialized_char_info_list = self.__communication_manager.deserialize_info(character_info_list, agent_num,
                                                                                    character_info_shape)
        deserialized_action_mask = self.__communication_manager.deserialize_action_mask(character_info_list, agent_num,
                                                                                        action_mask_shape)
        deserialized_field_info_list = self.__communication_manager.deserialize_info(field_info_list,
                                                                                     agent_num,
                                                                                     field_info_shape)
        # feature 시각화 함수 실행
        if self.__debug:
            self.__display_multi_features(deserialized_field_info_list)
        deserialized_step_info_list = self.__communication_manager.deserialize_info(step_results, agent_num,
                                                                                    step_info_shape)
        print("Deserializing: ", (time.time() - begin) * 1000, "ms")

        begin = time.time()
        done = torch.tensor(deserialized_step_info_list[:, 0], dtype=torch.bool)
        reward = self.__calculate_reward(deserialized_step_info_list)
        print("done & reward: ", (time.time() - begin) * 1000, "ms")
        state = {'matrix': deserialized_field_info_list,
                 'vector': deserialized_char_info_list,
                 'action_mask': deserialized_action_mask}
        return state, reward, done, None

    def __initialize_feature_display(self, batch_size, feature_size):
        figure, ax = plt.subplots(batch_size, feature_size, figsize=(15, 15))
        self.__plots = (figure, ax)
        for i in range(batch_size):
            for j in range(feature_size):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)
        subplot_num = 7
        subplot_title_list = ['last_enemy_location', 'grenade_damage', 'treat_points', 'obj_id', 'movable', 'visual_field_of_map', 'char_index']
        for i in range(subplot_num):
            ax[0, i].set_title(subplot_title_list[i], fontsize = 12)
        plt.tight_layout()
        plt.show(block=False)
        figure.canvas.flush_events()
        self.__plots = (figure, ax)

    def __display_multi_features(self, spatial_data):
        (fig, ax) = self.__plots
        batch_size = len(spatial_data)
        feature_size = len(spatial_data[0])
        for i in range(batch_size):
            for j in range(feature_size):
                ax[i, j].imshow(np.array(spatial_data[i, j, :, :]))

        plt.show(block=False)
        plt.pause(0.000000001)
        self.__plots = (fig, ax)

    def __calculate_reward(self, step_result):
        reward = torch.zeros(len(step_result))
        reward += 1 * step_result[:, 1]  # team win
        reward += 1 * step_result[:, 2]  # kill score
        reward -= 1 * step_result[:, 3]  # dead score
        # reward += 1 * step_result[:, 4]  # damage score
        # reward -= 1 * step_result[:, 5]  # hitted score
        reward += 1 * step_result[:, 6]  # healthy ratio
        return reward

    def render(self, mode="human", **kwargs):
        return

    def seed(self, seed=None):
        return

    def close(self):
        pass
