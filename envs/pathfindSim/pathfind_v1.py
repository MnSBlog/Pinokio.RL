import time
import gym
import torch
import numpy as np
from typing import Optional, Union, Tuple

from gym.core import ObsType
from matplotlib import pyplot as plt
from utils.pf_Interface import pf_comm_manger


class PathFindSim(gym.Env):
    def __init__(self, env_config):

        self.__envConfig = env_config
        self.__communication_manager = pf_comm_manger(self.__envConfig['port'])

        # self.__pause_mode = self.__envConfig['mode'] == "Pause"
        self.__period = self.__envConfig['period']
        # self.__self_play = self.__envConfig['self_play']
        # self.__debug = self.__envConfig['debug']

        # self.__agents_of_map = len(self.__envConfig['first_weapon_type'])
        # self.__total_agent_count = self.__agents_of_map * len(self.__envConfig['map_type'])
        # self.__visual_imgs = [[] * self.__envConfig['spatial_dim']] * self.__agents_of_map

        # if self.__debug:
        #     self.__initialize_feature_display(self.__total_agent_count, self.__envConfig['spatial_dim'])

        self.initialize()

    def initialize(self):
        print("Init")
        # self.__communication_manager.send_info(
        #     'Initialize:MapType"' + str(','.join(str(e) for e in self.__envConfig['map_type']))
        #     + '"Pause"' + str(self.__pause_mode)
        #     + '"Period"' + str(self.__envConfig['period'])
        #     + '"Acceleration"' + str(self.__envConfig['acceleration'])
        #     + '"')

    def step(self, action: Optional[list] = None):
        # Send action list

        # 받아온 액션을 넘겨준다.
        action_buffer = self.__communication_manager.serialize_action_list(action.cpu(), self.__agents_of_map)
        self.__communication_manager.send_info(action_buffer)
        return self.__get_observation()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:

        # To gathering spatial-feature
        state, _, _, _ = self.__get_observation()
        return state

    def __get_observation(self):
        # state 가져오면 된다.
        # non-spatial 정보

        message = self.__communication_manager.recv_info()

        deserialized_char_info_list = self.__communication_manager.deserialize_info(character_info_list, agent_num,
                                                                                    character_info_shape)
        deserialized_action_mask = self.__communication_manager.deserialize_action_mask(character_info_list, agent_num,
                                                                                        action_mask_shape)
        # feature 시각화 함수 실행
        if self.__debug:
            self.__display_multi_features(deserialized_field_info_list)

        deserialized_step_info_list = self.__communication_manager.deserialize_info(step_results, agent_num,
                                                                                    step_info_shape)

        done = torch.tensor(deserialized_step_info_list[:, 0], dtype=torch.bool)

        reward = self.__calculate_reward(deserialized_step_info_list)

        vector_shape = deserialized_char_info_list.shape
        state = {'matrix': deserialized_field_info_list,
                 'vector': deserialized_char_info_list.reshape([vector_shape[0], vector_shape[1] * vector_shape[2]]),
                 'action_mask': deserialized_action_mask}
        return state, reward, done, None

    def __initialize_feature_display(self, batch_size, feature_size):
        figure, ax = plt.subplots(batch_size, feature_size, figsize=(15, 15))
        self.__plots = (figure, ax)
        for i in range(batch_size):
            for j in range(feature_size):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)
        subplot_num = 8
        subplot_title_list = ['char_index', 'visual_field_of_map', 'movable', 'obj_id', 'treat_points',
                              'grenade_damage', 'visible_enemy_location', 'enemy_location']
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

