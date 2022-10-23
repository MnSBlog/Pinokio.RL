import copy
import time
import gym
import torch
import numpy as np
from typing import Optional, Union, Tuple

from gym.core import ObsType
from matplotlib import pyplot as plt
from utils.comm_manger import CommunicationManager
import threading
from utils.zero_mq import ZmqServer
from Batch.FlatData import FlatData


class CombatStrategy(gym.Env):
    def __init__(self, env_config):
        self.__version = 1
        self.__envConfig = env_config
        self.__pause_mode = self.__envConfig['mode'] == "Pause"
        self.__period = self.__envConfig['period']
        self.__self_play = self.__envConfig['self_play']
        self.__debug = self.__envConfig['debug']
        self.__visual_imgs = [[] * self.__envConfig['spatial_dim']] * self.__envConfig['agent_count']
        #self.__map_capacity = self.__envConfig['map_capacity']
        self.__agents_of_map = self.__envConfig['agent_count']
        self.server = ZmqServer(self.__envConfig['port'], self.on_received)
        self.state_buffer = (None, None, None, None, None)
        if self.__envConfig['self_play']:
            self.__agents_of_map += self.__envConfig['enemy_count']
       #self.__total_agent_count = self.__agents_of_map * self.__map_capacity
        if self.__debug:
            self.__initialize_feature_display(self.__total_agent_count, self.__envConfig['spatial_dim'])

        self.initialize()

    def initialize(self):
       # self.__communication_manager.send_info(
       #    'Initialize:MapType"' + str(','.join(str(e) for e in self.__envConfig['map_type']))
        #    + '"Pause"' + str(self.__pause_mode)
         #   + '"Period"' + str(self.__envConfig['period'])
          #  + '"Acceleration"' + str(self.__envConfig['acceleration'])
           # + '"')
        t = threading.Thread(target=self.server.listen)
        t.start()

    def step(self, action: torch.tensor = None):
        # Send action list
        begin = time.time()
        action_buffer = CommunicationManager.serialize_action("Action", action.cpu())
        self.server.send(action_buffer)
        begin = time.time()
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
        # To gathering spatial-feature
        state, _, _ = self.__get_observation()
        return state

    def __get_observation(self):
        # feature 시각화 함수 실행
        while self.state_buffer[0] is None:
            time.sleep(0.001)

        state, reward, done, _, _ = copy.deepcopy(self.state_buffer)
        self.state_buffer = (None, None, None, None, None)
        return state, reward, done

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
        reward = torch.zeros(1, 1)
        reward += (step_result[2] + step_result[3]) % 2  # on radar on screen 1
        reward += 5*step_result[1]  # targetting 5
        reward += 10*step_result[8]  # on damage 10
        reward += 20*step_result[5]  # kill 20
        # reward += 1 * step_result[:, 4]  #
        reward -= (-1)*(step_result[3]-1)  # just radar -1
        reward -= 5*step_result[7]  # targeted -5
        reward -= 10*step_result[10]  # get damage -10
        reward -= (-20)*(step_result[6]-1)  # death -20
        return reward

    def render(self, mode="human", **kwargs):
        return

    def seed(self, seed=None):
        return

    def close(self):
        pass

    def deserialize(self, data):
        msg = FlatData.GetRootAsFlatData(data)
        data_array = msg.Info(0)
        step_array = msg.Info(1)
        data_data = data_array.DataAsNumpy()
        data_shape = data_array.ShapeAsNumpy()

        mask_array = None
        if msg.Mask != None:
            mask_array = msg.Mask(0)
            mask_shape = mask_array.ShapeAsNumpy()
            mask_data = mask_array.DataAsNumpy()
            mask_result = torch.tensor(mask_data.reshape(mask_shape), dtype=torch.float)

        step_data = step_array.DataAsNumpy()
        step_shape = step_array.ShapeAsNumpy()

        return torch.tensor(data_data.reshape(data_shape), dtype=torch.float), mask_result, torch.tensor(step_data.reshape(step_shape), dtype=torch.float)
        #(tensor or np) 등 real state

    def on_received(self, message):
         origin_data, origin_mask, origin_step = self.deserialize(message) # reset에 할당해서 state전달
         done = torch.tensor(origin_step[0], dtype=torch.bool)
         reward = self.__calculate_reward(origin_step)
         state = {'matrix': [],
                  'vector':  origin_data.unsqueeze(dim=0),
                  'action_mask': origin_mask.unsqueeze(dim=0)}
         self.state_buffer = (state, reward, done, False, None)






