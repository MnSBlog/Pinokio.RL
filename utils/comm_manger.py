import flatbuffers
import numpy as np
import torch

import Batch
from Batch.FlatData import FlatData
from Batch.DataArray import DataArray
from Batch.MaskArray import MaskArray
from utils.zero_mq import ZmqClient


class CommunicationManager:
    def __init__(self, port):
        self.__flat_builder = flatbuffers.Builder(1024)
        self.__client = ZmqClient(port)

    def serialize_action_list(self, actions: list, agent_num_per_map):
        alpha_action = actions[0]  # move_action
        beta_action = actions[1]  # attack_action
        theta_action = actions[2]  # view_action
        seperated_alpha_action = np.reshape(alpha_action, (-1, agent_num_per_map))
        seperated_beta_action = np.reshape(beta_action, (-1, agent_num_per_map))
        seperated_theta_action = np.reshape(theta_action, (-1, agent_num_per_map))

        total_agent_num = len(alpha_action)
        agent_id_list = list(range(agent_num_per_map))
        total_action_list = []
        for map_id in range(total_agent_num // agent_num_per_map):
            action_numpy = np.zeros((agent_num_per_map, 5))  # map_id, agent_id, alpha, beta, theta
            action_numpy[:, 0] = map_id
            action_numpy[:, 1] = np.reshape(agent_id_list, (agent_num_per_map,))
            action_numpy[:, 2] = np.reshape(seperated_alpha_action[map_id], (agent_num_per_map,))
            action_numpy[:, 3] = np.reshape(seperated_beta_action[map_id], (agent_num_per_map,))
            action_numpy[:, 4] = np.reshape(seperated_theta_action[map_id], (agent_num_per_map,))
            flatten_action = action_numpy.reshape(-1)
            name = self.__flat_builder.CreateString('Action')
            Batch.DataArray.StartShapeVector(self.__flat_builder, len(action_numpy.shape))
            for i in reversed(range(len(action_numpy.shape))):
                self.__flat_builder.PrependInt32(int(action_numpy.shape[i]))
            shape = self.__flat_builder.EndVector()
            Batch.DataArray.StartDataVector(self.__flat_builder, len(flatten_action))
            for i in reversed(range(len(flatten_action))):
                self.__flat_builder.PrependFloat32(flatten_action[i])
            data = self.__flat_builder.EndVector()
            Batch.DataArray.DataArrayStart(self.__flat_builder)
            Batch.DataArray.AddName(self.__flat_builder, name)
            Batch.DataArray.AddShape(self.__flat_builder, shape)
            Batch.DataArray.AddData(self.__flat_builder, data)
            data_array = Batch.DataArray.DataArrayEnd(self.__flat_builder)
            total_action_list.append(data_array)

        Batch.FlatData.StartInfoVector(self.__flat_builder, len(total_action_list))
        for i in reversed(range(len(total_action_list))):
            self.__flat_builder.PrependUOffsetTRelative(total_action_list[i])
        total_info = self.__flat_builder.EndVector()
        Batch.FlatData.FlatDataStart(self.__flat_builder)
        Batch.FlatData.AddInfo(self.__flat_builder, total_info)
        end_offset = Batch.FlatData.FlatDataEnd(self.__flat_builder)
        self.__flat_builder.Finish(end_offset)
        buffer = self.__flat_builder.Output()
        return buffer

    def deserialize_info(self, data, agent_num, info_shape):
        total_agent_num = info_shape[0]
        msg = FlatData.GetRootAsFlatData(data)
        total_info_list_numpy = np.zeros(info_shape)
        map_num = total_agent_num // agent_num
        for map_id in range(map_num):
            data_array = msg.Info(map_id)
            data_shape = data_array.ShapeAsNumpy()
            float_data = data_array.DataAsNumpy()
            total_info_list_numpy[map_id * agent_num:(map_id + 1) * agent_num, :] = float_data.reshape(data_shape)
        return torch.tensor(total_info_list_numpy, dtype=torch.float)

    def deserialize_action_mask(self, data, agent_num, mask_shape):
        total_agent_num = mask_shape[0]
        msg = FlatData.GetRootAsFlatData(data)
        total_mask_numpy = np.ones(mask_shape)
        if msg.MaskIsNone() is False:
            map_num = total_agent_num // agent_num
            for map_id in range(map_num):
                mask_array = msg.Mask(map_id)
                mask_shape = mask_array.ShapeAsNumpy()
                mask_data = mask_array.DataAsNumpy()
                total_mask_numpy[map_id * agent_num:(map_id + 1) * agent_num, :] = mask_data.reshape(mask_shape)
        return torch.tensor(total_mask_numpy, dtype=torch.float)

    def send_info(self, msg: str):
        if isinstance(msg, str):
            reply = self.__client.send(msg.encode())
        else:
            reply = self.__client.send(msg)
        return reply
