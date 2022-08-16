import copy

import flatbuffers
import numpy as np
import torch

import Batch
from Batch.ActionList import ActionList
from Batch.Info import Info
from Batch.InfoList import InfoList
from Batch.FieldInfo import FieldInfo
from Batch.FieldInfoList import FieldInfoList
from SubModules.Utilities.zero_mq import ZmqClient


class CommunicationManager:
    def __init__(self, port):
        self.__flat_builder = flatbuffers.Builder(1024)
        self.__client = ZmqClient(port)

    def serialize_action_list(self, action_list: list):
        # 추후 변경 예정
        Batch.ActionList.StartDataVector(self.__flat_builder, len(action_list))
        for action in action_list:
            self.__flat_builder.PrependUOffsetTRelative(action)
        action_list_vector = self.__flat_builder.EndVector()

        Batch.ActionList.ActionListStart(self.__flat_builder)
        Batch.ActionList.AddLength(self.__flat_builder, len(action_list))
        Batch.ActionList.AddData(self.__flat_builder, action_list_vector)
        end_offset = Batch.ActionList.ActionListEnd(self.__flat_builder)
        self.__flat_builder.Finish(end_offset)
        buffer = self.__flat_builder.Output()
        return buffer

    def deserialize_info_list(self, data):
        msg = InfoList.GetRootAsInfoList(data)
        data_len = msg.Length()
        total_info_list = []
        for idx in range(data_len):
            character_info_list = []
            character_info = msg.Data(idx)
            for feature_num in range(character_info.DataLength()):
                character_info_list.append(character_info.Data(feature_num))
            total_info_list.append(character_info_list)
        temp = np.array(total_info_list)
        return torch.tensor(temp, dtype=torch.float)

    def deserialize_field_info_list(self, data, agent_num, batch_size):
        field_info_list = FieldInfoList.GetRootAsFieldInfoList(data)
        length = field_info_list.Data(0).Length()
        field_info_list_numpy = np.zeros((agent_num, batch_size, length, length))

        count = 0
        for agent_count in range(agent_num):
            for feature_count in range(batch_size):
                field_info_numpy = self.__deserialize_field_info(field_info_list.Data(count))
                count += 1
                field_info_list_numpy[agent_count, feature_count, :, :] = field_info_numpy

        return torch.tensor(field_info_list_numpy, dtype=torch.float)

    def __deserialize_field_info(self, field_info):
        length = field_info.Length()
        field_info_numpy = np.zeros((length, length))
        for i in range(field_info.Length()):
            int_vector = field_info.Data(i)
            int_vector_numpy = np.zeros((1, length))
            for j in range(int_vector.DataLength()):
                int_vector_numpy[0, j] = int_vector.Data(j)
            field_info_numpy[i, :] = copy.deepcopy(int_vector_numpy)
        return field_info_numpy

    def send_info(self, msg: str):
        reply = self.__client.send_info(msg)
        return reply