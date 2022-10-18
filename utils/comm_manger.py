import flatbuffers
import numpy as np
import torch

import Batch
from Batch.FlatData import FlatData
from Batch.DataArray import DataArray


class CommunicationManager:
    @staticmethod
    def serialize_action(name_str: str, actions: torch.tensor):
        flat_builder = flatbuffers.Builder(1024)

        send_tensor = actions.flatten()
        send_numpy = np.array(send_tensor)
        name = flat_builder.CreateString(name_str)
        Batch.DataArray.StartShapeVector(flat_builder, len(actions.shape))
        for i in reversed(range(len(actions.shape))):
            flat_builder.PrependInt32(int(actions.shape[i]))
        shape = flat_builder.EndVector()
        Batch.DataArray.StartDataVector(flat_builder, len(send_numpy))
        for i in reversed(range(len(send_numpy))):
            flat_builder.PrependFloat32(send_numpy[i])
        data = flat_builder.EndVector()
        Batch.DataArray.DataArrayStart(flat_builder)
        Batch.DataArray.AddName(flat_builder, name)
        Batch.DataArray.AddShape(flat_builder, shape)
        Batch.DataArray.AddData(flat_builder, data)
        data_array = Batch.DataArray.DataArrayEnd(flat_builder)

        Batch.FlatData.StartInfoVector(flat_builder, 1)
        flat_builder.PrependUOffsetTRelative(data_array)
        total_info = flat_builder.EndVector()
        Batch.FlatData.FlatDataStart(flat_builder)
        Batch.FlatData.AddInfo(flat_builder, total_info)
        end_offset = Batch.FlatData.FlatDataEnd(flat_builder)
        flat_builder.Finish(end_offset)
        buffer = flat_builder.Output()
        return buffer

    @staticmethod
    def deserialize_info(data):
        msg = FlatData.GetRootAsFlatData(data)

        data_array = msg.Info()
        data_name = data_array.Name()
        data_shape = data_array.ShapeAsNumpy()
        float_data = data_array.DataAsNumpy()
        total_info = torch.tensor(float_data.reshape(data_shape), dtype=torch.float)

        total_mask = None
        if msg.MaskIsNone() is False:  # mask 있을 때
            mask_array = msg.Mask()
            mask_shape = mask_array.ShapeAsNumpy()
            mask_data = mask_array.DataAsNumpy()
            total_mask = torch.tensor(mask_data.reshape(mask_shape), dtype=torch.float)

        return data_name, total_info, total_mask

    # @classmethod
    # def send_info(cls, msg: str):
    #     if isinstance(msg, str):
    #         reply = cls.__client.send(msg.encode())
    #     else:
    #         reply = cls.__client.send(msg)
    #     return reply