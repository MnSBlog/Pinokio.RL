# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Batch

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FieldInfoList(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FieldInfoList()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFieldInfoList(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # FieldInfoList
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FieldInfoList
    def Length(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FieldInfoList
    def Data(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from Batch.FieldInfo import FieldInfo
            obj = FieldInfo()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # FieldInfoList
    def DataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FieldInfoList
    def DataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def Start(builder): builder.StartObject(2)
def FieldInfoListStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddLength(builder, length): builder.PrependInt32Slot(0, length, 0)
def FieldInfoListAddLength(builder, length):
    """This method is deprecated. Please switch to AddLength."""
    return AddLength(builder, length)
def AddData(builder, data): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(data), 0)
def FieldInfoListAddData(builder, data):
    """This method is deprecated. Please switch to AddData."""
    return AddData(builder, data)
def StartDataVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def FieldInfoListStartDataVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartDataVector(builder, numElems)
def End(builder): return builder.EndObject()
def FieldInfoListEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)