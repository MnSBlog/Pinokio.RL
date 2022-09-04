# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Batch

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FieldInfo(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FieldInfo()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFieldInfo(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # FieldInfo
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FieldInfo
    def MapId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FieldInfo
    def CharIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FieldInfo
    def Length(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FieldInfo
    def Data(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from Batch.IntVector import IntVector
            obj = IntVector()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # FieldInfo
    def DataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FieldInfo
    def DataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

def Start(builder): builder.StartObject(4)
def FieldInfoStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddMapId(builder, mapId): builder.PrependInt32Slot(0, mapId, 0)
def FieldInfoAddMapId(builder, mapId):
    """This method is deprecated. Please switch to AddMapId."""
    return AddMapId(builder, mapId)
def AddCharIndex(builder, charIndex): builder.PrependInt32Slot(1, charIndex, 0)
def FieldInfoAddCharIndex(builder, charIndex):
    """This method is deprecated. Please switch to AddCharIndex."""
    return AddCharIndex(builder, charIndex)
def AddLength(builder, length): builder.PrependInt32Slot(2, length, 0)
def FieldInfoAddLength(builder, length):
    """This method is deprecated. Please switch to AddLength."""
    return AddLength(builder, length)
def AddData(builder, data): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(data), 0)
def FieldInfoAddData(builder, data):
    """This method is deprecated. Please switch to AddData."""
    return AddData(builder, data)
def StartDataVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def FieldInfoStartDataVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartDataVector(builder, numElems)
def End(builder): return builder.EndObject()
def FieldInfoEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)