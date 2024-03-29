# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Batch

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class FlatData(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FlatData()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFlatData(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    # FlatData
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FlatData
    def Info(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from Batch.DataArray import DataArray
            obj = DataArray()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # FlatData
    def InfoLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FlatData
    def InfoIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # FlatData
    def Mask(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from Batch.MaskArray import MaskArray
            obj = MaskArray()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # FlatData
    def MaskLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FlatData
    def MaskIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0


def FlatDataStart(builder): builder.StartObject(2)


def Start(builder):
    return FlatDataStart(builder)


def FlatDataAddInfo(builder, info): builder.PrependUOffsetTRelativeSlot(0,
                                                                        flatbuffers.number_types.UOffsetTFlags.py_type(
                                                                            info), 0)


def AddInfo(builder, info):
    return FlatDataAddInfo(builder, info)


def FlatDataStartInfoVector(builder, numElems): return builder.StartVector(4, numElems, 4)


def StartInfoVector(builder, numElems):
    return FlatDataStartInfoVector(builder, numElems)


def FlatDataAddMask(builder, mask): builder.PrependUOffsetTRelativeSlot(1,
                                                                        flatbuffers.number_types.UOffsetTFlags.py_type(
                                                                            mask), 0)


def AddMask(builder, mask):
    return FlatDataAddMask(builder, mask)


def FlatDataStartMaskVector(builder, numElems): return builder.StartVector(4, numElems, 4)


def StartMaskVector(builder, numElems):
    return FlatDataStartMaskVector(builder, numElems)


def FlatDataEnd(builder): return builder.EndObject()


def End(builder):
    return FlatDataEnd(builder)
