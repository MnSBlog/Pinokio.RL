from . import OHTReplyMsg, OHTRequestMsg, MsgType
import flatbuffers

class OHTMsgBuilder():
    def __init__(self):
        pass

    def BuildReplyMessage(self, ohtId : int, msgType : MsgType, action : int) -> bytearray:
        builder = flatbuffers.Builder(1024)

        OHTReplyMsg.OHTReplyMsgStart(builder)
        OHTReplyMsg.OHTReplyMsgAddType(builder, msgType)
        OHTReplyMsg.OHTReplyMsgAddOhtId(builder, ohtId)
        OHTReplyMsg.OHTReplyMsgAddAction(builder, action)

        msg = OHTReplyMsg.OHTReplyMsgEnd(builder)
        builder.Finish(msg)

        return builder.Output()

    def GetRequestMsg(self, buff : bytearray) -> OHTRequestMsg.OHTRequestMsg:
        return OHTRequestMsg.OHTRequestMsg.GetRootAsOHTRequestMsg(buff)


