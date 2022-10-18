import time
import zmq
from Batch.FlatData import FlatData

class ZmqServer:
    def __init__(self, port: int, func):
        self._port = port
        self._callback = func
        context = zmq.Context()
        self.socket = context.socket(zmq.SocketType.REP)
    def listen(self):
        addr = "tcp//127.0.0.1:" + str(self._port)
        print("Server Start : " + addr)

        self.socket.bind("tcp://*:%s" % self._port)

        while True:
            msg = self.socket.recv()
            flat_data = FlatData.GetRootAsFlatData(msg)
            print(flat_data.Info(0).DataAsNumpy())
            self._callback(msg)
            time.sleep(0.01)

    def send(self):
        self.socket.send()


class ZmqClient:
    def __init__(self, port: int):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://localhost:" + str(port))

    def send(self, msg):
        self._socket.send(msg)
        print(msg)
        reply = self._socket.recv()
        return reply
