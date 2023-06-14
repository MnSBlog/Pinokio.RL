import time
import zmq
from Batch.FlatData import FlatData


class ZmqServer:
    def __init__(self, port: int, func):
        self._port = port
        self.send_done = True
        self._callback = func
        context = zmq.Context(1)
        self.socket = context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        addr = "tcp//127.0.0.1:" + str(self._port)
        print("Server Start : " + addr)
        self.socket.bind("tcp://*:%s" % self._port)

    def listen(self):
        while True:
            if self.send_done:
                msg = self.socket.recv()
                self.send_done = False
                return self._callback(msg)
            time.sleep(0.001)

    def send(self, reply):
        self.socket.send(reply)
        self.send_done = True
        # time.sleep(0.2)


class ZmqClient:
    def __init__(self, port: int):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://localhost:" + str(port))

    def send(self, msg):
        # 여기서 멈춤 리시브 못 받네
        if isinstance(msg, str):
            self._socket.send_string(msg)
            reply = self._socket.recv()
        else:
            self._socket.send(msg)
            reply = self._socket.recv()
        return reply
