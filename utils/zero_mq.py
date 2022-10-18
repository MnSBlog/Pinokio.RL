import time
import zmq


class ZmqServer:
    def __init__(self, port: int, func):
        self._port = port
        self._callback = func
        context = zmq.Context()
        self.socket = context.socket(zmq.SocketType.REP)
        addr = "tcp//127.0.0.1:" + str(self._port)
        print("Server Start : " + addr)

        self.socket.bind("tcp://*:%s" % self._port)

    def listen(self):
        while True:
            msg = self.socket.recv()
            reply = self._callback(msg)
            time.sleep(0.01)
            if reply is not None:
                self.send(reply)

    def send(self, reply):
        self.socket.send(reply)


class ZmqClient:
    def __init__(self, port: int):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://localhost:" + str(port))

    def send(self, msg):
        self._socket.send(msg)
        reply = self._socket.recv()
        return reply