import time
import zmq


class ZmqServer:
    def __init__(self, port: int, func):
        self._port = port
        self._callback = func

    def listen(self):
        context = zmq.Context()
        socket = context.socket(zmq.SocketType.REP)
        addr = "tcp//127.0.0.1:" + str(self._port)
        print("Server Start : " + addr)

        socket.bind("tcp://*:%s" % self._port)

        while True:
            msg = socket.recv()
            reply = self._callback(msg)
            time.sleep(0.01)
            socket.send(reply)


class ZmqClient:
    def __init__(self, port: int):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SocketType.REQ)
        self._socket.connect("tcp://localhost:" + str(port))

    def send(self, msg):
        self._socket.send(msg)
        reply = self._socket.recv()
        return reply
