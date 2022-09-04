import time
import zmq


class ZmqServer:
    def __init__(self, port: int):
        self.__Context = zmq.Context()
        self.__Socket = self.__Context.socket(zmq.REP)
        self.__Socket.bind("tcp://*:" + str(port))

    def __listener(self):
        while True:
            #  Wait for next request from client
            message = self.__Socket.recv()
            # print(f"Received request: {message}")

            #  Do some 'work'
            time.sleep(0.003)

            #  Send reply to client
            self.__Socket.send(b"World")


class ZmqClient:
    def __init__(self, port: int):
        self.__Context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to hello world server…")
        self.__Socket = self.__Context.socket(zmq.REQ)
        self.__Socket.connect("tcp://localhost:" + str(port))

    def send_info(self, msg):
        # print(f"Sending request {msg} …")
        if isinstance(msg, str):
            self.__Socket.send(msg.encode())
        elif isinstance(msg, bytearray):
            self.__Socket.send(msg)

        reply = self.__Socket.recv()
        # print(f"Received reply")
        return reply
