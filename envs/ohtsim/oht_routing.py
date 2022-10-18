import gym
import torch
import subprocess
import threading
from gym.core import ObsType, ActType
from typing import Optional, Union, Tuple
from utils.sha256_encrypt import is_valid
from utils.zero_mq import ZmqServer


class OhtBase(gym.Env):
    def __init__(self, env_config):
        put = input("DBMS ID 입력 : ")
        if is_valid(itr=env_config['iteration'], value=put, key=env_config['env_config']['id']) is False:
            raise "Wrong ID value. check your configuration"
        env_config['env_config']['id'] = put

        put = input("DBMS Password 입력 : ")
        if is_valid(itr=env_config['iteration'], value=put, key=env_config['env_config']['pw']) is False:
            raise "Wrong Password value. check your configuration"
        env_config['env_config']['pw'] = put

        self._envConfig = env_config['env_config']

        # Base Parameters
        self.server = ZmqServer(self._envConfig['port'], self.on_received)
        self.subEnvs = {}
        self.envId = 0
        self.envCount = self._envConfig['env_count']

    # region local custom
    def start_env(self):
        args = [self._envConfig['env_path']]
        for key, value in self._envConfig['args'].item():
            arg = key + ":" + value
            args.append(arg)
        subprocess.call(args)

    def initialize(self):
        # Start Multiple Environments
        for i in range(self.envCount):
            env_thread = threading.Thread(target=self.start_env)
            env_thread.start()

        # Start ZMQ Server to listen
        t = threading.Thread(target=self.server.listen)
        t.start()

    def get_transition(self, request):
        prev_state = self.get_screen(request.PrevStateAsNumpy())
        prev_action = request.PrevAction()
        reward = request.Reward()
        state = self.get_screen(request.StateAsNumpy())
        done = request.Done()

        return prev_state, prev_action, reward, state, done

    @staticmethod
    def get_screen(buffer):
        img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        img = torch.Tensor(img.transpose((2, 0, 1)))
        return img.unsqueeze(0)

    def on_received(self, buff: bytearray):
        request = self.msgBuilder.GetRequestMsg(buff)

        ohtId = request.OhtId()
        (prev_state, prev_action, reward, state, done) = self.get_transition(request)

        if request.Type() == MsgType.MsgType.Step:
            if prev_action >= 0:  # Reset 상황 제외.
                a = torch.tensor([[prev_action]], device=device)
                r = torch.tensor([reward], device=device)
                self.memory.push(prev_state, a, state, r)
                self.optimize_model()

        if done:
            self.episode_count += 1
            if self.episode_count % TARGET_UPDATE == 0:
                print("target model update " + str(self.episode_count))
                self.model.load_state_dict(self.policy_model.state_dict())
                self.save_model()

            return self.msgBuilder.BuildReplyMessage(ohtId, MsgType.MsgType.DoneCheck, -1)
        else:
            return self.msgBuilder.BuildReplyMessage(ohtId, MsgType.MsgType.DoneCheck, self.select_action(state))

    # endregion
    # region Gym environment override
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        pass

    def render(self, mode="human"):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        pass

    def close(self):
        pass

    # endregion
