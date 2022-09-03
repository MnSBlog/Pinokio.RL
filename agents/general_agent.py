import numpy as np
import tensorflow as tf
from agents.tf.actorcritic import Actor, Critic


class GeneralAgent:
    def __init__(self, parameters: dict):
        self._config = parameters
        # 배치 설정
        self.batch_state = []
        self.batch_action = []
        self.batch_reward = []
        self.batch_next_state = []
        self.batch_done = []
        self.batch_log_old_policy_pdf = []

    def select_action(self, state):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def save(self, checkpoint_path: str):
        raise NotImplementedError

    def load(self, checkpoint_path: str):
        raise NotImplementedError


class PolicyAgent(GeneralAgent):
    def __init__(self, parameters: dict, actor: Actor, critic: Critic):
        super(PolicyAgent, self).__init__(parameters=parameters)
        self.actor = Actor
        self.critic = Critic

    def select_action(self, state):
        return super(PolicyAgent, self).select_action(state)

    def update(self, next_state=None, done=None):
        return super(PolicyAgent, self).update()

    def save(self, checkpoint_path: str):
        return super(PolicyAgent, self).save(checkpoint_path=checkpoint_path)

    def load(self, checkpoint_path: str):
        return super(PolicyAgent, self).load(checkpoint_path=checkpoint_path)


# 시간차 타깃 계산
def td_target(rewards, next_v_values, dones, gamma=0.99):
    y_i = np.zeros(next_v_values.shape)
    for i in range(next_v_values.shape[0]):
        if dones[i]:
            y_i[i] = rewards[i]
        else:
            y_i[i] = rewards[i] + gamma * next_v_values[i]
    return y_i


# 배치에 저장된 데이터 추출
def unpack_batch(batch):
    unpack = batch[0]
    for idx in range(len(batch) - 1):
        unpack = np.append(unpack, batch[idx + 1], axis=0)

    return unpack


# 로그-정책 확률밀도함수
def log_pdf(mu, std, action, std_bound):
    std = tf.clip_by_value(std, std_bound[0], std_bound[1])
    var = std ** 2
    log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
    return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
