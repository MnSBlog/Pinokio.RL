import numpy as np
import tensorflow as tf


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


# 시간차 타깃 계산
def td_target(rewards, next_v_values, dones, gamma=0.99):
    y_i = np.zeros(next_v_values.shape)
    for i in range(next_v_values.shape[0]):
        if dones[i]:
            y_i[i] = rewards[i]
        else:
            y_i[i] = rewards[i] + gamma * next_v_values[i]
    return y_i