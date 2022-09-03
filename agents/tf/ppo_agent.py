import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from agents.general_agent import unpack_batch, log_pdf
from agents.tf.actorcritic import Actor, Critic
from agents.general_agent import PolicyAgent


class PpoAgent(PolicyAgent):
    def __init__(self, parameters: dict, actor: Actor, critic: Critic):
        super(PpoAgent, self).__init__(parameters=parameters, actor=actor, critic=critic)
        # Hyper-parameters
        self.gamma = self._config['gamma']
        self.gae_lambda = self._config['gae_lambda']
        self.actor_lr = self._config['actor_lr']
        self.critic_lr = self._config['critic_lr']
        self.epochs = self._config['epochs']
        self.ratio_clipping = self._config['ratio_clipping']
        # 표준편차의 최솟값과 최대값 설정
        self.std_bound = self._config['std_bound']

        # 옵티마이저 설정
        self.actor_opt = Adam(self.actor_lr)
        self.critic_opt = Adam(self.critic_lr)

        self.action_bound = self._config['action_bound']

    def select_action(self, state):
        mu_a, std_a = self.actor(state)
        mu_a = mu_a.numpy()[0]
        std_a = std_a.numpy()[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.actor.action_dim)
        action = np.clip(action, -self.action_bound, self.action_bound)

        # 이전 정책의 로그 확률밀도함수 계산
        var_old = std_a ** 2
        log_old_policy_pdf = -0.5 * (action - mu_a) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
        log_old_policy_pdf = np.sum(log_old_policy_pdf)
        log_old_policy_pdf = np.reshape(log_old_policy_pdf, [1, 1])

        self.batch_state.append(state)
        self.batch_action.append(action)
        self.batch_log_old_policy_pdf.append(log_old_policy_pdf)
        return action

    def update(self, next_state=None, done=None):
        states = unpack_batch(self.batch_state)
        actions = unpack_batch(self.batch_action)
        rewards = unpack_batch(self.batch_reward)
        log_old_policy_pdfs = unpack_batch(self.batch_log_old_policy_pdf)

        # GAE와 시간차 타깃 계산
        next_v_value = self.critic(tf.convert_to_tensor([next_state], dtype=tf.float32))
        v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
        gaes, y_i = self.gae_target(rewards, v_values.numpy(), next_v_value.numpy(), done)

        # 에포크만큼 반복
        for _ in range(self.epochs):
            # 액터 신경망 업데이트
            self.actor_learn(tf.convert_to_tensor(log_old_policy_pdfs, dtype=tf.float32),
                             tf.convert_to_tensor(states, dtype=tf.float32),
                             tf.convert_to_tensor(actions, dtype=tf.float32),
                             tf.convert_to_tensor(gaes, dtype=tf.float32))
            # 크리틱 신경망 업데이트
            self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                              tf.convert_to_tensor(y_i, dtype=tf.float32))

        self.batch_state = []
        self.batch_action = []
        self.batch_reward = []
        self.batch_log_old_policy_pdf = []

    def save(self, checkpoint_path: str):
        self.actor.save_weights(checkpoint_path + "_actor.h5")
        self.critic.save_weights(checkpoint_path + "_critic.h5")

    def load(self, checkpoint_path: str):
        self.actor.load_weights(checkpoint_path + '_actor.h5')
        self.critic.load_weights(checkpoint_path + '_critic.h5')

    # 액터 신경망 학습
    def actor_learn(self, log_old_policy_pdf, states, actions, gaes):
        with tf.GradientTape() as tape:
            # 현재 정책 확률밀도함수
            mu_a, std_a = self.actor(states, training=True)
            log_policy_pdf = log_pdf(mu_a, std_a, actions, std_bound=self.std_bound)

            # 현재와 이전 정책 비율
            ratio = tf.exp(log_policy_pdf - log_old_policy_pdf)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.ratio_clipping, 1.0 + self.ratio_clipping)
            surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
            loss = tf.reduce_mean(surrogate)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    # 크리틱 신경망 학습
    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(td_hat - td_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    # GAE와 시간차 타깃 계산
    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * forward_val - v_values[k]
            gae_cumulative = self.gamma * self.gae_lambda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets


