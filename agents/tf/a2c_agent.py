import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from agents.general_agent import td_target, unpack_batch, log_pdf
from agents.tf.actorcritic import Actor, Critic
from agents.general_agent import PolicyAgent


class A2cAgent(PolicyAgent):
    def __init__(self, parameters: dict, actor: Actor, critic: Critic):
        super(A2cAgent, self).__init__(parameters=parameters, actor=actor, critic=critic)
        self.gamma = self._config['gamma']
        self.actor_lr = self._config['actor_lr']
        self.critic_lr = self._config['critic_lr']
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
        self.batch_state.append(state)
        self.batch_action.append(action)
        return action

    def update(self, next_state=None, done=None):
        states = unpack_batch(self.batch_state)
        actions = unpack_batch(self.batch_action)
        train_rewards = unpack_batch(self.batch_reward)
        next_states = unpack_batch(self.batch_next_state)
        dones = unpack_batch(self.batch_done)

        # 시간차 타깃 계산
        next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
        td_targets = td_target(train_rewards, next_v_values.numpy(), dones, gamma=self.gamma)

        # 크리틱 신경망 업데이트
        self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                          tf.convert_to_tensor(td_targets, dtype=tf.float32))

        # 어드밴티지 계산
        v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
        next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
        advantages = train_rewards + self.gamma * next_v_values - v_values

        # 액터 신경망 업데이트
        self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                         tf.convert_to_tensor(actions, dtype=tf.float32),
                         tf.convert_to_tensor(advantages, dtype=tf.float32))

        self.batch_state = []
        self.batch_action = []
        self.batch_reward = []
        self.batch_next_state = []
        self.batch_done = []

    def save(self, checkpoint_path: str):
        self.actor.save_weights(checkpoint_path + "_actor.h5")
        self.critic.save_weights(checkpoint_path + "_critic.h5")

    def load(self, checkpoint_path: str):
        self.actor.load_weights(checkpoint_path + '_actor.h5')
        self.critic.load_weights(checkpoint_path + '_critic.h5')

    # 액터 신경망 학습
    def actor_learn(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            # 정책 확률밀도함수
            mu_a, std_a = self.actor(states, training=True)
            log_policy_pdf = log_pdf(mu_a, std_a, actions, std_bound=self.std_bound)

            # 손실함수
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy)

        # 그래디언트
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    # 크리틱 신경망 학습
    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(td_targets - td_hat))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))


