import torch
import torch.nn as nn
from agents.general_agent import GeneralAgent


class PPO(GeneralAgent):
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                 has_continuous_action_space=False, action_std_init=0.6):

        device = get_device("auto")
        policy = ActorCritic(state_dim, action_dim, has_continuous_action_space,
                             action_std_init, device).to(device)

        policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space,
                                 action_std_init, device).to(device)

        parameters = {'policy': policy, 'policy_old': policy_old}

        # 습관이지만 파이썬에서는 어떤 것의 super인지 명명하는 것이 중요(다중 상속에 용이하게 사용됨)
        super(PPO, self).__init__(parameters=parameters)

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.optimizer = torch.optim.Adam([
            {'params': self._policy.actor.parameters(), 'lr': lr_actor},
            {'params': self._policy.critic.parameters(), 'lr': lr_critic}
        ])

        self._policy_old.load_state_dict(self._policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.device = device

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self._policy_old.act(state)

            self.buffer.States.append(state)
            self.buffer.Actions.append(action)
            self.buffer.LogProbs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self._policy_old.act(state)
            self.buffer.States.append(state)
            self.buffer.Actions.append(action)
            self.buffer.LogProbs.append(action_logprob)
            # 하나의 액션이 나오도록 *****
            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.Rewards), reversed(self.buffer.IsTerminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.States, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.Actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.LogProbs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self._policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self._policy_old.load_state_dict(self._policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self._policy.set_action_std(new_action_std)
            self._policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
            print("-------------------------------------------------------------------------------------------")


class TripleActorPPO:
    def __init__(self, state_dim, alpha_action_dim, beta_action_dim, theta_action_dim,
                 lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                 has_continuous_action_space=False, action_std_init=0.6, pause_mode=False):

        self.device = get_device("auto")
        self.__alpha_policy = ActorCritic(state_dim, alpha_action_dim, has_continuous_action_space,
                                          action_std_init, self.device).to(self.device)
        self.__alpha_policy_old = ActorCritic(state_dim, alpha_action_dim, has_continuous_action_space,
                                              action_std_init, self.device).to(self.device)

        self.__beta_policy = ActorCritic(state_dim, beta_action_dim, has_continuous_action_space,
                                         action_std_init, self.device).to(self.device)
        self.__beta_policy.critic = self.__alpha_policy.critic

        self.__beta_policy_old = ActorCritic(state_dim, beta_action_dim, has_continuous_action_space,
                                             action_std_init, self.device).to(self.device)
        self.__beta_policy_old.critic = self.__alpha_policy_old.critic

        self.__theta_policy = ActorCritic(state_dim, theta_action_dim, has_continuous_action_space,
                                          action_std_init, self.device).to(self.device)
        self.__theta_policy.critic = self.__alpha_policy.critic

        self.__theta_policy_old = ActorCritic(state_dim, theta_action_dim, has_continuous_action_space,
                                              action_std_init, self.device).to(self.device)
        self.__theta_policy_old.critic = self.__alpha_policy_old.critic

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()
        self.__buffer_clear()

        self.optimizer = torch.optim.Adam([
            {'params': self.__alpha_policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.__beta_policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.__theta_policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.__alpha_policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.__sync_policy()
        self.MseLoss = nn.MSELoss()
        self.__pause_mode = pause_mode

    def select_action(self, state):
        if self.has_continuous_action_space:
            raise NotImplementedError
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                a_action, a_action_logprob = self.__alpha_policy_old.act(state)
                b_action, b_action_logprob = self.__beta_policy_old.act(state)
                c_action, c_action_logprob = self.__theta_policy_old.act(state)

            if self.__pause_mode:
                self.buffer.States.append(state)
                alpha_actions: list = self.buffer.Actions[0]
                alpha_actions.append(a_action)
                beta_actions: list = self.buffer.Actions[1]
                beta_actions.append(b_action)
                theta_actions: list = self.buffer.Actions[2]
                theta_actions.append(c_action)

                alpha_logprobs: list = self.buffer.LogProbs[0]
                alpha_logprobs.append(a_action_logprob)
                beta_logprobs: list = self.buffer.LogProbs[1]
                beta_logprobs.append(b_action_logprob)
                theta_logprobs: list = self.buffer.LogProbs[2]
                theta_logprobs.append(c_action_logprob)

                return a_action.item(), b_action.item(), c_action.item()
            else:
                return (a_action, a_action_logprob), (b_action, b_action_logprob), (c_action, c_action_logprob)

    # def select_action(self, state):
    #     if self.has_continuous_action_space:
    #         raise NotImplementedError
    #     else:
    #         with torch.no_grad():
    #             state = torch.FloatTensor(state).to(self.device)
    #             a_action, a_action_logprob = self.__alpha_policy_old.act(state)
    #             b_action, b_action_logprob = self.__beta_policy_old.act(state)
    #             c_action, c_action_logprob = self.__theta_policy_old.act(state)
    #
    #         if self.__pause_mode:
    #             self.buffer.States.append(state)
    #             alpha_actions: list = self.buffer.Actions[0]
    #             alpha_actions.append(a_action)
    #             beta_actions: list = self.buffer.Actions[1]
    #             beta_actions.append(b_action)
    #             theta_actions: list = self.buffer.Actions[2]
    #             theta_actions.append(c_action)
    #
    #             alpha_logprobs: list = self.buffer.LogProbs[0]
    #             alpha_logprobs.append(a_action_logprob)
    #             beta_logprobs: list = self.buffer.LogProbs[1]
    #             beta_logprobs.append(b_action_logprob)
    #             theta_logprobs: list = self.buffer.LogProbs[2]
    #             theta_logprobs.append(c_action_logprob)
    #
    #             return a_action.item(), b_action.item(), c_action.item()
    #         else:
    #             return (a_action, a_action_logprob), (b_action, b_action_logprob), (c_action, c_action_logprob)

    def dummy_action(self):
        return NotImplementedError

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.Rewards), reversed(self.buffer.IsTerminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        alpha_actions: list = self.buffer.Actions[0]
        beta_actions: list = self.buffer.Actions[1]
        theta_actions: list = self.buffer.Actions[2]
        alpha_logprobs: list = self.buffer.LogProbs[0]
        beta_logprobs: list = self.buffer.LogProbs[1]
        theta_logprobs: list = self.buffer.LogProbs[2]

        old_states = torch.squeeze(torch.stack(self.buffer.States, dim=0)).detach().to(self.device)
        old_alpha_actions = torch.squeeze(torch.stack(alpha_actions, dim=0)).detach().to(self.device)
        old_alpha_logprobs = torch.squeeze(torch.stack(alpha_logprobs, dim=0)).detach().to(self.device)
        old_beta_actions = torch.squeeze(torch.stack(beta_actions, dim=0)).detach().to(self.device)
        old_beta_logprobs = torch.squeeze(torch.stack(beta_logprobs, dim=0)).detach().to(self.device)
        old_theta_actions = torch.squeeze(torch.stack(theta_actions, dim=0)).detach().to(self.device)
        old_theta_logprobs = torch.squeeze(torch.stack(theta_logprobs, dim=0)).detach().to(self.device)
        old_logprobs = old_alpha_logprobs + old_beta_logprobs + old_theta_logprobs
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            a_logprobs, state_values, a_dist_entropy = self.__alpha_policy.evaluate(old_states, old_alpha_actions)
            b_logprobs, _, b_dist_entropy = self.__beta_policy.evaluate(old_states, old_beta_actions)
            c_logprobs, _, c_dist_entropy = self.__theta_policy.evaluate(old_states, old_theta_actions)

            logprobs = a_logprobs + b_logprobs + c_logprobs
            dist_entropy = a_dist_entropy + b_dist_entropy + c_dist_entropy
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            # check 필요
            # self.__beta_policy.critic = self.__alpha_policy.critic

        # Copy new weights into old policy
        self.__sync_policy()

        # clear buffer
        self.__buffer_clear()

    def save(self, checkpoint_path: str):
        alpha_path = checkpoint_path.replace(".pth", "alpha.pth")
        beta_path = checkpoint_path.replace(".pth", "beta.pth")
        theta_path = checkpoint_path.replace(".pth", "theta.pth")
        torch.save(self.__alpha_policy_old.state_dict(), alpha_path)
        torch.save(self.__beta_policy_old.state_dict(), beta_path)
        torch.save(self.__theta_policy_old.state_dict(), theta_path)

    def load(self, checkpoint_path: str):
        if "alpha" in checkpoint_path:
            alpha_path = checkpoint_path
            beta_path = checkpoint_path.replace("alpha.pth", "beta.pth")
            theta_path = checkpoint_path.replace("alpha.pth", "theta.pth")
        elif "beta" in checkpoint_path:
            beta_path = checkpoint_path
            alpha_path = checkpoint_path.replace("beta.pth", "alpha.pth")
            theta_path = checkpoint_path.replace("beta.pth", "theta.pth")
        else:
            theta_path = checkpoint_path
            alpha_path = checkpoint_path.replace("theta.pth", "alpha.pth")
            beta_path = checkpoint_path.replace("theta.pth", "beta.pth")

        self.__alpha_policy_old.load_state_dict(torch.load(alpha_path, map_location=lambda storage, loc: storage))
        self.__alpha_policy.load_state_dict(torch.load(alpha_path, map_location=lambda storage, loc: storage))
        self.__beta_policy_old.load_state_dict(torch.load(beta_path, map_location=lambda storage, loc: storage))
        self.__beta_policy.load_state_dict(torch.load(beta_path, map_location=lambda storage, loc: storage))
        self.__theta_policy_old.load_state_dict(torch.load(theta_path, map_location=lambda storage, loc: storage))
        self.__theta_policy.load_state_dict(torch.load(theta_path, map_location=lambda storage, loc: storage))

    def __sync_policy(self):
        self.__alpha_policy_old.load_state_dict(self.__alpha_policy.state_dict())
        self.__beta_policy_old.load_state_dict(self.__beta_policy.state_dict())
        self.__theta_policy_old.load_state_dict(self.__theta_policy.state_dict())

    def __buffer_clear(self):
        self.buffer.clear()
        self.buffer.Actions.append([])
        self.buffer.Actions.append([])
        self.buffer.Actions.append([])
        self.buffer.LogProbs.append([])
        self.buffer.LogProbs.append([])
        self.buffer.LogProbs.append([])