import copy
import numpy as np
import torch
import torch.nn as nn
from agents.pytorch.utilities import get_device
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from agents.general_agent import PolicyAgent

class Discrete_SAC(PolicyAgent):
    def __init__(self, parameters: dict, actor: nn.Module, critic: nn.Module):
        device = get_device("auto")
        super(Discrete_SAC, self).__init__(parameters=parameters, actor=actor.to(device), critic=critic.to(device))
        self.critic2 = copy.deepcopy(critic).to(device)
        self.target_critic = copy.deepcopy(critic).to(device)
        self.target_critic2 = copy.deepcopy(critic).to(device)

        # HyperParameter
        self.gamma = self._config['gamma']
        self.tau = self._config['tau']
        self.lr = self._config['lr']
        self.clip_grad_param = self._config['clip_grad_param']

        # Entropy Temperature alpha
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()

        # Optimizer
        opt_arg = [
            {'params': self.actor.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr},
            {'params': self.critic2.parameters(), 'lr': self.lr},
            {'params': self.target_critic.parameters(), 'lr': self.lr},
            {'params': self.target_critic2.parameters(), 'lr': self.lr},
            {'params': self.log_alpha, 'lr': self.lr}
        ]
        self.optimizer = getattr(torch.optim, parameters['optimizer'])(opt_arg)
        self.critic2.load_state_dict(self.critic.state_dict())
        self.loss = getattr(nn, parameters['loss_function'])()
        self.device = device
        self.hidden_state = copy.deepcopy(self.actor.init_h_state)
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.to(self.device)

    def select_action(self, state):
        state = self.convert_to_torch(state)
        with torch.no_grad():
            if len(state['action_mask']) > 0:
                self.set_mask(state['action_mask'])
            output, actions, action_logprobs, next_hidden = self.act(state=state, hidden=self.hidden_state)

        self.batch_state_matrix.append(state['matrix'])
        self.batch_state_vector.append(state['vector'])

        self.batch_action.append(actions)
        self.batch_actions.append(output)

        self.batch_hidden_state.append(next_hidden)
        self.batch_log_old_policy_pdf.append(action_logprobs)
        self.hidden_state = next_hidden

        return actions

    def act(self, state, hidden=None):
        rtn_action = []
        action_logprob = None
        outputs, hidden = self.actor(x=state, h=hidden)
        last = 0
        if len(self.actor.action_mask) > 0:
            outputs *= self.actor.action_mask
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            dist = Categorical(outputs[:, last:last + output_dim])
            action = dist.sample()
            if action_logprob is None:
                action_logprob = dist.log_prob(action)
            else:
                action_logprob += dist.log_prob(action)
            rtn_action.append(action.detach())
            last += output_dim
        return outputs, torch.stack(rtn_action, dim=0).detach(), action_logprob.detach(), hidden

    def evaluate(self, state, actions, hidden=None):
        outputs, _ = self.actor(x=state, h=hidden)
        last = 0
        action_logprobs = None
        dist_entropy = None
        for idx, output_dim in enumerate(self.actor.outputs_dim):
            action = actions[:, idx, :].flatten()
            dist = Categorical(outputs[:, last:last + output_dim])

            if action_logprobs is None:
                action_logprobs = dist.log_prob(action)
                dist_entropy = dist.entropy()
            else:
                action_logprobs += dist.log_prob(action)
                dist_entropy += dist.entropy()
            last += output_dim

        # action에 대한 log(prob), entropy에 대한 log(prob) 뱉어냄.
        return outputs, action_logprobs, dist_entropy

    def update(self, next_state=None, done=None):
        # Monte Carlo estimate of returns
        # Agent 수 만큼 생성
        converted_rewards = np.zeros(self.batch_reward[0].shape[0])
        rewards = np.zeros((len(self.batch_reward), self.batch_reward[0].shape[0]))
        batch_count = len(self.batch_reward) - 1
        # b, n 구조로 계산
        for reward, is_terminal in zip(reversed(self.batch_reward), reversed(self.batch_done)):
            # batch iteration n about r(or d) shape
            for idx in range(reward.shape[0]):
                if is_terminal[idx]:
                    converted_rewards[idx] = 0
                converted_rewards[idx] = reward[idx] + (converted_rewards[idx])
                rewards[batch_count, idx] = converted_rewards[idx]
            batch_count -= 1

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = rewards.flatten()
        # convert list to tensor
        part_matrix = None
        next_part_matrix = None
        if 'spatial_feature' in self.actor.networks:
            part_matrix = torch.cat(self.batch_state_matrix, dim=0).detach().to(self.device)
            next_part_matrix = torch.cat(self.batch_next_state_matrix, dim=0).detach().to(self.device)
        part_vector = torch.cat(self.batch_state_vector, dim=0).detach().to(self.device)
        next_part_vector = torch.cat(self.batch_next_state_vector, dim=0).detach().to(self.device)

        old_states = {'matrix': part_matrix, 'vector': part_vector}
        old_action = torch.stack(self.batch_action, dim=0).detach().to(self.device)
        old_actions = torch.stack(self.batch_actions, dim=0).detach().to(self.device)
        next_states = {'matrix': next_part_matrix, 'vector': next_part_vector}
        old_logprobs = torch.stack(self.batch_log_old_policy_pdf, dim=0).flatten().to(self.device)
        old_hiddens = None
        if self.actor.recurrent:
            old_hiddens = self.batch_hidden_state[-1].detach().to(self.device)

        dump = torch.zeros(len(rewards), 1)
        metrics = {'reward': dump, 'entropy': dump, 'actor_loss': dump, 'alpha_loss': dump,
                   'critic1_loss': dump, 'critic2_loss': dump}
        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(old_states, old_action, current_alpha.to(self.device))

        # Compute alpha loss
        alpha_loss = -(self.log_alpha.exp() * (log_pis.cpu()).detach().cpu()).mean()

        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            action_probs, log_pis, dist_entropy = self.evaluate(next_states, old_action, old_hiddens)
            Q_target1_next, _ = self.target_critic(next_states)
            Q_target2_next, _ = self.target_critic2(next_states)
            Q_target_next = (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_target = rewards + (self.gamma * Q_target_next.unsqueeze(-1))

        # Compute critic loss
        c1, _ = self.critic(old_states)
        c2, _ = self.critic2(old_states)
        q1 = None
        q2 = None
        splited_actions = old_actions.split(1, dim=1)
        for a in splited_actions:
            splited_action = a.squeeze(2)
            q1 = c1.gather(1, splited_action.long())
            q2 = c1.gather(1, splited_action.long())
#196, 1 vs 196, 196 문제 생김 ㅅㅂ
        critic1_loss = 0.5 * self.loss(q1, Q_target)
        critic2_loss = 0.5 * self.loss(q2, Q_target)

        # take gradient step
        self.optimizer.zero_grad()
        actor_loss.backward()
        alpha_loss.backward()
        critic1_loss.backward()
        critic2_loss.backward()

        clip_grad_norm_(self.critic.parameters(), self.clip_grad_param)
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)

        self.alpha = self.log_alpha.exp().detach()

        self.optimizer.step()

        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.critic2, self.target_critic2)

        metrics = {'reward': rewards.detach().cpu(),
                   'entropy': dist_entropy.detach().cpu(),
                   'actor_loss': actor_loss.detach().cpu(),
                   'alpha_loss': alpha_loss.detach().cpu(),
                   'critic1_loss': critic1_loss.detach().cpu(),
                   'critic2_loss': critic2_loss.detach().cpu(),}

        # clear buffer
        self.batch_clear()

        # insert metric
        self.insert_metrics(metrics)

    def calc_policy_loss(self, states, actions, alpha):
        action_probs, log_pis, dist_entropy = self.evaluate(states, actions)

        q1, _ = self.critic(states)
        q2, _ = self.critic2(states)
        minQ = torch.min(q1, q2)

        actor_loss = (action_probs * ((alpha * log_pis) - minQ)).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=0)

        return actor_loss, log_action_pi

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, checkpoint_path: str):
        if ".pth" not in checkpoint_path:
            checkpoint_path = checkpoint_path + '.pth'
        actor_path = checkpoint_path.replace(".pth", "actor.pth")
        critic_path = checkpoint_path.replace(".pth", "critic.pth")
        torch.save(self.critic2.state_dict(), critic_path)

    def load(self, checkpoint_path: str):
        if "actor" in checkpoint_path:
            actor_path = checkpoint_path
            critic_path = checkpoint_path.replace("actor.pth", "critic.pth")
        elif "critic" in checkpoint_path:
            critic_path = checkpoint_path
            actor_path = checkpoint_path.replace("critic.pth", "actor.pth")

        self.actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(critic_path, map_location=lambda storage, loc: storage))
        self.critic2.load_state_dict(torch.load(critic_path, map_location=lambda storage, loc: storage))

    def convert_to_torch(self, state):
        spatial_x = state['matrix']
        non_spatial_x = state['vector']
        mask = state['action_mask']

        if torch.is_tensor(non_spatial_x) is False:
            non_spatial_x = torch.FloatTensor(non_spatial_x)
        non_spatial_x = non_spatial_x.to(self.device)

        if torch.is_tensor(spatial_x) is False:
            if len(spatial_x) > 0:
                spatial_x = torch.FloatTensor(spatial_x).to(self.device)
        else:
            spatial_x = spatial_x.to(self.device)

        if torch.is_tensor(mask) is False:
            if len(mask) > 0:
                mask = torch.FloatTensor(mask).to(self.device)
                mask = mask.unsqueeze(dim=0)
        else:
            mask = mask.to(self.device)

        state['matrix'] = spatial_x
        state['vector'] = non_spatial_x
        state['action_mask'] = mask

        return state

    def set_mask(self, mask):
        if len(mask) > 0:
            self.actor.action_mask = mask
