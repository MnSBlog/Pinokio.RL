from agents.tf.actorcritic import Actor, Critic


class GeneralAgent:
    def __init__(self, parameters: dict):
        self._config = parameters
        self.exconfig = dict()
        # 배치 설정
        self.batch_state = []
        self.batch_action = []
        self.batch_reward = []
        self.batch_next_state = []
        self.batch_done = []
        self.batch_log_old_policy_pdf = []

    def select_action(self, state):
        raise NotImplementedError

    def update(self, next_state=None, done=None):
        raise NotImplementedError

    def save(self, checkpoint_path: str):
        raise NotImplementedError

    def load(self, checkpoint_path: str):
        raise NotImplementedError


class PolicyAgent(GeneralAgent):
    def __init__(self, parameters: dict, actor, critic):
        super(PolicyAgent, self).__init__(parameters=parameters)
        self.actor = actor
        self.critic = critic

    def select_action(self, state):
        return super(PolicyAgent, self).select_action(state)

    def update(self, next_state=None, done=None):
        return super(PolicyAgent, self).update()

    def save(self, checkpoint_path: str):
        return super(PolicyAgent, self).save(checkpoint_path=checkpoint_path)

    def load(self, checkpoint_path: str):
        return super(PolicyAgent, self).load(checkpoint_path=checkpoint_path)
