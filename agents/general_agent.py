

class GeneralAgent:
    def __init__(self, parameters: dict):
        self._config = parameters

    def select_action(self, state):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def save(self, checkpoint_path: str):
        raise NotImplementedError

    def load(self, checkpoint_path: str):
        raise NotImplementedError