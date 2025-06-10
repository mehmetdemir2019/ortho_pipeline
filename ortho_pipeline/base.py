class Stage:
    def __init__(self, cfg): self.cfg = cfg
    def process(self, data): raise NotImplementedError
