from importlib import metadata
class Pipeline:
    def __init__(self, cfg_path):
        import yaml, pathlib
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.stages = {e.name: e.load() for e in
                       metadata.entry_points(group="ortho_pipeline.stages")}
    def run(self):
        data = None
        for s in self.cfg["order"]:
            data = self.stages[s](self.cfg[s]).process(data)
        return data
