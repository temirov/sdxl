from dataclasses import dataclass


@dataclass
class CNModel:
    version: str
    description: str
    model_path: str

    def __repr__(self):
        return f"ControlNet {self.description} {self.version}"
