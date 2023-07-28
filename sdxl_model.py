from dataclasses import dataclass


@dataclass
class SdxlModel:
    version: str
    base_model_path: str
    refiner_model_path: str

    def __repr__(self):
        return f"SD XL {self.version}"
