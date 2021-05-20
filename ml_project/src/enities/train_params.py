from dataclasses import dataclass, field


@dataclass()
class TrainParams:
    model_type: str = field(default="LogisticRegression")
