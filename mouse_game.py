from dataclasses import dataclass
from typing import Literal

from tqdm import tqdm


N = 100  # number of mice
MAX_TURNS = 50

type Cheese = Literal["parmesan", "gouda"]

@dataclass
class Action:
    name: str

actions = [
    Action(name="forage"),
    Action(name="forage-specialized")
]


@dataclass
class MouseMemCell:
    action: Action
    reward: float


# TODO: some function that turns cheese into float reward


def main():
    for turn in tqdm(range(MAX_TURNS)):
        pass


if __name__ == "__main__":
    main()
