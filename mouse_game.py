from dataclasses import dataclass
from typing import Literal

import numpy as np
from tqdm import tqdm


N = 100  # number of mice
MAX_TURNS = 50
MEM = 10  # mouse memory length (in turns)
RANDOM_ACTION_PROB = 0.1  # probability of picking a random action instead of from memory

type Cheese = Literal["parmesan", "gouda"]


@dataclass
class Action:
    name: str


@dataclass
class Exchange:
    give: dict[Cheese, int]
    receive: dict[Cheese, int]


@dataclass
class DealAction(Action):
    exchange: Exchange | None = None


random_action_pool = np.array([
    Action(name="forage"),
    Action(name="forage-specialized"),
    # DealAction(name="propose"),
    # Action(name="accept-any"),
])


@dataclass
class MouseMemCell:
    action: Action
    reward: float


def calculate_reward(belongings: dict[Cheese, int]) -> float:
    """
    Mice need both parmesan and gouda to be happy. 
    More cheese is better, but there are diminishing returns.
    """
    return min(
        belongings.get("parmesan", 0),
        belongings.get("gouda", 0)
    ) ** 0.5


def pick_action(memory: list[MouseMemCell]) -> Action:
    """
    Pick past action with probability proportional to its reward.
    There is also a small chance to pick a random action.
    If memory is not completely filled yet, always pick a random action.
    """
    if len(memory) < MEM:
        return np.random.choice(random_action_pool)

    rewards = np.array([cell.reward for cell in memory])
    probabilities = rewards / rewards.sum()
    if np.random.rand() < RANDOM_ACTION_PROB:  # small chance to pick random action
        return np.random.choice(random_action_pool)
    else:
        return np.random.choice(
            np.array([cell.action for cell in memory]),
            p=probabilities
        )


def main() -> None:
    memories: list[list[MouseMemCell]] = [[]] * N  # what each mouse remembers

    for turn in tqdm(range(MAX_TURNS)):
        # what each mouse has (at turn start: empty)
        belongings: list[dict[Cheese, int]] = [{}] * N

        # pick actions (needs to be done beforehand for deal actions)
        actions = [pick_action(memories[mouse]) for mouse in range(N)]

        # perform actions
        for mouse in np.random.permutation(N):
            # perform action (update belongings)
            # calculate reward
            # update memory

            pass

        # log the actions done and the rewards received


if __name__ == "__main__":
    main()
