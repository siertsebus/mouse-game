from dataclasses import dataclass
import random
from typing import Literal

import numpy as np
from tqdm import tqdm


N = 4  # number of mice
MAX_TURNS = 50
MEM = 10  # mouse memory length (in turns)
RANDOM_ACTION_PROB = 0.1  # probability of picking a random action instead of from memory

type Cheese = Literal["parmesan", "gouda"]


@dataclass
class Forage:
    pass


@dataclass
class ForageSpecialized:
    cheese: Cheese


@dataclass
class Exchange:
    give: dict[Cheese, int]
    receive: dict[Cheese, int]


@dataclass
class DealAction:
    exchange: Exchange | None = None


type Action = Forage | ForageSpecialized | DealAction


@dataclass
class MouseMemCell:
    action: Action
    reward: float


random_action_pool: list[type[Action]] = [
    Forage,
    ForageSpecialized,
    # DealAction(name="propose"),
    # Action(name="accept-any"),
]


def create_random_action() -> Action:
    action_type = random.choice(random_action_pool)
    if action_type is Forage:
        return Forage()
    elif action_type is ForageSpecialized:
        cheese: Cheese = random.choice(["parmesan", "gouda"])
        return ForageSpecialized(cheese=cheese)
    else:
        raise NotImplementedError(f"Action type {action_type} not implemented")


def pick_action(memory: list[MouseMemCell]) -> Action:
    """
    Pick past action with probability proportional to its reward.
    There is also a small chance to pick a random action.
    If memory is not completely filled yet, always pick a random action.
    """
    if len(memory) < MEM:
        return create_random_action()

    rewards = np.array([cell.reward for cell in memory])
    probabilities = rewards / rewards.sum()
    if np.random.rand() < RANDOM_ACTION_PROB:
        return create_random_action()
    else:
        return np.random.choice(
            np.array([cell.action for cell in memory]),
            p=probabilities
        )


def perform_action(action: Action) -> dict[Cheese, int]:
    match action:
        case Forage():
            return {"parmesan": 1, "gouda": 1}
        case ForageSpecialized(cheese):
            return {cheese: 4}
        case _:
            raise NotImplementedError(f"Action {action} not implemented")


def calculate_reward(inventory: dict[Cheese, int]) -> float:
    """
    Mice need both parmesan and gouda to be happy.
    More cheese is better, but there are diminishing returns.
    """
    return min(
        inventory.get("parmesan", 0),
        inventory.get("gouda", 0)
    ) ** 0.5


def main() -> None:
    memories: list[list[MouseMemCell]] = [[] for _ in range(N)]

    for turn in tqdm(range(MAX_TURNS)):
        # pick actions (needs to be done beforehand for deal actions)
        actions = [pick_action(memories[mouse]) for mouse in range(N)]

        # perform actions
        for mouse in np.random.permutation(N):
            # perform action (update inventory)
            action = actions[mouse]
            inventory = perform_action(action)

            # calculate reward
            reward = calculate_reward(inventory)

            # update memory
            memory = memories[mouse]
            memory.append(MouseMemCell(action, reward))
            if len(memory) > MEM:
                memory.pop(0)

            pass

        print(
            f"Turn {turn}: "
            f"{[(mem[-1].action, mem[-1].reward)
                for mem in memories if mem]}"
        )


if __name__ == "__main__":
    main()
