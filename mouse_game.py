from dataclasses import dataclass
import random
from typing import Literal, cast

import numpy as np
from tqdm import tqdm


N = 4  # number of mice
MAX_TURNS = 50
MEM = 10  # mouse memory length (in turns)
MUTATION_PROB = 0.1  # probability of mutating an action
RANDOM_ACTION_PROB = 0.1  # probability of picking a random action

N_ACTIONS = 1  # the length of action lists

type Cheese = Literal["parmesan", "gouda"]


@dataclass
class Forage:
    pass


@dataclass
class ForageSpecialized:
    cheese: Cheese


@dataclass
class Give:
    amount: dict[Cheese, int]


@dataclass
class Deal:
    me: "Action"
    you: "Action"


type Action = Forage | ForageSpecialized | Give | Deal


@dataclass
class MouseMemCell:
    actions: list[Action]
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


def pick_action_list(memory: list[MouseMemCell]) -> list[Action]:
    """
    Pick a past action list with probability proportional to its reward.
    There is a small chance of mutating the action list slightly.
    There is also a small chance to pick a completely random action list.
    If memory is not completely filled yet, always pick a random action list.
    """
    if len(memory) < MEM or np.random.rand() < RANDOM_ACTION_PROB:
        return [create_random_action() for _ in range(N_ACTIONS)]

    # pick an action list from memory
    rewards = np.array([cell.reward for cell in memory])
    probabilities = rewards / rewards.sum()
    chosen = random.choices(
        [cell.actions for cell in memory],
        weights=probabilities,
        k=1
    )[0]

    # mutate with some probability
    if np.random.rand() < MUTATION_PROB:
        chosen = cast(list[Action], chosen.copy())
        mutation_index = np.random.randint(len(chosen))
        chosen[mutation_index] = create_random_action()

    return chosen


def perform_action(
    action: Action,
    inventory: dict[Cheese, int],
    done_actions: list[Action]
) -> dict[Cheese, int]:
    gain: dict[Cheese, int]
    match action:
        case Forage():
            gain = (
                {"parmesan": 1, "gouda": 1}
                if action not in done_actions  # forage only once per turn
                else {}
            )
        case ForageSpecialized(cheese):
            gain = (
                {cheese: 4}
                if action not in done_actions  # forage only once per turn
                else {}
            )
        case _:
            raise NotImplementedError(f"Action {action} not implemented")
    
    # update inventory
    for cheese, amount in gain.items():
        inventory[cheese] = inventory.get(cheese, 0) + amount
    
    return inventory


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
        actions = [pick_action_list(memories[mouse]) for mouse in range(N)]
        inventories: list[dict[Cheese, int]] = [dict() for _ in range(N)]

        # perform actions
        for i in range(N_ACTIONS):
            for mouse in np.random.permutation(N):
                # perform action (update inventory)
                action = actions[mouse][i]
                inventory = inventories[mouse]
                inventories[mouse] = perform_action(
                    action, inventory, actions[mouse][:i]
                )

        # calculate reward
        for mouse in range(N):
            reward = calculate_reward(inventories[mouse])

            # update memory
            memory = memories[mouse]
            memory.append(MouseMemCell(actions[mouse], reward))
            if len(memory) > MEM:  # keep memory size constant
                memory.pop(0)

        print(
            f"Turn {turn}: "
            f"{[(mem[-1].actions, mem[-1].reward)
                for mem in memories if mem]}"
        )


if __name__ == "__main__":
    main()
