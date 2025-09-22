from dataclasses import dataclass
import random
from typing import Literal, cast

import numpy as np
from tqdm import tqdm


N = 20  # number of mice
MAX_TURNS = 10000
MEM = 100  # mouse memory length (in turns)
MUTATION_PROB = 0.1  # probability of mutating an action
RANDOM_ACTION_PROB = 0.1  # probability of picking a random action
GIVE_MIN, GIVE_MAX = 0, 2

N_ACTIONS = 2  # the length of action lists

type Cheese = Literal["parmesan", "gouda"]

CHEESES: list[Cheese] = ["parmesan", "gouda"]


@dataclass
class Forage:
    pass


@dataclass
class ForageSpecialized:
    cheese: Cheese


@dataclass
class Give:
    items: dict[Cheese, int]


@dataclass
class Deal:
    me: "Action"
    you: "Action"
    done: bool = False


@dataclass
class AnyDeal:
    pass


type Action = Forage | ForageSpecialized | Give | Deal | AnyDeal


def flip_deal(deal: Deal) -> Deal:
    return Deal(me=deal.you, you=deal.me, done=deal.done)


def match_deals(a1: Action, a2: Action) -> tuple[Deal, Deal] | None:
    """Check if the given two actions are matching deals.
    If so, return their concrete versions. E.g. that means that if given
    Deal(a, b) and AnyDeal then return (Deal(a, b), Deal(b, a)).
    """
    match a1:
        case Deal(me1, you1, done=False):
            match a2:
                case Deal(me2, you2, done=False):
                    return (a1, a2) if (me1 == you2 and me2 == you1) else None
                case AnyDeal():
                    return (a1, flip_deal(a1))
                case _:
                    return None
        case AnyDeal():
            match a2:
                case Deal(me2, you2, done=False):
                    return (flip_deal(a2), a2)
                case AnyDeal():
                    return None
                case _:
                    return None
        case _:
            return None


@dataclass
class MouseMemCell:
    actions: list[Action]
    reward: float


random_action_pool: list[type[Action]] = [
    # Forage,
    ForageSpecialized,
    Deal,
    AnyDeal,
]


deal_random_action_pool: list[type[Action]] = [Give]
# ^ keep in sync with random_deal_action and valid_offer


def valid_offer(deal: Deal, inventory: dict[Cheese, int]) -> bool:
    match deal.me:
        case Give(items):
            return all(inventory[cheese] >= items.get(cheese, 0) for cheese in CHEESES)
        case _:
            return False


@dataclass
class TurnState:
    memories: list[list[MouseMemCell]]
    actions: list[list[Action]]
    inventories: list[dict[Cheese, int]]


def create_random_action() -> Action:
    action_type = random.choice(random_action_pool)
    if action_type is Forage:
        return Forage()
    elif action_type is ForageSpecialized:
        cheese: Cheese = random.choice(["parmesan", "gouda"])
        return ForageSpecialized(cheese=cheese)
    elif action_type is Deal:
        return Deal(me=create_random_deal_action(), you=create_random_deal_action())
    elif action_type is AnyDeal:
        return AnyDeal()
    else:
        raise NotImplementedError(f"Action type {action_type} not implemented")


def create_random_deal_action() -> Action:
    action_type = random.choice(deal_random_action_pool)
    if action_type is Give:
        return Give(
            {
                "parmesan": random.randint(GIVE_MIN, GIVE_MAX),
                "gouda": random.randint(GIVE_MIN, GIVE_MAX),
            }
        )
    else:
        raise NotImplementedError(f"Deal action type {action_type} not implemented")


def pick_action_list(memory: list[MouseMemCell]) -> list[Action]:
    """
    Pick a past action list with probability proportional to its reward.
    There is a small chance of mutating the action list slightly.
    There is also a small chance to pick a completely random action list.
    If memory is not completely filled yet, always pick a random action list.
    """
    rewards = np.array([cell.reward for cell in memory])
    if len(memory) < MEM or np.random.rand() < RANDOM_ACTION_PROB or rewards.sum() == 0:
        return [create_random_action() for _ in range(N_ACTIONS)]

    # pick an action list from memory
    probabilities = rewards / rewards.sum()
    chosen = random.choices(
        [cell.actions for cell in memory], weights=probabilities, k=1
    )[0]

    # reset actions
    for action in chosen:
        reset_action(action)

    # mutate with some probability
    if np.random.rand() < MUTATION_PROB:
        chosen = cast(list[Action], chosen.copy())
        mutation_index = np.random.randint(len(chosen))
        chosen[mutation_index] = create_random_action()

    return chosen


def reset_action(action: Action) -> None:
    match action:
        case Deal(_, _, done=True):
            action.done = False  # When doing a memorized action again, reset to undone


def perform_action(
    mouse: int,
    action: Action,
    state: TurnState,
    action_idx: int | None = None,
    target_mouse: int | None = None,  # The target for e.g. the Give action
):
    inventory = state.inventories[mouse]
    done_actions = state.actions[mouse][:action_idx] if action_idx else []

    def contains_forage(actions: list[Action]) -> bool:
        return any(
            isinstance(a, Forage) or isinstance(a, ForageSpecialized) for a in actions
        )

    match action:
        case Forage():
            if not contains_forage(done_actions):  # forage only once per turn
                inventory["parmesan"] += 1
                inventory["gouda"] += 1

        case ForageSpecialized(cheese):
            if not contains_forage(done_actions):  # forage only once per turn
                inventory[cheese] += 4

        case Deal(_, _, done=False) | AnyDeal():
            if action_idx is not None:  # Deal requires an action_idx
                for other_mouse in range(N):
                    other_mouse_action = state.actions[other_mouse][action_idx]
                    deals = match_deals(action, other_mouse_action)
                    if (
                        deals is not None  # not None means a match
                        and valid_offer(deals[0], inventory)
                        and valid_offer(deals[1], state.inventories[other_mouse])
                    ):
                        perform_action(
                            mouse, deals[0].me, state, target_mouse=other_mouse
                        )
                        deals[0].done = True
                        state.actions[mouse][action_idx] = deals[0]
                        perform_action(
                            other_mouse, deals[0].you, state, target_mouse=mouse
                        )
                        deals[1].done = True
                        state.actions[other_mouse][action_idx] = deals[1]
                        break

        case Give(items):
            if target_mouse is not None and all(  # give requires a target mouse
                inventory[cheese] >= items.get(cheese, 0) for cheese in CHEESES
            ):  # you cannot give what you don't have
                target_mouse_inventory = state.inventories[target_mouse]
                for cheese, amount in items.items():
                    inventory[cheese] -= amount
                    target_mouse_inventory[cheese] += amount

        case _:
            pass  # Do nothing


def calculate_reward(inventory: dict[Cheese, int]) -> float:
    """
    Mice need both parmesan and gouda to be happy.
    More cheese is better, but there are diminishing returns.
    """
    return min(inventory.get("parmesan", 0), inventory.get("gouda", 0)) ** 0.5


def main() -> None:
    from plotter import Plotter

    plotter = Plotter()

    memories: list[list[MouseMemCell]] = [[] for _ in range(N)]

    for turn in tqdm(range(MAX_TURNS)):
        # pick actions (needs to be done beforehand for deal actions)
        actions = [pick_action_list(memories[mouse]) for mouse in range(N)]
        inventories: list[dict[Cheese, int]] = [
            {"parmesan": 0, "gouda": 0} for _ in range(N)
        ]
        state = TurnState(memories, actions, inventories)

        # perform actions
        # -> start with first action for all mice, then second action, etc.
        for i in range(N_ACTIONS):
            for mouse in np.random.permutation(N):
                action = actions[mouse][i]
                perform_action(mouse, action, state, action_idx=i)

        # calculate reward
        rewards = [-1.0] * N
        for mouse in range(N):
            reward = calculate_reward(inventories[mouse])
            rewards[mouse] = reward

            # update memory
            memory = memories[mouse]
            memory.append(MouseMemCell(actions[mouse], reward))
            if len(memory) > MEM:  # keep memory size constant
                memory.pop(0)

        # display results
        plotter.store(actions, rewards)
        if turn % 100 == 0:
            plotter.plot()

        # print([mem[-1].actions for mem in memories])

    input("Press Enter to close...")


if __name__ == "__main__":
    main()
