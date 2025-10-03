
from dataclasses import dataclass
from typing import Literal


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


def flip_deal(deal: Deal) -> Deal:
    return Deal(me=deal.you, you=deal.me, done=deal.done)


@dataclass
class AnyDeal:
    pass


@dataclass
class Trade:
    pass


@dataclass
class RunFactory:
    cheese: Cheese
    pass


@dataclass
class WorkInFactory:
    cheese: Cheese


type ChoiceAction = Forage | ForageSpecialized | AnyDeal | Trade | RunFactory
type Action = Forage | ForageSpecialized | Give | Deal | AnyDeal | WorkInFactory