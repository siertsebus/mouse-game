import matplotlib.pyplot as plt

from data import Action, Deal


PLOT_ACTIONS: list[str] = [
    "Forage",
    "ForageSpecialized",
    "Deal(Give->Give)",
    "Deal(Give->WorkInFactory)",
    "Deal(WorkInFactory->Give)",
    "AnyDeal",
]
MOVING_AVG_WINDOW = 50


def action_tag(action: Action) -> str:

    match action:
        case Deal(me, you, _):
            return f"Deal({action_tag(me)}->{action_tag(you)})"
        case _:
            return type(action).__name__


class Plotter:
    def __init__(self) -> None:
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.ion()  # Turn on interactive mode

        self.past_action_counts: dict[str, list[int]] = {a: [] for a in PLOT_ACTIONS}
        self.past_action_counts_mavg: dict[str, list[float]] = {
            a: [] for a in PLOT_ACTIONS
        }
        self.past_avg_rewards: list[float] = []
        self.past_avg_rewards_mavg: list[float] = []

    def store(self, actions: list[list["Action"]], rewards: list[float]) -> None:
        self.past_avg_rewards.append(sum(rewards) / len(rewards))
        window = min(MOVING_AVG_WINDOW, len(self.past_avg_rewards))
        self.past_avg_rewards_mavg.append(sum(self.past_avg_rewards[-window:]) / window)

        action_counts = {a: 0 for a in PLOT_ACTIONS}
        for action_list in actions:
            for a in action_list:
                if action_tag(a) in action_counts:
                    action_counts[action_tag(a)] += 1

        for a_type, count in action_counts.items():
            self.past_action_counts[a_type].append(count)
            self.past_action_counts_mavg[a_type].append(
                sum(self.past_action_counts[a_type][-window:]) / window
            )

    def plot(self) -> None:

        ax1, ax2 = self.ax1, self.ax2

        # Clear the axes
        ax1.clear()
        ax2.clear()

        # Plot 1: Average reward over iterations
        iterations = list(range(1, len(self.past_avg_rewards) + 1))

        ax1.plot(
            iterations,
            self.past_avg_rewards_mavg,
            "b-",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Average Reward")
        ax1.set_title("Average Reward Over Time")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Action popularity
        for a_type, counts in self.past_action_counts_mavg.items():
            ax2.plot(iterations, counts, label=str(a_type))
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Action Counts")
        ax2.set_title("Action Popularity Over Time")
        ax2.legend()

        # Adjust layout and refresh
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Small pause to allow plot to update
