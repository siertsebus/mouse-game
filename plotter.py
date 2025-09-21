import matplotlib.pyplot as plt
# from collections import Counter

from mouse_game import Action


class Plotter:
    def __init__(self) -> None:
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.ion()  # Turn on interactive mode

        self.past_actions: list[list[list["Action"]]] = []
        self.past_avg_rewards: list[float] = []

    def store(
        self, actions: list[list["Action"]], rewards: list[float]
    ) -> None:
        self.past_actions.append(actions)
        self.past_avg_rewards.append(sum(rewards) / len(rewards))

    def plot(self) -> None:

        ax1, ax2 = self.ax1, self.ax2

        # Clear the axes
        ax1.clear()
        ax2.clear()
        
        # Plot 1: Average reward over iterations
        iterations = list(range(1, len(self.past_avg_rewards) + 1))
        ax1.plot(iterations, self.past_avg_rewards, 'b-',
                 linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Average Reward Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Action popularity
        # Count all actions across all iterations
        # all_actions = []
        # for iteration_actions in actions:
        #     for action in iteration_actions:
        #         all_actions.append(str(action))
        
        # if all_actions:
        #     action_counts = Counter(all_actions)
        #     action_names = list(action_counts.keys())
        #     action_frequencies = list(action_counts.values())
            
        #     # Create bar plot
        #     bars = ax2.bar(action_names, action_frequencies, 
        #                 color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            
        #     ax2.set_xlabel('Action Type')
        #     ax2.set_ylabel('Frequency')
        #     ax2.set_title('Action Popularity')
        #     ax2.tick_params(axis='x', rotation=45)
            
        #     # Add value labels on top of bars
        #     for bar, freq in zip(bars, action_frequencies):
        #         ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
        #                 str(freq), ha='center', va='bottom', fontweight='bold')
    
        # Adjust layout and refresh
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Small pause to allow plot to update
