import matplotlib.pyplot as plt
class QVisualiser:
    def plot_learning_curve(self, rewards):
        """
        Plot the learning curve (rewards over episodes)
        
        Args:
            rewards (list): List of rewards per episode
        """
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title('Q-Learning Performance')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.show()