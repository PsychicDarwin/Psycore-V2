# Base Q Learning Class that contains the Q Learning information to be used by the QAgent and QTrainer classes
import numpy as np
import random

#this will switch between the q leanring training mode and the q learning agent decider 


class QModel:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize Q-learning parameters
        
        Args:
            states (int): Number of states in the environment
            actions (int): Number of possible actions
            alpha (float): Learning rate (0 to 1)
            gamma (float): Discount factor (0 to 1)
            epsilon (float): Exploration rate (0 to 1)
        """
        self.q_table = np.zeros((states, actions))
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.states = states
        self.actions = actions
        self.rewards_history = []
        
    def choose_action(self, state):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state (int): Current state of the agent
            
        Returns:
            int: Selected action
        """
        # Exploration: choose a random action
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)
        # Exploitation: choose the best action from Q-table
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-value using the Q-learning formula
        
        Args:
            state (int): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (int): Next state after action
            
        Returns:
            float: Updated Q-value
        """
        # Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        return self.q_table[state][action]
    
    def train(self, env, episodes=500):
        """
        Train the agent on the environment
        
        Args:
            env: Environment that provides state, reward, etc.
            episodes (int): Number of episodes to train
            
        Returns:
            list: History of rewards per episode
        """
        total_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                
                self.update(state, action, reward, next_state)
                
                state = next_state
                episode_reward += reward
                
            total_rewards.append(episode_reward)
            
            # Optional: decay epsilon over time
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
        return total_rewards
    
