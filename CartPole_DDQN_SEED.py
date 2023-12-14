import gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
from enum import Enum
from scipy.stats import sem, t
from scipy import mean
import seaborn as sns

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Hyperparameters
class Config:
    learning_rate = 0.0005   # Learning rate
    gamma = 0.98             # Discount factor for future rewards
    buffer_limit = 50000     # Maximum size of the replay buffer
    batch_size = 32          # Mini-batch size
    print_interval = 20      # Interval for logging information

# Different configurations for experiments
class Config2(Config):
    learning_rate = 0.0001
    gamma = 0.99
    buffer_limit = 100000
    batch_size = 64

class Config3(Config):
    learning_rate = 0.001
    gamma = 0.95
    buffer_limit = 25000
    batch_size = 16

# Enum for actions
class Actions(Enum):
    LEFT = 0
    RIGHT = 1

# Replay buffer class
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=Config.buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a.value])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst)
        )

    def size(self):
        return len(self.buffer)

# Q-network class
class QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        with torch.no_grad():
            out = self(obs)
            coin = random.random()
            if coin < epsilon:
                return random.choice(list(Actions))
            else:
                return Actions(out.argmax().item())

    def train(self, q_target, memory, optimizer):
        for i in range(10):
            s, a, r, s_prime, done_mask = memory.sample(Config.batch_size)
            q_out = self(s)
            q_a = q_out.gather(1, a)
            
            with torch.no_grad():
                argmax_Q = self(s_prime).max(1)[1].unsqueeze(1)
                max_q_prime = q_target(s_prime).gather(1, argmax_Q)
            
            target = r + Config.gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the CartPole environment
def create_environment():
    return gym.make('CartPole-v1')

# Save model function
def save_model(model, episode):
    torch.save(model.state_dict(), f"cartpole_dqn_model_episode_{episode}.pt")
    logger.info("The model has been saved.")

# Load model function
def load_model(model, episode):
    model.load_state_dict(torch.load(f"cartpole_dqn_model_episode_{episode}.pt"))
    logger.info(f"Model from episode {episode} has been loaded.")

# Run experiment function
def run_experiment(seed, config_class):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    env = create_environment()
    q = QNet(input_size=4, output_size=len(Actions))
    q_target = QNet(input_size=4, output_size=len(Actions))
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=config_class.learning_rate)
    episode_scores = []

    for n_epi in range(100000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        s, _ = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, terminated, truncated, info = env.step(a.value)
            done = (terminated or truncated)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            q.train(q_target, memory, optimizer)

        if n_epi % Config.print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            average_score = score / Config.print_interval
            episode_scores.append(average_score)
            logger.info("n_episode: %d, score: %.1f, n_buffer: %d, eps: %.1f%%", n_epi, average_score, memory.size(), epsilon * 100)

            if all(score >= 500 for score in episode_scores[-3:]):
                save_model(q, n_epi + 1)

            score = 0.0

    env.close()
    return episode_scores

# 실험 횟수 및 시드 값
num_experiments = 10
# seed_values = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
seed_values = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1111]
all_experiment_results = []

# 실험 실행 및 95% 신뢰 구간 계산
for i in range(num_experiments):
    config_instance = Config()
    seed_value = seed_values[i]
    experiment_results = run_experiment(seed_value, config_instance)
    all_experiment_results.append(experiment_results)

# 95% 신뢰 구간 계산 및 시각화
confidence = 0.95
data_means = [mean(experiment_results) for experiment_results in all_experiment_results]
data_sems = [sem(experiment_results) for experiment_results in all_experiment_results]
confidence_intervals = [t.interval(confidence, len(experiment_results) - 1, loc=mean_val, scale=sem_val) for mean_val, sem_val in zip(data_means, data_sems)]

# 결과 시각화
plt.figure(figsize=(10, 6))
for i in range(num_experiments):
    sns.lineplot(x=range(len(all_experiment_results[i])), y=all_experiment_results[i], label=f'Experiment {i + 1}')
    plt.fill_between(range(len(all_experiment_results[i])), confidence_intervals[i][0], confidence_intervals[i][1], alpha=0.3)
plt.title('Episode Scores over Time (Multiple Experiments)')
plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.legend()
plt.show()