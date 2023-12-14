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

# 하이퍼파라미터
class Config1:
    learning_rate = 0.0005   # 학습률
    gamma = 0.98             # 할인 계수 (미래 보상에 대한 할인 계수)
    buffer_limit = 50000     # 리플레이 버퍼의 최대 크기
    batch_size = 32          # 미니배치 크기
    print_interval = 20      # 로그 정보를 출력하는 간격

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

config = Config1()

# 행동 열거형
class Actions(Enum):
    LEFT = 0
    RIGHT = 1

class ReplayBuffer():
    def __init__(self):
        # 리플레이 버퍼를 초기화하고, 최대 길이가 config.buffer_limit인 deque를 생성함
        self.buffer = collections.deque(maxlen=config.buffer_limit)

    def put(self, transition):
        # 새로운 transition을 리플레이 버퍼에 추가함
        self.buffer.append(transition)

    def sample(self, n):
        # 리플레이 버퍼에서 크기가 n인 미니배치를 무작위로 샘플링함
        mini_batch = random.sample(self.buffer, n)

        # 각각 상태(state), 행동(action), 보상(reward), 다음 상태(next state), 종료 여부(done mask)를 저장할 리스트를 초기화함
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            # 미니배치에서 각 transition을 가져옴
            s, a, r, s_prime, done_mask = transition

            # 현재 상태를 리스트에 추가함
            s_lst.append(s)

            # 행동을 리스트에 추가. Enum을 숫자로 변환하여 저장함
            a_lst.append([a.value])

            # 보상을 리스트에 추가함
            r_lst.append([r])

            # 다음 상태를 리스트에 추가함
            s_prime_lst.append(s_prime)

            # 종료 여부를 리스트에 추가함
            done_mask_lst.append([done_mask])

        # 각 리스트를 PyTorch 텐서로 변환하여 반환함
        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst)
        )

    def size(self):
        # 현재 리플레이 버퍼의 크기를 반환
        return len(self.buffer)

# Q네트워크
class QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNet, self).__init__()
        # 신경망의 층을 정의. 입력 크기(input_size)에서 128개의 뉴런을 가진 첫 번째 은닉층,
        # 128개의 뉴런을 가진 두 번째 은닉층, 그리고 출력 크기(output_size)의 출력층으로 구성
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        # 순전파(forward pass) 메서드를 정의. 활성화 함수 ReLU를 사용하는 두 개의 은닉층이 적용된 후, 세 번째 층에서 출력을 반환
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        # Double Q-learning에서는 q_target으로부터 행동을 선택
        with torch.no_grad():
            # 현재 상태를 입력으로 받아 신경망에 전달하고 출력을 계산
            out = self(obs)
            coin = random.random()
            if coin < epsilon:
                # 무작위 행동 선택: epsilon보다 작은 확률로 무작위 행동을 선택
                return random.choice(list(Actions))
            else:
                # Q 값이 가장 높은 행동을 선택
                return Actions(out.argmax().item())

    def train(self, q_target, memory, optimizer):
        for i in range(10):
            # 리플레이 버퍼에서 미니배치를 샘플링
            s, a, r, s_prime, done_mask = memory.sample(config.batch_size)
            # 현재 상태를 입력으로 받아 신경망에 전달하고 출력을 계산
            q_out = self(s)
            # 선택된 행동에 대한 Q 값만을 선택
            q_a = q_out.gather(1, a)
            
            # q_target으로부터 행동을 선택하여 해당 행동의 가치를 계산
            with torch.no_grad():
                # 다음 상태에서 Q 값이 가장 높은 행동을 선택
                argmax_Q = self(s_prime).max(1)[1].unsqueeze(1)
                # q_target에서 해당 행동의 가치를 가져옴
                max_q_prime = q_target(s_prime).gather(1, argmax_Q)
            
            # Double Q-learning에서의 타깃 계산
            target = r + config.gamma * max_q_prime * done_mask
            # Smooth L1 Loss를 사용하여 손실을 계산
            loss = F.smooth_l1_loss(q_a, target)
            # 기울기를 초기화
            optimizer.zero_grad()
            # 역전파를 통해 기울기를 계산
            loss.backward()
            # 옵티마이저를 사용하여 모델 파라미터를 업데이트
            optimizer.step()

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 생성
def create_environment():
    return gym.make('CartPole-v1')

# 모델 저장 함수
def save_model(model, episode):
    torch.save(model.state_dict(), f"cartpole_dqn_model_episode_{episode}.pt")
    logger.info("The model has been saved.")

# 모델 로드 함수
def load_model(model, episode):
    model.load_state_dict(torch.load(f"cartpole_dqn_model_episode_{episode}.pt"))
    logger.info(f"Model from episode {episode} has been loaded.")

# 메인 함수
def main():
    # CartPole 환경을 생성
    env = create_environment()
    # Q 네트워크를 생성
    q = QNet(input_size=4, output_size=len(Actions))
    # 타겟 Q 네트워크를 생성하고 초기 상태로 설정
    q_target = QNet(input_size=4, output_size=len(Actions))
    q_target.load_state_dict(q.state_dict())
    # 리플레이 버퍼를 생성
    memory = ReplayBuffer()

    # Adam 옵티마이저를 사용하여 Q 네트워크를 훈련
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=config.learning_rate)

    # 최근 3개의 에피소드 점수를 저장하기 위한 deque를 생성
    recent_scores = collections.deque(maxlen=3)

    # 에피소드별 점수를 저장하기 위한 리스트를 생성
    episode_scores = []

    for n_epi in range(100000):
        # 탐험(exploration)을 위한 입실론 값을 계산
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        # 환경을 초기화하고 초기 상태를 얻음
        s, _ = env.reset()
        done = False

        while not done:
            # Q 네트워크를 사용하여 입실론 탐험 정책에 따라 행동을 선택
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            # 선택한 행동을 환경에 적용하여 다음 상태, 보상 등을 얻음
            s_prime, r, terminated, truncated, info = env.step(a.value)
            # 종료 상태인 경우 done_mask를 0.0으로, 그렇지 않으면 1.0으로 설정
            done = (terminated or truncated)
            done_mask = 0.0 if done else 1.0
            # 리플레이 버퍼에 현재 transition을 저장
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            # 현재 상태를 다음 상태로 업데이트
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            # 리플레이 버퍼 크기가 일정 이상이면 Q 네트워크를 훈련
            q.train(q_target, memory, optimizer)

        if n_epi % config.print_interval == 0 and n_epi != 0:
            # 일정 주기마다 타겟 Q 네트워크를 업데이트
            q_target.load_state_dict(q.state_dict())
            # 에피소드 점수를 저장
            average_score = score / config.print_interval
            episode_scores.append(average_score)
            recent_scores.append(average_score)
            # 현재 에피소드의 정보를 로그에 출력
            logger.info("n_episode: %d, score: %.1f, n_buffer: %d, eps: %.1f%%", n_epi, average_score, memory.size(), epsilon * 100)

            # 최근 3개의 점수가 모두 500 이상인 경우 모델 저장
            if all(score >= 500 for score in recent_scores):
                # 최근 3개 에피소드의 점수가 모두 500 이상이면 Q 네트워크 모델을 저장
                save_model(q, n_epi + 1)

            # 에피소드 점수를 초기화
            score = 0.0

    # 에피소드 점수를 그래프로 플로팅
    plt.plot(episode_scores)
    plt.title('Episode Scores over Time')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.savefig('cartpole_double_dqn_Config1.png')
    plt.show()

    env.close()

if __name__ == '__main__':
    main()
