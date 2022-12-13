import torch
import random
import numpy as np
from collections import deque

from snake_game_ai import SnakeGameAi, Direction, Point
from model_conv import Conv_QNet
from helper import plot
import torch.autograd as autograd

from torch import optim

# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
Variable = (
    lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
    if USE_CUDA
    else autograd.Variable(*args, **kwargs)
)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class AgentConvDQN:
    def __init__(self, model, optimizer) -> None:

        self.epsilon_by_frame = self._get_decay_function(start=1.0, end=0.01, decay=50)
        self.n_experiments = 0
        self.n_frames = 0

        self.exploration = True
        self.gamma = 0.9  # discount rate
        self.replay_buffer = ReplayBuffer(100000)

        self.model = model
        self.optimizer = optimizer

    def _get_decay_function(self, start, end, decay):
        return lambda x: end + (start - end) * np.exp(-1.0 * x / decay)

    def get_state(self, game):
        # the state will be a matrix of the game state
        # 1 for snake head
        # 2 for snake body
        # 3 for food

        state = np.zeros(
            (game.w // game.block_size, game.h // game.block_size), dtype=np.float32
        )

        head = game.snake[0]
        state[
            int(head.x // game.block_size) - 1, int(head.y // game.block_size) - 1
        ] = 1
        for point in game.snake[1:]:
            state[
                int(point.x // game.block_size) - 1, int(point.y // game.block_size) - 1
            ] = 2
        state[
            int(game.food.x // game.block_size) - 1,
            int(game.food.y // game.block_size) - 1,
        ] = 3

        state /= 4  # normalize

        # return as [1, 1, 32, 32] toch tensor
        # state = torch.from_numpy(state)
        # state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        return state

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        
        state = Variable(torch.FloatTensor(np.float32(state)))
        state = state.unsqueeze(0)

        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        next_state = next_state.unsqueeze(0)
        
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def get_action(self, state, expl_stop=100):
        eps = self.epsilon_by_frame(self.n_experiments)

        # stop random exploration after expl_stop episodes
        if eps > expl_stop:
            self.epsilon = 0
            self.exploration = False

        # get either a random move for exploration or an expected move from the model
        if random.random() > eps:
            # print("Shape of tensor before squeeze: ", state.shape)
            state = Variable(
                torch.FloatTensor(np.float32(state)), volatile=True
            )
            state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
            # print("Shape of tensor: ", state.size())
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].data[0]
            action = action.item()
        else:
            action = random.randrange(self.model.num_actions)
            # action = random.randrange(env.action_space.n)

        # transform single value to vector
        action_vec = [0, 0, 0]
        action_vec[action] = 1

        return action_vec


def train():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    init_replay_buffer = 2000
    batch_size = 1
    losses = []

    game = SnakeGameAi(speed=20000, display_game=False)

    # read deep double dqn paper,
    # categorical most used

    # use sequence of 4 frames
    # number of chanels is 4
    # channels, width, height
    input_shape = (1, 32, 32)
    output_shape = 3
    model = Conv_QNet(input_shape, output_shape)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    agent = AgentConvDQN(model=model, optimizer=optimizer)
    print("Starting loop")
    while True:

        # get old state
        state = agent.get_state(game)

        # get move
        action = agent.get_action(state)

        # perform move and get new state
        reward, done, score = game.play_step(action)
        next_state = agent.get_state(game)

        # train
        if len(agent.replay_buffer) > init_replay_buffer:
            loss = agent.compute_td_loss(batch_size)
            losses.append(loss.item())

        # remember
        agent.replay_buffer.push(state, action, reward, next_state, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_experiments += 1

            if score > record:
                record = score
                # agent.model.save()

            ### print some stats
            print("Game", agent.n_experiments, "Score", score, "Record:", record)
            print(
                f"Epsilon: {agent.epsilon_by_frame(agent.n_experiments)}, Exploration: {agent.exploration}"
            )

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_experiments
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
