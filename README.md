# snake_game_rl
A repository containing different work used to learn about RL using a simple snake game with pygame

## Usage

Run the agent script and observe the training process.

### Naive solution
Have the snake travel in a spiral pattern around the environment. No heuristics are considered.

### Simple DQN
The first DQN solution as presented in the credited tutorial. 
The agent is able to learn to hunt for the food, but since it does not keep track of its tail it ends up entraping itself.

### Conv DQN
COnvolutional DQN where the input is a matrix describing the complete environment state. This approach aims to allow the snake to avoid getting trapped by its own tail.

Credits: code developed starting from https://github.com/python-engineer/snake-ai-pytorch
