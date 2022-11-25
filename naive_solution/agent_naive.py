import torch
import random
import numpy as np
from collections import deque
from enum import Enum

from snake_game_ai import SnakeGameAi, Direction, Point
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
Lr = 0.001


class SolutionState(Enum):
    SPIRAL = 1
    FALL = 2


class AgentNaive:
    def __init__(self) -> None:
        self.sol_state = SolutionState.SPIRAL

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)

        state = [
            # Position
            head,
            [
                point_l,
                point_r,
                point_u,
                point_d,
            ],
            # Move direction
            game.direction,
            [game.w, game.h, game.block_size],
        ]

        return state

    def get_action(self, state):
        """Naive action that solves the snake problem"""

        final_move = [1, 0, 0]

        head_pos = state[0]
        around_pos = state[1]
        dir = state[2]
        game_size = state[3]

        if (game_size[1] / game_size[2]) % 2 == 0:
            fill_row = True 
        else: 
            fill_row = False

        if self.sol_state == SolutionState.SPIRAL:
            if fill_row:
                padding = game_size[2]
            else:
                padding = 2 * game_size[2]

            if head_pos.x >= game_size[0] - 2*game_size[2]:
                if dir == Direction.RIGHT:
                    final_move = [0, 0, 1]
                elif dir == Direction.UP:
                    final_move = [0, 0, 1]

            if head_pos.x <= padding - game_size[2]:
                if dir == Direction.LEFT:
                    final_move = [0, 1, 0]
                elif dir == Direction.UP:
                    final_move = [0, 1, 0]

            if head_pos.y <= 0 and (
            dir == Direction.RIGHT or dir == Direction.LEFT
        ):
                self.sol_state = SolutionState.FALL

        if self.sol_state == SolutionState.FALL:
            if head_pos.y <= 0:
                if head_pos.x <= 0:
                    final_move = [0, 0, 1]
                if head_pos.x >= game_size[0] - game_size[2]:
                    final_move = [0, 1, 0]

            if head_pos.y >= game_size[1] - game_size[2]:
                if head_pos.x <= game_size[2]:
                    final_move = [0, 0, 1]
                elif head_pos.x >= game_size[0] - 2 * game_size[2]:
                    final_move = [0, 1, 0]
                self.sol_state = SolutionState.SPIRAL            

        return final_move


def solve():
    agent = AgentNaive()
    game = SnakeGameAi(speed=2000, display_game=True)

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        if done:
            break

if __name__ == "__main__":
    solve()

