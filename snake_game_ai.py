import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont("arial", 25)

# TODO: merge with snake_game_human.py and make otpion to jump from automatic to manual mode


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLUEH = (0, 100, 150)
BLACK = (0, 0, 0)

SPEED = 4000


class SnakeGameAi:
    def __init__(self, w=640, h=480, speed=20, block_size=20):
        self.w = w
        self.h = h
        self.speed = speed
        self.block_size = block_size

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - (2 * self.block_size), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = (
            random.randint(0, (self.w - self.block_size) // self.block_size)
            * self.block_size
        )
        y = (
            random.randint(0, (self.h - self.block_size) // self.block_size)
            * self.block_size
        )
        self.food = Point(x, y)

        # retry if food is on the snake
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        reward = 0
        game_over = False

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # self.change_heading(action)

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # end game if there is a collision or if too much time passed without anything happening
        # if self.is_collision():
        #     game_over = True
        #     reward = -10
        #     return reward, game_over, self.score

        # if self.frame_iteration > 100 * len(self.snake):
        #     game_over = True
        #     reward = -10
        #     return reward, game_over, self.score

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if (
            pt.x > self.w - self.block_size
            or pt.x < 0
            or pt.y > self.h - self.block_size
            or pt.y < 0
        ):
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # render the snake head
        pygame.draw.rect(
            self.display,
            BLUEH,
            pygame.Rect(
                self.snake[0].x, self.snake[0].y, self.block_size, self.block_size
            ),
        )

        # render the snake tail
        for pt in self.snake[1:]:
            pygame.draw.rect(
                self.display,
                BLUE1,
                pygame.Rect(pt.x, pt.y, self.block_size, self.block_size),
            )
            pygame.draw.rect(
                self.display,
                BLUE2,
                pygame.Rect(
                    pt.x + self.block_size // 5,
                    pt.y + self.block_size // 5,
                    self.block_size // 5 * 3,
                    self.block_size // 5 * 3,
                ),
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size),
        )

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left] actions
        # what about [right, down, left, up] directly?

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        action_idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[action_idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (action_idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:
            next_idx = (action_idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)


if __name__ == "__main__":
    game = SnakeGameAi()

    # game loop
    while True:
        game_over, score = game.play_step()

        if game_over == True:
            break

    print("Final Score", score)
    pygame.quit()
