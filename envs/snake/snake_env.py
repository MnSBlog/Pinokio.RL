from typing import Optional, Union, List

import gym
import pygame
import random

# for communication on override
from gym.core import ActType, ObsType, RenderFrame
from gym.spaces import Tuple


# class Snake(gym.Env):
#     def __init__(self, mode, **kwargs):
#         test = 1
#     def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
#         test =1
#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[dict] = None,
#     ) -> Tuple[ObsType, dict]:
#         test = 1
#     def close(self):
#         test = 1
#     def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
#         test = 1

class SnakeSingle:
    def __init__(self, **kwargs):
        # 게임 초기화
        pygame.init()

        # 게임 창 설정
        self.cell_size = kwargs['cell_size']
        self.cols = kwargs['columns']
        self.rows = kwargs['rows']
        width = self.cell_size * self.cols
        height = self.cell_size * self.rows
        self.screen = pygame.display.set_mode((width, height))

        # 뱀 초기 위치 설정
        self.snake = [(self.cols // 2, self.rows // 2)]


        # 사과 초기 위치 설정
        self.apple = (random.randint(0, self.cols-1), random.randint(0, self.rows-1))
        self.clock = pygame.time.Clock()

    def action(self):
        # 게임 루프
        running = True
        dx, dy = 0, 0
        while running:
            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        dx, dy = -1, 0
                    elif event.key == pygame.K_RIGHT:
                        dx, dy = 1, 0
                    elif event.key == pygame.K_UP:
                        dx, dy = 0, -1
                    elif event.key == pygame.K_DOWN:
                        dx, dy = 0, 1

            if dx + dy == 0:
                continue
            # 뱀 이동 처리
            x, y = self.snake[-1]
            x += dx
            y += dy
            x %= self.cols
            y %= self.rows
            self.snake.append((x, y))

            # 뱀 충돌 검사
            if self.snake.count((x, y)) > 1:
                running = False

            if (x, y) == self.apple:
                self.apple = (random.randint(0, self.cols-1), random.randint(0, self.rows-1))
            else:
                self.snake.pop(0)
            self.update()
        self.close()

    def update(self):
        # 화면 업데이트
        self.screen.fill((0, 0, 0))  # 검은색 배경
        for x, y in self.snake:
            pygame.draw.rect(self.screen, (255, 255, 255), (x * self.cell_size,
                                                            y * self.cell_size,
                                                            self.cell_size,
                                                            self.cell_size))  # 흰색 뱀
        pygame.draw.rect(self.screen, (255, 0, 0), (self.apple[0] * self.cell_size,
                                                    self.apple[1] * self.cell_size,
                                                    self.cell_size, self.cell_size))  # 빨간색 사과
        pygame.display.update()
        # 초당 프레임 수 설정
        self.clock.tick(10)

    def close(self):
        # 게임 종료
        pygame.quit()



if __name__ == "__main__":
    snake = SnakeSingle(cell_size=20, columns=30, rows=20)
    snake.action()