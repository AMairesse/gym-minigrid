from gym_minigrid.minigrid import *
from gym_minigrid.register import register

DEFAULT_LIFE_EXPECTANCY = 20
DEFAULT_ENERGY_PER_SQUARE = 1
DEFAULT_FOOD_PER_SQUARE = 0.2

class FoodEnv(MiniGridEnv):
    """
    Simple grid environment, no obstacles, some food available, no goal
    Agent has an energy level and must gather food to keep it > 0
    Reward is progressive, max if agent live to is life expectancy
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_start_energy = DEFAULT_ENERGY_PER_SQUARE * size
        self.agent_life_expectancy = DEFAULT_LIFE_EXPECTANCY

        super().__init__(
            grid_size=size,
            max_steps=self.agent_life_expectancy,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _reward(self):
        return self.step_count / self.agent_life_expectancy

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Set the starting energy level of the agent
        self.agent_energy = self.agent_start_energy

        # Place some food objects
        nb_food = int(((width + height) // 2 ) * DEFAULT_FOOD_PER_SQUARE)
        for _ in range(0, nb_food):
            self.place_obj(Food(energy=self.agent_start_energy))

        self.mission = f"Pickup food to stay alive, make it to {self.agent_life_expectancy} steps"

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if self.carrying.type == 'food':
                self.agent_energy += self.carrying.energy
                self.carrying = None

        if self.steps_remaining == 0 and self.agent_energy > 0.0:
            done = True

        if self.agent_energy <= 0.0:
            done = True
        else:
            self.agent_energy += -1.0

        reward = self._reward()
        info["agent_energy"] = self.agent_energy

        return obs, reward, done, info

class FoodEnv5x5(FoodEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)

class FoodRandomEnv5x5(FoodEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class FoodEnv6x6(FoodEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)

class FoodRandomEnv6x6(FoodEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class FoodEnv16x16(FoodEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)

register(
    id='MiniGrid-Food-5x5-v0',
    entry_point='gym_minigrid.envs:FoodEnv5x5'
)

register(
    id='MiniGrid-Food-Random-5x5-v0',
    entry_point='gym_minigrid.envs:FoodRandomEnv5x5'
)

register(
    id='MiniGrid-Food-6x6-v0',
    entry_point='gym_minigrid.envs:FoodEnv6x6'
)

register(
    id='MiniGrid-Food-Random-6x6-v0',
    entry_point='gym_minigrid.envs:FoodRandomEnv6x6'
)

register(
    id='MiniGrid-Food-16x16-v0',
    entry_point='gym_minigrid.envs:FoodEnv16x16'
)
