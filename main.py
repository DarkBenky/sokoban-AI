import gym
import wandb
from stable_baselines3 import PPO
import torch
import random
from gym import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from queue import PriorityQueue


class Game():
    def __init__(self, player_x, player_y, player_char='X', wall_char='#', empty_char='.', box_char='O', final_char='$', final_cords=[(3, 3)],
                 box_cords=[(5, 6)], wall_cords = [(0, 0),(0, 9),(9, 0),(9, 9)], map_size=(10, 10) , path_char = 'P'):
        self.x = player_x
        self.y = player_y
        self.player_char = player_char
        self.wall_char = wall_char
        self.empty_char = empty_char
        self.final_char = final_char
        self.box_char = box_char
        self.path_char = path_char
        self.box_cords = box_cords
        self.final_cords = final_cords
        self.wall_cords = wall_cords
        self.map_size = map_size

    def move(self, direction) -> bool:
        def convert_direction(direction):
            if direction == 'up' or direction == 'w' or direction == 'W':
                return (0, -1)
            if direction == 'down' or direction == 's' or direction == 'S':
                return (0, 1)
            if direction == 'left' or direction == 'a' or direction == 'A':
                return (-1, 0)
            if direction == 'right' or direction == 'd' or direction == 'D':
                return (1, 0)

        def is_valid_move(x, y):
            return 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]

        new_x = self.x + convert_direction(direction)[0]
        new_y = self.y + convert_direction(direction)[1]

        def check_wall_collision(x, y):
            return (x, y) in self.wall_cords
        def check_box_collision(x, y):
            return (x, y) in self.box_cords
    
        if is_valid_move(new_x, new_y):
            if not check_wall_collision(new_x, new_y) and not check_box_collision(new_x, new_y):
                self.x = new_x
                self.y = new_y
                return True
            elif check_box_collision(new_x, new_y):
                for i, cord in enumerate(self.box_cords):
                    if cord == (new_x, new_y):
                        if not check_wall_collision(convert_direction(direction)[0] + new_x, convert_direction(direction)[1] + new_y) and not check_box_collision(convert_direction(direction)[0] + new_x, convert_direction(direction)[1] + new_y) and  is_valid_move(convert_direction(direction)[0] + new_x, convert_direction(direction)[1] + new_y):
                            self.box_cords[i] = (convert_direction(direction)[0] + new_x, convert_direction(direction)[1] + new_y)
                            self.x = new_x
                            self.y = new_y
                            return True
                        else:
                            return False
            else:
                return False

    def __str__(self):
        map  = [[self.empty_char for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        map[self.y][self.x] = self.player_char
        for cord in self.box_cords:
            map[cord[1]][cord[0]] = self.box_char
        for cord in self.wall_cords:
            map[cord[1]][cord[0]] = self.wall_char
        for cord in self.final_cords:
            map[cord[1]][cord[0]] = self.final_char
        return '\n'.join([' '.join(str(row)) for row in map])
    
    def calculate_distance(self,start_x:int,start_y:int, end_x:int,end_y:int):
        return (abs(start_x - end_x) + abs(start_y - end_y)) / ((end_x - start_x) + (end_y - start_y))
    
    def generate_heatmap(self):
        array_2d = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        box_x , box_y = self.box_cords[0]
        final_x , final_y = self.final_cords[0]
        
        array_2d_box = array_2d.copy()
        array_2d_final = array_2d.copy()
        
        for i in range(len(array_2d_box)):
            for j in range(len(array_2d_box)):
                distance = self.calculate_distance(box_x, box_y, i, j)
                array_2d_box[j][i] = 1/distance
                distance = self.calculate_distance(final_x, final_y, i, j)
                array_2d_final[j][i] = 1/distance
        
        for i in range(len(array_2d)):
            for j in range(len(array_2d)):
                array_2d[i][j] = array_2d_box[i][j] + array_2d_final[i][j] / 2
                
        for cord  in self.wall_cords:
            array_2d[cord[1]][cord[0]] = 0
                   
        return array_2d
        
        
        
        
    
    def return_map_3d_array(self):
        
        # path = self.find_path_to_goal()
        
        # path_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        # for cord in path:
        #     path_map[cord[1]][cord[0]] = 1
        
        heatmap = self.generate_heatmap()
            
        player_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        player_map[self.y][self.x] = self.player_char

        box_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        for cord in self.box_cords:
            box_map[cord[1]][cord[0]] = self.box_char
            
        wall_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])]
        for cord in self.wall_cords:
            wall_map[cord[1]][cord[0]] = self.wall_char
        
        final_map = [[0 for _ in range(self.map_size[0])] for _ in range(self.map_size[1])] 
        for cord in self.final_cords:
            final_map[cord[1]][cord[0]] = self.final_char
        
        map = [player_map, box_map, wall_map, final_map , heatmap]
        return map
    
    def find_path_to_goal(self):
        def heuristic(node, goal):
            return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

        def get_neighbors(node):
            x, y = node
            return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        start = self.box_cords[0]
        goal = self.final_cords[0]

        open_set = PriorityQueue()
        open_set.put((0, start, []))  # (priority, node, path)

        closed_set = set()

        while not open_set.empty():
            _, current, path = open_set.get()

            if current == goal:
                return path

            if current in closed_set:
                continue
            
            if len(path) > 50:
                return []

            closed_set.add(current)

            for neighbor in get_neighbors(current):
                if neighbor not in self.wall_cords and neighbor not in closed_set:
                    new_path = path + [current]
                    priority = len(new_path) + heuristic(neighbor, goal)
                    open_set.put((priority, neighbor, new_path))

        return []  # No path found
        
    
    def check_win(self):
        for cord in self.final_cords:
            if cord not in self.box_cords:
                return False
        return True
    
    
class Env(gym.Env):
    def __init__(self , map_size = (20, 20)):
        self.game = Game(player_x=3, player_y=6, player_char=1, wall_char=2, empty_char=0, box_char=3, final_char=4 , map_size=map_size)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(20, 20 , 5), dtype=np.float32)
        self.map_size = map_size

    def step(self, action):
        action = ['up', 'down', 'left', 'right'][action]
        
        old_x, old_y = self.game.box_cords[0]
        old_distance_map = self.game.generate_heatmap()
        
        valid_move = self.game.move(action)
        
        
        if not valid_move:
            return self.game.return_map_3d_array(), -10, False, {}
        
        if self.game.check_win():
            return self.game.return_map_3d_array(), 100, True, {}
        
        if valid_move:
            new_x, new_y = self.game.box_cords[0]
            if old_distance_map[new_y][new_x] < old_distance_map[old_y][old_x]:
                return self.game.return_map_3d_array(), -1, False, {}
            else:
                return self.game.return_map_3d_array(), +1, False, {}


    def reset(self):
        size = self.map_size[0]
        wall_cords = []
        for _ in range((size * size) // 5):
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            if (x, y) not in wall_cords:
                wall_cords.append((x, y))

        final_cords = []
        while not final_cords:
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            if (x, y) not in wall_cords:
                final_cords.append((x, y))

        box_cords = []
        while True:
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            if (x, y) not in wall_cords and (x, y) not in final_cords:
                box_cords.append((x, y))
                break

        player_cords = []
        while True:
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            if (x, y) not in wall_cords and (x, y) not in final_cords and (x, y) not in box_cords:
                player_cords.append((x, y))
                break

        self.game = Game(player_x=player_cords[0][0], player_y=player_cords[0][1], player_char=1, wall_char=2, empty_char=0, box_char=3, final_char=4, wall_cords=wall_cords, final_cords=final_cords, box_cords=box_cords , map_size=(size, size), path_char=5)
        new_map = self.game.return_map_2d_array()
        return np.array(new_map)

    def render(self):
        print(self.game)
      
class CustomCallback(BaseCallback):
    pass
        
        
        
def generate_model_name():
    import time
    return f"model-{int(time.time())}"
            
config_dict = {
    'learning_rate': 0.001,
    'net_arch': {'pi': [1024,1024,1024,1024], 'vf': [1024,1024,1024,1024]},
    'batch_size': 256,
    'model_name': generate_model_name(),
    'map_size': (20, 20),
    'moves': 1_00_000
}

wandb.init(project="box_pusher", config=config_dict)   

env = Monitor(Env(moves=config_dict['moves'],map_size=config_dict['map_size']))

model = PPO("MlpPolicy", env, verbose=1 , learning_rate=config_dict['learning_rate'], policy_kwargs=dict(net_arch=config_dict['net_arch']) , batch_size=config_dict['batch_size'])
model.learn(total_timesteps=10000, callback=CustomCallback(env, eval_frequency=1000, tests = 10))
    