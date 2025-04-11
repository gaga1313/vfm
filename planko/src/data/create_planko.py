import numpy as np
import pygame
from pygame import gfxdraw
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import Box2D  # The main library
from Box2D import (b2World, b2PolygonShape, b2CircleShape, b2FixtureDef, b2_pi)
from Box2D.b2 import staticBody, dynamicBody
import dgread
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import math
import csv
import ast

FOLDERNAME = "board_change_new"
P_ID = "s21"
PPM = 20.0
TARGET_FPS = 160
TIME_STEP = 1.0 / TARGET_FPS
CANVAS_SIZE = 512
INITIA_BALL_Y = 15
INITIA_BALL_X = CANVAS_SIZE / 2
SCREEN_WIDTH, SCREEN_HEIGHT = CANVAS_SIZE, CANVAS_SIZE + 200
COORD_EXTENT_X, COORD_EXTENT_Y = CANVAS_SIZE, CANVAS_SIZE + 200
STIMULATION_TIME=10
colors = {
    'plankColor': (0, 0, 0, 255),
    'ballColor': (5, 200, 250, 255),
}
xscalar = CANVAS_SIZE / PPM
yscalar = CANVAS_SIZE / PPM
bias = CANVAS_SIZE / 2
adjustx = 0.5
adjusty = 0.5
BALLRADIUS = 0.5 * yscalar
N_PLANK = 20

class MyContactListener(Box2D.b2ContactListener):
    def __init__(self):
        Box2D.b2ContactListener.__init__(self)
        self.collided_planks = []
        self.ball_positions = ball_positions  # Use the global ball_positions list

    def BeginContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body
        if not bodyA.userData or not bodyB.userData:
            return
        if bodyA.userData["type"] == "ball" and bodyB.userData["type"] == "plank":
            self.collided_planks.append(bodyB.userData['id'])
        elif bodyB.userData["type"] == "ball" and bodyA.userData["type"] == "plank":
            self.collided_planks.append(bodyA.userData['id'])
        if bodyA.userData["type"] == "ball" and bodyB.userData["type"] == "basket":
            self.handleBallBasketCollision(bodyA, bodyB)
        elif bodyB.userData["type"] == "ball" and bodyA.userData["type"] == "basket":
            self.handleBallBasketCollision(bodyB, bodyA)

    def handleBallBasketCollision(self, ball, basket):
        if not ball.userData["hit"]:
            ball.userData["hit_basket"] = basket.userData['basket_id']
            print(f"Ball hit basket number {basket.userData['basket_id']}!")
            basket_pos = basket.userData.get("my_x_pos")
            ball.userData["hit"] = True
            ball.userData["post_hit_positions"] = self.ball_positions.copy()
            ball.userData["basket_id"] = basket.userData['basket_id']  # Store basket_id

def create_world(tx, ty, sx, sy, angles, n_planks=10, basket_draw=1, import_world=False):
    global ball_positions
    ball_positions = []  # Initialize ball_positions for each run
    world = b2World(gravity=(0, -200), doSleep=True)
    if import_world:
        for i in range(n_planks):
            x1, y1, x2, y2 = tx[i], ty[i], sx[i], sy[i]
            plank_pos_x = x1 * xscalar + bias
            plank_pos_y = y1 * yscalar + bias
            plank_angle = -angles[i] * b2_pi / 180
            box = b2PolygonShape(box=(x2 * xscalar * adjustx, y2 * yscalar * adjusty, (plank_pos_x, plank_pos_y), plank_angle))
            plank_texture = b2FixtureDef(shape=box, friction=0.05)
            plank_body = world.CreateBody()
            plank_body.CreateFixture(plank_texture)
            plank_body.userData = {'type': 'plank', 'id': i}
        def create_basket(leftOrRight=1):
            if leftOrRight == 1:
                x1, y1, x2, y2 = -3, -7.25, 3.0, 1
                plank_pos_x = x1 * xscalar + bias
                plank_pos_y = y1 * yscalar + bias
                plank_angle = 0 * b2_pi / 180
                box = b2PolygonShape(box=(x2 * xscalar * adjustx, y2 * yscalar * adjusty, (plank_pos_x, plank_pos_y), plank_angle))
                basket = world.CreateBody(shapes=[box])
                basket.userData = {"type": "basket", "basket_id": 1, "my_x_pos": x1}
            elif leftOrRight == 2:
                x1, y1, x2, y2 = 3, -7.25, 3.0, 1
                plank_pos_x = x1 * xscalar + bias
                plank_pos_y = y1 * yscalar + bias
                plank_angle = 0 * b2_pi / 180
                box = b2PolygonShape(box=(x2 * xscalar * adjustx, y2 * yscalar * adjusty, (plank_pos_x, plank_pos_y), plank_angle))
                basket = world.CreateBody(shapes=[box])
                basket.userData = {"type": "basket", "basket_id": 2, "my_x_pos": x1}
        create_basket(basket_draw)

    else:
        for i in range(n_planks):
            plank_pos_x = np.random.randint(50, 462)
            plank_pos_y = np.random.randint(75, 412)
            plank_angle = float(np.random.rand(1)[0])
            plank_length = np.random.randint(10, 50)
            box = b2PolygonShape(box=(plank_length, 4, (plank_pos_x, plank_pos_y), plank_angle * b2_pi))
            world.CreateBody(shapes=box)

    circle = b2CircleShape(pos=(INITIA_BALL_X, INITIA_BALL_Y * yscalar + bias), radius=BALLRADIUS)
    ballfixture = b2FixtureDef(shape=circle, density=0.00005, restitution=0.27)
    ball = world.CreateDynamicBody(fixtures=ballfixture, bullet=True, userData={"type": "ball", "hit": False, "hit_basket": 0, "trajectory": []})

    print(f"Initial ball position: {(INITIA_BALL_X, INITIA_BALL_Y * yscalar + bias)}")  # Log the initial position
    return world, ball

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def update_collision_dict(trial_number, collision_list):
    collision[trial_number] = collision_list

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Planko board')
clock = pygame.time.Clock()

def my_draw_polygon(polygon, body, fixture):
    vertices = [(body.transform * v) for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    gfxdraw.filled_polygon(screen, vertices, colors['plankColor'])
    gfxdraw.aapolygon(screen, vertices, colors['plankColor'])

b2PolygonShape.draw = my_draw_polygon

def my_draw_circle(circle, body, fixture):
    position = body.transform * circle.pos
    position = (position[0], SCREEN_HEIGHT - position[1])
    if not body.userData.get("hit", False):  # Only record if ball has not hit the basket
        ball_positions.append(position)
    # for pos in ball_positions:
    #     pygame.draw.circle(screen, (205, 200, 200), [int(x) for x in pos], BALLRADIUS)
    pygame.draw.circle(screen, colors['ballColor'], [int(x) for x in position], int(circle.radius))

def draw_trajectory(surface, trajectory, radius, color=(205, 0, 0)):
    if len(trajectory) > 1:
        pygame.draw.lines(surface, color, False, trajectory, radius)


def save_world_as_png(world, trindex):
    total_distance = 0
    post_hit_positions = []
    basket_id = 0

    def draw_world(surface, world, draw_trajectory_func):
        surface.fill((255, 255, 255, 255))
        trajectory = []
        for body in world.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, b2PolygonShape):
                    my_draw_polygon(shape, body, fixture)
                elif isinstance(shape, b2CircleShape):
                    pos = body.transform * shape.pos
                    pos = (pos[0], SCREEN_HEIGHT - pos[1])
                    trajectory.append(pos)
                    my_draw_circle(shape, body, fixture)
        return trajectory

    trajectory = draw_world(screen, world, draw_trajectory)
    
    # Save the first image (initial position)
    filename = "../../data/dataset/" + str(trindex) + "_start.png"
    pygame.image.save(screen, filename)

    # Calculate the vertical distance between the ball's initial y-position and the basket
    # ball_initial_y = ball.position[1]
    # basket_y = world.bodies[-2].position[1]
    distance_to_basket = 640 - 72
   
    # Calculate the key y-distances (d/4, d/2, 3d/4)
    distances = {
        "d_4": 72 + distance_to_basket / 4,
        "d_2": 72 + distance_to_basket / 2,
        "3d_4": 72 + 3 * distance_to_basket / 4
    }
    # Run the simulation and save images when the ball reaches specific distances
    for i in range(TARGET_FPS * STIMULATION_TIME):
        world.Step(TIME_STEP, 10, 10)
        world.ClearForces()
        trajectory = draw_world(screen, world, draw_trajectory)
        
        current_y = trajectory[-1][1]
        
        # Check if the ball has reached d/4, d/2, or 3d/4 distances and save the image
        if distances["d_4"] - 1 < current_y <= distances["d_4"] + 1:
            filename = "../../data/dataset/" + str(trindex) + "_d4.png"
            pygame.image.save(screen, filename)
        
        elif distances["d_2"] - 1 < current_y <= distances["d_2"] + 1:
            filename = "../../data/dataset/" + str(trindex) + "_d2.png"
            pygame.image.save(screen, filename)
            
        elif distances["3d_4"] - 1 < current_y <= distances["3d_4"] + 1:
            filename = "../../data/dataset/" + str(trindex) + "_3d4.png"
            pygame.image.save(screen, filename)

        # Stop the simulation when the ball hits the basket
        if ball.userData["hit"]:
            basket_id = ball.userData["hit_basket"]
            filename = "../../data/dataset/" + str(trindex) + "_end_b" + str(basket_id) + ".png"
            pygame.image.save(screen, filename)
            break

    if ball.userData["hit"]:
        post_hit_positions = ball.userData["post_hit_positions"]
        for j in range(1, len(post_hit_positions)):
            total_distance += calculate_distance(post_hit_positions[j - 1], post_hit_positions[j])
        print(f"Total actual distance traveled after hitting the basket: {total_distance}")
    
    return total_distance, post_hit_positions, basket_id



if __name__ == "__main__":

    collision={}
    root = '/media/data_cifs_lrs'
    # root = '/cifs/data/tserre_lrs'
    with open('../../data/combined_final_infor.csv', mode ='r') as file:    
       csvFile = csv.DictReader(file)
       for lines in csvFile:

            n_planks = int(lines['n_planks'])
            trialidx = lines['trindex']
            tx = ast.literal_eval(lines['tx'])  # Converts string to list of floats
            ty = ast.literal_eval(lines['ty'])
            sx = ast.literal_eval(lines['sx'])
            sy = ast.literal_eval(lines['sy'])
            angles = ast.literal_eval(lines['angles'])
            baskettodraw = int(lines['basket_draw'])
            # Your world creation logic
            pygame.init()
            world, ball = create_world(tx, ty, sx, sy, angles, n_planks, baskettodraw, True)
            listener = MyContactListener()
            world.contactListener = listener

            update_collision_dict(trialidx, listener.collided_planks)
            # Save the world as a PNG
            save_world_as_png(world, trialidx)