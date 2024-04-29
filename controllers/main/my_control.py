# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from utils import euler2rotmat
import cv2
import copy

# Global variables
on_ground = True
height_desired = 1.0
timer = None
startpos = None
timer_done = None
state = 0
target = None
path = None
first_time = True
time1 = 0
square_yaw = None
position_at_detection = None
height_square = 0
mean_height_square = 1
square_region_entering = None
clipping = False #True
grid_search_points = None
visited_cells = []
idx_Gridpoint = 0

SAFETY_MARGIN = 0.15     # meters
MOVE_WITH_ROTATION = 1
ENABLE_SQUARE_DETECTION = 0

index_current_setpoint = 1

# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 3.0 # meter
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.1 # meter
conf = 0.2 # certainty given by each measurement
t = 0 # only for plotting
map = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied


# All available ground truth measurements can be accessed by calling sensor_data[item], where "item" can take the following values:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "v_forward": Forward velocity (body frame)
# "v_left": Leftward velocity (body frame)
# "v_down": Downward velocity (body frame)
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration
# "range_front": Front range finder distance
# "range_down": Donward range finder distance
# "range_left": Leftward range finder distance 
# "range_back": Backward range finder distance
# "range_right": Rightward range finder distance
# "range_down": Downward range finder distance
# "rate_roll": Roll rate (rad/s)
# "rate_pitch": Pitch rate (rad/s)
# "rate_yaw": Yaw rate (rad/s)

# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global on_ground, startpos, state, path, index_current_setpoint, first_time, time1, height_desired
    global ENABLE_SQUARE_DETECTION, square_yaw, position_at_detection, height_square, mean_height_square, square_region_entering, clipping
    global grid_search_points, visited_cells, idx_Gridpoint

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    #cv2.imshow('Camera Feed', camera_data)
    #cv2.waitKey(1)
    
    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]  
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    map_computed = occupancy_map(sensor_data)

    if ENABLE_SQUARE_DETECTION:
        # I can check here if the pink square is detected: if it's detected -> change state to A* and set pink square as goal. When it'll be reached -> remove it and go back to normal A* path
        image, coordinates_center, angle, height = detect_pink_square(camera_data, sensor_data)
        if coordinates_center[0] is not None:
            square_yaw = angle # this angle will be used to compute euristic goal to square
            position_at_detection = [sensor_data['x_global'], sensor_data['y_global']]
            if height>0.20 and height<1.5: # filter wrong detections
                height_square = (height_square + height)/mean_height_square
                mean_height_square += 1
                #print(f"drone height: {sensor_data['z_global']}, square height: {height_square}")
            #print("square_yaw °: ", square_yaw*180/np.pi)
            # these two will be used in pair to compute the goal cell on the map
            
            #cv2.imshow('Camera Feed', image)
            #cv2.waitKey(1)
    else:
        square_yaw = None
        position_at_detection = None

        #print(f"Yaw: {sensor_data['yaw']}")
        #if coordinates_center[0] is not None:
        #    # save current yaw to find furthest 1 in the map in that direciton
        #    # square_yaw = sensor_data['yaw']
        #    position_at_detection = [sensor_data['x_global'], sensor_data['y_global']]
            #print("Pink square detected")
            #if sensor_data['t'] < 6.0:
            #    print("square_yaw", square_yaw)
            #    print(f"position_at_detection: {int(np.round(position_at_detection[0]/res_pos))}, {int(np.round(position_at_detection[1]/res_pos))}")
         

    if state == 0:                          # turning in place
        control_command = [0.0, 0.0, height_desired, 1]
        if sensor_data['t'] >= 6.0:
            state += 1
    elif state == 1:                        # call A* to generate path (go to landing area)
        
        print("A* called")
        direction = 'forth'
        
        if ENABLE_SQUARE_DETECTION:     # TODO: da lasciare quando CV deve funzionare?
            if sensor_data['x_global']>=2.3 and sensor_data['x_global']<=2.4 and clipping == True:
                # make drone's camera point towards positive x
                state = 11
                control_command = [0.0, 0.0, height_desired, MOVE_WITH_ROTATION]
                print("go to state 11")
                return control_command
                #print("Inside the last recall of cv path planning, height specified, clipping set to False")
                # recompute path to pass through the center of square (x is not clipped anymore)
                # square_region_entering = [int(np.round((sensor_data['x_global'] - min_x )/res_pos,0)), int(np.round((sensor_data['y_global'] - min_y)/res_pos,0)) ]
                #if ENABLE_SQUARE_DETECTION:
                #    clipping = False
                #    #height_desired = height_square
                #    ENABLE_SQUARE_DETECTION = 0
        
        path = astar_path(map_computed, sensor_data, direction, square_yaw, position_at_detection, clipping) #clipping: if False the drone shall go through the square
        #print(f"Path: {path}")
        control_command = [0.0, 0.0, height_desired, MOVE_WITH_ROTATION]
        index_current_setpoint = 1
        state += 1

    elif state == 2:                        # follow path until euristic goal is reached
        #print("----- follow_setpoint -----")
        #print(f"Path: {path}")
        #print(f"Current cells: {int(np.round((sensor_data['x_global'] - min_x )/res_pos,0))}, {int(np.round((sensor_data['y_global'] - min_y )/res_pos,0))}")
        direction = 'forth'

        vx,vy, reached = follow_setpoints(sensor_data, direction)
        control_command = [vx, vy, height_desired, MOVE_WITH_ROTATION]

        if sensor_data['x_global'] >= 3.7:
            state += 1
        else:   
            if reached and (sensor_data['x_global'] < 3.7):
                    state -= 1

    elif state == 3:                        # rotation in landing area for better mapping
        control_command = [0.0, 0.0, height_desired, 1]
        
        if first_time:
            time1 = sensor_data['t']
            first_time = False
        if sensor_data['t'] - time1 >= 8.0:
            first_time = True
            state += 1
            #state = 6
    
    elif state == 4:                        # generate goals to create path for grid search in the landing area
        control_command = [0.0, 0.0, height_desired, 0]
        #path = single_agent_coverage(map_computed, sensor_data)
        grid_search_points = grid_search_gen(map_computed)
        state += 1
        print(f"Path of grid search: {grid_search_points}")
    
    elif state == 5:                        # use A* to follow the grid search path
        direction = 'grid_search'
        path = astar_path(map_computed, sensor_data, direction, square_yaw, position_at_detection, clipping)

        control_command = [0.0, 0.0, height_desired, MOVE_WITH_ROTATION]
        index_current_setpoint = 0
        idx_Gridpoint = 0
        state += 1

    elif state == 6:                        # follow path in the landing area
        #control_command = [0.0, 0.0, height_desired, 0]
        if sensor_data['z_global']-sensor_data['range_down'] >= 0.07:
            print("Landing pad found")
            state += 1
            return [0, 0, height_desired, 0]

        print(f"Path of grid search: {grid_search_points}")
        print(f"Current goal: {grid_search_points[idx_Gridpoint]}")
        #print(f"Grid search path: {path}")
        print(f"Current cells: {int(np.round((sensor_data['x_global'] - min_x )/res_pos,0))}, {int(np.round((sensor_data['y_global'] - min_y )/res_pos,0))}")
        direction = 'grid_search'
        vx,vy, reached = follow_setpoints(sensor_data, direction)
        control_command = [vx, vy, height_desired, 1]

        if reached:
            state -= 2
    
    elif state == 7:
        print("Landing pad found")
        control_command = [0, 0, height_desired, 0]

    elif state == 7:                        # call A* to generate path (return to starting point)
        direction = 'back'
        path = astar_path(map_computed, sensor_data, direction, square_yaw = None, position_at_detection = None, clipping = None)
        #print(f"Path: {path}")
        control_command = [0.0, 0.0, height_desired, MOVE_WITH_ROTATION]
        index_current_setpoint = 1
        state += 1

    elif state == 8:                        # follow path until starting pad is reached
        #print("----- going back -----")
        #print(f"Path: {path}")
        #print(f"Current cells: {int(np.round((sensor_data['x_global'] - min_x )/res_pos,0))}, {int(np.round((sensor_data['y_global'] - min_y )/res_pos,0))}")
        direction = 'back'
        vx,vy, reached = follow_setpoints(sensor_data, direction)
        control_command = [vx, vy, height_desired, MOVE_WITH_ROTATION]
        if reached:
            state += 1

    elif state == 9:                        # landing
        if height_desired > startpos[2]+0.15:
            height_desired -= 0.01
        else:
            height_desired = startpos[2]+0.05
            print("Landed")
            state += 1
        vx,vy = keep_aligned(sensor_data, startpos)
        control_command = [vx, vy, height_desired, 0]
        
    elif state == 10:                        # land after some hovering
        control_command = [0, 0, 0.1, 0]
        if first_time:
            time1 = sensor_data['t']
            first_time = False
        if sensor_data['t'] - time1 >= 3.0:
            control_command = [0.0, 0.0, 0.02, 0]
    
    elif state == 11:                       # make the drone point towards positive x
        print("square yaw: ", square_yaw, "yaw: ", sensor_data['yaw'])
        if square_yaw is not None:
            if (square_yaw-sensor_data['yaw']) <= 0.1 and (square_yaw-sensor_data['yaw']) >= -0.1:
                control_command = [0.0, 0.0, height_desired, 0]
                state += 1
            else:
                control_command = [0.0, 0.0, height_desired, MOVE_WITH_ROTATION]
                print("rotating to face the square")
        else:
            control_command = [0.0, 0.0, height_desired, MOVE_WITH_ROTATION]
    
    elif state == 12:                       # slide along y until facing the square [square_yaw ~ 0]
        control_command = [0.0, 0, height_desired, 0]
        print("Finished. Now it has to go straight")
        state = 1
        clipping = False




        # when reached -> if final area: state+1 | else: state-1 -> rigenero nuovo path da seguire

    # loop(
    # step 1: leggi mappa
    # step 2: crea vettori e trova il punto da raggiungere
    # step 3: crea comandi che lo facciano arrivare lì facendolo anche ruotare
    

    #print(sensor_data['x_global'], sensor_data['y_global'] )
    #no = astar_path(map, sensor_data)

    # Convert desired direction to velocity commands
    # vx = np.cos(desired_direction)  # Move towards the desired direction
    # vy = np.sin(desired_direction)  # Move towards the desired direction
    # print(vx, vy)

    #control_command = [0, 0, height_desired, 0.0]


    # posso tenere uno yaw_rate basso ma costante così che continui sempre a ruotare. Definisco comandi di vx e vy globali e li passo
    # con matrice di rotazione in locale per il drone (dovrebbero gestire anche lo yaw_rate)

    # come ottengo la distanza tra il drone e gli ostacoli rilevati? sulla mappa vengono visualizzati nella posizione giusta, quindi in qualche
    # modo la loro posizione e quindi la distanza dal drone

    # ogni volta che devo ricalcolare le forze attrattive (serve quindi avere un punto come goal) posso prendere la posizione y corrente
    # e settare come goal [x=3.6, y=corrente]

    #control_command = [0.0, 0.0, height_desired, 0]
    #on_ground = False
    #map = occupancy_map(sensor_data)
    
    return control_command # [vx, vy, alt, yaw_rate]
# ------------------------------------------------------------------------------------------------------------------------------------------------

def grid_search_gen(map):
    global visited_cells
    points = []

    for x in range(int(np.round(3.6/res_pos)), map.shape[0] -1, 3):
        for y in range(1, map.shape[1] -1, 2):
            #print(f"x: {x}, y: {y}")
            if map[x, y] >= 0.8 and ((x,y) not in visited_cells):
                points.append((x, y))
    # TODO: add check if around there are at least 2 free cells
    return points


def cost_estimate(start, goal):
    #return math.sqrt( (start[0]-goal[0])**2 + (start[1]-goal[1])**2 )
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])  # Manhattan distance

def goal_finder(start,map):
    min_cost = math.inf
    # find the furthest mapped free point, closest to the final goal area
    for i in range(int(range_max/res_pos), 1, -1):
        # search horizontally for the closest point to the starting poin on the furthest line
        #if i == 0: ?? aggiungere controllo sulla stessa riga e shift ??
            
        for col in range(int((max_y-min_y)/res_pos)):
            if start[0]+i >= int((max_x-min_x)/res_pos):
                i = int((max_x-min_x)/res_pos) - start[0] - 1 # clip to max value of the map
            possible_sol = map[start[0]+i, col]
            #print(f"checking column {col}, cell {(i,col)}, value {possible_sol}")
            #print(possible_sol)
            if possible_sol >= 0.8: 
                cost = cost_estimate(start,(start[0]+i,col))
                if  cost < min_cost:
                    min_cost = cost
                    goal = (start[0]+i, col)
        if min_cost != math.inf:
            return goal
    return

def goal_square(yaw,start,map, clipping):                   # modifico questa per settare come goal x=2.3+-3% e y=x*tan(yaw) cella + cella -

    # se sono a 2.3: scorri lungo x=2.3 o 2.4 per trovare la cella che mi dà yaw = 0 mantenendo drone rivolto verso x positivo
    if start[0] == int(np.round(2.3/res_pos)):
        for i in range(0, int((max_y-min_y)/res_pos)):
            if start[1]+i >= int((max_y-min_y)/res_pos):
                i = int((max_y-min_y)/res_pos) - start[1] - 1




    # find the furthest point in the yaw direction

    # This set the goal in front of the square      [check if cells around x=2.3m and y=x*tan(yaw) are free]
    print("Yaw: ", yaw)
    if yaw >= -np.pi/3 and yaw <= np.pi/3: # values outside this range are not reliable
        front_square_cell = (int(np.round(2.3/res_pos)), int(np.round(2.3*np.tan(yaw)/res_pos)))
        print(f"Front square cell: {front_square_cell}")
        if front_square_cell[0] > 0 and front_square_cell[0] < int((max_x-min_x)/res_pos) and front_square_cell[1] > 0 and front_square_cell[1] < int((max_y-min_y)/res_pos):
            if map[front_square_cell[0], front_square_cell[1]] < 0.8:
                neighbors = get_neighbors(front_square_cell, map)
                print(f"Neighbors: {neighbors}")
                for neigh in neighbors:
                    if map[neigh[0], neigh[1]] >= 0.8:
                        return neigh
            else:
                return front_square_cell

    # This makes the drone move towards the square when far
    for i in range(int(range_max/res_pos), 1, -1):
        dist = i*res_pos
        #if start[0]+dist > int((max_x-min_x)/res_pos):
        #    dist = int((max_x-min_x)/res_pos) - start[0] - 1 # clip to max value of the map

            # make sure the current_setpoint is within the map
        x = int(np.round((start[0] + dist*np.cos(yaw))/res_pos,0))
        y = int(np.round((start[1] + dist*np.sin(yaw))/res_pos,0))
        print(f"Checking cell {(x,y)}")
        
        if x < 0 or x >= map.shape[0] or y < 0 or y >= map.shape[1]:
            if x < 0 or x >= map.shape[0] or y < 0 or y >= map.shape[1]:
                continue

        if clipping: # set to false only when drone shall pass the square
            if x > 2.3/res_pos:
                x = int(np.round(2.3/res_pos)) # the drone will stop at the beginning of the square area
        else:
            ENABLE_SQUARE_DETECTION = 0
        
        if map[x,y] >= 0.8:
            print("Clipped? ", clipping)
            return (x,y) 
        

def get_neighbors(cell, map): # todo: try to remove diagonal neighbors
    x, y = cell
    directions = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx != 0 or dy != 0)]
    neighbors = [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < int((max_x-min_x)/res_pos) and 0 <= y + dy < int((max_y-min_y)/res_pos) and map[x + dx][y + dy] >= 0.8]
    return neighbors

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path

def detect_pink_square(camera_data, sensor_data):
    # Convert RGBA to RGB
    rgb_image = camera_data[:, :, :3].copy()
    alpha_channel = camera_data[:, :, 3].copy()

    lower_pink = np.array([150, 0, 150])
    upper_pink = np.array([255, 100, 255])

    # Threshold the image to isolate pink color
    mask = cv2.inRange(rgb_image, lower_pink, upper_pink)
    _, alpha_mask = cv2.threshold(alpha_channel, 80, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_and(mask, alpha_mask)
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_mid = None
    y_mid = None
    square_yaw = None
    height_square = None

    for contour in contours:
        cv2.drawContours(rgb_image, contour, -1, (0, 255, 0), 4)
        x, y, w, h = cv2.boundingRect(contour) # This will return the 4 values of the rectangle's corners
        
        # center of the rectangle
        x_mid = int(x + w/2)
        y_mid = int(y + h/2)
        #print(f"Center of the rectangle: {x_mid}, {y_mid}")

        # yaw needed by the drone to reach the pink square
        horiz_offset = x_mid - camera_data.shape[1]/2
        square_yaw = sensor_data['yaw'] - np.arctan(horiz_offset/(camera_data.shape[1]/2))# * np.tan(1.5/2)

        # Desidered drone height based on square center
        height_square = sensor_data['z_global'] - 1.5*(y_mid - camera_data.shape[0]/2)

    return rgb_image, (x_mid,y_mid), square_yaw, height_square


def astar_path(map_provided, sensor_data, direction, square_yaw, position_at_detection, clipping):
    global grid_search_points,idx_Gridpoint, target

    start = (int(np.round((sensor_data['x_global'] - min_x )/res_pos,0)), int(np.round((sensor_data['y_global'] - min_y)/res_pos,0)) )
    #print(f"start: {start}")
    if direction == 'forth':
        goal = goal_finder(start,map_provided)
        
        if square_yaw is not None:
            # find the furthest mapped free point, closest to the final goal area
            goal_to_square = goal_square(square_yaw, position_at_detection, map_provided, clipping)
            print(f"Goal to square: {goal_to_square}")
            if goal_to_square is not None:
                goal = goal_to_square
                print("Goal overridden by pink square coordinates")

    elif direction == 'grid_search':
        goal = grid_search_points[idx_Gridpoint]
        idx_Gridpoint += 1
        if idx_Gridpoint == len(grid_search_points):
            idx_Gridpoint = 0
    else:
        # TODO: First goal will be pink square, then landing paf
        goal = (int(np.round((startpos[0]/res_pos))),int(np.round((startpos[1]/res_pos)))) # set starting pad as goal

    target = goal
    #print(f"goal:{goal}")

    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: cost_estimate(start, goal)}

    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        if current == goal:
            #print("path found")
            path = reconstruct_path(came_from, current)
            return path

        open_set.remove(current)
        for neighbor in get_neighbors(current, map_provided):
            tentative_g_score = g_score.get(current, float('inf')) + cost_estimate(current, neighbor)
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + cost_estimate(neighbor, goal)
                if neighbor not in open_set:
                    open_set.add(neighbor)

    # No path found
    return None

def enlarge_obstacles(map_array, enlargement_factor):
    enlarged_map = np.copy(map_array)
    
    # Iterate over each element of the map array
    for i in range(map_array.shape[0]):
        for j in range(map_array.shape[1]):
            # Check if the current element is an obstacle (<= threshold)
            if map_array[i, j] <= -0.6:
                # Enlarge the obstacle by setting neighboring cells to the obstacle value
                for dx in range(-enlargement_factor, enlargement_factor + 1):
                    for dy in range(-enlargement_factor, enlargement_factor + 1):
                        # Calculate the indices of the neighboring cell
                        new_i = i + dx
                        new_j = j + dy
                        
                        # Check if the neighboring cell is within the map boundaries (map boundaries are not enlarged)
                        if 1 < new_i < map_array.shape[0]-2 and 1 < new_j < map_array.shape[1]-1:
                            # Update the neighboring cell with the obstacle value
                            enlarged_map[new_i, new_j] = -1#map_array[i, j] | every cell becomes full -1   
    return enlarged_map


def occupancy_map(sensor_data):
    global map, t, target
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    yaw = sensor_data['yaw']

    for j in range(4): # 4 sensors
        yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']
        
        for i in range(int(range_max/res_pos)): # range is 2 meters
            dist = i*res_pos
            idx_x = int(np.round((pos_x - min_x + dist*np.cos(yaw_sensor))/res_pos,0))
            idx_y = int(np.round((pos_y - min_y + dist*np.sin(yaw_sensor))/res_pos,0))

            # make sure the current_setpoint is within the map
            if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > range_max:
                break

            # update the map
            if dist < measurement:
                map[idx_x, idx_y] += conf
            else:
                map[idx_x, idx_y] -= conf
                break

    map = np.clip(map, -1, 1) # certainty can never be more than 100%

    # Create a map copy to enlarge obstacles
    map_enlarged = enlarge_obstacles(map, int(SAFETY_MARGIN/res_pos))

    # always recreate borders and map them as obstacles
    map_enlarged[0,:] = -1
    map_enlarged[-1,:] = -1
    map_enlarged[:,0] = -1
    map_enlarged[:,-1] = -1

    # only plot every Nth time step (comment out if not needed)
    if t % 50 == 0:
        plt.imshow(np.flip(map_enlarged,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        #plt.imshow(map, vmin=-1, vmax=1, cmap='gray', origin='lower')

        # show current point in map, and target point 
        start_x = int(np.round((sensor_data['x_global'] - min_x)/res_pos,0))
        start_y = int(np.round((sensor_data['y_global'] - min_y)/res_pos,0))
        if target != None:
            rect_arr = plt.Rectangle(((max_y-min_y)/res_pos - target[1] -0.5, map_enlarged.shape[0] - target[0] -0.5), 1, 1, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect_arr)
        rect_beg = plt.Rectangle(((max_y-min_y)/res_pos - start_y -0.5, start_x - 0.5), 1, 1, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect_beg)
        

        plt.savefig("map.png")
        plt.close()
        #matrix = pd.DataFrame(map)
        #matrix.to_excel(excel_writer="/Users/riccardolionetto/Documents/EPFL/aerial-robotics/controllers/main/map.xlsx")
        matrix_enlarged = pd.DataFrame(map_enlarged)
        matrix_enlarged.to_excel(excel_writer="/Users/riccardolionetto/Documents/EPFL/aerial-robotics/controllers/main/map_enlarged.xlsx")

    t +=1

    return map_enlarged


def follow_setpoints(sensor_data, direction):
    global path, index_current_setpoint, visited_cells
    #print(f"Path: {path}", f"Current setpoint: {index_current_setpoint}")
    
    try:
        # Get the goal position and drone position
        current_setpoint = path[index_current_setpoint]
    except:
        return 0, 0, 1
    
    #print(f"Point to reach: {current_setpoint}")
    #print(f"Dist x is {current_setpoint[0]} - {sensor_data['x_global']/res_pos}, x global:{sensor_data['x_global']}")
    #print(f"Dist y is {current_setpoint[1]} - {sensor_data['y_global']/res_pos}, y global:{sensor_data['y_global']}")

    # Add visited cells (used for grid search)
    if (int(np.round(sensor_data['x_global']/res_pos)), int(np.round(sensor_data['y_global']/res_pos))) not in visited_cells:
        visited_cells.append((int(np.round(sensor_data['x_global']/res_pos)), int(np.round(sensor_data['y_global']/res_pos))))

    # The drone will reach the center of the cells -> -0.5
    #dist_x = current_setpoint[0]-0.5 - sensor_data['x_global']/res_pos
    #dist_y = current_setpoint[1]-0.5 - sensor_data['y_global']/res_pos
    dist_x = current_setpoint[0] - sensor_data['x_global']/res_pos
    dist_y = current_setpoint[1] - sensor_data['y_global']/res_pos
    vx = 0.25*dist_x
    vy = 0.25*dist_y
    
    #Clipping
    if vx > 0.5:
        vx = 0.5
    if vy > 0.5:
        vy = 0.5


    distance_drone_to_goal = np.linalg.norm([dist_x, dist_y])
    #print(f"Distance to goal: {distance_drone_to_goal}")
    # When the drone reaches the goal setpoint, e.g., distance < 0.1m   
    if distance_drone_to_goal < 0.1: # this could be set to 2*res_pos?
        #print("Point reached")
        
        # Hover at the final setpoint if target is reached
        if direction == 'forth':

            if ENABLE_SQUARE_DETECTION==0:
                if index_current_setpoint == math.ceil(len(path)/4): #len(path)-1: this would take the drone to the furthest point, reduce it to increase safety (paht will be recalculated earlier)
                    return 0,0,1
                else:
                    index_current_setpoint += 1 # Select the next setpoint as the goal position
            else:
                if index_current_setpoint == math.ceil(len(path)/1.5): #len(path)-1: this would take the drone to the furthest point, reduce it to increase safety (paht will be recalculated earlier)
                    return 0,0,1
                else:
                    index_current_setpoint += 1 # Select the next setpoint as the goal position
        
        else: # valid for 'grid_search' and 'back'
            if index_current_setpoint == len(path)-1: #reach the final goal (landing pad)
                return 0,0,1
            else:
                index_current_setpoint += 1
    
    # transform commands from inertial frame (ground) to body frame (drone)
    euler_angles = [sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw']]
    R = euler2rotmat(euler_angles)
    vel_body = np.linalg.inv(R) @ [vx, vy, 0]

    return vel_body[0],vel_body[1], 0 # vx,vy | reached = 1 if the drone reached the final goal, 0 otherwise


def is_valid_cell(cell, map):
    return 5.6/res_pos <= cell[0] < map.shape[0] and 0 <= cell[1] < map.shape[1] and map[cell] >= 0.8


def get_neighbors_noDiag(cell, map):
    x, y = cell
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    neighbors = [(x + dx, y + dy) for dx, dy in directions if is_valid_cell((x + dx, y + dy), map)]
    return neighbors

def single_agent_coverage(map, sensor_data):
    #start = (int(np.round((sensor_data['x_global'] - min_x )/res_pos,0)), int(np.round((sensor_data['y_global'] - min_y)/res_pos,0)) )
    ## Initialize a stack to store the cells to visit
    #stack = [start]
    #
    ## Initialize a set to keep track of visited cells
    #visited = set()
    #
    ## Initialize a list to store the path
    #path = []
    #
    ## Perform DFS until the stack is empty
    #while stack:
    #    # Pop the top cell from the stack
    #    current = stack.pop()
    #    
    #    # Check if the current cell is valid and not visited yet
    #    if current not in visited and is_valid_cell(current, map):
    #        # Mark the current cell as visited
    #        visited.add(current)
    #        
    #        # Add the current cell to the path
    #        path.append(current)
    #        
    #        # Get neighboring cells
    #        neighbors = get_neighbors_noDiag(current, map)
    #        
    #        # Add valid neighbors to the stack
    #        for neighbor in neighbors:
    #            stack.append(neighbor)
    
    # Get the current cell as the start position
    start = (int(np.round((sensor_data['x_global'] - min_x )/res_pos,0)), int(np.round((sensor_data['y_global'] - min_y)/res_pos,0)) )
    
    # Initialize a list to store the path
    path = [start]
    
    # Initialize a variable to toggle between searching rows and skipping rows
    search_row = True
    
    # Iterate through all rows
    for row in range(start[0], map.shape[0]):
        # Toggle between searching rows and skipping rows
        if search_row:
            # Iterate through all columns in the row
            for col in range(map.shape[1]):
                cell = (row, col)
                # Check if the current cell is valid and not visited yet
                if is_valid_cell(cell, map):
                    # Add the current cell to the path
                    path.append(cell)
                    # Get neighboring cells
                    neighbors = get_neighbors_noDiag(cell, map)
                    # Add valid neighbors to the path
                    for neighbor in neighbors:
                        path.append(neighbor)
            search_row = False
        else:
            search_row = True

    return path


def keep_aligned(sensor_data, startpos):
    vx = (sensor_data['x_global'] - startpos[0])*0.1
    vy = (sensor_data['y_global'] - startpos[1])*0.1
    return vx,vy



# ------------------------------------------------------------------------------------------------------------------------------------------------


# Control from the exercises
# index_current_setpoint = 1 # Uncomment for proper working of this function
def path_to_setpoint(path,sensor_data,dt):
    global on_ground, height_desired, index_current_setpoint, timer, timer_done, startpos

    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]    
    if on_ground and sensor_data['z_global'] < 0.49:
        current_setpoint = [startpos[0], startpos[1], height_desired, 0.0]
        return current_setpoint
    else:
        on_ground = False

    # Start timer
    if (index_current_setpoint == 1) & (timer is None):
        timer = 0
        print("Time recording started")
    if timer is not None:
        timer += dt
    # Hover at the final setpoint
    if index_current_setpoint == len(path):
        # Uncomment for KF
        control_command = [startpos[0], startpos[1], startpos[2]-0.05, 0.0]

        if timer_done is None:
            timer_done = True
            print("Path planing took " + str(np.round(timer,1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[index_current_setpoint]
    x_drone, y_drone, z_drone, yaw_drone = sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']
    distance_drone_to_goal = np.linalg.norm([current_setpoint[0] - x_drone, current_setpoint[1] - y_drone, current_setpoint[2] - z_drone, clip_angle(current_setpoint[3]) - clip_angle(yaw_drone)])

    # When the drone reaches the goal setpoint, e.g., distance < 0.1m
    if distance_drone_to_goal < 0.1:
        # Select the next setpoint as the goal position
        print(f"{index_current_setpoint} setpoint reached, moving.")
        index_current_setpoint += 1
        # Hover at the final setpoint
        if index_current_setpoint == len(path):
            current_setpoint = [0.0, 0.0, height_desired, 0.0]
            return current_setpoint

    return current_setpoint

def clip_angle(angle):
    angle = angle%(2*np.pi)
    if angle > np.pi:
        angle -= 2*np.pi
    if angle < -np.pi:
        angle += 2*np.pi
    return angle
