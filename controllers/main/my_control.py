# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import ndimage
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
SAFETY_MARGIN = 0.15     # meters

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
    global on_ground, startpos, state, path, index_current_setpoint, first_time, time1

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)
    
    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]  
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----
    map_computed = occupancy_map(sensor_data)

    if state == 0:                          # turning in place
        control_command = [0.0, 0.0, height_desired, 1]
        if sensor_data['t'] >= 6.0:
            state += 1
    elif state == 1:                        # call A* to generate path
        path = astar_path(map_computed, sensor_data)
        print(f"Path: {path}")
        control_command = [0.0, 0.0, height_desired, 1]
        index_current_setpoint = 1
        state += 1
    elif state == 2:                        # follow path until euristic goal is reached
        print("----- follow_setpoint -----")
        print(f"Path: {path}")
        print(f"Current cells: {int(np.round((sensor_data['x_global'] - min_x )/res_pos,0))}, {int(np.round((sensor_data['y_global'] - min_y )/res_pos,0))}")
        vx,vy, reached = follow_setpoints(sensor_data)
        control_command = [vx, vy, height_desired, 1]

        #x_position = int(np.round((sensor_data['x_global'] - min_x )/res_pos,0))
        if sensor_data['x_global'] >= 3.7:
            #first_time = True
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
            state += 1
    
    elif state == 4:                        # plan path for grid search in the landing area
        control_command = [0.0, 0.0, height_desired, 0]
        path = single_agent_coverage(map_computed, sensor_data)
        state += 1
    elif state == 5:                        # follow path in the landing area
        control_command = [0.0, 0.0, height_desired, 0]
        print(f"Path: {path}")
        print(f"Current cells: {int(np.round((sensor_data['x_global'] - min_x )/res_pos,0))}, {int(np.round((sensor_data['y_global'] - min_y )/res_pos,0))}")
        vx,vy, reached = follow_setpoints(sensor_data)
        control_command = [vx, vy, height_desired, 1]


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

def astar_path(map_provided, sensor_data):
    global target
    # estraggo starting node
    # setto goal: cerco a partire dallo starting point il punto 1.8 metri più avanti che abbia casella a 0?
    
    # if spia = 1 
        # algoritmo per trovare il path da seguire -> spia = 0
        # genero i waypoints da seguire (coordinate)
    # finchè la lista di waypoints non è esaurita -> muovi il drone
    # qunado è esaurita -> richiama a* e genera un nuovo path
    print("astar in")
    start = (int(np.round((sensor_data['x_global'] - min_x )/res_pos,0)), int(np.round((sensor_data['y_global'] - min_y)/res_pos,0)) )
    print(f"start: {start}")
    goal = goal_finder(start,map_provided)
    target = goal
    print(f"goal:{goal}")
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: cost_estimate(start, goal)}

    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        if current == goal:
            print("path found")
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
                # enlarge the obstacle for safety
                #for k in range(1, int(SAFETY_MARGIN/res_pos)):
                #    idx_x = int(np.round((pos_x - min_x + (i*res_pos-k)*np.cos(yaw_sensor))/res_pos,0))
                #    idx_y = int(np.round((pos_y - min_y + (i*res_pos-k)*np.sin(yaw_sensor))/res_pos,0))
                #    map[idx_x, idx_y] -= conf
                #break
    

    map = np.clip(map, -1, 1) # certainty can never be more than 100%

    # Create a map copy to enlarge obstacles
    # Enlarge obstacles
    map_enlarged = enlarge_obstacles(map, int(SAFETY_MARGIN/res_pos))
    #map = copy.deepcopy(map_enlarged)
    #map = map_enlarged.copy()
    #print("Map")
    #print(map.astype(int))
    #map_copy = np.copy(map)
    #map_copy[map_copy > -0.6] = 0
    #map_copy[map_copy <= -0.6] = 1
    #print("Map copy")
    #print(map_copy)
    #struc = ndimage.generate_binary_structure(2, 2)
    #obstacles_dilatated = ndimage.binary_dilation(map_copy, struc, iterations=1)#iterations=int(SAFETY_MARGIN / res_pos))
    #print("Obstacles dilatated")
    #print(obstacles_dilatated)
    #map_enlarged = np.copy(map)
    #map_enlarged[obstacles_dilatated] = -1
    #map[obstacles_dilatated] = -1
    #print("Map after dilatation")
    #print(map)


    # always recreate borders and map them as obstacles
    #map[0,:] = -1
    #map[-1,:] = -1
    #map[:,0] = -1
    #map[:,-1] = -1
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
        matrix = pd.DataFrame(map)
        matrix.to_excel(excel_writer="/Users/riccardolionetto/Documents/EPFL/aerial-robotics/controllers/main/map.xlsx")
        matrix_enlarged = pd.DataFrame(map_enlarged)
        matrix_enlarged.to_excel(excel_writer="/Users/riccardolionetto/Documents/EPFL/aerial-robotics/controllers/main/map_enlarged.xlsx")

    t +=1

    return map_enlarged




def follow_setpoints(sensor_data):
    global path, index_current_setpoint, target

    # Get the goal position and drone position
    reached = 0
    current_setpoint = path[index_current_setpoint]
    print(f"Point to reach: {current_setpoint}")
    #print(f"Dist x is {current_setpoint[0]} - {sensor_data['x_global']/res_pos}, x global:{sensor_data['x_global']}")
    #print(f"Dist y is {current_setpoint[1]} - {sensor_data['y_global']/res_pos}, y global:{sensor_data['y_global']}")

    # The drone will reach the center of the cells -> -0.5
    #dist_x = current_setpoint[0]-0.5 - sensor_data['x_global']/res_pos
    #dist_y = current_setpoint[1]-0.5 - sensor_data['y_global']/res_pos
    dist_x = current_setpoint[0] - sensor_data['x_global']/res_pos
    dist_y = current_setpoint[1] - sensor_data['y_global']/res_pos
    vx = 0.25*dist_x
    vy = 0.25*dist_y

    distance_drone_to_goal = np.linalg.norm([dist_x, dist_y])
    print(f"Distance to goal: {distance_drone_to_goal}")
    # When the drone reaches the goal setpoint, e.g., distance < 0.1m   
    if distance_drone_to_goal < 0.1: # this could be set to 2*res_pos?
        print("Point reached")
        
        # Hover at the final setpoint if target is reached
        if index_current_setpoint == math.ceil(len(path)/4): #len(path)-1: this would take the drone to the furthest point, reduce it to increase safety (paht will be recalculated earlier)
            #if distance_drone_to_goal < 0.1:
            reached = 1
                #print(f"Arrived at the end: {sensor_data['x_global'], sensor_data['y_global']}")
                #index_current_setpoint = 1
                #current_setpoint = [0.0, 0.0, height_desired, 0.0]
            return 0,0,1
        else:
            # Select the next setpoint as the goal position
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