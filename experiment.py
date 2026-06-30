def main(seed, baseline, alignment, communication, weights, swarm_size = 10, folder_name = None): 
    import pygame
    from sys import exit #not used rn
    import sys
    import random
    import numpy as np
    import math
    import csv
    import scipy
    
    #BASELINE WITH RANDOM ROBOT MOTION, NO COMMUNICATION

    random.seed(seed)

    def display_time():
        current_time = pygame.time.get_ticks() - start_time # time in ms
        time_str = str(round(current_time/1000, 1)) + '/' + str(sim_duration/1000) + 'sec'

        courier = pygame.font.SysFont('couriernew', 20)
        time_surface = courier.render(time_str, False, (64, 64, 64))
        time_rect = time_surface.get_rect(topright = (display_width-10, 10))
        screen.blit(time_surface, time_rect)

        return current_time

    def display_score():
        courier = pygame.font.SysFont('couriernew', 20)
        score_surface = courier.render(f'Collected Food: {global_food_count}', False, (64, 64, 64))
        score_rect = score_surface.get_rect(topright = (display_width-10, 30))
        screen.blit(score_surface, score_rect)
        return current_time

    def draw_network():
        robot_ids = []
        if communication:
            for robot1 in robots:
                robot_ids.append(robot1.id)
                for robot2 in robots:
                    if robot2.id in robot_ids:
                        continue
                    if robot1.id in robot2.send_communication or robot2.id in robot1.send_communication: pygame.draw.line(screen, 'darkblue', robot1.rect.center, robot2.rect.center, 2)
        else: 
            for robot in robots:
                for robot2 in robots:
                    if robot.id != robot2.id:
                        (x1, y1) = robot2.rect.center
                        (x2, y2) = robot.rect.center
                        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if dist < vision_radius: pygame.draw.line(screen, 'darkblue', robot.rect.center, robot2.rect.center, 2)
                            
                        

    def normalize_vector(my_vector):
        (x, y) = my_vector
        length = math.sqrt(x**2 + y**2)
        result = my_vector/length
        if (x, y) == (0, 0):
            print(f'ERROR NORMALIZING VECTOR {x, y} with length {length}')
        return my_vector/length

    def change_angle(my_vector, max_angle):
        current_angle = np.arctan2(
            my_vector[1],
            my_vector[0]
        )

        # Random deviation between -max_angle° and +max_angle° (in degrees, here converted to rad)
        angle = max_angle * np.pi/180
        deviation = random.uniform(-angle, angle)

        # New angle
        new_angle = current_angle + deviation

        # Updated direction vector
        my_vector = np.array([
            np.cos(new_angle),
            np.sin(new_angle)
        ])
        return my_vector

    def log_activity(current_time, activity):
        #header = ['time', 'robot_id', 'location', 'velocity', 'carries_food', 'neighbors', 'food_location_knowledge', 'food_count_knowledge', 'delivered_food', 'sending_communication', 'activity']
        filename = folder_name + "/timelog.csv"
        with open(filename, 'a', newline='') as timelog:
            timelog_writer = csv.writer(timelog, delimiter=',')
            #timelog_writer.writerow([str(current_time)])
            for robot in robots:
                row = [str(current_time), robot.id, robot.rect.center, robot.normal_direction, robot.food, robot.neighbors, robot.food_locations, robot.food_count_knowledge, robot.food_count_knowledge[robot.id], robot.send_communication, activity]
                timelog_writer.writerow(row)

    def log_weights():
        filename = folder_name + "/weights.csv"
        with open(filename, 'w', newline='') as weightlog:
            weightlog_writer = csv.writer(weightlog, delimiter=',')
            robot_ids = [i for i in range(num_robots)]
            weightlog_writer.writerow(['this_robot_id'] + robot_ids)
            
            for robot in robots:
                row = []
                row.append(robot.id)
                for i in range(num_robots):
                    row.append(robot.weights[str(i)])
                weightlog_writer.writerow(row)
        weightlog.close()

        current_time = pygame.time.get_ticks() - start_time
        filename = folder_name + "/stats.csv"
        with open(filename, 'w', newline='') as statslog:
            statslog_writer = csv.writer(statslog, delimiter=',')
            statslog_writer.writerow(['this_robot_id', 'delivered_food', 'total_collected_food', 'all_food_found_time', 'end_time'])
            
            for robot in robots:
                row = [robot.id, robot.food_count_knowledge[robot.id], global_food_count, found_time, current_time]
                statslog_writer.writerow(row)
        statslog.close()

    def generate_food_positions(num_food):
        l_bounds = [30, 30]
        u_bounds = [display_width-30, display_height-30]
        sampler = scipy.stats.qmc.PoissonDisk(d=2, radius=120, l_bounds = l_bounds, u_bounds = u_bounds, seed = seed)

        samples = sampler.random(num_food*2)  # oversample to compensate filtering
        #print('samples:', len(samples))

        x = samples[:, 0]
        y = samples[:, 1]

        return np.column_stack((x, y))
    
    def is_valid_point(p):
            my_surface = pygame.Surface((15, 15))
            rect = my_surface.get_rect(center=(p[0], p[1]))

            if rect.colliderect(home_margin_rect): return False
            return True
    

    # SETUP
    # pygame setup
    pygame.init()
    display_width = 900
    display_height = 900
    screen = pygame.display.set_mode((display_width, display_height))
    clock = pygame.time.Clock()
    #pygame.display.set_caption('Foraging Simulation')
    running = True
    dt = 0

    pygame.display.set_caption(f'Foraging Simulation - baseline weights: {baseline}, alignment: {alignment}, comm: {communication}')

    num_robots = swarm_size
    num_food = 20
    #num_obstacles = 5
    lines = True
    global_food_count = 0
    vision_radius = 60 # vision radius for the robots (before 50)
    spawn_margin = 0
    if swarm_size == 30: spawn_margin = 100
    food_respawn_time = 1000

    home_diameter = 150
    margin = 40

    picked_up_food = [] # control picked up food
    found_food = []
    found_time = None

    # set up logs
    with open('timelog.csv', 'w', newline='') as timelog:
        timelog_writer = csv.writer(timelog, delimiter=',')

        header = ['time', 'robot_id', 'location', 'velocity', 'carries_food', 'neighbors', 'food_location_knowledge', 'food_count_knowledge', 'delivered_food', 'sending_communication', 'activity']
        timelog_writer.writerow(header)

    #duration and timer
    sim_duration = 300000 # max 100000000
    timer = pygame.USEREVENT + 1
    pygame.time.set_timer(timer, sim_duration)
    start_time = pygame.time.get_ticks()

    log_interval = 100
    log_timer = pygame.USEREVENT + 2
    pygame.time.set_timer(log_timer, log_interval)
    start_log = pygame.time.get_ticks()
    log_num = 0

    # FOOD CLASS
    class Food(pygame.sprite.Sprite):
        def __init__(self, food_id):
            super().__init__()
            self.surface = pygame.Surface((15, 15))
            self.radius = 15
            #self.rect = self.surface.get_rect(center = (random.randint(30, display_width-30), random.randint(30, display_height-30)))
            nonlocal valid_points
            self.rect = self.surface.get_rect(center = valid_points[food_id])
            self.id = food_id
            self.color = (143, 213, 108)
            self.spawned = False
            self.color_cooldown = 0
            self.exists = True
            self.found = False
            self.respawn_counter = 0

        def draw_food(self):
            #pygame.draw.circle(screen, self.color, self.rect.center, self.radius)
            self.surface.fill(self.color)
            
            #change color if collision
            if self.exists:
                    for robot in robots:
                        if self.rect.colliderect(robot.rect) and robot.food == False:
                            self.color_cooldown += 10
                            self.exists = False
                    
            if self.color_cooldown > 0: 
                self.surface.fill('darkred')
                self.color_cooldown -= 1

            elif self.found: self.surface.fill('darkolivegreen')

            screen.blit(self.surface, self.rect)
        
        def spawn_food(self):
            center_x = display_width / 2
            center_y = display_height / 2

            for food in food_group:
                if food.id != self.id and food.spawned == False:
                    while self.rect.colliderect(home_margin_rect) or self.rect.colliderect(food.rect) or ((self.rect.centerx >= center_x - home_diameter - margin and self.rect.centerx <= center_x + home_diameter + margin) and (self.rect.centery >= center_y - home_diameter - margin and self.rect.centery <= center_y + home_diameter + margin)): # also checks collision manually
                        self.rect = self.surface.get_rect(center = (random.randint(30, display_width-30), random.randint(30, display_height-30)))
                    self.spawned = True


        def be_collected(self):
            pass

        def update(self):
            if self.exists or self.color_cooldown > 0: self.draw_food()

            if not self.exists: self.respawn_counter += 1
            if self.respawn_counter >= food_respawn_time:
                self.exists = True
                try:
                    picked_up_food.remove(self.rect.center)
                except:
                    print('Item was not in the list and thus couldnt be removed.')
                self.respawn_counter = 0

    # OBSTACLE CLASS (not used, can potentially be expanded for future experiments)
    class Obstacle(pygame.sprite.Sprite):
        def __init__(self, obstacle_id):
            super().__init__()
            self.surface = pygame.Surface((15, 15))
            self.rect = self.surface.get_rect(center = (random.randint(30, display_width-30), random.randint(30, display_height-30)))
            self.id = obstacle_id

    # ROBOT CLASS
    class Robot(pygame.sprite.Sprite):
        def __init__(self, robot_id):
            super().__init__()
            self.image = pygame.image.load('graphics/robot_black_32.png').convert_alpha()
            #self.rect = self.image.get_rect(center = (random.randint(30, display_width-30), random.randint(30, display_height-30)))
            self.rect = self.image.get_rect(center = (random.randint(home_margin_rect.left, home_margin_rect.right), random.randint(home_margin_rect.top, home_margin_rect.bottom)))
            self.id = robot_id
            self.spawned = False

            #movement
            angle = random.uniform(0, 2 * np.pi)
            self.normal_direction = np.array([np.cos(angle), np.sin(angle)])
            self.speed = random.uniform(1, 1.5) #right now each robot just has a slightly different speed, change later?
            self.speed = 1
            self.drop_cooldown = 0 # drop cooldown to avoid wrongly dropping the same food multiple times
            self.collide_cooldown = 0 # collide cooldown to avoid overcorrection and "wiggleing"

            # locations and states
            self.food = False # True = currently carrying food, False = not carrying food rn
            self.food_locations = [] # known locations of existent and collected food
            self.neighbors = {} # current neighbors {robot.id: [robot.rect.center, dist]}
            self.weights = {} # how much the robot is influenced by other robots
            self.send_communication = {} # with whom which communication is currently shared {to_id: [food_locations, food_count_knowledge, timer]} 

            self.food_count_knowledge = {} # {id: (believed) food_drops}
            self.belief_food_count = 0
            self.food_count_knowledge[self.id] = 0


        def spawn(self):
            '''
            Makes sure robots don't spawn on top of each other.
            Spawn inside of home base (with margin)
            Initializes weights.
            '''
            #loction
            nonlocal spawn_margin
            counter = 0
            collide = True
            while collide:
                collide = False
                for robot in robots:
                    if robot.id != self.id:
                        if self.rect.colliderect(robot.rect):
                            counter += 1
                            collide = True
                            #self.rect = self.image.get_rect(center = (random.randint(30, display_width-30), random.randint(30, display_height-30)))
                            self.rect = self.image.get_rect(center = (random.randint(home_margin_rect.left + spawn_margin, home_margin_rect.right + spawn_margin), random.randint(home_margin_rect.top + spawn_margin, home_margin_rect.bottom + spawn_margin)))
                            if counter >= 5:
                                counter = 0
                                spawn_margin += 5
            self.spawned = True

            #weights
            if not baseline:
                self.weights = weights[self.id]
            else:
                for robot in robots:
                    if robot.id == self.id: self.weights.setdefault(str(robot.id), 1)
                    else: self.weights.setdefault(str(robot.id), 1)

        def move_robot(self):
            '''
            Moves robot.
            '''
            if self.collide_cooldown == 0:
                self.change_direction()
                self.normal_direction = normalize_vector(self.normal_direction)
                self.adapt_movement()
                self.normal_direction = normalize_vector(self.normal_direction)
                self.align_movement()

            # EXPLORATION STATE
            if self.food == False:
                pass

            # RETURNING STATE ADD FORCE TOWARDS HOME
            else:
                self.go_home()
            
            # dont move outside of the border
            self.normal_direction = normalize_vector(self.normal_direction)
            (x,y) = self.rect.center + self.normal_direction * self.speed
            while x < 5 or x > display_width-5 or y < 5 or y > display_height-5:
                angle = random.uniform(0, 2 * np.pi)
                self.normal_direction = normalize_vector(np.array([np.cos(angle), np.sin(angle)]))
                #self.normal_direction = change_angle(self.normal_direction, 10)
                (x,y) = self.rect.center + self.normal_direction * self.speed
            self.rect.center += + self.normal_direction * self.speed

        def go_home(self):
            '''
            Move back to base if carrying food.
            '''
            center = np.array([display_width / 2, display_height / 2])

            # vector from robot to center
            to_center = center - np.array(self.rect.center)
            self.normal_direction += to_center

        def align_movement(self): 
            '''
            Adapt movement according to other robots and interaction weights.
            Adapt movement towards known food.
            '''
            if alignment:
                for robot in robots:
                    if robot.id in self.neighbors and robot.id != self.id:

                        #alignment
                        #prevent zero vector when two robots from directly opposite directions meet and at least one of them has a weight of 1
                        weight = self.weights[str(robot.id)]
                        if self.weights[str(robot.id)] == 1: 
                            if self.normal_direction[0] == -robot.normal_direction[0] and self.normal_direction[1] == -robot.normal_direction[1]:
                                #print('ERROR PREVENTED, WEIGHT CHANGED')
                                weight = 0.99
                        if robot.food: weight = weight * 0.5 # follow robots who go home less

                        self.normal_direction += (robot.normal_direction * weight)
                        
        def adapt_movement(self): 
            # movement to existing food
            if not self.food:   
                try:
                    direction = np.array([0.0, 0.0])
                    pos = np.array(self.rect.center)

                    for food_id, food_location, exists, time in self.food_locations:
                        if exists:
                            to_food = np.array(food_location) - pos

                            dist = np.linalg.norm(to_food) + 1e-8
                            unit = to_food / dist

                            # distance weighting (choose one below)
                            influence = unit / dist          # linear falloff
                            # influence = unit / (dist**2)   # stronger falloff

                            direction += influence

                    if np.linalg.norm(direction) > 0:
                        direction /= np.linalg.norm(direction)

                    self.normal_direction += direction
                except: pass

        def change_direction(self):
            '''
            Determines direction of robot movement and avoids collisions with other robots and environment walls.
            '''
            #bounce off wall 
            if self.rect.bottom >= display_height or self.rect.top <= 0 or self.rect.left <= 0 or self.rect.right >= display_width:
                if self.collide_cooldown == 0: self.collide_cooldown = 50
                center = np.array([display_width / 2, display_height / 2])
                to_center = center - np.array(self.rect.center)
                self.normal_direction *= -1
                self.normal_direction += to_center
                self.normal_direction = change_angle(self.normal_direction, 5)

            #random 20% chance of chnaging direction slightly
            else: 
                chance = random.uniform(0,1)
                if chance >= 0.8: 

                    self.normal_direction = change_angle(self.normal_direction, 20)
                    
            # avoid robot collision by changing the direction if colliding
            for robot in robots:
                if robot.id != self.id:
                    if self.rect.colliderect(robot.rect):
                        #bounce off of other robot
                        if self.collide_cooldown == 0: self.collide_cooldown = 50
                        self.normal_direction = change_angle(self.normal_direction, 10)

        # communication currently not used in the experiment, but a setup for communication delay beased on the interaction weights was implemented for future experiments                               
        def communicate(self):
            '''
            Send knowledge to neighbor robots. Process that knowledge once the data has been received.
            '''
            try:
                if len(self.neighbors) > 0:
                    for neighbor_id in self.neighbors:
                        if neighbor_id not in self.send_communication:
                            for robot in robots:
                                if robot.id == neighbor_id:
                                    timer_scale = robot.weights[str(self.id)]
                                    break
                            self.send_communication[neighbor_id] = [self.food_locations, self.food_count_knowledge, 300 - (299 * timer_scale)]

            except KeyError:
                current_time = pygame.time.get_ticks() - start_time
                #print('COMMUNICATION FAILED, KEY ERROR', current_time)

            completed = []
            for neighbor_id, data in self.send_communication.items():
                    data[-1] -= 1
                    if communication and data[-1] <= 0: 
                        self.integrate_knowledge(neighbor_id, self.send_communication[neighbor_id][:-1])
                        completed.append(neighbor_id)

            for neighbor_id in completed:
                del self.send_communication[neighbor_id]

        def integrate_knowledge(self, receiver_id, knowledge):
            """
            Integrate knowledge that this robot has sent into the knowledge base of the other robot.
            """
            #print(f'Robot {receiver_id} integrates knowledge from robot {self.id}')

            for robot in robots:
                if robot.id == receiver_id:
                    break
            food_loc, food_counts = knowledge
            
            #print(f'Knowledge before: {len(robot.food_locations)}, belief: {robot.belief_food_count}')
            before = robot.food_locations.copy()

            # update food count believe
            for id, food_count in food_counts.items():
                if id in robot.food_count_knowledge:
                    if food_count > robot.food_count_knowledge[id]: robot.food_count_knowledge[id] = food_count
                else: robot.food_count_knowledge[id] = food_count

            food_belief = 0
            for _, food_count in robot.food_count_knowledge.items():
                food_belief += food_count
            robot.belief_food_count = food_belief

            # update food location list with new knowledge if knowledge is more recent or new
            for my_food_id, my_food_loc, my_exists, my_time in food_loc:
                food_in_list = False
                for i in range(len(robot.food_locations)):
                    if my_food_id == robot.food_locations[i][0]:
                        food_in_list = True
                        if my_time > robot.food_locations[i][3]: robot.food_locations[i] = [my_food_id, my_food_loc, my_exists, my_time]
                        break
                if not food_in_list:
                    robot.food_locations.append([my_food_id, my_food_loc, my_exists, my_time])

            
            #print(f'Knowledge after: {len(robot.food_locations)}, belief: {robot.belief_food_count}. Knowledge changed: {before != robot.food_locations}')

        def sense_surroundings(self):

            #sense surrounding robots
            neighbors = {}
            (x1, y1) = self.rect.center
            for robot in robots:
                if robot.id != self.id:
                    (x2, y2) = robot.rect.center
                    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < vision_radius and robot.id not in neighbors:
                        neighbors.setdefault(robot.id, [robot.rect.center, dist])
            #if len(self.neighbors) > 0 and len(self.neighbors) != len(neighbors): print(f'ID: {self.id}, N: {self.neighbors}')
            if neighbors != self.neighbors: self.neighbors = neighbors

            # find surrounding food
            current_time = pygame.time.get_ticks() - start_time
            if True:
                for food in food_group:
                    (x3, y3) = food.rect.center
                    dist = math.sqrt((x1-x3)**2 + (y1-y3)**2)

                    #add food that you have never seen/heard about if you see it
                    if dist < vision_radius and food.id not in [food[0] for food in self.food_locations] and food.exists: 
                        self.food_locations.insert(0, [food.id, food.rect.center, True, current_time])
                        #pygame.draw.line(screen, 'orange', (x1, y1), food.rect.center, 2)
                        if not food.found: 
                            food.found = True
                            found_food.append(food.id)
                            nonlocal found_time
                            if len(found_food) == num_food: found_time = current_time

                    #update food you have already seen/heard about if you see it or see that its not there
                    if dist < vision_radius and food.id in [food[0] for food in self.food_locations]: 
                        for i in range(len(self.food_locations)):
                            if [food.id, food.rect.center] == self.food_locations[i][:-2]:
                                self.food_locations[i] = [food.id, food.rect.center, food.exists, current_time]
                                #print('updated food location')
                                break


        def collect_food(self):
            '''
            Collecting food. Update food locations lists (show that food has been collected)
            '''
            if self.food == False:
                for food in food_group:
                    if self.rect.colliderect(food.rect) and (food.exists or food.color_cooldown > 0) and food.rect.center not in picked_up_food:
                        self.food = True
                        picked_up_food.append(food.rect.center)
                        #print(f'Robot number {self.id} collected food at {food.rect.center}')
                        current_time = pygame.time.get_ticks() - start_time
                        log_activity(current_time, {'activity': 'food_collect', 'robot_id': self.id, 'location': food.rect.center})

                        #update food locations list
                        for i in range(len(self.food_locations)):
                            food_id, food_loc, exists, time = self.food_locations[i]
                            if food_id == food.id:
                                self.food_locations[i][2] = False
                                self.food_locations[i][3] = current_time
                                break
                        break

        def drop_food(self):
            '''
            Dropping off food at home base
            MAKE COOLDOWN TIMER AFTER DROPPING OFF FOOD!!!
            '''
            nonlocal global_food_count
            if self.food == True and self.drop_cooldown == 0:
                if self.rect.colliderect(home_rect):
                    self.food = False
                    if self.drop_cooldown == 0: self.drop_cooldown = 100
                    global_food_count += 1
                    self.food_count_knowledge[self.id] += 1
                    self.belief_food_count += 1

                    current_time = pygame.time.get_ticks() - start_time
                    log_activity(current_time, {'activity': 'food_drop', 'robot_id': self.id})
                    #print(f'Robot number {self.id} dropped off food. Food delivered: {self.food_count_knowledge[self.id]}, belief: {self.belief_food_count}')

            if not self.rect.colliderect(home_rect) and self.food == False:
                if self.drop_cooldown != 0: self.drop_cooldown = 0

        def draw_robot(self):
            if self.food == True: screen.blit(pygame.image.load('graphics/robot_green_32.png').convert_alpha(), self.rect)
            #elif self.belief_food_count >= num_food: screen.blit(pygame.image.load('graphics/robot_purple_32.png').convert_alpha(), self.rect)
            else: screen.blit(self.image, self.rect)

        def update(self):
            if not self.spawned: self.spawn()
            self.sense_surroundings()
            self.collect_food()
            self.drop_food()
            self.communicate()
            #self.change_direction()
            self.move_robot()
            self.draw_robot()

            if self.drop_cooldown > 0: self.drop_cooldown -= 1
            if self.collide_cooldown > 0: self.collide_cooldown -= 1

    # SPAWN HOME BASE
    home_with_margin_surface = pygame.Surface((home_diameter + margin, home_diameter + margin))
    home_margin_rect = home_with_margin_surface.get_rect(center = (display_width/2, display_height/2))
    home_with_margin_surface.fill('lightpink')

    home_surface = pygame.Surface((home_diameter, home_diameter))
    home_rect = home_surface.get_rect(center = (display_width/2, display_height/2))
    home_surface.fill('darkgray')

    # SPAWN ROBOTS
    robots = pygame.sprite.Group()
    robot_id = 0
    for i in range(num_robots):
        robots.add(Robot(robot_id))
        robot_id += 1

    # SPAWN FOOD
    food_points = generate_food_positions(num_food)
    valid_points = []
    for p in food_points:
        if len(valid_points) >= num_food:
            break
        if is_valid_point(p):
            valid_points.append(p)

    food_group = pygame.sprite.Group()
    food_id = 0
    for i in range(num_food):
        food_group.add(Food(food_id))
        food_id += 1


    # TEST OBSTACLE
    #test_obst_surface = pygame.Surface((15, 15))
    #test_obst_surface.fill('red')
    #test_obs_rect = test_obst_surface.get_rect(center = (200, 200))


    # SIMULATION LOOP
    while running:
        for event in pygame.event.get():
            
            #stop simulation when clicking x
            if event.type == pygame.QUIT:
                running = False

            if event.type == timer:
                running = False

            if event.type == log_timer:
                log_num += 1
                current_time = log_num * log_interval
                log_activity(current_time, False)
                pygame.time.set_timer(log_timer, log_interval)
        
        #screen background
        screen.fill('lightgray')

        #display time
        current_time = display_time()
        display_score()

        #home base
        #screen.blit(home_with_margin_surface, home_margin_rect)
        screen.blit(home_surface, home_rect)

        #draw and update food
        food_group.update()

        #update and draw robos
        robots.update()
        #robots.draw(screen)

        #network lines between robos
        if lines:
            draw_network()

        pygame.display.flip()

        dt = clock.tick(60) #set framerate to max 60
        pygame.display.update()

        if not running: log_weights()
    pygame.quit()

    print(f"Total food collected: {global_food_count}")
    timelog.close()