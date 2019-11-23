import traci
import os, sys
import numpy as np
from random import randint, random
from time import sleep


gamma = 0.95
# gamma_to_the_power is an array containing gamma powers
# gamma_to_the_power = [1, gamma, gamma^2, gamma^3, ..., gamma^49]
gamma_to_the_power = np.power(gamma, range(50))
actions = ['go_right', 'go_left', 'stay']

# default lane change mode 0b011001010101
# cars lane change mode 0b011001010001

class car:
    carid = ''
    distance_to_ego = 0
    position_x = 0
    position_y = 0
    velocity = 0
    acceleration = 0
    status = -1
    lane_id = ''

    def __init__(self, car_id):
        self.carid = car_id

    def print(self):
        print('\tcar ID : ', self.carid)
        print('\tdistance to ego : ', self.position)
        print('\tvelocity : ', self.velocity)
        print('\tacceleration : ', self.acceleration)
        print()

    def car_left(self):
        # returns true if the vehicle does not appear in the simulation (reached the end of the road or didnt load yet)
        return False if self.carid in traci.vehicle.getIDList() else True

    def get_values(self, ego_x):
        if self.carid != '':
            self.position_x, self.position_y = traci.vehicle.getPosition(self.carid)
            self.velocity = traci.vehicle.getSpeed(self.carid)
            self.acceleration = traci.vehicle.getAcceleration(self.carid)
            self.lane_id = traci.vehicle.getLaneID(self.carid)
            if self.carid == 'ego':
                self.distance_to_ego = 0
            else:
                self.distance_to_ego = self.position_x - ego_x
    
    def get_leader_and_follower_ID(self, lane_id):
        # input (string) : lane_id                  lane ids follow the following form "1to2_0"
        # output (tuple of 2 strings) : car_ids     car ids follow the form "f3.4"
        # returns the ids of both vehicles that are directly ahead or behind of our vehicle
        all_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        ego_x = traci.vehicle.getPosition(self.carid)[0]
        leader_id = ''
        follower_id = ''
        leader_x = 1000000
        follower_x = -1000000
        
        for car_id in all_ids:
            car_x = traci.vehicle.getPosition(car_id)[0]
            if car_x > ego_x and car_x < leader_x:
                leader_id, leader_x = car_id, car_x
            if car_x < ego_x and car_x > follower_x:
                follower_id, follower_x = car_id, car_x
        return leader_id, follower_id

    def get_neighbour_IDs(self):
        # input : none
        # output (list of 6 strings) : car ids of neighbouring vehicles
        # neighbours are 6 vehicles defined as follows:
        #     > leading and following vehicles in the same lane
        #     > leading and following vehicles in adjacent (left anf right) lanes
        if not self.car_left():
            ego_lane = traci.vehicle.getLaneIndex(self.carid)
            ego_lane_id = traci.vehicle.getLaneID(self.carid)
            edge_id = traci.lane.getEdgeID(ego_lane_id)

            neighbour_ids = []
            if ego_lane < 2:
                left_lane_id = edge_id + '_' + str(ego_lane + 1)
                left_leader_id , left_follower_id = self.get_leader_and_follower_ID(left_lane_id)
                neighbour_ids.extend([left_leader_id, left_follower_id])
            else :
                neighbour_ids.extend(['', ''])

            if ego_lane > 0:
                right_lane_id = edge_id + '_' + str(ego_lane - 1)
                right_leader_id , right_follower_id = self.get_leader_and_follower_ID(right_lane_id)
                neighbour_ids.extend([right_leader_id, right_follower_id])
            else :
                neighbour_ids.extend(['', ''])
            
            leader_id, follower_id = self.get_leader_and_follower_ID(ego_lane_id)
            neighbour_ids.extend([leader_id, follower_id])
            return neighbour_ids

    def perform_action(self, action):
        if not self.car_left():
            ego_lane = traci.vehicle.getLaneIndex(self.carid)
            try:
                if action == 'go_right':
                    traci.vehicle.changeLane(self.carid, ego_lane-1, 2)
                elif action == 'go_left':
                    traci.vehicle.changeLane(self.carid, ego_lane+1, 2)
                elif action == 'stay':
                    traci.vehicle.changeLane(self.carid, ego_lane, 2)
                elif action == 'accel':     # increase car speed smoothly by 5% over the course of 1 second
                    traci.vehicle.slowDown(self.carid, self.velocity*1.05, 1)
                elif action == 'decel':     # decrease car speed smoothly by 5% over the course of 1 second
                    traci.vehicle.slowDown(self.carid, self.velocity*0.95, 1)
            except:
                pass    # no lane available to go into

    def perform_random_action(self, proba):
        action = 'stay'
        if random() < proba:                # 1% probability of performing a lane change
            if self.lane_id == '1to2_0':
                action = 'go_left'
            elif self.lane_id == '1to2_2':
                action = 'go_right'
            elif self.lane_id == '1to2_1':
                action = 'go_left' if randint(0,1) == 0 else 'go_right' # 50% chance of either left/right
        self.perform_action(action)
        return action



class Environment:

    def __init__(self, render:False, ego_id : 'ego'):
        self.ego = ego_id
        self.render = render
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("Please declare environment variable 'SUMO_HOME'.")

    def start(self, filename):        # start the simulation and subscribe to get variables
        try:
            sumoBinary = 'sumo-gui' if self.render else 'sumo'
            sumoCmd = [sumoBinary, "-c", filename]
            traci.start(sumoCmd)
        except:
            sys.exit("Sumo could not start. Please check your file name.")
            # print("Sumo could not start. Please check your file name.")
            # sleep(3)
            # print('trying to restart')
            # self.start('Highway.sumocfg')
        # try:
        #     traci.vehicle.subscribe(self.ego, (
        #                 traci.constants.VAR_LANE_ID, traci.constants.VAR_ACCELERATION,
        #                 traci.constants.VAR_POSITION, traci.constants.VAR_SPEED
        #                 ))
        # except:
        #     sys.exit("Sumo could not subscribe to get variables. Check your vehicle IDs and variable constants.")

    def instant_reward(self, ego, leader, follower, left_follower, right_follower, action, last_distance):
        # should include gamma later
        reward = 0
        if leader.position_x - ego.position_x < 3:      # safety distance of 3m # could be replaced with emergency brake detection
            reward -= 100
        if self.collision_happened():                   # prevent collisions
            reward -= 1000
        if self.emergency_brake(ego, follower, left_follower, right_follower):  # prevent emergency brakes
            reward -= 100
        if action == 'go_left' or action == 'go_right': # prevent unnecessary lane changes
            reward -= 10
        reward -= 1                                     # arrive faster
        reward += ego.position_x - last_distance        # reward driving with no collisions
        return reward

    def emergency_brake(self, ego, follower, left_follower, right_follower):
        threshold = -7
        if ego.acceleration < threshold or follower.acceleration < threshold or left_follower.acceleration < threshold or right_follower.acceleration < threshold:
            return True
        return False

    def get_values(self):
        return traci.vehicle.getSubscriptionResults(self.ego)

    def get_vehicle_IDs(self):
        # returns a list of IDs of all vehicules that are currently in the simulation (on the road)
        return traci.vehicle.getIDList()

    def end_simulation(self):
        return self.collision_happened() or self.ego_left()

    def collision_happened(self):
        # returns true if ego vehicle collided with another vehicle
        return True if self.ego in traci.simulation.getCollidingVehiclesIDList() else False

    def ego_left(self):
        # returns true if ego vehicle has left the simulation (reached the end of the road)
        return False if self.ego in traci.vehicle.getIDList() else True

    def close(self):
        traci.close()
