import os
import traci
import pandas as pd
import numpy as np
from classes import car, Environment
from random import randint
import matplotlib.pyplot as plt
import seaborn as sns
from random import random
from time import sleep
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from keras.models import load_model

columns_list = ['egoY', 'egoV','egoA',
                'leftFollowerDistanceToEgo', 'leftFollowerY', 'leftFollowerV','leftFollowerA',
                'leftLeaderDistanceToEgo', 'leftLeaderY', 'leftLeaderV','leftLeaderA',
                'rightFollowerDistanceToEgo', 'rightFollowerY', 'rightFollowerV', 'rightFollowerA',
                'rightLeaderDistanceToEgo', 'rightLeaderY', 'rightLeaderV', 'rightLeaderA',
                'LeaderDistanceToEgo', 'LeaderY', 'LeaderV','LeaderA',
                'FollowerDistanceToEgo', 'FollowerY', 'FollowerV', 'FollowerA']

# default lane change mode 0b011001010101
# cars lane change mode 0b011001010001 # prevent cooperative lane change

def show_PCA_plot(cumsum):
    plt.ylabel('Variance cumulative sum')
    plt.xlabel('Number of features')
    plt.title('PCA Analysis')
    plt.ylim(0,1.05)
    plt.plot(cumsum, 'o-')
    plt.show()

performance = []

# Read data
df = pd.read_csv("data.csv", index_col=False)

# apply PCA to training data
X = df[columns_list]
sc = StandardScaler()
X = sc.fit_transform(X)
pca = PCA(n_components=11)
X = pca.fit_transform(X)

# Convert string labels into numbers.
le = LabelEncoder()
Y = le.fit_transform(df['action'])

model = load_model('ANN_model.h5')

# if __name__ == '__main__':

for episode in range(100):
    all_data = []
    time = 0
    pred_actions = []
    ego = car(car_id='ego')
    env = Environment(render=False if episode%10 == 0 else False, ego_id=ego.carid)
    env.start('Highway.sumocfg')
    # prevent ego vehicle from doing any lane change by itself
    traci.vehicle.setLaneChangeMode(ego.carid, 0x00)

    last_distance = 0
    rows_list = []
    risk_detected = False
    risk_start = -1
    risk_distance = 0
    collided = False
    for step in range(1400):
        traci.simulationStep()
        if env.collision_happened():
            collided = True
        if step > 210:      #ego vehicle should be now on the road
            if not ego.car_left():
                neighbour_ids = ego.get_neighbour_IDs()
                left_leader = car(neighbour_ids[0])
                left_follower = car(neighbour_ids[1])
                right_leader = car(neighbour_ids[2])
                right_follower = car(neighbour_ids[3])
                leader = car(neighbour_ids[4])
                follower = car(neighbour_ids[5])

                ego.get_values(0)
                left_leader.get_values(ego.position_x)
                left_follower.get_values(ego.position_x)
                right_leader.get_values(ego.position_x)
                right_follower.get_values(ego.position_x)
                leader.get_values(ego.position_x)
                follower.get_values(ego.position_x)

                # correct NA values
                if ego.lane_id == '1to2_2':     # right most lane
                    left_follower.position_x = ego.position_x - 50
                    left_follower.position_y = -1.88 * 4     # 4th lane
                    left_follower.velocity = ego.velocity
                    left_follower.acceleration = ego.acceleration
                    left_follower.distance_to_ego = left_follower.position_x - ego.position_x
                    
                    left_leader.position_x = ego.position_x + 50
                    left_leader.position_y = -1.88 * 4     # 4th lane
                    left_leader.velocity = ego.velocity
                    left_leader.acceleration = ego.acceleration
                    left_leader.distance_to_ego = left_leader.position_x - ego.position_x

                elif ego.lane_id == '1to2_0':
                    right_follower.position_x = ego.position_x - 50
                    right_follower.position_y = 1.88        # -1th lane
                    right_follower.velocity = ego.velocity
                    right_follower.acceleration = ego.acceleration
                    right_follower.distance_to_ego = right_follower.position_x - ego.position_x
                    
                    right_leader.position_x = ego.position_x + 50
                    right_leader.position_y = 1.88          # -1th lane
                    right_leader.velocity = ego.velocity
                    right_leader.acceleration = ego.acceleration
                    right_leader.distance_to_ego = right_leader.position_x - ego.position_x

                
                # #compute risk
                if not risk_detected or ego.position_x - risk_start > risk_distance:
                    if random() < 0.002:
                        risk_detected = True
                        start = ego.position_x
                        risky_lane = randint(0, 2)
                        risk_distance = randint(30, 200)
                    else:
                        risk_detected = False
                
                data_dict = {
                    'timestep' : traci.simulation.getTime(),
                    'egoX' : ego.position_x,
                    'egoY' : ego.position_y,
                    'egoV' : ego.velocity,
                    'egoA' : ego.acceleration,
                    'egolaneID' : ego.lane_id,
                    'leftFollowerID' : 	left_follower.carid,
                    'leftFollowerX' : left_follower.position_x ,
                    'leftFollowerY' : left_follower.position_y,
                    'leftFollowerV' : left_follower.velocity,
                    'leftFollowerA' : left_follower.acceleration,
                    'leftFollowerLaneID' : left_follower.lane_id,
                    'leftFollowerDistanceToEgo' : left_follower.distance_to_ego,
                    'leftLeaderID' : left_leader.carid,
                    'leftLeaderX' : left_leader.position_x,
                    'leftLeaderY' : left_leader.position_y,
                    'leftLeaderV' : left_leader.velocity,
                    'leftLeaderA' : left_leader.acceleration,
                    'leftLeaderLaneID' : left_leader.lane_id,
                    'leftLeaderDistanceToEgo' : left_leader.distance_to_ego,
                    'rightFollowerID' : right_leader.carid,
                    'rightFollowerX' : left_follower.position_x,
                    'rightFollowerY' : left_follower.position_y,
                    'rightFollowerV' : left_follower.velocity,
                    'rightFollowerA' : left_follower.acceleration,
                    'rightFollowerLaneID' : left_follower.lane_id,
                    'rightFollowerDistanceToEgo' : right_follower.distance_to_ego,
                    'rightLeaderID' : right_leader.carid,
                    'rightLeaderX' : right_leader.position_x,
                    'rightLeaderY' : right_leader.position_y,
                    'rightLeaderV' : right_leader.velocity,
                    'rightLeaderA' : right_leader.acceleration,
                    'rightLeaderLaneID' : right_leader.lane_id,
                    'rightLeaderDistanceToEgo' : right_leader.distance_to_ego,
                    'LeaderID' : leader.carid,
                    'LeaderX' : leader.position_x,
                    'LeaderY' : leader.position_y,
                    'LeaderV' : leader.velocity,
                    'LeaderA' : leader.acceleration,
                    'LeaderLaneID' : leader.lane_id,
                    'LeaderDistanceToEgo' : leader.distance_to_ego,
                    'FollowerID' : follower.carid,
                    'FollowerX' : follower.position_x,
                    'FollowerY' : follower.position_y,
                    'FollowerV' : follower.velocity,
                    'FollowerA' : follower.acceleration,
                    'FollowerLaneID' : follower.lane_id,
                    'FollowerDistanceToEgo' : follower.distance_to_ego,
                    'action' : -1,
                    # 'risk' : -1 if not risk_detected else risky_lane
                }

                row = []
                row.append(data_dict)
                all_data.append(data_dict)
                state = pd.DataFrame(row)
                state = state[columns_list]
                state = pca.transform(sc.transform(state))
                state = pd.DataFrame(state)
                # state = state.assign(risk=risk)

                last_distance = ego.position_x
                rows_list.append(data_dict)

                pred_action = le.inverse_transform(model.predict_classes(state))        # Predict the response for test dataset
                pred_actions.append(pred_action)
                if pred_action[0] == 'go_right' and ego.lane_id == '1to2_0':
                    pred_action[0] = 'stay'
                elif pred_action[0] == 'go_left' and ego.lane_id == '1to2_1':
                    pred_action[0] == 'stay'
                
                if risk_detected:
                    lane = int(ego.lane_id[5])
                    print("Risk deteted in lane ", risky_lane)
                    if (risky_lane == lane + 1) and pred_action[0] == 'go_left':
                        action = 'stay'
                        print("Lane change request denied because of risk in lane ", risky_lane)
                    elif (risky_lane == lane - 1) and pred_action[0] == 'go_right':
                        action = 'stay'
                        print("Lane change request denied because of risk in lane ", risky_lane)
                    elif lane == risky_lane:
                        time += 1
                else:
                    print("no risk")
                ego.perform_action(pred_action[0])

    env.close()
    all_data = pd.DataFrame(all_data)
    count = 0
    for x in pred_actions:
        if x == 'go_right' or x == 'go_left':
            count += 1

    performance_dict = {'time_to_end' : data_dict['timestep'] if not collided else -1,
                        'collision' : collided,
                        'nb_lane_change_requests' : count,
                        'nb_emergency_brakes': (all_data.egoA.values < -7).sum() + (all_data.FollowerA.values < -7).sum() +
                                    (all_data.rightFollowerA.values < -7).sum() + (all_data.leftFollowerA.values < -7).sum(),
                        'time_driven_in_a_risky_lane' : time
                        }

    performance = []
    performance.append(performance_dict)
    performance_df = pd.DataFrame(performance)

    try:
        old_perf_df = pd.read_csv("performance.csv")
        performance_df = performance_df.append(old_perf_df, ignore_index=True, sort=False)
    except:
        pass
    performance_df.to_csv('performance.csv', index=False)
    print('\n\n\nepisode : {}\n\n\n'.format(episode))

print('done')









