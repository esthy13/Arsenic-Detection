#GRASP_linear_predictor.py
#Contains functions for the optimization of the linear predictor in linear_predictor.py

##Imports

import random
import numpy as np
from linear_predictor_functions import detect_anom_power, getTestData, calculateMetrics, detect_anom
import copy

## Functions

###Returns the number of adjacent edges, which is important to determine the best nodes
###Input: C (all the nodes and their neighbors), candidate (the candidate/node whose number of adjacent edges is needed)
###Output: Nunber of adjacent edges of candidate in C
def greedy_function(C, candidate):
    return len(C[candidate])

###Implementation of the semi-greedy algorithm
###Input: Threshold (the minimal number of edges a node has to have to be considered), C (all the nodes and their neighbors)
###Output: S (random selection of nodes from RCL), RCL (list of the nodes which pass the threshold)
def semi_greedy_construction(threshold, C):
    S = []
    RCL = []
    i = 0 
    #Copy C to be able to modify it without changing the original C
    new_C = copy.deepcopy(C)
    while len(new_C) > 0:
        #Create a temporary RCL to store the candidates that pass the threshold in this iteration
        RCL_temporary = []
        for candidate in new_C:
            #print("Candidate: ", candidate)
            #calculate the number of adjacent edges for the candidate
            gc = greedy_function(new_C, candidate)
            #If the number of adjacent edges is greater than or equal to the threshold, add the candidate to RCL_temporary and/or RCL
            if gc >= threshold:
                if i == 1:
                    RCL.append(candidate)
                    i+=1
                RCL_temporary.append(candidate)
        #print("RCL: ", RCL)
        #print("RCL_temporary: ", RCL_temporary)
        #If solution already has 3 nodes, break the loop. Solution only needs 3 nodes (1/3 of all nodes)
        if len(S) == 3:
            break
        #If RCL_temporary is empty, break the loop. This means that there are no more candidates that pass the threshold.
        if RCL_temporary == []:
            break
        #Choose random node from temporary RCL and add it to the solution. Remove it from new_C to avoid duplicates.
        c_star = random.choice(RCL_temporary)
        #print("c_star: ", c_star)
        S.append(c_star)
        #print("S: ", S)
        new_C.pop(c_star)
    return S, RCL

###Implementation of a repair function for the semi-greedy algorithm. If necessary, reduces number of nodes in solution
###Input: S (nodes/sensor placements selected for the solution), RCL (list of all nodes which pass the threshold)
###Output: S (same as before, but with reduced number of nodes)
def repair(S, RCL, threshold):
    MAX_SENSORS = 3
    #If more then three sensors, remove worst sensors until 3 left. Should not happen, but just in case
    if len(S)>MAX_SENSORS:
        new_S = {}
        for sensor in S:
            quality = RCL[sensor]
            if len(new_S) < threshold:
                new_S[sensor] = quality
            elif quality > min(new_S.values()):
                delete_key = [key for key, value in new_S.items() if value == min(new_S.values())]
                new_S.pop(delete_key)
                new_S[sensor] = quality
    new_S = S
    return S

###Get partial data (only from the sensors that are part of the suggested solution)
###Input: data (data to filter), S (the nodes that are part of the suggested solution), node_sensor (relationships between nodes and sensors), sensor_data (relationship between sensors and positions in data)
###Output: partial_data (data with only the selected sensors), node_sensor (relationships between remaining nodes and sensors), sensor_data (relationships between remaining sensors and data)
def get_partial_data(data, S, node_sensor, sensor_data):
    #Select the nodes that are part of the solution and their corresponding sensors
    selected_sensors = [node_sensor[node] for node in S]
    print("S: ", S)
    print("Selected sensors: ", selected_sensors)
    

    partial_data = []

    for subdata in data:
        #copy the subdata to avoid changing the original data
        new_subdata = subdata.copy()
        #get indices in subdata of selected sensors
        selected_indices = [i for i, sensor in enumerate(subdata["Cl_sensors"]) if sensor in selected_sensors]
        print("Selected indices: ", selected_indices)
        #Keep only selected sensors and their readings in new subdata
        new_subdata["Cl_sensors"] = new_subdata["Cl_sensors"] = [subdata["Cl_sensors"][i] for i in selected_indices]
        new_subdata["X_cl_st_per_sensor"] = subdata["X_cl_st_per_sensor"][selected_indices].copy()
        #print("New subdata: ", new_subdata["Cl_sensors"])
        #print("New subdata: ", new_subdata["X_cl_st_per_sensor"])

        #add this to partial_data
        partial_data.append(new_subdata)

    #modify node_sensor and sensor_data to only include the selected sensors
    new_node_sensor = {
        node: sensor
        for node, sensor in node_sensor.items()
        if node in S
    }

    new_sensor_data = {
        sensor: i
        for i, sensor in enumerate(partial_data[0]["Cl_sensors"])
    }

    return partial_data, new_node_sensor, new_sensor_data


###Implementation of local search. Searches best solutions based on semi-greedy solution.
###Input: S (nodes/sensor placements selected by semi-greedy algorithm and repair), regression_model (a linear regression model to test the efficiency of sensor placements), C (list of nodes and their neighbors), data(data to test on)
###Output: S (optimal sensor placements according to local search)
def local_search(S, regression_model, C, data, node_sensor, sensor_data, threshold):
    #Accuracy
    #print(data[0]["X_cl_st_per_sensor"])
    #test regression with original data and get metrics (tp_rel)
    #Comment: tp_rel is used because it is the most important how many true positives are detected for arsenic contamination
    original_partial_data, original_node_sensor, original_sensor_data = get_partial_data(data, S, node_sensor, sensor_data)
    #print(original_partial_data[0]["X_cl_st_per_sensor"])
    #print("Cl sensors: ", original_partial_data[0]["Cl_sensors"])
    original_X_test, original_y_test, original_as_y  = getTestData(original_partial_data)
    original_metrics = calculateMetrics(regression_model, original_X_test, original_y_test, original_as_y, lambda pred_val, real_val, id: detect_anom_power(pred_val, real_val, id, threshold))
    tp_rel_final = original_metrics["tp_rel"]
    i = 0

    #Iterate through nodes in solution (S)
    while i < len(S):
        node = S[i]
        #print("C: ", C)
        neighbors = C[node]
        #Iterate through neighbors of a solution node
        for neighbor in neighbors:
            #If neighbor not in solution, replace node with neighbor and test the new solution
            if neighbor not in S:
                new_S = S.copy()
                new_S.remove(node)
                new_S.append(neighbor)
                partial_data, new_node_sensor, new_sensor_data = get_partial_data(data, new_S, node_sensor, sensor_data)
                X_test, y_test, as_y  = getTestData(partial_data)
                ##Test these sensors with the regression model
                test_metrics = calculateMetrics(regression_model, X_test, y_test, as_y, lambda pred_val, real_val, id: detect_anom_power(pred_val, real_val, id, threshold))
                ##If accuracy better, keep it and break to restart the same process with the new node in S
                if original_metrics["tp_rel"] < test_metrics["tp_rel"]:
                    S = new_S
                    tp_rel_final = test_metrics["tp_rel"]
                    break;
        i += 1
    return S, tp_rel_final

###Implementation of path relinking. Changes one solution to resemble the other
###Input: elitepool (dictionary of best solutions and their tp values), S (nodes/sensor placements selected by semi-greedy algorithm and repair), data(data to test on), node_sensor (relationships between nodes and sensors), sensor_data (relationships between sensors in general and in data) regression_model (a linear regression model to test the efficiency of sensor placements), 
###Output: S (optimal sensor placements according to local search)
def path_relinking(elitepool, S, data, node_sensor, sensor_data, regression_model, threshold):
    #Copy S to avoid changing the original S
    solution_one = S.copy()
    #Choose a random solution from the elitepool that is not the same as solution_one
    solution_two = list(random.choice(list(elitepool.keys())))
    while frozenset(solution_one) == frozenset(solution_two):
        solution_two = list(random.choice(list(elitepool.keys())))
    print("Solution one: ", solution_one)
    print("Solution two: ", solution_two)

    #Transform solutions into sets (useful for further algorithm)
    solution_one_set = set(solution_one)
    solution_two_set = set(solution_two)

    #As long as two solutions not the same, change content of solution to resemble second solution
    while solution_one_set != solution_two_set:

        solution_one_set_test = solution_one_set.copy()

        #Check which nodes are different
        remove_nodes = solution_one_set_test - solution_two_set
        add_nodes = solution_two_set - solution_one_set_test

        #Replace one node at a time
        remove_node = list(remove_nodes)[0]
        add_node = list(add_nodes)[0]
        solution_one_set_test.remove(remove_node)
        solution_one_set_test.add(add_node)

        #Get according partial data, test it
        partial_data, _, _ = get_partial_data(data, solution_one_set_test, node_sensor, sensor_data)
        X_test, y_test, as_y  = getTestData(partial_data)
        test_metrics = calculateMetrics(regression_model, X_test, y_test, as_y, lambda pred_val, real_val, id: detect_anom_power(pred_val, real_val, id, threshold))
        
        #If elitepool has at least 10 nodes
        if len(elitepool) >= 10:
            #get worst solution in elitepool
            min_solution = min(elitepool, key=elitepool.get)
            #If new solution better than worst solution in elitepool,
            if elitepool[min_solution] < test_metrics["tp_rel"] and len(set(solution_one_set_test)) == 3:
                #replace worst solution by new solution
                elitepool.pop(min_solution)
                elitepool[frozenset(solution_one_set_test)] = test_metrics["tp_rel"]
        #If elitepool has less than 10 nodes, add new solution if not in elitepool 
        else:
            elitepool[frozenset(solution_one_set_test)] = test_metrics["tp_rel"]

        #Transform back to list for next iteration
        solution_one = list(solution_one_set_test) 
        solution_one_set = solution_one_set_test.copy()
    return elitepool


        
                