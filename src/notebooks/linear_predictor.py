##Linear predictor
##This file uses a linear predictor to predict the chlorine values based on flow and chlorine values.
##The chlorine values that are predicted are then compared to the real chlorine values
##If the predicted chlorine values are significantly lower than the real values, an anomaly is detected.

##Imports

from sklearn.linear_model import LinearRegression
from linear_predictor_functions import *
from GRASP_linear_predictor import *

##Script

###Read the files with the chlorine, flow and arsenic sensor readings
###data: files with little arsenic contamination
###data_no_cont: files with no arsenic contamination
###data_strong_cont: files with strong arsenic contamination
data = read_files("../data/scada_data_")
data_no_cont = read_files("../data/scada_data_no_cont")
data_strong_cont = read_files("../data/scada_data_strong_cont")

###Get mean and standard deviation for the chlorine, flow and arsenic sensor readings for all the values
cl_mean, flow_mean, as_mean, cl_std, flow_std, as_std = get_cardinal_metrics_cl_as_flow(data)
cl_mean_no_cont, flow_mean_no_cont, as_mean_no_cont, cl_std_no_cont, flow_std_no_cont, as_std_no_cont = get_cardinal_metrics_cl_as_flow(data_no_cont)
cl_mean_strong_cont, flow_mean_strong_cont, as_mean_strong_cont, cl_std_strong_cont, flow_std_strong_cont, as_std_strong_cont = get_cardinal_metrics_cl_as_flow(data_strong_cont)

###Standardize the chlorine, flow, and arsenic sensor readings in each file
###Standardization formula: Z(t) = X(t) - µ / σ (Detecting Changes in Water Quality Data)
for i in range(0,20):
    
    X_cl_st = standardize(data[i]["X_cl"][0], cl_mean, cl_std)
    X_flow_st = standardize(data[i]["X_flow"][0], flow_mean, flow_std)
    y_st = standardize(data[i]["y"][0], as_mean, as_std)

    data[i]["X_cl_st"] = X_cl_st
    data[i]["X_flow_st"] = X_flow_st
    data[i]["y_st"] = y_st

    X_cl_st_no_cont = standardize(data_no_cont[i]["X_cl"][0], cl_mean_no_cont, cl_std_no_cont)
    X_flow_st_no_cont = standardize(data_no_cont[i]["X_flow"][0], flow_mean_no_cont, flow_std_no_cont)
    y_st_no_cont = standardize(data_no_cont[i]["y"][0], as_mean_no_cont, as_std_no_cont)

    data_no_cont[i]["X_cl_st"] = X_cl_st_no_cont
    data_no_cont[i]["X_flow_st"] = X_flow_st_no_cont
    data_no_cont[i]["y_st"] = y_st_no_cont

    X_cl_st_strong_cont = standardize(data_strong_cont[i]["X_cl"][0], cl_mean_strong_cont, cl_std_strong_cont)
    X_flow_st_strong_cont = standardize(data_strong_cont[i]["X_flow"][0], flow_mean_strong_cont, flow_std_strong_cont)
    y_st_strong_cont = standardize(data_strong_cont[i]["y"][0], as_mean_strong_cont, as_std_strong_cont)

    data_strong_cont[i]["X_cl_st"] = X_cl_st_strong_cont
    data_strong_cont[i]["X_flow_st"] = X_flow_st_strong_cont
    data_strong_cont[i]["y_st"] = y_st_strong_cont


###Get the values per chlorine and flow, and arsenic sensor

####Normal contamination
NUMBER_CL_SENSORS = len(data[0]["Cl_sensors"])
NUMBER_FLOW_SENSORS = len(data[0]["Flow_sensors"])
NUMBER_AS_SENSORS = len(data[0]["As_sensors"])

get_readings_per_sensor(data, NUMBER_CL_SENSORS, "X_cl_st")
get_readings_per_sensor(data, NUMBER_FLOW_SENSORS, "X_flow_st")
get_readings_per_sensor(data, NUMBER_AS_SENSORS, "y_st")

####No contamination
NUMBER_CL_SENSORS_no_cont = len(data_no_cont[0]["Cl_sensors"])
NUMBER_FLOW_SENSORS_no_cont = len(data_no_cont[0]["Flow_sensors"])
NUMBER_AS_SENSORS_no_cont = len(data_no_cont[0]["As_sensors"])

get_readings_per_sensor(data_no_cont, NUMBER_CL_SENSORS_no_cont, "X_cl_st")
get_readings_per_sensor(data_no_cont, NUMBER_FLOW_SENSORS_no_cont, "X_flow_st")
get_readings_per_sensor(data_no_cont, NUMBER_AS_SENSORS_no_cont, "y_st")

####Strong contamination
NUMBER_CL_SENSORS_strong_cont = len(data_strong_cont[0]["Cl_sensors"])
NUMBER_FLOW_SENSORS_strong_cont = len(data_strong_cont[0]["Flow_sensors"])
NUMBER_AS_SENSORS_strong_cont = len(data_strong_cont[0]["As_sensors"])

get_readings_per_sensor(data_strong_cont, NUMBER_CL_SENSORS_strong_cont, "X_cl_st")
get_readings_per_sensor(data_strong_cont, NUMBER_FLOW_SENSORS_strong_cont, "X_flow_st")
get_readings_per_sensor(data_strong_cont, NUMBER_AS_SENSORS_strong_cont, "y_st")

###Divide the data into training and test set (by scada data files)
###There are only about 10 sensors in each of the categories, which is why I will only split the sensors 
###in training and test set and not create a validation set (2/3 training set, 1/3 test set)

####Normal contamination
train_set = data[0:14]
test_set = data[14:20]

####No contamination
train_set_no_cont = data_no_cont[0:14]
test_set_no_cont = data_no_cont[14:20]

####Strong contamination
train_set_strong_cont = data_strong_cont[0:14]
test_set_strong_cont = data_strong_cont[14:20]

###Get the training data for the linear predictor (chlorine and flow sensor readings) 
X_train_no_cont = []
y_train_no_cont = []

for file in range(len(train_set_no_cont)):

    chlor_data_no_cont = train_set_no_cont[file]["X_cl_st_per_sensor"]
    flow_data_no_cont = train_set_no_cont[file]["X_flow_st_per_sensor"]

    n_sensors_no_cont = len(chlor_data_no_cont)
    n_time_no_cont = len(chlor_data_no_cont[0]) 

    for sensor in range(n_sensors_no_cont):

        for timestep in range(2, n_time_no_cont):

            x_no_cont = [
                chlor_data_no_cont[sensor][timestep-2],
                chlor_data_no_cont[sensor][timestep-1]
            ]

            for sensor2 in range(len(flow_data_no_cont)):
                x_no_cont.append(flow_data_no_cont[sensor2][timestep-2])
                x_no_cont.append(flow_data_no_cont[sensor2][timestep-1])


            y_no_cont = chlor_data_no_cont[sensor][timestep]

            X_train_no_cont.append(x_no_cont)
            y_train_no_cont.append(y_no_cont)


####Get the test data for a small, no and strong contamination
X_test_2, y_test_2, as_y = getTestData(test_set)
X_test_no_cont, y_test_no_cont, as_y_no_cont = getTestData(test_set_no_cont)
X_test_strong_cont, y_test_strong_cont, as_y_strong_cont = getTestData(test_set_strong_cont)

###Create and train the regression model
reg_no_cont = LinearRegression()
reg_no_cont.fit(X_train_no_cont, y_train_no_cont)

###Test with no contamination data to get appropriate threshold for anomaly detection
best_threshold = 1
while best_threshold > 0:
    no_cont_metrics = calculateMetrics(reg_no_cont, X_test_no_cont, y_test_no_cont, as_y_no_cont, lambda pred_val, real_val, id: detect_anom(pred_val, real_val, id, 0.4))
    if len(no_cont_metrics["anomalies"]) <= 0:
        break
    else:
        best_threshold -= 0.1
print("Best threshold for anomaly detection: ", best_threshold) 

###Test with small contamination data, with threshold
print("Normal test:")
no_cont_metrics = calculateMetrics(reg_no_cont, X_test_2, y_test_2, as_y, lambda pred_val, real_val, id: detect_anom(pred_val, real_val, id, 0.4))
reg_no_cont.score(X_train_no_cont, y_train_no_cont)
no_cont_abs, no_cont_rel, no_cont_cond_true, no_cont_cond_false = createOverviewTables(no_cont_metrics["H1_H0"], no_cont_metrics["tp_rel"], no_cont_metrics["fn_rel"], no_cont_metrics["fp_rel"], no_cont_metrics["tn_rel"], no_cont_metrics["tp_true"], no_cont_metrics["fn_true"], no_cont_metrics["fp_false"], no_cont_metrics["tn_false"])
printOverviewTables(no_cont_abs, no_cont_rel, no_cont_cond_true, no_cont_cond_false)
#saveOverviewTables(no_cont_abs, no_cont_rel, no_cont_cond_true, no_cont_cond_false, "no_cont")

###Test with strong contamination data
print("Strong data test:")
strong_cont_metrics = calculateMetrics(reg_no_cont, X_test_strong_cont, y_test_strong_cont, as_y_strong_cont, lambda pred_val, real_val, id: detect_anom(pred_val, real_val, id, 0.4))
strong_cont_abs, strong_cont_rel, strong_cont_cond_true, strong_cont_cond_false = createOverviewTables(strong_cont_metrics["H1_H0"], strong_cont_metrics["tp_rel"], strong_cont_metrics["fn_rel"], strong_cont_metrics["fp_rel"], strong_cont_metrics["tn_rel"], strong_cont_metrics["tp_true"], strong_cont_metrics["fn_true"], strong_cont_metrics["fp_false"], strong_cont_metrics["tn_false"])
printOverviewTables(strong_cont_abs, strong_cont_rel, strong_cont_cond_true, strong_cont_cond_false)
#saveOverviewTables(strong_cont_abs, strong_cont_rel, strong_cont_cond_true, strong_cont_cond_false, "strong_cont")
createOverviewDiagrams(strong_cont_metrics["real_vals_cl"], strong_cont_metrics["pred_vals_cl"], strong_cont_metrics["anomalies_truth_values"], strong_cont_metrics["no_anomalies_truth_values"])


###Test with strong contamination data and power transformation
print("Strong data and power test:")
strong_power_metrics = calculateMetrics(reg_no_cont, X_test_strong_cont, y_test_strong_cont, as_y_strong_cont, lambda pred_val, real_val, id: detect_anom_power(pred_val, real_val, id, 0.4))
strong_power_abs, strong_power_rel, strong_power_cond_true, strong_power_cond_false = createOverviewTables(strong_power_metrics["H1_H0"], strong_power_metrics["tp_rel"], strong_power_metrics["fn_rel"], strong_power_metrics["fp_rel"], strong_power_metrics["tn_rel"], strong_power_metrics["tp_true"], strong_power_metrics["fn_true"], strong_power_metrics["fp_false"], strong_power_metrics["tn_false"])
printOverviewTables(strong_power_abs, strong_power_rel, strong_power_cond_true, strong_power_cond_false)
#saveOverviewTables(strong_power_abs, strong_power_rel, strong_power_cond_true, strong_power_cond_false, "strong_power")
createOverviewDiagrams(strong_power_metrics["real_vals_cl"], strong_power_metrics["pred_vals_cl"], strong_power_metrics["anomalies_truth_values"], strong_power_metrics["no_anomalies_truth_values"])

###Test with gliding timeframe (Timeframe size = 10 differences)

print("Gliding time frame test:")
gliding_strong_power_metrics = calculateMetrics_last_10_diffs(strong_power_metrics, X_test_strong_cont, as_y_strong_cont)
gliding_strong_cont_abs, strong_cont_rel, strong_cont_cond_true, strong_cont_cond_false = createOverviewTables(gliding_strong_power_metrics["H1_H0"], gliding_strong_power_metrics["tp_rel"], gliding_strong_power_metrics["fn_rel"], gliding_strong_power_metrics["fp_rel"], gliding_strong_power_metrics["tn_rel"], gliding_strong_power_metrics["tp_true"], gliding_strong_power_metrics["fn_true"], gliding_strong_power_metrics["fp_false"], gliding_strong_power_metrics["tn_false"])
printOverviewTables(gliding_strong_cont_abs, strong_cont_rel, strong_cont_cond_true, strong_cont_cond_false)
#saveOverviewTables(gliding_strong_cont_abs, strong_cont_rel, strong_cont_cond_true, strong_cont_cond_false)

print("Optimization with GRASP:")
#Grasp for optimizing the linear regression
C = {10: [11], 11 : [10,12,21], 12: [11,22,13], 13: [12,23], 21: [11,22,31], 22: [12,21,23,32], 23: [13,22], 31: [21,32], 32: [31,22]}
node_sensor = {10: 24, 11: 25, 12: 26, 13: 27, 21: 28, 22: 29, 23: 30, 31: 31, 32: 32}
sensor_data = {24: 0, 25: 1, 26: 2, 27: 3, 28: 4, 29: 5, 30: 6, 31: 7, 32: 8}

P = {}
GRASP_THRESHOLD = 3
i = 0
while i < 50:
    print("Step: ", i)
    #Construct semi-greedy solution
    S, RCL = semi_greedy_construction(3, C)
    print("S: ", S)
    print("RCL: ", RCL)
    #Repair if necessary
    if len(S) > GRASP_THRESHOLD:
        S = repair(S,RCL, GRASP_THRESHOLD)
    #Perform local search on new solution
    S, tp_rel_final = local_search(S, reg_no_cont, C, test_set_strong_cont, node_sensor, sensor_data)
    print("S after local search: ", S)

    #if more than one iteration
    if i > 0:
        #if solution is not already in elitepool
        if frozenset(S) not in P:
            #add it
            P[frozenset(S)] = tp_rel_final
            #If elitepool has more than one solution
            if len(P) > 1:
                #, perform path relinking
                print("Elitepool before path relinking:", P)
                P = path_relinking(P, S, test_set_strong_cont, node_sensor, sensor_data, reg_no_cont)
                print("Elitepool after path relinking:", P)
    #if first iteration
    else:
        #add solution to elitepool, no path relinking (does not work for one solution)
        P[frozenset(S)] = tp_rel_final
    i+=1

print(P)
P2 = {}
#Postoptimization, just deleting duplicates
for solution in P:
    if solution not in P2:
        P2[solution] = P[solution]
#Argmax instead of argmin (like in the paper) because tp_rel is supposed to be the highest possible
print(P2)
#Get the best solution (highest tp_rel)
best_solution = [key for key, value in P2.items() if value == max(P2.values())]
print(best_solution)


    