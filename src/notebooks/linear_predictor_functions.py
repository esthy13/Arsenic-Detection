#Linear predictor functions
#The functions used in the file linear_predictor.py

##Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Functions

###Reads the files from the provided file path and saves chlorine, flow and arsenic sensors.
###Input: filepath: path to the folder where the files are stored
###Output: data: list of dictionaries with file number, list of sensors(chlorine, flow, arsenic), list of readings(chlorine, flow, arsenic)
def read_files(filepath):
    #Iterate over the different numpy files
    data = []
    for i in range(1,21):
        chlorine_sensors = []
        flow_sensors = []
        arsenic_sensors = []
        X_cl = []
        X_flow = []
        y = []

        file_data = np.load(f"{filepath}{i}.npz", allow_pickle=True)

        #Get the indices/column numbers of the chlorine, flow, and arsenic sensors in the file

        for j in range(0,len(file_data["col_desc"])):
            if "Chlorine" in file_data["col_desc"][j][1]:
                chlorine_sensors.append(j)
            elif "AsIII" in file_data["col_desc"][j][1]:
                arsenic_sensors.append(j)
            elif "flow" in file_data["col_desc"][j][0]:
                flow_sensors.append(j)

        #Get sensor readings of chlorine and flow sensors
        cl_first = chlorine_sensors[0]
        cl_last = chlorine_sensors[-1]
        flow_first = flow_sensors[0]
        flow_last = flow_sensors[-1]
        cols = np.r_[cl_first:cl_last+1,
                    flow_first:flow_last+1]
        X_cl.append(file_data["sensor_readings"][:, cl_first:cl_last+1])
        X_flow.append(file_data["sensor_readings"][:, flow_first:flow_last+1])

        #Get sensor readings of arsenic sensors
        as_first = arsenic_sensors[0]
        as_last = arsenic_sensors[-1]
        y.append(file_data["sensor_readings"][:, as_first:as_last+1])

        data.append({"file": file_data, 
                    "Cl_sensors": chlorine_sensors, 
                    "Flow_sensors": flow_sensors, 
                    "As_sensors": arsenic_sensors, 
                    "X_cl": X_cl, 
                    "X_flow": X_flow, 
                    "y": y})
        
    return data


###Calculates the mean and standard deviation of the chlorine, flow, and arsenic sensor readings across all files.
###Input: data: list of dictionaries with file number, list of sensors(chlorine, flow, arsenic), list of readings(chlorine, flow, arsenic)
###Output: cl_mean: mean of chlorine sensor readings, flow_mean: mean of flow sensor readings, as_mean: mean of arsenic sensor readings, 
### cl_std: standard deviation of chlorine sensor readings, flow_std: standard deviation of flow sensor readings, as_std: standard deviation of arsenic sensor readings
def get_cardinal_metrics_cl_as_flow(data):
    cl_all = []
    as_all = []
    flow_all = []

    #get all readings for each sensor type across all files
    for i in range(0,20):
        cl_all.append(data[i-1]["X_cl"][0])
        flow_all.append(data[i-1]["X_flow"][0])
        as_all.append(data[i-1]["y"][0])

    #Combine all readings into one array for sensor type
    cl_all = np.concatenate(cl_all, axis=0)
    flow_all = np.concatenate(flow_all, axis=0)
    as_all = np.concatenate(as_all, axis=0)

    #get mean
    cl_mean = np.mean(cl_all)
    flow_mean = np.mean(flow_all)
    as_mean = np.mean(as_all)

    #get standard deviation
    cl_std = np.std(cl_all)
    flow_std = np.std(flow_all)
    as_std = np.std(as_all)

    return cl_mean, flow_mean, as_mean, cl_std, flow_std, as_std


###Standardizes the chlorine, flow, and arsenic sensor readings using the provided mean and standard deviation.
###Input: unst_data: list of readings(chlorine, flow, arsenic), mean: mean of the sensor readings, std: standard deviation of the sensor readings
###Output: st_data: standardized sensor readings
def standardize(unst_data, mean, std):

    #standardize the data using the provided mean and standard deviation
    st_data = np.zeros(unst_data.shape)
    for j in range(0, len(unst_data)):
        for k in range(0, len(unst_data[j])):
            st_data[j][k] = (unst_data[j][k] - mean) / std  
    return st_data


###Gets the readings per sensor for the chlorine, flow, and arsenic sensors.
###Input: data: list of dictionaries with file number, list of sensors(chlorine, flow, arsenic), list of readings(chlorine, flow, arsenic), number_sensors: number of sensors, sensor_type: type of sensor (chlorine, flow, arsenic)
###Output: data with appended sensor readings per sensor for the chlorine, flow, and arsenic sensors. (no return)
def get_readings_per_sensor(data, number_sensors, sensor_type):

    #go by file
    for subdata in data: 
        #get all sensor readings for sensor, 
        all_sensor_readings = []
        for i in range(0, (number_sensors)):
            #by sensor type, but then all values of one sensor, so get all values of sensor i 
            sensor_readings = subdata[sensor_type][:,i]
            #append to list of all sensor readings
            all_sensor_readings.append(sensor_readings)
        #name sensor_type in subdata 
        subdata[f"{sensor_type}_per_sensor"] = np.array(all_sensor_readings)


###Detects anomalies based on the predicted and real values of the chlorine sensor readings. Uses a threshold of 0.5 to detect anomalies.
###Input: pred_val: predicted value of the chlorine sensor reading, real_val: real value of the chlorine sensor reading, id: id of the reading, threshold: threshold for anomaly detection
###Output: id: id of the reading, diff: difference between predicted and real value, anom_val: boolean indicating if an anomaly is detected
def detect_anom(pred_val, real_val, id, threshold):
    #if difference to big, anomaly detected
    if abs(pred_val - real_val) > threshold:
        #anomaly = "probable anomaly detected at", id
        return id, (pred_val - real_val), True
    else:
        return id, (pred_val - real_val), False


###Detects anomalies based on the predicted and real values of the chlorine sensor readings. Uses a power transformation to amplify the differences between the predicted and real values of the chlorine sensor readings.
###Input: pred_val: predicted value of the chlorine sensor reading, real_val: real value of the chlorine sensor reading, id: id of the reading, threshold: threshold for anomaly detection
###Output: id: id of the reading, r_trans: anomaly score, anom_val: boolean indicating if an anomaly is detected
def detect_anom_power(pred_val, real_val, id, threshold):
    #apply power to difference to amplify differences:
    r_trans = np.copysign((pred_val+real_val)**10, (pred_val+real_val))
    #same as detect_anom
    if r_trans > threshold:
        #anomaly = "probable anomaly detected at", id
        return id, r_trans, True
    else:
        return id, r_trans, False
    

###Calculates the metrics for the anomaly detection based on the predicted and real values of the chlorine sensor readings.
###Input: model: linear regression model, X_test: test data for the chlorine and flow sensor readings, y_test: test data for the arsenic sensor readings, as_y: test data for the arsenic sensor readings, threshold_function: function to detect anomalies based on the predicted and real values of the chlorine sensor readings
###Output: returnData: dictionary with the metrics (e.g. precision, recall) for the anomaly detection
def calculateMetrics(model, X_test, y_test, as_y, threshold_function):

    anomalies = []
    anomalies_truth_values = {}
    anomalies_truth_values["true"] = []
    anomalies_truth_values["false"] = []

    no_anomalies = []
    no_anomalies_truth_values = {}
    no_anomalies_truth_values["true"] = []
    no_anomalies_truth_values["false"] = []

    pred_vals_cl = []
    real_vals_cl = []
    real_cl_val_t_minus_one = 0

    for situation in range(len(X_test)):
        
        #put X_test into regression, get predicted value
        pred_cl_val = model.predict(np.array(X_test[situation][1]).reshape(1, -1))[0]
        
        #get real value
        real_cl_val = y_test[situation][1]
        real_vals_cl.append(real_cl_val)

        if real_cl_val_t_minus_one == real_cl_val:
            pred_cl_val = real_cl_val

        pred_vals_cl.append(pred_cl_val)

        real_cl_val_t_minus_one = real_cl_val
        
        #get id
        id = X_test[situation][0]

        #get actual as value in the system
        real_as_val = np.sum(as_y[situation][1])

        #get anomaly scores for the detected anomalies, threshold 0.5        
        anom_id, anom_score, anom_truth_val = threshold_function(pred_cl_val, real_cl_val, id)

        #if situation == 2:
            #print("pred_cl_val: ", pred_cl_val)
            #print("real_cl_val: ", real_cl_val)
            #print("real_as_val: ", real_as_val)
            #print("anom_score: ", anom_score)

        #save anomaly ids and scores, see whether there actually is an anomaly and save the comparison as a truth value
        if anom_truth_val == True:
            anomalies.append((anom_id, anom_score, pred_cl_val, real_cl_val))

            ##true positives/sensitivity
            if real_as_val > 0:
                anomalies_truth_values["true"].append((anom_id, anom_score, pred_cl_val, real_cl_val))
            ##false positives/beta error
            else:
                anomalies_truth_values["false"].append((anom_id, anom_score, pred_cl_val, real_cl_val))
        else:
            no_anomalies.append((anom_id, anom_score, pred_cl_val, real_cl_val))

            ##false negatives/alpha error
            if real_as_val > 0:
                no_anomalies_truth_values["false"].append((anom_id, anom_score, pred_cl_val, real_cl_val))
            ##true negatives/specificity
            else:
                no_anomalies_truth_values["true"].append((anom_id, anom_score, pred_cl_val, real_cl_val))
        
    #get absolute and relative frequencies of true positives, false positives, false negatives and true negatives
    n_situations = len(X_test)

    ##true positives/sensitivity
    tp = len(anomalies_truth_values["true"])
    tp_rel = round(((tp/n_situations)*100), 2)

    ##false positives/alpha error
    fp = len(anomalies_truth_values["false"])
    fp_rel = round(((fp/n_situations)*100), 2)

    ##false negatives/beta error
    fn = len(no_anomalies_truth_values["false"])
    fn_rel = round(((fn/n_situations)*100), 2)

    ##true negatives/specificity
    tn = len(no_anomalies_truth_values["true"])
    tn_rel = round(((tn/n_situations)*100), 2)

    #get conditional frequencies, sorted by as true or false

    n_as_true = tp+fn
    n_as_false = tn+fp

    #tp/true
    if n_as_true == 0:
        tp_true = 0.00
    else:
        tp_true = round(((tp/n_as_true)*100), 2)

    #fp/false
    if n_as_false == 0:
        fp_false = 0.00
    else:
        fp_false = round(((fp/n_as_false)*100), 2)

    #fn/true
    if n_as_true == 0:
        fn_true = 0.00
    else:
        fn_true = round(((fn/n_as_true)*100), 2)

    #tn/false
    if n_as_false == 0:
        tn_false = 0.00
    else:
        tn_false = round(((tn/n_as_false)*100), 2)


    #create overview tables 
    #H1: There is an arsenic contamination
    #H0: There is no arsenic contamination 

    #absolute frequency
    H1 = [tp, fn]
    H0 = [fp, tn]

    H1_H0 = [H1, H0]

    returnData = {
        "H1_H0": H1_H0, 
        "anomalies": anomalies,
        "anomalies_truth_values": anomalies_truth_values,
        "no_anomalies": no_anomalies,
        "no_anomalies_truth_values": no_anomalies_truth_values,
        "pred_vals_cl": pred_vals_cl,
        "real_vals_cl": real_vals_cl,
        "H1_H0": H1_H0,
        "tp_rel": tp_rel,
        "fn_rel": fn_rel,
        "fp_rel": fp_rel,
        "tn_rel": tn_rel,
        "tp_true": tp_true,
        "fn_true": fn_true,
        "fp_false": fp_false,
        "tn_false": tn_false,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

    return returnData

###Creates tables to visualize the functionality of the linear predictor and the anomaly detection.
###Input: H1_H0: absolute frequencies of true positives, false positives, false negatives and true negatives, tp_rel: relative frequency of true positives, fn_rel: relative frequency of false negatives, fp_rel: relative frequency of false positives, tn_rel: relative frequency of true negatives, tp_true: conditional frequency of true positives given actual arsenic contamination, fn_true: conditional frequency of false negatives given actual arsenic contamination, fp_false: conditional frequency of false positives given no actual arsenic contamination, tn_false: conditional frequency of true negatives given no actual arsenic contamination
###Output: the absolute, relative and conditional frequencies in tables
def createOverviewTables(H1_H0, tp_rel, fn_rel, fp_rel, tn_rel, tp_true, fn_true, fp_false, tn_false):
    df_abs = pd.DataFrame(H1_H0, index=["is positive", "is negative"], columns=["detected as positive", "detected as negative"])


    #relative frequency
    H1_rel = [tp_rel, fn_rel]
    H0_rel = [fp_rel, tn_rel]

    H1_H0_rel = [H1_rel, H0_rel]

    df_rel = pd.DataFrame(H1_H0_rel, index=["is positive", "is negative"], columns=["detected as positive", "detected as negative"])

    #conditional frequencies
    H1_cond = [tp_true, fn_true]
    H0_cond = [fp_false, tn_false]
    df_cond_true = pd.DataFrame(H1_cond, index=["Detected as positive", "Detected as negative"], columns=["is positive"])

    df_cond_false = pd.DataFrame(H0_cond, index=["Detected as positive", "Detected as negative"], columns=["is negative"])

    return df_abs, df_rel, df_cond_true, df_cond_false

###Prints the tables to visualize the functionality of the linear predictor and the anomaly detection.
###Input: The tables with the absolute, relative and conditional frequencies (df_abs, df_rel, df_cond_true, df_cond_false)
def printOverviewTables(df_abs, df_rel, df_cond_true, df_cond_false):
    print("Absolute frequencies: \n", df_abs, "\n \n")
    print("Relative frequencies (in percent): \n", df_rel, "\n \n")
    print("Conditional frequencies for actual arsenic contamination (in percent): \n", df_cond_true, "\n \n")
    print("Conditional frequencies for no arsenic contamination (in percent): \n", df_cond_false, "\n \n")

###Saves the overview tables
###Input: The tables with the absolute, relative and conditional frequencies (df_abs, df_rel, df_cond_true, df_cond_false)
def saveOverviewTables(df_abs, df_rel, df_cond_true, df_cond_false, filename):
    df_abs.to_csv(f"{filename}_abs.csv")
    df_rel.to_csv(f"{filename}_rel.csv")
    df_cond_true.to_csv(f"{filename}_cond_true.csv")
    df_cond_false.to_csv(f"{filename}_cond_false.csv")

###Creates diagrams to visualize the functionality of the linear predictor and the anomaly detection.
###Input: real_vals_cl: real values of the chlorine sensor readings, pred_vals_cl: predicted values of the chlorine sensor readings, anomalies_truth_values: dictionary with the metrics (e.g. precision, recall) for the anomaly detection, no_anomalies_truth_values: dictionary with the metrics (e.g. precision, recall) for the anomaly detection
###Output: prints the diagrams to visualize the functionality of the linear predictor and the anomaly detection
def createOverviewDiagrams(real_vals_cl, pred_vals_cl, anomalies_truth_values, no_anomalies_truth_values):

    #Vergleich Ground-Truth und Predicted Value

    situations = np.array(list(range(len(real_vals_cl))))
    plt.plot(situations[500:1500] ,np.array(real_vals_cl[500:1500]), label ="Real values")
    plt.plot(situations[500:1500], np.array(pred_vals_cl[500:1500]), label = "Predicted values")

    plt.xlabel("Situations")
    plt.ylabel("Cl concentration (standardized)")
    plt.title("Predicted Cl values in comparison to actual Cl values")
    plt.legend()
    plt.show()

    #Threshold Split

    tp_diffs = np.array([tp_val[1] for tp_val in anomalies_truth_values["true"]])
    fp_diffs = np.array([fp_val[1] for fp_val in anomalies_truth_values["false"]])
    tn_diffs = np.array([tn_val[1] for tn_val in no_anomalies_truth_values["true"]])
    fn_diffs = np.array([fn_val[1] for fn_val in no_anomalies_truth_values["false"]])


    full_diffs = np.concatenate((tp_diffs, fn_diffs, tn_diffs, fp_diffs))

    categories = np.concatenate((
        np.zeros((len(tp_diffs) + len(fn_diffs)), dtype=int),
        np.ones((len(tn_diffs) + len(fp_diffs)), dtype=int)
    ))


    plt.scatter(categories, full_diffs)
    plt.xlabel('Category (0: anomaly; 1: no anomaly)')
    plt.ylabel('Value (<0.5: no anomaly predicted; >0.5 anomaly predicted)')
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.title('Differences and their categories')
    plt.show()

    #Regression line check

    plt.scatter(np.array(real_vals_cl), np.array(pred_vals_cl))
    plt.plot([np.array(real_vals_cl).min(), np.array(real_vals_cl).max()],
            [np.array(real_vals_cl).min(), np.array(real_vals_cl).max()],
            "r--")

###Calculates the metrics for the anomaly detection based on the last 10 differences between the predicted and real values of the chlorine sensor readings.
###Input: data: with predicted and real values of the chlorine sensor readings, X_test: test data for the chlorine and flow sensor readings, as_y: test data for the arsenic sensor readings
###Output: returnData: dictionary with the metrics (e.g. precision, recall) for the anomaly detection based on the last 10 differences between the predicted and real values of the chlorine sensor readings
def calculateMetrics_last_10_diffs(data, X_test, as_y):

    last_10_diffs = []
    ten_anomalies = []
    ten_no_anomalies = []
    id_list = []
    fp_10 = []
    tp_10 = []
    tn_10 = []
    fn_10 = []
    before_last_diff_mean = None

    pred_vals_cl = data["pred_vals_cl"]
    real_vals_cl = data["real_vals_cl"]

    #for each situation, add situation to list and get real as value
    for situation in range(len(pred_vals_cl)):
        id = X_test[situation][0]
        id_list.append(id)
        real_as_val = np.sum(as_y[situation][1])
        
        #if the list of last 10 differences is full, reset it and the id list
        if len(last_10_diffs) == 10:
            last_10_diffs = []
            id_list = []
        last_10_diffs.append(pred_vals_cl[situation] - real_vals_cl[situation])

        #if the list of last 10 differences is full, 
        if len(last_10_diffs) == 10:
            #calculate the mean of the last 10 differences and 
            last_10_diffs_mean = np.mean(last_10_diffs)
            if before_last_diff_mean is None:
                before_last_diff_mean = last_10_diffs_mean
            #compare it to the mean of the previous 10 differences.
            #If too big difference, anomaly, save it as such
            if abs(last_10_diffs_mean-before_last_diff_mean) > 0.1:
                if real_as_val > 0:
                    anom_val = True
                    tp_10.append(id_list)
                else:
                    anom_val = False
                    fp_10.append(id_list)
                ten_anomalies.append((id_list, last_10_diffs_mean, before_last_diff_mean, anom_val))
            #If ok, no anomaly, save it as such
            else:
                if real_as_val > 0:
                    anom_val = True
                    fn_10.append(id_list)
                else:
                    anom_val = False
                    tn_10.append(id_list)
                ten_no_anomalies.append((id_list, last_10_diffs_mean, before_last_diff_mean, anom_val))

            before_last_diff_mean = last_10_diffs_mean

    n_situations = len(X_test)

    #true positives/sensitivity
    tp = len(tp_10)
    tp_rel = round(((tp/n_situations)*100), 2)

    #false positives/alpha error
    fp = len(fp_10)
    fp_rel = round(((fp/n_situations)*100), 2)

    #false negatives/beta error
    fn = len(fn_10)
    fn_rel = round(((fn/n_situations)*100), 2)

    #true negatives/specificity
    tn = len(tn_10)
    tn_rel = round(((tn/n_situations)*100), 2)

    #get conditional frequencies, sorted by as true or false

    n_as_true = tp+fn
    n_as_false = tn+fp

    #tp/true
    tp_true = round(((tp/n_as_true)*100), 2)

    #fp/false
    fp_false = round(((fp/n_as_false)*100), 2)

    #fn/true
    fn_true = round(((fn/n_as_true)*100), 2)

    #tn/false
    tn_false = round(((tn/n_as_false)*100), 2)


    #create overview tables 
    #H1: There is an arsenic contamination
    #H0: There is no arsenic contamination 

    #print("\n \n \n")
    #absolute frequency
    H1 = [tp, fn]
    H0 = [fp, tn]

    H1_H0 = [H1, H0]
            

    returnData = {
        "H1_H0": H1_H0, 
        "anomalies": ten_anomalies,
        "no_anomalies": ten_no_anomalies,
        "fp_list": fp_10,
        "tp_list": tp_10,
        "tn_list": tn_10,
        "fn_list": fn_10,
        "pred_vals_cl": pred_vals_cl,
        "real_vals_cl": real_vals_cl,
        "H1_H0": H1_H0,
        "tp_rel": tp_rel,
        "fn_rel": fn_rel,
        "fp_rel": fp_rel,
        "tn_rel": tn_rel,
        "tp_true": tp_true,
        "fn_true": fn_true,
        "fp_false": fp_false,
        "tn_false": tn_false,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

    return returnData


def getTestData(files):
    X_test = []
    y_test = []
    as_y = []

    #iterate over the files
    for file in range(len(files)):
        #get data per sensor per sensor type 
        chlor_data = files[file]["X_cl_st_per_sensor"]
        flow_data = files[file]["X_flow_st_per_sensor"]
        arsenic_data = files[file]["y"][0]

        #get number of chlorine sensors and number of time steps
        n_cl_sensors = len(chlor_data)
        n_time = len(chlor_data[0]) 

        #For each sensor,
        for cl_sensor in range(n_cl_sensors):
            #For each timestep:
            for timestep in range(2, n_time):
                #x = chlorine reading of sensor at timestepe before and double before that (for predicting next timestep) 
                x = [
                    chlor_data[cl_sensor][timestep-2],
                    chlor_data[cl_sensor][timestep-1]
                ]

                #x also all flow readings of these two timesteps 
                for flow_sensor in range(len(flow_data)):
                    x.append(flow_data[flow_sensor][timestep-2])
                    x.append(flow_data[flow_sensor][timestep-1])

                #y actual chlorine value (should be predicted if regression works), of course not considering anomalies
                y = chlor_data[cl_sensor][timestep]

                #arsenic value to be able to check whether the model works
                as_y_t = ([file, cl_sensor, timestep], arsenic_data[timestep])
                
                #set all x claues together
                full_x = ([file, cl_sensor, timestep], x)
                X_test.append(full_x)

                #save y values
                y_test.append(([file, cl_sensor, timestep], y))
                as_y.append(as_y_t)

    return X_test, y_test, as_y
