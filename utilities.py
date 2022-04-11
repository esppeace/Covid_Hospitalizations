import pandas as pd
import numpy as np
import random
import torch
import os
import csv

import torch.nn as nn

from torch.utils.data import DataLoader,TensorDataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)

def wisWriter(description, args, quantiles, prediction_list_wis, target_list_wis, state_name_list,county_name_list, start_date_list, first_week_list, second_week_list, third_week_list, fourth_week_list):
    description = description.replace(" ", "_")
    wis_filename = args.model_dir+'/'+description+'_wis.csv'

    prediction_list_wis = prediction_list_wis.squeeze(-1).cpu().detach().numpy()
    
    sample_size = len(first_week_list)

    first_week_list = np.tile(first_week_list, 23)
    second_week_list = np.tile(second_week_list, 23)
    third_week_list = np.tile(third_week_list, 23)
    fourth_week_list = np.tile(fourth_week_list, 23)

    datelist = np.concatenate((first_week_list, second_week_list, third_week_list, fourth_week_list), axis=0)
    

    start_date_list = np.tile(start_date_list, 23)
    start_date_list = np.tile(start_date_list, 4)

    state_name_list = np.tile(state_name_list, 23)
    state_name_list = np.tile(state_name_list, 4)

    county_name_list = np.tile(county_name_list, 23)
    county_name_list = np.tile(county_name_list, 4)

    quantiles = np.repeat(quantiles, sample_size, axis=0)
    quantiles = np.tile(quantiles, 4)
    
    prediction_list = []
    target_list = []

    for i in range(prediction_list_wis.shape[1]):
        if i == 0:
            prediction_list = prediction_list_wis[:, i]
        else:
            prediction_list = np.concatenate((prediction_list, prediction_list_wis[:, i]), axis=0)
    
    for i in range(target_list_wis.shape[1]-1):
        if i == 0:
            target_list = np.tile(target_list_wis[:, i+1], 23)
        else:
            target_list = np.concatenate((target_list, np.tile(target_list_wis[:, i+1], 23)), axis=0)

    with open(wis_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["State", "County", "Start", 
                        "target_date", 'quantile',
                        "prediction", "target_prediction"])
        writer.writerows(zip(state_name_list,county_name_list, start_date_list, \
                            datelist, quantiles, prediction_list, target_list))
    

def eval_model_hospitalization(args, model, device, testloader, path, state_dict, county_dict, description):
    checkpoint = torch.load(path)
    print(description)
    printWis = False
    if description == 'final model state prediction' or description == 'best model state prediction':
        printWis = True
    print("Epoch: ", checkpoint['epoch'])
    model.to(device)
    model.eval()

    state_name_list = []
    county_name_list = []
    start_date_list = []
    first_week_list = []
    second_week_list = []
    third_week_list = []
    fourth_week_list = []

    prediction_list = []
    prediction_list_wis = []
    target_list = []
    counter = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            prediction, prediction2 =  model(data)
            if counter == 0:
                prediction_list = prediction[:, :4]
                prediction_list_wis = prediction2
                target_list = target
                counter += 1
            else:
                prediction_list = torch.cat((prediction_list,prediction[:, :4]), dim=0)
                prediction_list_wis = torch.cat((prediction_list_wis,prediction2), dim=0)
                target_list = torch.cat((target_list, target), dim=0)
   
    target_list_sum = torch.sum(target_list[:,:, 0:5], dim=1)
    index_list = target_list[:,0, 5].reshape(target_list.shape[0]).cpu().detach().numpy()
    state_index = target_list[:,0, 6].reshape(target_list.shape[0]).cpu().detach().numpy()
    start_week_index = target_list[:,0, 7].reshape(target_list.shape[0]).cpu().detach().numpy()
    first_week_index = target_list[:,0, 8].reshape(target_list.shape[0]).cpu().detach().numpy()
    second_week_index = target_list[:,0, 9].reshape(target_list.shape[0]).cpu().detach().numpy()
    third_week_index = target_list[:, 0, 10].reshape(target_list.shape[0]).cpu().detach().numpy()
    fourth_week_index = target_list[:,0, 11].reshape(target_list.shape[0]).cpu().detach().numpy()
    
    for i in range(target_list.shape[0]):
        if index_list[i] == 0:
            # this is state
            state_name = state_dict[str(int(state_index[i]))]['state'][int(start_week_index[i])]
            state_name_list.append(state_name)
            county_name = state_dict[str(int(state_index[i]))]['state'][int(start_week_index[i])]
            county_name_list.append(county_name)
            start_week = state_dict[str(int(state_index[i]))]['date'][int(start_week_index[i])+6]
            start_date_list.append(start_week)
            first_week = state_dict[str(int(state_index[i]))]['date'][int(first_week_index[i])+6]
            first_week_list.append(first_week)
            second_week = state_dict[str(int(state_index[i]))]['date'][int(second_week_index[i])+6]
            second_week_list.append(second_week)
            third_week = state_dict[str(int(state_index[i]))]['date'][int(third_week_index[i])+6]
            third_week_list.append(third_week)
            fourth_week = state_dict[str(int(state_index[i]))]['date'][int(fourth_week_index[i])+6]
            fourth_week_list.append(fourth_week)
        else:
            # this is counties
            state_name = county_dict[str(int(state_index[i]))]['state'][int(start_week_index[i])]
            state_name_list.append(state_name)
            county_name = county_dict[str(int(state_index[i]))]['county'][int(start_week_index[i])]
            county_name_list.append(county_name)
            start_week = county_dict[str(int(state_index[i]))]['date'][int(start_week_index[i])+6]
            start_date_list.append(start_week)
            first_week = county_dict[str(int(state_index[i]))]['date'][int(first_week_index[i])+6]
            first_week_list.append(first_week)
            second_week = county_dict[str(int(state_index[i]))]['date'][int(second_week_index[i])+6]
            second_week_list.append(second_week)
            third_week = county_dict[str(int(state_index[i]))]['date'][int(third_week_index[i])+6]
            third_week_list.append(third_week)
            fourth_week = county_dict[str(int(state_index[i]))]['date'][int(fourth_week_index[i])+6]
            fourth_week_list.append(fourth_week)

    prediction_list2 = prediction_list.squeeze(-1).cpu().detach().numpy()
    target_list2 = target_list_sum.squeeze(-1).cpu().detach().numpy()
    
    if printWis:
        wisWriter(description, args, model.quantiles, prediction_list_wis, target_list2, state_name_list,county_name_list, start_date_list, first_week_list, second_week_list, third_week_list, fourth_week_list)
    
    description = description.replace(" ", "_")
    csvfile = args.model_dir+'/'+description+'.csv'

    p1, p2, p3, p4 = zip(*prediction_list2)
    t0, t1, t2, t3, t4 = zip(*target_list2)
    with open(csvfile, 'w') as f:
        writer = csv.writer(f, delimiter=',')

        writer.writerow(["State", "County", "Start", 
                        "First week", "Naive 7 days",  "Prediction 7 days", "Targets 7 days",
                        "Second week","Naive 14 days", "Prediction 14 days", "Targets 14 days",
                        "Third week", "Naive 21 days", "Prediction 21 days", "Targets 21 days",
                        "Fourth week", "Naive 28 days", "Prediction 28 days","Targets 28 days"])
        writer.writerows(zip(state_name_list,county_name_list, start_date_list, \
                            first_week_list, t0, p1, t1,
                            second_week_list, t0, p2, t2,
                            third_week_list,t0, p3, t3,
                            fourth_week_list, t0, p4, t4))
    
def create_dataset_hospitalization(input_file, output_file, size=14, size_y=7):
    datasetx = []
    datasety = []
    input_dim = input_file[0].shape

    data_x = np.stack(input_file, axis=-1)
    data_y = np.stack(output_file, axis=-1)

    output_dim = data_y.shape[1]

    for i in range(input_dim[0]-size-4*size_y):
        x = size
        y = size + 4*size_y
        if i+y <= (input_dim[0]):

            xvalue = data_x[i:i+size, 0:-3] # week 0 0-13

            list_value = []
            for j in range(output_dim):
                yvalue0 = data_y[i+size-size_y:i+size, j] # week naive 7-13 (we should use the second week value), if size=7, then still valid 0-6
                yvalue0 = yvalue0.reshape(yvalue0.shape[0], 1)
                list_value.append(yvalue0)

                yvalue1 = data_y[i+size:i+size+size_y, j] # week 1 14-20 / 7-13
                yvalue1 = yvalue1.reshape(yvalue1.shape[0], 1)
                list_value.append(yvalue1)

                yvalue2 = data_y[i+size+size_y:i+2*size_y+size, j] # week 2 21-27 / 14 - 20
                yvalue2 = yvalue2.reshape(yvalue2.shape[0], 1)
                list_value.append(yvalue2)

                yvalue3 = data_y[i+2*size_y+size:i+3*size_y+size, j] # week 3 28-34 / 21 - 27
                yvalue3 = yvalue3.reshape(yvalue3.shape[0], 1)
                list_value.append(yvalue3)

                yvalue4 = data_y[i+3*size_y+size:i+4*size_y+size, j] # week 4 35-41 / 28 - 34
                yvalue4 = yvalue4.reshape(yvalue4.shape[0], 1)
                list_value.append(yvalue4)

            # date
            
            #type
            yvalue5 = data_x[i:i+size_y,-3]
            yvalue5 = yvalue5.reshape(yvalue5.shape[0], 1)
            list_value.append(yvalue5)

            #index
            yvalue6 = data_x[i:i+size_y, -2]
            yvalue6 = yvalue6.reshape(yvalue6.shape[0], 1)
            list_value.append(yvalue6)

            #date testing
            yvalue7 = data_x[i:i+size_y, -1]
            yvalue7 = yvalue7.reshape(yvalue7.shape[0], 1)
            list_value.append(yvalue7)

            #date 1week
            yvalue8 = data_x[i+size:i+size+size_y, -1]
            yvalue8 = yvalue8.reshape(yvalue8.shape[0], 1)
            list_value.append(yvalue8)

            #date 2nd week
            yvalue9 = data_x[i+size+size_y:i+size+2*size_y, -1]
            yvalue9 = yvalue9.reshape(yvalue9.shape[0], 1)
            list_value.append(yvalue9)

            #date 3rd week
            yvalue10 = data_x[i+size+2*size_y:i+size+3*size_y, -1]
            yvalue10 = yvalue10.reshape(yvalue10.shape[0], 1)
            list_value.append(yvalue10)

            #date 4th week
            yvalue11 = data_x[i+size+3*size_y:i+size+4*size_y, -1]
            yvalue11 = yvalue11.reshape(yvalue11.shape[0], 1)
            list_value.append(yvalue11)

            yvalue = np.concatenate(list_value, axis=-1)

            datasetx.append(xvalue)
            datasety.append(yvalue)

        else:
            print('Not enough: ', i, len(input_file))

    x, y = torch.FloatTensor(datasetx), torch.FloatTensor(datasety)
    return x, y 

def createLoader_split_train_val_hospitalization(args):
    shuffle = args.shuffle
    standardize = args.standardize
    BATCH_SIZE = args.batch_size
    TRAIN_SIZE = args.train_size
    TRAIN_VAL_SIZE = args.train_val_size
    remove_day = args.remove_days
    state_folder = os.path.join(args.state_data_dir, args.state_dir)
    counties_folder = os.path.join(args.county_data_dir, args.counties_dir)
    full_state_file = os.path.join(args.state_data_dir, "us-states.csv")
    us_states_df = pd.read_csv(full_state_file)
    
    states_list = list(us_states_df.state.unique())
    print('Original: ', np.array(states_list))
    
    remove_state = ["Puerto Rico", "Northern Mariana Islands", "Rhode Island", "District of Columbia", "Hawaii", "Alaska", "Guam", "Virgin Islands"]
    for state in remove_state:
        print("\nRemoved state: ", state)
        states_list.remove(state)


    print("\nupdated: ", np.array(states_list))

    full_train_cases_x = []
    full_test_cases_x = []

    full_train_cases_y = []
    full_test_cases_y = []

    full_train_cases_states_x = []
    full_test_cases_states_x = []

    full_train_cases_states_y = []
    full_test_cases_states_y = []

    state_dict = {}
    county_dict = {}
    count = 0

    # we have two extra remove day for the state and county since they have different collection day
    state_remove_day = args.state_remove_day
    county_remove_day = args.county_remove_day
   
    for state in states_list:
        filename = state_folder+ '/' + str(state) +'/'+ str(state) + ".csv"
        state_df = pd.read_csv(filename)

        state_df['type'] = 0
        state_df['state_index'] = count
        state_df['month'] = pd.to_datetime(state_df.date).dt.month
        state_df['index'] = range(0, len(state_df.date.values))
        state_dict[str(count)] = {'date': list(state_df.date.values), 'state': list(state_df.state.values), 'month': list(state_df.month.values)}
        count += 1

        type = state_df.type.values[remove_day:]
        state_index = state_df.state_index.values[remove_day:]
        index = state_df.index.values[remove_day:]

        cases = state_df['cases_diff'].values[remove_day:]
        deaths = state_df['deaths_diff'].values[remove_day:]

        smoothing_deaths = state_df.deaths_diff.rolling(7).mean()
        smoothing_cases = state_df.cases_diff.rolling(7).mean()
        cases_s = smoothing_cases[remove_day:]
        deaths_s = smoothing_deaths[remove_day:]

        train_size = int(TRAIN_SIZE*len(deaths))-state_remove_day
    
        deaths_train = deaths[state_remove_day:train_size]
        cases_train = cases[state_remove_day:train_size]
        deaths_test = deaths[train_size:]
        cases_test = cases[train_size:]

        deaths_s_train = deaths_s[state_remove_day:train_size]
        cases_s_train = cases_s[state_remove_day:train_size]
        deaths_s_test = deaths_s[train_size:]
        cases_s_test = cases_s[train_size:]

        input_list=[]
        if args.use_cases:
            input_list.append(cases_train)            
            input_list.append(cases_s_train)

        if args.use_deaths:
            input_list.append(deaths_train)
            input_list.append(deaths_s_train)

        input_list_test=[]
        
        if args.use_cases:
            input_list_test.append(cases_test)
            input_list_test.append(cases_s_test)
        
        if args.use_deaths:
            input_list_test.append(deaths_test) 
            input_list_test.append(deaths_s_test)

        if args.use_economy:
            economy = state_df['household_median_income'].values[remove_day:]
            economy_train = economy[state_remove_day:train_size]
            economy_test = economy[train_size:]
            input_list.append(economy_train)
            input_list_test.append(economy_test)

        if args.use_hospitalization:
            for i in args.hospitalization_list:
                hospitalization_data = state_df[i].values[remove_day:]
                hospitalization_data_train = hospitalization_data[state_remove_day:train_size]
                hospitalization_data_test = hospitalization_data[train_size:]
                input_list.append(hospitalization_data_train)
                input_list_test.append(hospitalization_data_test)

                if args.use_hospitalization_smoothing:
                    # smoothing for hospitalization
                    hospitalization_data_smoothing = state_df[i].rolling(7).mean()
                    hospitalization_data_smoothing = hospitalization_data_smoothing[remove_day:]

                    hospitalization_data_s_train = hospitalization_data_smoothing[state_remove_day:train_size]
                    hospitalization_data_s_test = hospitalization_data_smoothing[train_size:]
                    input_list.append(hospitalization_data_s_train)
                    input_list_test.append(hospitalization_data_s_test)
    
        input_list.append(type[state_remove_day:train_size])
        input_list.append(state_index[state_remove_day:train_size])
        input_list.append(index[state_remove_day:train_size])

        input_list_test.append(type[train_size:])
        input_list_test.append(state_index[train_size:])
        input_list_test.append(index[train_size:])

        output_list =[]
        if args.use_hospitalization:
            for i in args.hospitalization_prediction_list:
                if args.use_smoothing_for_train and args.use_hospitalization_smoothing:
                   # smoothing for hospitalization
                    hospitalization_data_smoothing = state_df[i].rolling(7).mean()
                    hospitalization_data_smoothing = hospitalization_data_smoothing[remove_day:]

                    hospitalization_data_s_train = hospitalization_data_smoothing[state_remove_day:train_size]
                    hospitalization_data_s_test = hospitalization_data_smoothing[train_size:]
                    output_list.append(hospitalization_data_s_train)
                else:
                    hospitalization_data = state_df[i].values[remove_day:]
                    hospitalization_data_train = hospitalization_data[state_remove_day:train_size]
                    hospitalization_data_test = hospitalization_data[train_size:]
                    output_list.append(hospitalization_data_train)
                    
        # put as training
        if not args.not_using_states:
            train_x, train_y = create_dataset_hospitalization(input_list, output_list, size=args.use_how_many_days)
            full_train_cases_x.append(train_x)
            full_train_cases_y.append(train_y)

        output_list =[]
        if args.use_hospitalization:
            for i in args.hospitalization_prediction_list:
                hospitalization_data = state_df[i].values[remove_day:]
                hospitalization_data_train = hospitalization_data[state_remove_day:train_size]
                hospitalization_data_test = hospitalization_data[train_size:]
                output_list.append(hospitalization_data_test)
        # output_list.append(cases_test)

        test_x, test_y = create_dataset_hospitalization(input_list_test, output_list, size=args.use_how_many_days)

        full_test_cases_states_x.append(test_x)
        full_test_cases_states_y.append(test_y)

    full_test_states_x, full_test_states_y = torch.cat(full_test_cases_states_x), torch.cat(full_test_cases_states_y)

    if not args.not_using_counties:
        county_count = 0
        for state in states_list:
            folder_name = counties_folder + '/' + state
            for county in sorted(os.listdir(folder_name)):
                filename = folder_name+ '/' + county
                state_df = pd.read_csv(filename)

                # this is to find counties dict to found later
                state_df['type'] = 1
                # this is the index of that state here
                state_df['state_index'] = county_count
                state_df['month'] = pd.to_datetime(state_df.date).dt.month
                # this is the index of each row
                state_df['index'] = range(0, len(state_df.date.values))
                county_dict[str(county_count)] = {'date': list(state_df.date.values), 'state': list(state_df.state.values), 'county': list(state_df.county.values),'month': list(state_df.month.values)}
                county_count += 1

                
                smoothing_deaths = state_df.deaths_diff.rolling(7).mean()
                smoothing_cases = state_df.cases_diff.rolling(7).mean()
                cases = state_df['cases_diff'].values[remove_day:]
                deaths = state_df['deaths_diff'].values[remove_day:]
                cases_s = smoothing_cases[remove_day:]
                deaths_s = smoothing_deaths[remove_day:]


                type = state_df.type.values[remove_day:]
                state_index = state_df.state_index.values[remove_day:]
                index = state_df.index.values[remove_day:]

                train_size = int(TRAIN_SIZE*len(deaths))-county_remove_day
                              
                deaths_train = deaths[county_remove_day:train_size]
                cases_train = cases[county_remove_day:train_size]
                deaths_test = deaths[train_size:]
                cases_test = cases[train_size:]

                deaths_s_train = deaths_s[county_remove_day:train_size]
                cases_s_train = cases_s[county_remove_day:train_size]
                deaths_s_test = deaths_s[train_size:]
                cases_s_test = cases_s[train_size:]

                input_list=[]
                if args.use_cases:
                    input_list.append(cases_train)
                    input_list.append(cases_s_train)
                
                if args.use_deaths:
                    input_list.append(deaths_train)
                    input_list.append(deaths_s_train)


                input_list_test=[]

                if args.use_cases:
                    input_list_test.append(cases_test)
                    input_list_test.append(cases_s_test)
                
                if args.use_deaths:
                    input_list_test.append(deaths_test)
                    input_list_test.append(deaths_s_test)


                if args.use_economy:
                    economy = state_df['household_median_income'].values[remove_day:]
                    economy_train = economy[county_remove_day:train_size]
                    economy_test = economy[train_size:]
                    input_list.append(economy_train)
                    input_list_test.append(economy_test)

                if args.use_hospitalization:
                    for i in args.hospitalization_list:
                        hospitalization_data = state_df[i].values[remove_day:]
                        hospitalization_data_train = hospitalization_data[county_remove_day:train_size]
                        hospitalization_data_test = hospitalization_data[train_size:]
                        input_list.append(hospitalization_data_train)
                        input_list_test.append(hospitalization_data_test)

                        if args.use_hospitalization_smoothing:
                            # smoothing for hospitalization
                            hospitalization_data_smoothing = state_df[i].rolling(7).mean()
                            hospitalization_data_smoothing = hospitalization_data_smoothing[remove_day:]

                            hospitalization_data_s_train = hospitalization_data_smoothing[county_remove_day:train_size]
                            hospitalization_data_s_test = hospitalization_data_smoothing[train_size:]
                            input_list.append(hospitalization_data_s_train)
                            input_list_test.append(hospitalization_data_s_test)



                input_list.append(type[county_remove_day:train_size])
                input_list.append(state_index[county_remove_day:train_size])
                input_list.append(index[county_remove_day:train_size])

                input_list_test.append(type[train_size:])
                input_list_test.append(state_index[train_size:])
                input_list_test.append(index[train_size:])

                output_list =[]
                if args.use_hospitalization:
                    for i in args.hospitalization_prediction_list:
                        #print('State: ', i)
                        if args.use_smoothing_for_train and args.use_hospitalization_smoothing:
                            # smoothing for hospitalization
                            hospitalization_data_smoothing = state_df[i].rolling(7).mean()
                            hospitalization_data_smoothing = hospitalization_data_smoothing[remove_day:]

                            hospitalization_data_s_train = hospitalization_data_smoothing[county_remove_day:train_size]
                            hospitalization_data_s_test = hospitalization_data_smoothing[train_size:]
                            output_list.append(hospitalization_data_s_train)
                        else:
                            hospitalization_data = state_df[i].values[remove_day:]
                            hospitalization_data_train = hospitalization_data[county_remove_day:train_size]
                            hospitalization_data_test = hospitalization_data[train_size:]
                            output_list.append(hospitalization_data_train)
                            
        
                train_x, train_y = create_dataset_hospitalization(input_list, output_list, size=args.use_how_many_days)
                full_train_cases_x.append(train_x)
                full_train_cases_y.append(train_y)

                output_list_test =[]
                if args.use_hospitalization:
                    for i in args.hospitalization_prediction_list:
                        hospitalization_data = state_df[i].values[remove_day:]
                        hospitalization_data_train = hospitalization_data[county_remove_day:train_size]
                        hospitalization_data_test = hospitalization_data[train_size:]
                        output_list_test.append(hospitalization_data_test)

                test_x, test_y = create_dataset_hospitalization(input_list_test, output_list_test, size=args.use_how_many_days)

                full_test_cases_x.append(test_x)
                full_test_cases_y.append(test_y)

    full_train_x_ori, full_train_y_ori = torch.cat(full_train_cases_x), torch.cat(full_train_cases_y)

    # split train val
    print(full_train_x_ori.shape)
    t_size = int(TRAIN_VAL_SIZE*len(full_train_x_ori))
    print(len(full_train_x_ori))

    index = np.arange(len(full_train_x_ori))
    random.shuffle(index)

    train_index = index[:t_size]
    val_index = index[t_size:]

    full_train_x = full_train_x_ori[train_index, :]
    full_val_x = full_train_x_ori[val_index, :]

    full_train_y = full_train_y_ori[train_index, :]
    full_val_y = full_train_y_ori[val_index, :]

    if not args.not_using_counties:
        full_test_x, full_test_y = torch.cat(full_test_cases_x), torch.cat(full_test_cases_y)
    else:
        full_test_x, full_test_y = full_test_states_x, full_test_states_y

    if standardize:
        mean = torch.mean(full_train_x.reshape((-1, full_train_x.shape[-1])),dim=0)
        print("means: ", mean)

        std = torch.std(full_train_x.reshape((-1, full_train_x.shape[-1])),dim=0)
        print("stds:", std)
        full_train_x = (full_train_x - mean)/std
        full_test_x = (full_test_x - mean)/std
        full_val_x = (full_val_x - mean)/std

        full_test_states_x = (full_test_states_x - mean)/std

    print("trainx: ", full_train_x.shape, "trainy: ", full_train_y.shape)
    print("valx: ", full_val_x.shape, "valy: ", full_val_y.shape)
    print("testx: ", full_test_x.shape, "testy: ", full_test_y.shape)

    print("state testx: ", full_test_states_x.shape, "testy: ", full_test_states_y.shape)

    input_dim = full_train_x.shape[-1]
    output_dim = full_train_y.shape[-1]

    train_loader = DataLoader(TensorDataset(full_train_x, full_train_y), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(TensorDataset(full_val_x, full_val_y), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(full_test_x, full_test_y), batch_size=BATCH_SIZE, shuffle=False)
    
    test_states_loader = DataLoader(TensorDataset(full_test_states_x, full_test_states_y), batch_size=BATCH_SIZE, shuffle=False)

    args.input_dim = input_dim

    if args.num_of_week == 4:
        args.output_dim = output_dim - 7   # fixed 8 for information
        args.output_dim -= int(args.output_dim/5)
        
    else:
        # this is only for single variable prediction
        args.output_dim = output_dim - 10 # predict only 2 weeks

    args.num_training_steps = len(train_loader)*args.epochs
    print('Output dim :' ,args.output_dim)
    return train_loader, val_loader, test_loader, test_states_loader, state_dict, county_dict


