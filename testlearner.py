""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import math  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import sys  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import timeit

  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    if len(sys.argv) != 2:  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        print("Usage: python testlearner.py <filename>")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sys.exit(1)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    inf = open(sys.argv[1])
    # Skip the first row of the csv file
    inf.readline()
    data = np.array(
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()]
    )

    # compute how much of the data is training and testing  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_rows = data.shape[0] - train_rows  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # separate out training and testing data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_x = data[train_rows:, 0:-1]  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_y = data[train_rows:, -1]  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			


    ###########################################################################
    # create a learner for DTLearner and train it
    ###########################################################################
    learner = dt.DTLearner(leaf_size = 1, verbose=False)  # create a DTLearner
    learner.add_evidence(train_x, train_y)  # train it  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    #print(learner.author())
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # evaluate in sample  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    pred_y = learner.query(train_x)  # get the predictions  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()  		  
    print("******* Results for DTLeaner ********") 	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print("In sample results")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"RMSE: {rmse}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")
    in_sample_variance = np.nanvar(pred_y, ddof=1)
    print(f"In Sample Variance: {in_sample_variance}")

    in_sample_mae = (np.abs((train_y - pred_y)/train_y)).mean(axis=0)
    print(f"In-sample Mean Absolute Error = {in_sample_mae}")
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # evaluate out of sample  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    pred_y = learner.query(test_x)  # get the predictions  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print("Out of sample results")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"RMSE: {rmse}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    c = np.corrcoef(pred_y, y=test_y)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"corr: {c[0,1]}")
    out_of_sample_variance = np.nanvar(pred_y, ddof=1)
    print(f"Out of Sample Variance: {out_of_sample_variance}")

    out_of_sample_mae = (np.abs((test_y - pred_y)/test_y)).mean(axis=0)
    print(f"Out-of-sample Mean Absolute Error = {out_of_sample_mae}")


    #############################################################################
    # create a learner for RTLearner and train it.
    #############################################################################
    learner = rt.RTLearner(leaf_size=1, verbose=False)  # create a RTLearner
    learner.add_evidence(train_x, train_y)  # train it
    #print(learner.author())

    # evaluate in sample
    pred_y = learner.query(
        train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("********* Results for RTLearner *********")
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")
    in_sample_variance = np.nanvar(pred_y, ddof=1)
    print(f"In Sample Variance: {in_sample_variance}")

    in_sample_mae = (np.abs((train_y - pred_y) / train_y)).mean(axis=0)
    print(f"In-sample Mean Absolute Error = {in_sample_mae}")

    # evaluate out of sample
    pred_y = learner.query(
        test_x)  # get the predictions

    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")
    out_of_sample_variance = np.nanvar(pred_y, ddof=1)
    print(f"Out of Sample Variance: {out_of_sample_variance}")
    out_of_sample_mae = (np.abs((test_y - pred_y) / test_y)).mean(axis=0)
    print(f"Out-of-sample Mean Absolute Error = {out_of_sample_mae}")

    #############################################################################
    # create a learner for BagLearner and train it.
    #############################################################################
    #learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=15}, bags=10, boost=False, verbose=False) # create a BagLearner with LinRegLearner
    learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size':1}, bags=20, boost=False, verbose=False) # create a BagLearner with DTLearner
    # learner = bl.BagLearner(learner=rt.RTLearner, kwargs={'leaf_size':100}, bags=10, boost=False, verbose=False) # create a BagLearner with RTLearner
    learner.add_evidence(train_x, train_y)  # train it
    #print(learner.author())

    # evaluate in sample
    pred_y = learner.query(
        train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("********* Results for BagLearner *********")
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(
        test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

    #############################################################################
    # create a learner for InsaneLearner and train it.
    #############################################################################
    learner = it.InsaneLearner(verbose=False) # create an InsaneLearner
    learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())

    # evaluate in sample
    pred_y = learner.query(
        train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("********* Results for InsaneLearner *********")
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(
        test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")


    ################################################
    # Plotting graphs for Experiment 1
    ###############################################
    # Line 1 points
    leaf_size = [5, 10, 15, 30, 50, 60, 80, 100]
    out_of_sample_RSME = [0.0056957, 0.0056957, 0.0053756, 0.0040336, 0.0062093, 0.0062300, 0.006274, 0.0062247]

    # Plotting leaf_size vs out-of-sample RSME values
    plt.plot(leaf_size, out_of_sample_RSME, label='Out-of-sample RMSE')

    # Line 2 points
    leaf_size = [5, 10, 15, 30, 50, 60, 80, 100]
    in_sample_RSME = [0.0042253, 0.0042253, 0.0053468, 0.0052860, 0.0074501, 0.00756239, 0.0087720, 0.0090121]

    # Plotting leaf_size vs in-sample RSME values
    plt.plot(leaf_size, in_sample_RSME, label='In-sample RMSE')

    plt.xlabel('Leaf size')
    plt.ylabel('RMSE values')
    plt.title('Leaf size vs Out-of-sample and In-sample RMSE values ')
    plt.legend()
    plt.show()



    ################################################
    # Plotting graphs for Experiment 2
    ###############################################
    # Line 1 points
    leaf_size = [5, 10, 15, 30, 50, 60, 80, 100]
    out_of_sample_RSME = [0.0044303, 0.0045442, 0.0045675, 0.0046818, 0.0048613, 0.0062300, 0.006274, 0.0062247]

    # Plotting leaf_size vs out-of-sample RSME values
    plt.plot(leaf_size, out_of_sample_RSME, label='Out-of-sample RMSE')

    # Line 2 points
    leaf_size = [5, 10, 15, 30, 50, 60, 80, 100]
    in_sample_RSME = [0.0034627, 0.0044694, 0.0050392, 0.0058578, 0.0065003, 0.0065906, 0.0079512, 0.00795118]

    # Plotting leaf_size vs in-sample RSME values
    plt.plot(leaf_size, in_sample_RSME, label='In-sample RMSE')

    plt.xlabel('Leaf size')
    plt.ylabel('RMSE values')
    plt.title('Experiment 2- Leaf size vs Out-of-sample and In-sample RMSE values ')
    plt.legend()
    plt.show()


    ################################################
    # Plotting graphs for Experiment 3A
    ###############################################
    # Line 1-DT points for DTLearner
    leaf_size = [5, 10, 20, 40, 50, 60, 80, 100]
    out_of_sample_variance_DT = [8.10550e-05, 6.33288e-05, 6.94411e-05, 6.57966e-05, 6.23759e-05, 6.07161e-05, 5.17165e-05, 4.66295e-05]

    # Plotting leaf_size vs out-of-sample variance values of DTLearner
    plt.plot(leaf_size, out_of_sample_variance_DT, label='Out-of-sample variance values of DTLearner')


    # Line 1-RT points for RTLearner
    leaf_size = [5, 10, 20, 40, 50, 60, 80, 100]
    out_of_sample_variance_RT = [6.56987e-05, 7.10736e-05, 6.00291e-05, 4.47358e-05, 5.29354e-05, 5.29354e-05, 4.18489e-05, 4.29914e-05]

    # Plotting leaf_size vs out-of-sample variance values of RTLearner
    plt.plot(leaf_size, out_of_sample_variance_RT, label='Out-of-sample variance values of RTLearner')

    plt.xlabel('Leaf size')
    plt.ylabel('Variance Values')
    plt.title('Experiment 3A - Leaf size vs Out-of-sample Variance Values of DT and RT ')
    plt.legend()
    plt.show()

    # Line 2-DT points for DTLearner
    leaf_size = [5, 10, 20, 40, 50, 60, 80, 100]
    in_sample_variance_DT = [0.000121058, 0.000110289, 0.00010029, 8.82499e-05, 8.32888e-05, 8.15989e-05, 6.17778e-05,
                             5.74966e-05]

    # Plotting leaf_size vs in-sample variance values of DTLearner
    plt.plot(leaf_size, in_sample_variance_DT, label='In-sample variance values of DTLearner')

    # Line 2-RT points for RTLearner
    leaf_size = [5, 10, 20, 40, 50, 60, 80, 100]
    in_sample_variance_RT = [ 9.70629e-05 ,8.87709e-05,7.23203e-05,6.30011e-05,6.543121e-05,6.54312e-05, 5.02808e-05,4.17260e-05 ]

    # Plotting leaf_size vs in-sample variance values of RTLearner
    plt.plot(leaf_size, in_sample_variance_RT, label='In-sample variance values of RTLearner')

    plt.xlabel('Leaf size')
    plt.ylabel('Variance Values')
    plt.title('Experiment 3A - Leaf size vs In-sample Variance Values of DT and RT ')
    plt.legend()
    plt.show()


    ################################################
    # Plotting graphs for Experiment 3B
    ###############################################
    # Line 1-DT points for DTLearner
    leaf_size = [5, 10, 20, 40, 50, 60, 80, 100]
    out_of_sample_MAE_DT = [3.55430, 2.78985, 3.36382, 5.87322, 8.70788, 8.48607, 8.59175, 5.17239]

    # Plotting leaf_size vs out-of-sample MAE values of DTLearner
    plt.plot(leaf_size, out_of_sample_MAE_DT, label='Out-of-sample MAE values of DTLearner')

    # Line 1-RT points for RTLearner
    leaf_size = [5, 10, 20, 40, 50, 60, 80, 100]
    out_of_sample_MAE_RT = [6.68141, 6.52495, 2.74108, 3.04721, 4.77533,
                                 2.40598, 3.59076, 3.35314]

    # Plotting leaf_size vs out-of-sample MAE values of RTLearner
    plt.plot(leaf_size, out_of_sample_MAE_RT, label='Out-of-sample MAE values of RTLearner')

    plt.xlabel('Leaf size')
    plt.ylabel('MAE Values')
    plt.title('Experiment 3B - Leaf size vs Out-of-sample MAE Values of DT and RT ')
    plt.legend()
    plt.show()

    # Line 2-DT points for DTLearner
    leaf_size = [5, 10, 20, 40, 50, 60, 80, 100]
    in_sample_MAE_DT = [1.80446, 2.19652, 2.52078, 2.82146,
                        3.47855, 3.15213, 3.67704, 3.39390]

    # Plotting leaf_size vs in-sample MAE values of DTLearner
    plt.plot(leaf_size, in_sample_MAE_DT, label='In-sample MAE values of DTLearner')

    # Line 2-RT points for RTLearner
    leaf_size = [5, 10, 20, 40, 50, 60, 80, 100]
    in_sample_MAE_RT = [3.0649988, 2.59517, 2.55892, 2.49175,
                        3.46043, 2.98886, 3.49741, 3.58203]

    # Plotting leaf_size vs in-sample MAE values of RTLearner
    plt.plot(leaf_size, in_sample_MAE_RT, label='In-sample MAE values of RTLearner')

    plt.xlabel('Leaf size')
    plt.ylabel('MAE Values')
    plt.title('Experiment 3B - Leaf size vs In-sample MAE Values of DT and RT ')
    plt.legend()
    plt.show()











