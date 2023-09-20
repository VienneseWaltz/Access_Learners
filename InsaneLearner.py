import numpy as np
import LinRegLearner as lrl
import BagLearner as bl

class InsaneLearner(object):
    def __init__(self, verbose=False):
        """
        Initialize an instance of InsaneLearner
        :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
        :type verbose: bool
        :return An initialized instance of InsaneLearner object
        """
        self.verbose = verbose
        self.bag_learners = [bl.BagLearner(lrl.LinRegLearner, {"verbose":verbose}, bags=20, boost=False, verbose=False) for i in range(20)]


    def author(self):
        """
        Auther string

        Returns
        -------
        string
            The GT username of the student.

        """
        return "lsoh3"  # Georgia Tech username


    def add_evidence(self, data_x, data_y):
        """
        Add training data to the learner
        :param data_x: NumPy array of X values
        :param data_y: NumPy array of Y values
        :return: An updated value of individual bag learners within InsaneLearner object
        """

        for self.bag_learner in self.bag_learners:
            self.bag_learner.add_evidence(data_x, data_y)


    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        Parameters
        ----------
        points : numpy.ndarray
            A numpy array with each row corresponding to a specific query. There are multiple rows in
            this numpy array.

        Returns
        -------
        result : numpy.ndarray
            The predicted result of the input data according to the trained model.

        """
        predicted_values = np.array([learner.query(points) for learner in self.bag_learners])
        return predicted_values.mean(axis=0)






