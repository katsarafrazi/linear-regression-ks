import numpy as np
from simple_linear_regr_utils import generate_data, evaluate
import pickle
import time, os

VERSION = 1.0
DEF_LR = 0.1
DEF_ITERATIONS = 15000

class SimpleLinearRegression:
    def __init__(self, iterations=DEF_ITERATIONS, lr=DEF_LR):
        self.iterations = iterations # number of iterations the fit method will be called
        self.lr = lr # The learning rate
        self.losses = [] # A list to hold the history of the calculated losses
        self.W, self.b = None, None # the slope and the intercept of the model
        self.version = VERSION

    def __loss(self, y, y_hat):
        """

        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: the sum of squared error

        """
        loss =((y-y_hat)**2).sum()

        self.losses.append(loss)
        return loss

    def __init_weights(self, X):
        """

        :param X: The training set
        """
        weights = np.random.normal(size=X.shape[1] + 1)
        self.W = weights[:X.shape[1]].reshape(-1, X.shape[1])
        self.b = weights[-1]

    def __sgd(self, X, y, y_hat):
        """

        :param X: The training set
        :param y: The actual output on the training set
        :param y_hat: The predicted output on the training set
        :return:
            sets updated W and b to the instance Object (self)
        """

        ind = np.random.randint(0,X.shape[0])
        dW = -2*X[ind]*(y[ind]-y_hat[ind])
        db = -2*(y[ind]-y_hat[ind])

        self.W -= self.lr * dW
        self.b -= self.lr * db


    def fit(self, X, y):
        """

        :param X: The training set
        :param y: The true output of the training set
        :return:
        """
        t1 = time.time()
        self.__init_weights(X)
        y_hat = self.predict(X)
        loss = self.__loss(y, y_hat)
        print(f"Initial Loss: {loss}")
        for i in range(self.iterations + 1):
            self.__sgd(X, y, y_hat)
            y_hat = self.predict(X)
            loss = self.__loss(y, y_hat)
            if not i % 10000:
                print(f"Iteration {i}, Loss: {loss}")
        
        print(f"Training Time: {time.time()-t1} secs")

    def predict(self, X):
        """

        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        #ToDO calculate the predicted output y_hat. remember the function of a line is defined as y = WX + b
        y_hat = X @ self.W + self.b
        return y_hat


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_data()
    
    model = SimpleLinearRegression()
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    evaluate(model, X_test, y_test, predicted)
    
    # Save model
    with open(os.path.join(os.getcwd(),'model','model.pickle'), 'wb') as f:
        pickle.dump(model, f)
        
    print("Model Saved Successfully.")
