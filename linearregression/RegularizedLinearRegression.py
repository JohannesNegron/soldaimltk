import numpy as np
import scipy.optimize as op
import soldaimltk.general.Model as Model

class RegularizedLinearRegression(object):

    def __init__(self, model, theta, alpha, l):
        self.model = model
        self.theta = theta
        self.alpha = alpha
        self.l = l
    
    @classmethod
    def fromModel(cls, model, alpha = 0.1, l=0.1):
        theta = np.rand(model.y.shape)
        return cls(model, theta, alpha, l)
    
    @classmethod
    def fromDataFile(cls, datafile, delimeter, alpha=0.1, l=0.1):
        model = Model.fromDataFile(datafile, delimeter)
        theta = np.rand(model.y.shape)
        return cls(model, theta, alpha, l)

    #m denotes the number of examples
    #gradient indicates the gradient matrix
    #regularization is the regularization parameter in order to prevent the over fitting
    #cost is the cotst of the logistic regression funciton
    def getCostAndGradient(self, theta):
        X = self.model.getBiasedX()
        gradient = np.zeros(np.size(theta))
        h_theta = self.evaluateHypothesis(X, theta).reshape(self.model.m, 1)
        differences = h_theta - self.model.y
        cost_m = differences **2
        regularization = (self.l * np.sum(theta[1 :]**2))/(2*self.model.m)
        cost = np.sum(cost_m)/(2*self.model.m) + regularization        
        gradient[0] = np.sum(differences * X[:, 0].reshape(len(X), 1))/self.model.m
        #We calculate the gradient for the rest of the parameters
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * X[:, j].reshape(len(X), 1))/self.model.m + self.l/self.model.m * theta[j]
        return cost, gradient
    
    @staticmethod
    def getCost(theta, x, y, l=0):
        m = len(x)
        differences = np.matmul(x, theta).reshape(m, 1) - y
        cost_m = differences **2
        regularization = (l * np.sum(theta[1 :]**2))/(2*m)
        cost = np.sum(cost_m)/(2*m) + regularization
        return cost
    
    @staticmethod
    def getGradient(theta, x, y, l=0):
        gradient = np.zeros(np.size(theta))
        m = len(x)
        differences = np.matmul(x, theta).reshape(m, 1) - y
        gradient[0] = np.sum(differences * x[:, 0].reshape(m, 1))/m
        #We calculate the gradient for the rest of the parameters
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * x[:, j].reshape(m, 1))/m + (l/m) * theta[j]
        return gradient

    def evaluateHypothesis(self, xi, theta):
        return np.matmul(xi, theta)

    def gradienDescent(self, iterations):
        theta = self.theta
        cost = []
        gradient = []
        print("****************************Computing gradient descent***************************")
        print(iterations)
        for i in range(iterations):
          costj, gradientj = self.getCostAndGradient(theta)
          cost.append(costj)
          gradient.append(gradientj)
          for j in range(len(theta)):
              theta[j] -= self.alpha * gradientj[j]
        self.theta = theta
        return self.theta, cost, gradient

    def normalEquation(self):
        X = self.model.getBiasedX()
        theta = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, self.model.y))
        return theta

    def optimizedGradientDescent(self):
        initial_theta = np.zeros(self.model.n + 1)
        result = op.minimize(fun = RegularizedLinearRegression.getCost, 
                             x0 = initial_theta, 
                             args = (self.model.getBiasedX(), self.model.y),
                             method = 'TNC',
                             jac = RegularizedLinearRegression.getGradient)
        return result

    def makePrediction(self, x):
        return self.evaluateHypothesis(x, self.theta)