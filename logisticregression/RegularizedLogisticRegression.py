import numpy as np
import scipy.optimize as op
import scipy as scipy
import soldaimltk.general.Model as Model

#This class compuite the cost function for a regularized logistig regression
class RegularizedLogisticRegression(object):

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
        gradient = np.zeros(np.size(theta))
        h_theta = self.computePredictions(theta).reshape(self.model.m, 1)
        regularization = (self.l * np.sum(theta[1 :]**2))/(2*self.model.m)
        
        CK = 0.000009
        h_theta[h_theta == 0] = CK
        h_theta[h_theta == 1] = 1 - CK
        
        cost_m = self.model.y * np.log(h_theta) + (1 - self.model.y) * np.log(1 - h_theta)
        
        cost = -1 * np.sum(cost_m)/self.model.m + regularization

        #We calculate the gradient of the first regression parameter
        differences = h_theta - self.model.y
        X = self.model.getBiasedX()
        gradient[0] = np.sum(differences * X[:, 0].reshape(self.model.m, 1))/self.model.m
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * X[:, j].reshape(self.model.m, 1))/self.model.m - self.l/self.model.m * theta[j]
        return cost, gradient

    def getCostAndGradient2(self, theta):
        gradient = np.zeros(np.size(theta))
        h_theta = self.computePredictions(theta).reshape(self.model.m, 1)
        regularization = (self.l * np.sum(theta[1 :]**2))/(2*self.model.m)
        cost_m = self.model.y * np.log(h_theta) + (1 - self.model.y) * np.log(1 - h_theta)
        cost = -1 * np.sum(cost_m)/self.model.m + regularization

        #We calculate the gradient of the first regression parameter
        differences = h_theta - self.model.y
        X = self.model.getBiasedX()
        gradient[0] = np.sum(differences * X[:, 0].reshape(self.model.m, 1))/self.model.m
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * X[:, j].reshape(self.model.m, 1))/self.model.m - self.l/self.model.m * theta[j]
        return cost, gradient
    
    @staticmethod
    def getCost(theta, x, y, l=0):
        m = len(x)
        sigmoid = lambda z: 1/ (1 + np.exp(-1 * z))
        #input("Wait a second...")
        h_theta = sigmoid(np.matmul(x, theta)).reshape(m, 1)
        regularization = (l * np.sum(theta[1 :]**2))/(2*m)
        cost_m = y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta)
        cost = -1 * np.sum(cost_m)/m + regularization
        return cost
    
    @staticmethod
    def getGradient(theta, x, y, l=0):
        gradient = np.zeros(np.size(theta))
        m = len(x)
        sigmoid = lambda z: 1/ (1 + np.exp(-1 * z))
        #input("Wait a second...")
        h_theta = sigmoid(np.matmul(x, theta)).reshape(m, 1)
        differences = h_theta - y
        gradient[0] = np.sum(differences * x[:, 0].reshape(m, 1))/m
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * x[:, j].reshape(m, 1))/m - l/m * theta[j]
        return gradient

    def computePredictions(self, theta):
        return self.sigmoid(np.matmul(self.model.getBiasedX(), theta))

    def evaluateHypotesis(self, xi, theta):
        return self.sigmoid(np.matmul(xi, theta))
                   
    def sigmoid(self, z):
        return scipy.special.expit(z)
        #return 1/ (1 + np.exp(-1 * z))

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
          theta[0] -= self.alpha * gradientj[0]
          for j in range(1, len(theta)):
              theta[j] -= self.alpha * (gradientj[j] + (self.l/self.model.m) * theta[j])
        self.theta = theta
        return self.theta, cost, gradient

    def optimizedGradientDescent(self):
        initial_theta = np.zeros(self.model.n + 1)
        result = op.minimize(fun = RegularizedLogisticRegression.getCost, 
                             x0 = initial_theta, 
                             args = (self.model.getBiasedX(), self.model.y),
                             method = 'TNC',
                             jac = RegularizedLogisticRegression.getGradient)
        return result

    def makePrediction(self, x):
        evaluation = self.evaluateHypotesis(x, self.theta)
        if evaluation >= 0.5:
            return 1
        return 0