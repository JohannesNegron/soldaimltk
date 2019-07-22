import numpy as np
import rand as rand
import soldaimltk.general.Model as Model
from sklearn.neural_network import MLPClassifier

class SimpleNeuralNetwork(object):

    def __init__(self, model, theta, alpha, l, layers):
        self.model = model
        self.theta = theta
        self.alpha = alpha
        self.l = l
        self.layers_description = layers
        self.num_labels = layers[-1]
        self.clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=tuple(layers), random_state=1)
        self.initLayers(layers)
    
    @classmethod
    def fromModel(cls, model, alpha = 0.1, l=0.1):
        theta = np.rand(model.y.shape)
        return cls(model, theta, alpha, l)
    
    @classmethod
    def fromDataFile(cls, datafile, delimeter, alpha=0.1, l=0.1):
        model = Model.fromDataFile(datafile, delimeter)
        theta = np.rand(model.y.shape)
        return cls(model, theta, alpha, l)
    
    def initLayers(self, layers):
        self.layers = []
        for i in range(len(layers) -1):
            layer = np.narray(layers[i], layers[i+1])
            #TODO randomly initialize the layer
            for j in range(layers[j]):
                for k in range(layers[j + 1]):
                    layer[j][k] = self.randInit(0.12)
            self.layers.append(layer)
            
    def randInit(self, epsilon_init = 0.12):
        return rand.rand() * (2*epsilon_init) - epsilon_init
    
    def getLayerSize(self, i):
        if i<= len(self.layers):
            return len(self.layers[i])
        return 0
    

    #m denotes the number of examples
    #gradient indicates the gradient matrix
    #regularization is the regularization parameter in order to prevent the over fitting
    #cost is the cost of the neural network cost function
    def getCostAndGradient(self, theta):
        #Feed formard propagation
        p = zeros(size(self.model.Y));
        #Initialize the thetas of the net
        theta_net = []
        pointer = 0
        for i in range(len(self.layers) -1):
            layer_size = (self.getLayerSize(i)  + 1) * self.getLayerSize(i + 1)
            theta_net[i] = theta[pointer:layer_size].reshape(((self.getLayerSize(i - 1)  + 1), self.getLayerSize(i)))
            pointer = pointer + layer_size
        
        output = self.feedForward(self.model.getBiasedX(), theta_net)
        cost = 0
        J_tmp = []
        for k in range(num_labels):
            y_k = self.model.Y[self.model.Y == k]
            cost += -y_k * np.log(output) - (1 -y_k) * np.log(1 - output)
            J_tmp.append(np.sum(cost[:, k])/self.mdoel.m)
        
        #Regularization term
        reg = 0
        for t in tehta_net:
            reg += np.sum(np.sum(t ** 2))
        
        reg = self.l  * (reg)/(len(theta_net)*self.model.m)
        J = np.sum(J_tmp)[0] + reg;
        
        #Back propagation
        delta = []
        for i in range(self.model.m):
            a_k = []
            z_k = []
            prev_output = self.model.getBiasedX()
            for k in len(self.layers):
                a_t = np.matmul(theta_net[k], prev_output)
                a_k.append(a_t)
                z_k.append(sigmoid(a_t))
        
        X = self.model.getBiasedX()
        gradient = np.zeros(np.size(theta))
        #h_theta = self.computePredictions(X, theta).reshape(len(self.model.X), 1)
        h_theta = self.evaluateHypothesis(X, theta).reshape(self.model.m, 1)
        differences = h_theta - self.model.y
        #print(differences)
        cost_m = differences **2
       # print(cost_m)
        #hey = input("Waiting...")
        regularization = (self.l * np.sum(theta[1 :]**2))/(2*self.model.m)
        cost = np.sum(cost_m)/(2*self.model.m) + regularization        
        #gradient[0] = np.sum(np.matmul(differences, self.X[:, 0]))/m
        print(differences.shape)
        gradient[0] = np.sum(differences * X[:, 0].reshape(len(X), 1))/self.model.m
        #We calculate the gradient for the rest of the parameters
        for j in range(1, len(theta)):
            gradient[j] = np.sum(differences * X[:, j].reshape(len(X), 1))/self.model.m + self.l/self.model.m * theta[j]
                
                
#    %Back propagation
#    delta_layer_2 = 0;
#    delta_layer_3 = 0;
#    delta_layer_1 = 0;
#    for i=1:m
#      a_1 = X(i, :);
#      z_2 = Theta1*a_1';
#      a_2 = sigmoid(z_2);
#      a_2 = [1; a_2];
#      z_3 = Theta2*a_2;
#      a_3 = sigmoid(z_3);
#      %For each output unit k in layer 3 set deltak_3 = ak_3 - yk
#      delta_3 = a_3 .- y_index(i, :)';
#      delta_2 = (Theta2' * delta_3) .* [0; sigmoidGradient(z_2)];
#      delta_layer_2 = delta_layer_2 + delta_3 * a_2';
#      delta_layer_1 = delta_layer_1 + delta_2(2:end) * a_1;
#    end
#    
#    Theta1(:, 1) = 0;
#    Theta2(:, 1) = 0;
#    %Hack to check without regularization
#    %lambda = 0;
#    reg_01 = Theta1 .* (lambda/m);
#    reg_02 = Theta2 .* (lambda/m);
#    %reg_02 = sum(sum(Theta2 .* (lambda/m)));
#    Theta1_grad = (1/m) * (delta_layer_1) + reg_01;
#    Theta2_grad = (1/m) * (delta_layer_2) + reg_02;
        return cost, gradient
    
    def feedForward(self, X, theta_net):
        #Make forward propagation and make the output
        prev_input = self.model.getBiasedX()
        for i in range(1, len(self.layers)):
            prev_input = self.sigmoid(np.matmul(prev_input, tehta_net[i].T))
        
        return prev_input


    def evaluateHypothesis(self, xi, theta):
        return np.matmul(xi, theta)
    
    def sigmoid(self, z):
        return scipy.special.expit(z)


    def gradienDescent(self, iterations):
        theta = self.theta
        #m = len(self.y)
        cost = []
        gradient = []
        print(iterations)
        for i in range(iterations):
          #  differences = np.matmul(self.X, theta) - self.y
          costj, gradientj = self.getCostAndGradient(theta)
          cost.append(costj)
          gradient.append(gradientj)
          for j in range(len(theta)):
              theta[j] -= self.alpha * gradientj[j]
              #theta[j] -= self.alpha * (np.sum(np.matmul(differences, self.X[:, j])) / m)
        self.theta = theta
        return self.theta, cost, gradient

    def executeOptimizedGradientDescent(self, iterations):
        return self.theta, 0, 0

    def makePrediction(self, x):
        return self.evaluateHypothesis(x, self.theta)