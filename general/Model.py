# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as pyplot

class Model(object):

    def __init__(self, X, y, xlabel='x', ylabel="y", legend=["y=1", "y=0"], title="Graph"):
        self.initialize(X, y, xlabel, ylabel, legend, title)
    
    @classmethod
    def fromDataFile(cls, filename, delimiter):
        data = np.loadtxt(filename, delimiter=delimiter)
        X = data[:, 0:-1]
        y = data[:, -1]
        return cls(X, y)

    def initialize(self, X, y, xlabel='x', ylabel="y", legend=["y=1", "y=0"], title="Graph"):
        self.X = X
        self.y = y
        self.m, self.n = self.X.shape
        self.y.shape = (self.m, 1)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xsize = len(X[0])
        self.ysize = len(y)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.legend = legend

    def loadData(self, datafile, delimiter):
        data = np.loadtxt(datafile, delimiter)
        self.X = data[:, 0:-1]
        self.y = data[:, -1]
        self.initialize(self.X, self.y)


    def plotData(self, file=None):
        #load the dataset
        pos = np.where(self.y == 1)
        neg = np.where(self.y == 0)
        fig, ax = pyplot.subplots()
        ax.scatter(self.X[pos, 0], self.X[pos, 1], marker='o', c='b')
        ax.scatter(self.X[neg, 0], self.X[neg, 1], marker='x', c='r')
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        ax.legend(self.legend)
        
        if file:
            pyplot.savefig(file + '.png')
        #pyplot.scatter(self.X[pos, 0], self.X[pos, 1], marker='o', c='b')
        #pyplot.scatter(self.X[neg, 0], self.X[neg, 1], marker='x', c='r')
        #pyplot.xlabel(self.xlabel)
        #pyplot.ylabel(self.ylabel)
        #pyplot.legend(self.legend)
        #pyplot.title(self.title)
        pyplot.show()
        
        
        #it = self.mapFeature(self.X[:, 0], self.X[:, 1])
        #return it
    
    def plotDataRegression(self, file=None):
        #load the dataset
        fig, ax = self.getPlotRegression()
                
        if file:
            fig.savefig(file + ".png")

        pyplot.show()

        return fig, ax

    def getPlotRegression(self):
        fig, ax = pyplot.subplots()
        ax.scatter(self.X, self.y, marker='x', c='r')
        #ax.plot(self.X[:, 0], self.y[:])
        #ax.plot(self.X, self.y)
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        ax.grid()
        return fig, ax

    def mapFeature(self, x1, x2):
        #Maps the two input features to quadratic features.
        #Returns a new feature array with more features, comprising of
        #1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
        #nputs X1, X2 must be the same size
        x1.shape = (x1.size, 1)
        x2.shape = (x2.size, 1)
        degree = 6
        out = np.ones(shape=(x1[:, 0].size, 1))

        m, n = out.shape

        for i in range(1, degree + 1):
            for j in range(i + 1):
                r = (x1 ** (i - j)) * (x2 ** j)
                out = np.append(out, r, axis=1)
        return out

    def normalizeFeatures(self):
        X_norm = self.X;
        mu = np.zeros((self.n, 1))
        sigma = np.zeros((self.n, 1))
        for i in range(self.n):
            mu[i] = np.mean(self.X[:, i])
            sigma[i] = np.std(self.X[:, i])
            X_norm[:, i] = (self.X[:, i] - mu[i]) / sigma[i]
        self.X = X_norm
        self.mu = mu
        self.sigma = sigma
        return X_norm, mu, sigma

    
    def getBiasedX(self):
        return np.hstack((np.ones((len(self.X), 1)), self.X))
        
