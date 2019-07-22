import random

class Coin:

	def __init__(self, probability):
		self.probability = probability

	def toss(self):
		 return random.random() <= self.probability
