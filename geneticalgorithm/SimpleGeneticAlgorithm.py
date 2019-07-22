import random
import datetime
from soldaimltk.geneticalgorithm.Individual import Individual
from soldaimltk.general.Coin import  Coin

class SimpleGeneticAlgorithm:

	def __init__(self, population_size, tournament_size, generations, chromosome_size, mutation_probability, cross_probability):
		self.population_size = population_size
		self.tournament_size = tournament_size
		self.generations = generations
		self.chromosome_size = chromosome_size
		self.mutation_probability = mutation_probability 
		self.cross_probability = cross_probability
		self.mse = 0.0
		self.population = []
		self.best = None
		self.initializePopulation()


	def initializePopulation(self):
		self.population = []
		for i in range(self.population_size):
			self.population.append(Individual(self.chromosome_size))
		self.best = self.population[0]
		self.best.evaluation = self.evaluateIndividual(self.population[0])

	def evaluateIndividual(self, individual):
		errors = 0
		for gen in individual.chromosome:
			if not gen:
				errors += 0.1
		return errors

	def evaluate(self):
		min_error = 100.0
		sum_error = 0.0

		for i in range(self.population_size):
			evaluation = self.evaluateIndividual(self.population[i])
			self.population[i].evaluation = evaluation

			if evaluation < min_error:
				min_error = evaluation
				print("Evaluating individual " + str(i) + " " + self.population[i].print() + " Score --> " + str(evaluation) )
				if self.best.evaluation >= evaluation:
					self.best = self.population[i].copy()

			sum_error += evaluation

		self.mse = sum_error/self.population_size
		return self.mse

	#Optimized tournament strategy
	def select(self):
		new_population = []
		for i in range(self.population_size):
			individuals = self.population[i:i + self.tournament_size]
			if i + self.tournament_size >= self.population_size:
				individuals += self.population[0:(i - self.population_size)]
			sorted_individuals = individuals.sort(key= lambda x: x.evaluation)
			new_population.append(individuals[0])
		self.population = new_population

	def cross(self):
		c = Coin(self.cross_probability)
		new_population = []

		for i in range(int(self.population_size)//2):
			individual_1 = self.population[i*2]
			individual_2 = self.population[i*2 + 1]

			if c.toss():
				cross_point = random.randint(0,self.chromosome_size)
				new_population += individual_1.cross(cross_point, individual_2)
			else:
				new_population += [individual_1, individual_2]
		self.population = new_population

	def mutate(self):
		c = Coin(self.mutation_probability)
		for i in range(self.population_size):
			for j in range(self.chromosome_size):
				if c.toss():
					self.population[i].toggleBit(j)

	def runGeneration(self):
		self.evaluate()
		self.select()
		self.cross()
		self.mutate()
		return self.mse

	def crossValidation(self):
		generation_time = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
		individuals = open("data/individuals_" + generation_time + ".log","w")
		error = open("data/errors_" + generation_time + ".log","w")
		errors = []
		for i in range(self.generations):
			self.runGeneration()
			print("Error " + str(self.mse))
			print("Best individual of generation " + str(i) + " " + self.best.print())
			error.write(str(self.mse) + ",")
			individuals.write(self.best.print() + "\n")
			errors.append(self.mse)
			if i%10 == 0:
				self.mutation_probability = 0.8*self.mutation_probability
		error.write(str(self.mse))
		individuals.write(self.best.print() + "\n")
		individuals.close()
		error.close()
		return errors


