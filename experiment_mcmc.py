import numpy as np
from skopt import gp_minimize
from tqdm import tqdm
from joblib import Parallel, delayed

from mcmc import *

def run_experiment(beta=[0.6], max_iter=500, metric=None, verbose=False):
	beta = beta[0]

	# metric = max_probability_metric
	# metric = successful_big_moves_metric
	metric = sharpe_metric

	correct_document = 'it was the best of times  it was the worst of times it was the age of wisdom it was the age of foolishness it was the epoch of belief  it was the epoch of incredulity it was the season of light it was the season of darkness it was the spring of hope it was the winter of despair we had everything before us we had nothing before us we were all going direct to heaven we were all going direct the other way in short the period was so far like the present period that some of its noisiest authorities insisted on its being received  for good or for evil in the superlative degree of comparison only '
	encrypted_document = encrypt_document(correct_document)   # randomly encrypts
	expected_letter_distribution = build_letter_transition_dist("war_and_peace.txt")

	res = test_metropolis_hastings(encrypted_document, expected_letter_distribution, correct_document, beta, max_iter, metric, verbose)
	print(res)
	return res[-1]

def max_probability_metric(history):
	# make negative bc our solver is a minimizer
	return -max(history, key=lambda x: x[1])[1]

def successful_big_moves_metric(history):
	def distance(a, b):
		c = 0
		for i in range(len(a)):
			if a[i] != b[i]:
				c+=1
		return c//2

	s = 0
	for i in range(1, len(history)):
		x = history[i-1]
		x_prime = history[i]
		d = distance(x[0], x_prime[0])
		s += d * (x_prime[1] - x[1]) # subtract because they're in log scale

	# make negative because our solver is a minimizer
	return -s

def sharpe_metric(history):
	vals = [x[1] for x in history]
	mean = -np.mean(vals) + 1     		# apply negative because our solver is a minimizer     
	var = np.var(vals) + 1				# add 1 to smooth (no divide by 0) i.e. laplace smoothing
	return (math.log(mean) - math.log(var)) 

def optimize():
	# gets best value of beta
	res = gp_minimize(run_experiment, [(0.01, 20)], n_calls=10)
	return res['x']

def run_trials(n, best_beta=None):
	if best_beta is None:
		print("Optimizing...")
		best_beta = optimize()[0]
		print()
		print("Best beta found: {}".format(best_beta))

	max_iter = 200000    # select a cutoff so at least we don't run forever
	num_proposed = []
	for i in range(n):
		# print("Trial: {}".format(i+1))
		correct_document = 'it was the best of times  it was the worst of times it was the age of wisdom it was the age of foolishness it was the epoch of belief  it was the epoch of incredulity it was the season of light it was the season of darkness it was the spring of hope it was the winter of despair we had everything before us we had nothing before us we were all going direct to heaven we were all going direct the other way in short the period was so far like the present period that some of its noisiest authorities insisted on its being received  for good or for evil in the superlative degree of comparison only '
		encrypted_document = encrypt_document(correct_document)  # randomly encrypts
		expected_letter_distribution = build_letter_transition_dist("war_and_peace.txt")

		res = test_metropolis_hastings(encrypted_document, expected_letter_distribution, correct_document, best_beta, max_iter, verbose=False)
		# print(res)
		num_proposed.append(res[-2])

	print("Tried: {}".format(best_beta))
	print(num_proposed)
	return np.mean(num_proposed)


if __name__ == "__main__":
	# r = run_experiment()
	# print(r)

	# # Estimate Beta
	# out = []
	# for i in tqdm(range(10)):
	# 	r = optimize()[0]
	# 	out.append(r)
	# print(out)

	# for beta in tqdm([0.6]):
	# 	print("Trying: {}".format(beta))
	# 	res = run_trials(30, best_beta=beta)
	# 	print(res)
	# 	print()
	# n_samples = 25
	# beta_list = [2.0, 2.2]

	# res = Parallel(n_jobs=2)(delayed(run_trials)(n_samples, beta) for beta in beta_list)
	# print()
	# print(beta_list)
	# print(res)

	# a = [('abcd', -1000), ('bacd',-1200), ('badc', -2050), ('acbd', -1102), ('abcd', -1102), ('badc', -3000)]

	# r = successful_big_moves_metric(a)
	# print(r)