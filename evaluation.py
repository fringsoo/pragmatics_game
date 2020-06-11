'''
MIT License

Copyright (c) 2018 Nicholas Leo Martin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
############################
Modified From: https://github.com/NickLeoMartin/emergent_comm_rl
'''

import numpy as np 
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import Levenshtein as levenshtein
import collections

def task_accuracy_metrics(reward_list):
	""" Accuracy as percentage of examples that received rewards """
	accuracy = sum(reward_list)*100/float(len(reward_list))
	#print("Total Reward: %s, Accuracy: %s %%"%(sum(reward_list),accuracy))
	return accuracy

def action_distribution(action_list):
	#print(action_list)
	counter = collections.Counter(action_list)
	return counter

def levenshtein_message_distance(m1, m2):
	""" Use python-levenshtein package to calculate edit distance """
	return levenshtein.distance(m1,m2) 

def message_sequence_to_alphabet(message, alphabet):
	if type(message[0]) is np.int64:
		return alphabet[int(message[0])]		
	elif type(message[0]) is list:
		return "".join(alphabet[int(idx)] for idx in message[0])
	else:
		return "".join(alphabet[int(idx)] for idx in message)

def topographic_similarity(input_vectors, messages):
	""" 
	Calculate negative spearman correlation between message levenshtein distances
	and cosine similarities of input vectors 
	"""
	## Calculate levenshtein distance between all message pairs
	message_similarities = []
	for idx, message in enumerate(messages):
		other_messages = messages
		other_messages.pop(idx)
		for other_message in messages:
			lev_dist = levenshtein_message_distance(message, other_message)
			message_similarities.append(lev_dist)	

	## Calculate cosine similarity of target and chosen vectors
	input_vect_similarities = []
	for idx, input_vect in enumerate(input_vectors):
		other_input_vectors = input_vectors
		other_input_vectors.pop(idx)
		for other_input in other_input_vectors:
			cos_dist = cosine(input_vect,other_input)
			input_vect_similarities.append(cos_dist)

	## Calculate negative Spearman correlation between message distances and vector similarities
	rho = spearmanr(message_similarities,input_vect_similarities) 
	return - rho.correlation

def obtain_metrics(training_stats, config_dict):
	""" Compute metrics given trianing stats list of dicts"""
	metrics = {}

	## Accuracy
	reward_list = [e['reward'] for e in training_stats]
	metrics['accuracy'] = task_accuracy_metrics(reward_list)

	## Speaker action distribution 
	message_list = []
	for e in training_stats:
		message_list.append(e["message"])
		#for m in e["message"]:
			## Variable message lengths
			#if type(m) is np.int64:
			#	message_list.append(m)
			#else:
			#	for m_ in m:
			#		message_list.append(m_)
			
	metrics["speaker_action_dist"] = action_distribution(message_list)
	#print("Speaker action distribution: %s"%(metrics["speaker_action_dist"]))
	#print("Message lexicon size: %s"%(len(metrics["speaker_action_dist"])))

	## Listener action distribution
	action_list = []
	for e in training_stats:
		#print(e["chosen_target_idx"])
		for t in e["chosen_target_idx"]:
			action_list.append(t)

	metrics["listener_action_dist"] = action_distribution(action_list)
	#print("Listener action distribution: %s"%(metrics["listener_action_dist"]))


	metrics["reward_dist"] = action_distribution(reward_list)
	#print("reward distribution: %s"%(metrics["reward_dist"]))


	## Topographic similarity
	#input_vectors = [e['input'] for e in training_stats]
	#messages = [message_sequence_to_alphabet(e['message'], config_dict['alphabet']) for e in training_stats]
	#metrics['topographical_sim'] = topographic_similarity(input_vectors, messages)
	#print("Topographical Similarity: %s"%(metrics['topographical_sim']))

	return metrics