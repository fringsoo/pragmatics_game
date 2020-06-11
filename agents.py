import random
import numpy as np
import scipy
import scipy.stats as st
import time 
import json
import pdb
import copy
import os
import pickle
import copy
import shutil
from progressbar import *
from collections import Counter
import matplotlib.pyplot as plt
import pandas
from itertools import product
import pprint
import scipy.spatial.distance
import pandas as pd
import datetime

from wrapper_visa import VisaDatasetWrapper
from wrapper_pybullet import BulletWrapper
from evaluation import obtain_metrics
from agent_speaker import PaperSpeakerNetwork, PaperSpeakerNetwork_rnn, PaperSpeakerNetwork_rnn_conv
from agent_listener import PaperListenerNetwork, PaperListenerNetwork_rnn, PaperListenerNetwork_rnn_conv
from utils import pretrain_conv

def get_pbar(maxval):
	widgets = ['Progress: ', Percentage(), ' ', Bar('#'),' ', Timer(),  ' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=maxval).start()
	return pbar

def arraytostring(nparray):
	message = ''
	for _ in nparray:
		message += str(_)
	return message	

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()
	#return x / np.linalg.norm(x)

def message_similarity_hard(m1, m2):
	"""
	Inputs: One dimension various length numpy array.
	"""
	return int(np.all(m1==m2))

def weighted_random_by_dct(dct):
	assert abs(sum(dct.values()) - 1) < 0.01
	rand_val = random.random()
	total = 0
	for k, v in dct.items():
		total += v
		if rand_val <= total:
			return k
	assert False, 'unreachable'
	
#######################################################################################
def utils_imrmbg(img):
	newimg = []
	for layer in range(3):
		im = img[:,:,layer]
		pixels = Counter(im.reshape(-1).tolist())
		keypixels = [k for k in pixels.keys() if pixels[k]<500]
		
		ft = im>1
		for k in keypixels:
			ft = ft | (im == k)
		#newim = im[ft]
		newim = im * ft
		newimg.append(newim)
	return np.array(newimg).transpose([1,2,0])

def utils_immean(imgs):
	effective_pixel_num = np.sum([(np.sum(img, axis=2)!=0) * 1 for img in imgs], axis=0)
	effective_pixel_num += (effective_pixel_num==0)* 1
	effective_pixel_num = np.repeat(effective_pixel_num[:,:,np.newaxis], 3, axis=2)
	return np.sum(imgs, axis=0) / effective_pixel_num

def utils_digit2alphabet(speaker_message):
	alphabet = 'abcdefghijklmnopqrstuvwxyz'
	nm = ''
	for m in speaker_message:
		nm += alphabet[m]
	return nm

def utils_judge_sameset(test_example, same_set):
	for s in range(6):
		if test_example[4][0] in same_set[s]:
			sn0 = s
		if test_example[4][1] in same_set[s]:
			sn1 = s
	if sn0!=sn1:
		return False
	else:
		return True

#######################################################################################
def utils_calculate_reward(chosen_target_idx, target_candidate_idx):
	if target_candidate_idx[chosen_target_idx]==1.:
		return 1
	else:
		return 0

def utils_calculate_reward_forgame_s(chosen_target_idx, target_candidate_idx):
	if target_candidate_idx[chosen_target_idx]==1.:
		return 0
	else:
		return -1

def utils_calculate_reward_forgame_l(chosen_target_idx, target_candidate_idx):
	if target_candidate_idx[chosen_target_idx]==1.:
		return 1
	else:
		return -1

def utils_sample_from_networks_on_batch(speaker_model, listener_model, target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set):
	"""
	All inputs: Just one instance. No bs dimensize.
	"""
	speaker_message, speaker_probs = speaker_model.sample_from_speaker_policy(target_input)
	chosen_target_idx, listener_probs, us = listener_model.sample_from_listener_policy(speaker_message, candidates)
	reward = utils_calculate_reward(chosen_target_idx, target_candidate_idx)
	speaker_model.remember_speaker_training_details(target_input, speaker_message, speaker_probs, reward)
	listener_model.remember_listener_training_details(speaker_message, chosen_target_idx, listener_probs, target_input, candidates, reward)
	return reward

def utils_fit(speaker_model, listener_model, batch, trainspeaker=True, trainlistener=True):
	training_reward = 0
	for b in batch:
		target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = b
		reward = utils_sample_from_networks_on_batch(speaker_model, listener_model, target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set)		
		training_reward += reward
	if trainspeaker:
		speaker_model.train_speaker_policy_on_batch()
	if trainlistener:
		listener_model.train_listener_policy_on_batch()
	return training_reward

#######################################################################################
def utils_simulate(speaker_model, listener_model, s4l, l4s, batch):
	for b in batch:
		target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = b

		speaker_message, speaker_probs = speaker_model.sample_from_speaker_policy(target_input)
		s, sp = s4l.sample_from_speaker_policy(target_input)
		s4l.remember_speaker_training_details(target_input, speaker_message, sp, 1)
		
		chosen_target_idx, listener_probs, us = listener_model.sample_from_listener_policy(speaker_message, candidates)
		c, lp, usv = l4s.sample_from_listener_policy(speaker_message, candidates)
		l4s.remember_listener_training_details(speaker_message, chosen_target_idx, lp, target_input, candidates, 1)
	
	s4l.train_speaker_policy_on_batch()
	l4s.train_listener_policy_on_batch()

def utils_simulate_s_simu(speaker_model, newspeaker, batch):
	for b in batch:
		target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = b

		speaker_message, speaker_probs = speaker_model.sample_from_speaker_policy(target_input)
		s, sp = newspeaker.sample_from_speaker_policy(target_input)
		newspeaker.remember_speaker_training_details(target_input, speaker_message, sp, 1)
		
	newspeaker.train_speaker_policy_on_batch()

def utils_simulate_s_learn(listener_model, newspeaker, batch):
	for b in batch:
		target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = b

		s, sp = newspeaker.sample_from_speaker_policy(target_input)
		chosen_target_idx, listener_probs, us = listener_model.infer_from_listener_policy(s, candidates)
		reward = utils_calculate_reward(chosen_target_idx, target_candidate_idx)
		newspeaker.remember_speaker_training_details(target_input, s, sp, reward)
		
	newspeaker.train_speaker_policy_on_batch()

def utils_simulate_l_simu(speaker_model, listener_model, newlistener, batch):
	for b in batch:
		target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = b

		speaker_message, speaker_probs = speaker_model.infer_from_speaker_policy(target_input)
		
		chosen_target_idx, listener_probs, us = listener_model.sample_from_listener_policy(speaker_message, candidates)
		c, lp, usv = newlistener.sample_from_listener_policy(speaker_message, candidates)
		newlistener.remember_listener_training_details(speaker_message, chosen_target_idx, lp, target_input, candidates, 1)
	
	newlistener.train_listener_policy_on_batch()

def utils_simulate_l_learn(speaker_model, newlistener, batch):
	for b in batch:
		target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = b

		speaker_message, speaker_probs = speaker_model.infer_from_speaker_policy(target_input)

		c, lp, usv = newlistener.sample_from_listener_policy(speaker_message, candidates)
		reward = utils_calculate_reward(c, target_candidate_idx)

		newlistener.remember_listener_training_details(speaker_message, c, lp, target_input, candidates, reward)
	
	newlistener.train_listener_policy_on_batch()

def utils_virtual_check_speaker(speaker_model, newspeaker, example):
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example

	samples_real = utils_sample_message_for_a_candidate(speaker_model, target_input)
	samples_new = utils_sample_message_for_a_candidate(newspeaker, target_input)

	for m in samples_new.keys():
		if m not in samples_real:
			samples_real[m] = 0
	for m in samples_real.keys():
		if m not in samples_new:
			samples_new[m] = 0

	samples_real_pd = pd.Series(samples_real)
	samples_new_pd = pd.Series(samples_new)
	speaker_similarity = 1 - scipy.spatial.distance.cosine(samples_real_pd, samples_new_pd)
	
	return speaker_similarity
	
def utils_virtual_check_listener(speaker_model, listener_model, newlistener, example):
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	
	speaker_message, speaker_probs = speaker_model.infer_from_speaker_policy(target_input)
	
	cs_real, cp_real, _ = utils_get_choice_scores(listener_model, candidates, speaker_message, np.argmax(target_candidate_idx))
	cs_virtual, cp_virtual, _ = utils_get_choice_scores(newlistener, candidates, speaker_message, np.argmax(target_candidate_idx))
	#listener_similarity = 1 - abs(cs_real-cs_virtual) / abs(cs_real)
	listener_similarity = 1 - abs(cp_real-cp_virtual)
	
	return listener_similarity
	
#######################################################################################
def utils_get_message_prob(speaker_message, speaker_probs, maskdigit):
	message_prob = 1
	#for ss in range(3):
	for ss in range(len(speaker_message)):
		if ss not in maskdigit:
			message_prob *= speaker_probs[ss][speaker_message[ss]]
	return message_prob

def utils_get_choice_scores(listener_model, candidates, speaker_message, assign_choice=None):
	chosen_target_idx, listener_probs, us = listener_model.infer_from_listener_policy(speaker_message, candidates)
	if assign_choice:
		choice_similarity = us[assign_choice]
		#listener_model.get_internal_model_torm(speaker_message, candidates)[3][0][assign_choice]
		choice_prob = listener_probs[assign_choice]
		choice_similarity = choice_prob
		return choice_similarity, choice_prob, None
	else:
		choice_similarity = us[chosen_target_idx]
		#choice_similarity = listener_model.get_internal_model_torm(speaker_message, candidates)[3][0][chosen_target_idx]
		choice_prob = listener_probs[chosen_target_idx]
		choice_similarity = choice_prob
		return choice_similarity, choice_prob, chosen_target_idx

def utils_mask_message(message, maskdigit, mask):
	message[maskdigit] = mask
	return message

def utils_sample_message_for_a_candidate(speaker_model, candidate, num_samples=100, prob_threshold=0.00):
	'''
	Sample policies from here
	'''
	action_prob = speaker_model.network_output(candidate)
	samples = {}
	#use history samples!?
	for _r in range(num_samples):
		s, sp = speaker_model.sample_message_from_prob(action_prob)
		s = utils_mask_message(s, speaker_model.config['maskdigit'], speaker_model.config['mask'])
		if tuple(s) not in samples:
			prob = utils_get_message_prob(s, sp, speaker_model.config['maskdigit'])
			if prob >= prob_threshold:
				samples[tuple(s)] = prob
	#
	
	#samples_to_return = dict(sorted(samples.items(), key=lambda x:x[1], reverse=True)[:3])
	#return samples_to_return

	sum = 0
	samples_to_return = {}
	for it in sorted(samples.items(), key=lambda x:x[1], reverse=True):
		samples_to_return[it[0]] = it[1]
		sum += it[1]
		if sum > speaker_model.config['threshold']:
			break

	#import pdb; pdb.set_trace()
	return samples_to_return

#######################################################################################
def utils_test(speaker_model, listener_model, example):
	"""
	All inputs: Just one instance. No bs dimensize.
	"""
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	speaker_message, _speaker_probs = speaker_model.infer_from_speaker_policy(target_input)
	chosen_target_idx, _listener_probs, us = listener_model.infer_from_listener_policy(speaker_message, candidates)
	reward = utils_calculate_reward(chosen_target_idx,target_candidate_idx)
		
	message_prob = utils_get_message_prob(speaker_message, _speaker_probs, speaker_model.config['maskdigit'])
	cs, cp, _ = utils_get_choice_scores(listener_model, candidates, speaker_message, chosen_target_idx)
	
	return reward, target_input, speaker_message, chosen_target_idx, message_prob, cp, [], {}

def utils_prag_sample(speaker_model, listener_model, example, lambda_setting=0.5):
	"""
	All inputs: Just one instance. No bs dimensize.
	"""
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	samples = utils_sample_message_for_a_candidate(speaker_model, target_input)
	samples_score = {}
	for sk in samples.keys():
		cs, cp, _ = utils_get_choice_scores(listener_model, candidates, np.array(sk), np.argmax(target_candidate_idx))
		score = samples[sk]**lambda_setting * cp**(1-lambda_setting)
		samples_score[sk] = score
	speaker_message = max(samples_score, key=samples_score.get)
	message_prob = samples[speaker_message]
	chosen_target_idx, listener_probs, us = listener_model.sample_from_listener_policy(np.array(speaker_message), candidates)
	cs, cp, _ = utils_get_choice_scores(listener_model, candidates, np.array(speaker_message), chosen_target_idx)
	reward = utils_calculate_reward(chosen_target_idx, target_candidate_idx)
	return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cp, [], {}

def utils_prag_sample_virtual(speaker_model, listener_model, virtual_l4s, example, lambda_setting=0.5):
	"""
	All inputs: Just one instance. No bs dimensize.
	"""
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	
	result = utils_prag_sample(speaker_model, virtual_l4s, example, lambda_setting)
	speaker_message = result[2]
	message_prob = result[4]
	
	chosen_target_idx, listener_probs, us = listener_model.sample_from_listener_policy(np.array(speaker_message), candidates)
	cs, cp, _ = utils_get_choice_scores(listener_model, candidates, np.array(speaker_message), chosen_target_idx)
	
	reward = utils_calculate_reward(chosen_target_idx, target_candidate_idx)
	return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cp, None, None

def utils_prag_argmax(speaker_model, listener_model, example):
	"""
	All inputs: Just one instance. No bs dimensize.
	"""
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	samples = utils_sample_message_for_a_candidate(speaker_model, target_input)
	samples_score = {}
	for sk in samples.keys():
		chosen_target_idx, listener_probs, us = listener_model.infer_from_listener_policy(np.array(sk), candidates)
		if chosen_target_idx != np.argmax(target_candidate_idx):
			score = 0
		else:
			score = samples[sk]
		samples_score[sk] = score
	speaker_message = max(samples_score, key=samples_score.get)
	message_prob = samples[speaker_message]
	cs, cp, chosen_target_idx = utils_get_choice_scores(listener_model, candidates, np.array(speaker_message))
	reward = utils_calculate_reward(chosen_target_idx, target_candidate_idx)
	return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cp, [], {}

def utils_prag_argmax_virtual(speaker_model, listener_model, virtual_l4s, example):
	"""
	All inputs: Just one instance. No bs dimensize.
	"""
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	
	result = utils_prag_argmax(speaker_model, virtual_l4s, example)
	speaker_message = result[2]
	message_prob = result[4]
	
	cs, cp, chosen_target_idx = utils_get_choice_scores(listener_model, candidates, np.array(speaker_message))
	reward = utils_calculate_reward(chosen_target_idx, target_candidate_idx)
	return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cp, None, None

def utils_prag_iter(speaker_model, listener_model, example, stopcondition, onehotlistener_0, onehotspeaker, onehotlistener, lambda_setting=0.5):
	#import time; t0=time.time()
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	n_classes = len(candidates)
	
	m4cs_withprob = [] #[{m1:p, m2:p,...}, {m1:p, m3:p,...}]
	for candidate in candidates:
		samples = utils_sample_message_for_a_candidate(speaker_model, candidate)
		#assert len(samples) >=2 
		m4cs_withprob.append(samples)
	
	#t1=time.time()

	m4cs_onlymessage = [set(m4c.keys()) for m4c in m4cs_withprob] #[{m1, m2, ...}, {m1, m3, ...}]
	messages_all = list(set.union(*m4cs_onlymessage)) #[m1, m2, m3]

	for samples in m4cs_withprob:
		for message in messages_all:
			if message not in samples.keys():
				samples[message] = 0

	messages_index = {} #{m1:1, ...}
	for nm in range(len(messages_all)):
		messages_index[messages_all[nm]] = nm
	
	ps_m_t = [[m4cs_withprob[t][messages_all[m]] for m in range(len(messages_all))] for t in range(n_classes)]
	ps_m_t = [p4t / (sum(p4t) + 0.0) for p4t in ps_m_t]

	pl_t_m = [listener_model.infer_from_listener_policy(np.array(messages_all[m]), candidates)[1] for m in range(len(messages_all))]
	if onehotlistener_0:
		pl_t_m = np.eye(n_classes)[np.argmax(pl_t_m, 1)]

	''''
	ps_m_t = [[m4cs_withprob[t][messages_all[m]]**lambda_setting * pl_t_m[m][t]**(1-lambda_setting) for m in range(len(messages_all))] for t in range(n_classes)]
	ps_m_t = [p4t / (sum(p4t) + 0.0) for p4t in ps_m_t]
	ps_m_t = np.eye(len(messages_all))[np.argmax(ps_m_t, 1)]
	'''
	pl_t_m = np.array(pl_t_m)
	ps_m_t = np.array(ps_m_t)
	
	pl_t_m_bak = copy.deepcopy(pl_t_m)

	iterround = 0
	while True:
		iterround += 1
		ps_m_t_bak = copy.deepcopy(ps_m_t)
		for t in range(n_classes):
			tempsum = 0
			for m in range(len(messages_all)):
				tempsum += m4cs_withprob[t][messages_all[m]]**lambda_setting * pl_t_m[m][t]**(1-lambda_setting)
			if tempsum > 0:
				for m in range(len(messages_all)):
					ps_m_t[t][m] = m4cs_withprob[t][messages_all[m]]**lambda_setting * pl_t_m[m][t]**(1-lambda_setting)
				ps_m_t[t] = ps_m_t[t] / (sum(ps_m_t[t]) + 0.0)
			if onehotspeaker:
				ps_m_t[t] = np.eye(len(messages_all))[np.argmax(ps_m_t[t])]

		if stopcondition=='converge':
			if (abs(ps_m_t_bak - ps_m_t) < 1e-4).all() and (abs(pl_t_m_bak - pl_t_m) < 1e-4).all():
				break
		else:
			if iterround==stopcondition:
				break
		
		pl_t_m_bak = copy.deepcopy(pl_t_m)
		for m in range(len(messages_all)):
			tempsum = 0
			for t in range(n_classes):
				tempsum += ps_m_t[t][m]
			if tempsum > 0:
				for t in range(n_classes):
					pl_t_m[m][t] = ps_m_t[t][m]
				pl_t_m[m] = pl_t_m[m] / (sum(pl_t_m[m]) + 0.0)
			if onehotlistener:
				pl_t_m[m] = np.eye(n_classes)[np.argmax(pl_t_m[m])]
	
	
	message_no = np.argmax(ps_m_t[np.argmax(target_candidate_idx)])
	speaker_message = messages_all[message_no]
	message_prob = m4cs_withprob[np.argmax(target_candidate_idx)][speaker_message]
	chosen_target_idx = np.argmax(pl_t_m[message_no])
	cs, cp, _ = utils_get_choice_scores(listener_model, candidates, np.array(speaker_message), chosen_target_idx)
	reward = utils_calculate_reward(chosen_target_idx, target_candidate_idx)
	
	#if reward!=1 and onehotlistener:
	#	pdb.set_trace()
	
	s_m4t_info = [speaker_message, message_prob]
	l_ms4t_info = {}
	for m in range(len(messages_all)):
		if np.argmax(pl_t_m[m]) == np.argmax(target_candidate_idx):
			l_sm = messages_all[m]
			l_ms4t_info[l_sm] = utils_get_choice_scores(listener_model, candidates, np.array(l_sm),  np.argmax(target_candidate_idx))[1]
			#l_ms4t_info[l_sm] = listener_model.get_internal_model_torm(np.array(l_sm), candidates)[3][0][np.argmax(target_candidate_idx)]

	#t2 = time.time()
	#print('0-1',t1-t0)
	#print('1-2',t2-t1)
	return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cp, s_m4t_info, l_ms4t_info
	#return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cs, None, None

	'''
	candidatenum2messageseq = {}
	for t in range(n_classes):
		candidatenum2messageseq[t] = np.argmax(ps_m_t[t])
	'''

def utils_prag_iter_virtual(speaker_model, listener_model, virtual_s4l, virtual_l4s, example, stopcondition, onehotlistener_0, onehotspeaker, onehotlistener, lambda_setting=0.5):
	result4s = utils_prag_iter(speaker_model, virtual_l4s, example, stopcondition, onehotlistener_0, onehotspeaker, onehotlistener, lambda_setting)
	s_m4t_info = result4s[-2]
	
	
	result4l = utils_prag_iter(virtual_s4l, listener_model, example, stopcondition, onehotlistener_0, onehotspeaker, onehotlistener, lambda_setting)
	l_ms4t_info = result4l[-1]
	

	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	if s_m4t_info[0] in l_ms4t_info.keys():
		reward = 1
		speaker_message = s_m4t_info[0]
		chosen_target_idx = np.argmax(target_candidate_idx)
		message_prob = s_m4t_info[1]
		cp = l_ms4t_info[speaker_message]
	else:
		reward = 0
		speaker_message = s_m4t_info[0]
		chosen_target_idx = -1
		message_prob = s_m4t_info[1]
		cp = 0
	return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cp, None, None

def utils_table(speaker_model, listener_model, example, special):
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	n_classes = len(candidates)
	
	#import time; t0 = time.time()

	m4cs_withprob = [] #[{m1:p, m2:p,...}, {m1:p, m3:p,...}]
	for candidate in candidates:
		samples = utils_sample_message_for_a_candidate(speaker_model, candidate)
		#assert len(samples) >=2 
		m4cs_withprob.append(samples)
	
	#t1=time.time()
	
	m4cs_onlymessage = [set(m4c.keys()) for m4c in m4cs_withprob] #[{m1, m2, ...}, {m1, m3, ...}]
	messages_all = list(set.union(*m4cs_onlymessage)) #[m1, m2, m3]
	
	speaker_strategies = []
	for ss in product(*m4cs_onlymessage):
		#ss: (m2, m3)
		payoff = 1
		for nc in range(n_classes):
			payoff *= m4cs_withprob[nc][ss[nc]]
		speaker_strategies.append((ss, payoff))

	c4ms = [] #[[0,1], [0], [1], ...]

	messages_index = {} #{m1:1, ...}
	for nm in range(len(messages_all)):
		messages_index[messages_all[nm]] = nm

	for message in messages_all:
		cs = [] #[0, 1, ...]
		for c in range(n_classes):
			if message in m4cs_onlymessage[c]:
				cs.append(c)
		c4ms.append(cs)
	
	listener_strategies = []
	for ls in product(*c4ms):
		#ls: (0, 0, 1, ...)
		#payoff = 0
		payoff = 1
		assert len(ls) == len(messages_all)
		for nm in range(len(messages_all)):
			#payoff += listener_model.get_internal_model_torm(np.array(messages_all[nm]), candidates)[3][0][ls[nm]]
			payoff *= listener_model.infer_from_listener_policy(np.array(messages_all[nm]), candidates)[1][ls[nm]]
		listener_strategies.append((ls, payoff))

	table = [[[-1, -10000, True] for col in range(len(speaker_strategies))] for row in range(len(listener_strategies))]
	for row in range(len(listener_strategies)):
		for col in range(len(speaker_strategies)):
			match = True
			for c in range(n_classes):
				if not listener_strategies[row][0][messages_index[speaker_strategies[col][0][c]]] == c:
					match = False
					break
		
			#lp = 1
			#for m in listener_strategies[row].items():
			#	lp *= self.listener_model.sample_from_listener_policy(np.array(m[0]), candidates)[1][m[1]]

			if match:
				table[row][col] = [speaker_strategies[col][1], listener_strategies[row][1], True]
	
	for row in range(len(listener_strategies)):
		maxsp = 0
		maxcol = []
		for col in range(len(speaker_strategies)):
			if table[row][col][0] > maxsp:
				maxcol = [col]
				maxsp = table[row][col][0]
			elif table[row][col][0] == maxsp:
				maxcol.append(col)
		for col in range(len(speaker_strategies)):
			if col not in maxcol:
				table[row][col][2] = False
	
	for col in range(len(speaker_strategies)):
		#maxlp = 1e-10
		maxlp = -9999
		maxrow = []
		for row in range(len(listener_strategies)):
			if table[row][col][1] > maxlp:
				maxrow = [row]
				maxlp = table[row][col][1]
			elif table[row][col][1] == maxlp:
				maxrow.append(row)
		for row in range(len(listener_strategies)):
			if row not in maxrow:
				table[row][col][2] = False


	#data = pandas.DataFrame(table)
	#data.insert(0,'',listener_strategies)
	#data.to_csv('torm.csv', header=['']+speaker_strategies, index=False)
	
	equilibria = []
	for row in range(len(listener_strategies)):
		for col in range(len(speaker_strategies)):
			if table[row][col][2]:
				equilibria.append([speaker_strategies[col], listener_strategies[row], table[row][col][0], table[row][col][1]])
	
	if len(equilibria) == 1:
		equi = equilibria[0]
	else:
		maxs = -1
		maxse = -1
		for e in range(len(equilibria)):
			if equilibria[e][2] > maxs:
				maxs = equilibria[e][2]
				maxse = e

		maxl = -10000
		maxle = -1
		for e in range(len(equilibria)):
			if equilibria[e][3] > maxl:
				maxl = equilibria[e][3]
				maxle = e

		#pdb.set_trace()

		if maxse == maxle and len(equilibria)>0:
			if maxse <0 or maxse>=len(equilibria):
				import pdb; pdb.set_trace()
			equi = equilibria[maxse]
			assert equi[1][0][messages_index[equi[0][0][np.argmax(target_candidate_idx)]]] == np.argmax(target_candidate_idx)

		else:
			equi = None

	if equi:
		speaker_message = equi[0][0][np.argmax(target_candidate_idx)]
		chosen_target_idx= np.argmax(target_candidate_idx)
		message_prob = m4cs_withprob[np.argmax(target_candidate_idx)][tuple(speaker_message)]
	else:
		equi_messages4candidates = [set() for nc in range(n_classes)]
		for e in range(len(equilibria)):
			equii = equilibria[e]
			for nc in range(n_classes):
				message_no4_target = messages_index[equii[0][0][nc]]
				equi_messages4candidates[nc].add(message_no4_target)
		remain_set = equi_messages4candidates[np.argmax(target_candidate_idx)]
		for nc in range(n_classes):
			if nc != np.argmax(target_candidate_idx):
				remain_set = remain_set - equi_messages4candidates[nc]

		if remain_set and special:
			got_special = True
			message_prob = 0
			for message_no in remain_set:
				mp = m4cs_withprob[np.argmax(target_candidate_idx)][messages_all[message_no]]
				if mp > message_prob:
					message_prob = mp
					speaker_message = messages_all[message_no]
			speaker_message_special = speaker_message
			message_prob_special = message_prob
			chosen_target_idx= np.argmax(target_candidate_idx)
		else:
			got_special = False
			speaker_message, _speaker_probs = speaker_model.infer_from_speaker_policy(target_input)
			chosen_target_idx, _listener_probs, us = listener_model.infer_from_listener_policy(speaker_message, candidates)
			message_prob = utils_get_message_prob(speaker_message, _speaker_probs, speaker_model.config['maskdigit'])
	
	reward = utils_calculate_reward(chosen_target_idx,target_candidate_idx)
	cs, cp, _ = utils_get_choice_scores(listener_model, candidates, np.array(speaker_message), chosen_target_idx)

	if equi:
		'''
		candidatenum2messageseq = {}
		for t in range(n_classes):
			s_payoff = m4cs_withprob[t][equi[0][0][t]]
			candidatenum2messageseq[t] = [equi[0][0][t], s_payoff]
		messageseq2candidatenum = {}
		for m in range(len(messages_all)):
			lpayoff = listener_model.get_internal_model_torm(np.array(messages_all[m]), candidates)[3][0][equi[1][0][m]]
			messageseq2candidatenum[messages_all[m]] = [equi[1][0][m], lpayofff]
		'''
		
		s_m4t = equi[0][0][np.argmax(target_candidate_idx)]
		s_payoff = m4cs_withprob[np.argmax(target_candidate_idx)][s_m4t]
		s_m4t_info = [s_m4t, s_payoff]

		l_ms4t_info = {}
		for m in range(len(messages_all)):
			if equi[1][0][m] == np.argmax(target_candidate_idx):
				l_sm = messages_all[m]
				l_ms4t_info[l_sm] = utils_get_choice_scores(listener_model, candidates, np.array(l_sm),  np.argmax(target_candidate_idx))[1]
				#l_payoff = listener_model.get_internal_model_torm(np.array(l_sm), candidates)[3][0][equi[1][0][m]]
				#l_ms4t_info[l_sm] = l_payoff

	elif special and got_special:
		s_m4t_info = [speaker_message_special, message_prob_special]
		l_ms4t_info = {}
		for message_no in remain_set:
			l_sm = messages_all[message_no]
			l_ms4t_info[l_sm] = utils_get_choice_scores(listener_model, candidates, np.array(l_sm),  np.argmax(target_candidate_idx))[1]
			#l_payoff = listener_model.get_internal_model_torm(np.array(l_sm), candidates)[3][0][np.argmax(target_candidate_idx)]
			#l_ms4t_info[l_sm] = l_payoff
	else:
		s_m4t_info = []
		l_ms4t_info = {}

	#t2 = time.time()
	#print('0-1',t1-t0)
	#print('1-2',t2-t1)
	return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cp, s_m4t_info, l_ms4t_info
	#return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cs, None, None

def utils_table_virtual(speaker_model, listener_model, virtual_s4l, virtual_l4s, example, special):
	result4s = utils_table(speaker_model, virtual_l4s, example, special)
	s_m4t_info = result4s[-2]
	
	result4l = utils_table(virtual_s4l, listener_model, example, special)
	l_ms4t_info = result4l[-1]
	
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example

	if s_m4t_info!=[]:
		speaker_message = s_m4t_info[0]
		message_prob = s_m4t_info[1]
	else:
		speaker_message, _speaker_probs = speaker_model.infer_from_speaker_policy(target_input)
		message_prob = utils_get_message_prob(speaker_message, _speaker_probs, speaker_model.config['maskdigit'])
		speaker_message = tuple(speaker_message) 
		
	if l_ms4t_info!={}:
		if speaker_message in l_ms4t_info.keys():
			reward = 1
			chosen_target_idx = np.argmax(target_candidate_idx)
			cp = l_ms4t_info[speaker_message]
		else:
			reward = 0
			chosen_target_idx = -1
			cp = 0
	else:
		chosen_target_idx, _listener_probs, us = listener_model.infer_from_listener_policy(np.array(speaker_message), candidates)
		reward = utils_calculate_reward(chosen_target_idx,target_candidate_idx)
		cs, cp, _ = utils_get_choice_scores(listener_model, candidates, np.array(speaker_message))

	return reward, candidates[np.argmax(target_candidate_idx)], np.array(speaker_message), chosen_target_idx, message_prob, cp, None, None

#######################################################################################
def utils_short_term_train(speaker_model, listener_model, candidates, policy4shortgame, stop=0.1, maxrounds=1000, trainspeaker=True, trainlistener=True):
	"""
	All inputs: Just one instance. No bs dimensize.
	"""
	new_candidates = copy.deepcopy(candidates)
	rr = 0
	rewards = []
	while rr < maxrounds:
		np.random.shuffle(new_candidates)
		choice = np.random.randint(len(candidates))
		target_input = new_candidates[choice]
		target_candidate_idx = np.zeros(len(candidates))
		target_candidate_idx[choice] = 1

		if policy4shortgame=='sample':
			speaker_message, speaker_probs = speaker_model.sample_from_speaker_policy(target_input)
			chosen_target_idx, listener_probs, us = listener_model.sample_from_listener_policy(speaker_message, new_candidates)
		elif policy4shortgame=='infer':
			speaker_message, speaker_probs = speaker_model.infer_from_speaker_policy(target_input)
			chosen_target_idx, listener_probs, us = listener_model.infer_from_listener_policy(speaker_message, new_candidates)
		else:
			assert False
		
		
		reward_test = utils_calculate_reward(chosen_target_idx, target_candidate_idx)
		rewards.append(reward_test)

		#print(reward)

		if trainspeaker:
			reward = utils_calculate_reward_forgame_s(chosen_target_idx, target_candidate_idx)
			speaker_model.remember_speaker_training_details(target_input, speaker_message, speaker_probs, reward)
			speaker_model.train_speaker_policy_on_batch()

		if trainlistener:
			reward = utils_calculate_reward_forgame_l(chosen_target_idx, target_candidate_idx)
			listener_model.remember_listener_training_details(speaker_message, chosen_target_idx, listener_probs, target_input, new_candidates, reward)
			listener_model.train_listener_policy_on_batch()
		rr += 1
	print(sum(rewards))

def utils_game_full(speaker_model, listener_model, example, policy4shortgame, maxrounds):
	self.speaker_model.save_memory()
	self.listener_model.save_memory()

	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	utils_short_term_train(speaker_model, listener_model, candidates, policy4shortgame, maxrounds=maxrounds)
	speaker_message, _speaker_probs = speaker_model.infer_from_speaker_policy(target_input)
	speaker_model.load_memory()
	listener_model.load_memory()
	utils_short_term_train(speaker_model, listener_model, candidates, policy4shortgame, maxrounds=maxrounds)
	chosen_target_idx, _listener_probs, us = listener_model.infer_from_listener_policy(speaker_message, candidates)					
	speaker_model.load_memory()
	listener_model.load_memory()
	reward = utils_calculate_reward(chosen_target_idx,target_candidate_idx)
	return reward, target_input, speaker_message, chosen_target_idx

def utils_game_ibr(speaker_model, listener_model, example, policy4shortgame, maxrounds):
	self.speaker_model.save_memory()
	self.listener_model.save_memory()

	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	for r in range(1):
		utils_short_term_train(speaker_model, listener_model, candidates, policy4shortgame, maxrounds=maxrounds, trainlistener=False)
		utils_short_term_train(speaker_model, listener_model, candidates, policy4shortgame, maxrounds=maxrounds, trainspeaker=False)
	speaker_message, _speaker_probs = speaker_model.infer_from_speaker_policy(target_input)
	chosen_target_idx, _listener_probs, us = listener_model.infer_from_listener_policy(speaker_message, candidates)					
	speaker_model.load_memory()
	listener_model.load_memory()
	reward = utils_calculate_reward(chosen_target_idx,target_candidate_idx)
	return reward, target_input, speaker_message, chosen_target_idx

def utils_game_zihe(speaker_model, listener_model, example, sample_round_explicit):
	
	no_correct = 0
	multiple_correct = 0
	
	target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = example
	n_classes = len(candidates)

	def get_m():
		ms = []
		sm_all = Counter()
		local_all = []
		
		for c in candidates:
			local = Counter()
			for _rp in range(sample_round_explicit):
				sm, _sp = speaker_model.infer_from_speaker_policy(c)
				sm = tuple(sm.tolist())
				if sm not in local:
					sm_all[sm] += 1
				local[sm] += 1
			local_all.append(local)
		
		for c in range(n_classes):
			local = local_all[c]
			for sm in list(local.keys()):
				if sm_all[sm] > 1:
					del(local[sm])
			if local:
				localsum = sum(local.values()) + 0.0
				for sm in local:
					local[sm] /= localsum

		for c in range(n_classes):
			local = local_all[c]
			if local:
				m = weighted_random_by_dct(local)
			else:
				m, _sp = speaker_model.infer_from_speaker_policy(candidates[c])
			m = np.array(m)
			ms.append(m)
		return ms

	#real shit starts
	ms_s = get_m()
	speaker_message = ms_s[np.argmax(target_candidate_idx)]

	ms_l = get_m()
	
	ds = []
	for m in ms_l:
		ds.append(message_similarity_hard(speaker_message, m))
	ds = np.array(ds, dtype=np.float)
	
	if np.sum(ds) == 1:
		solution = 'correct'
		chosen_target_idx = np.argmax(ds)

	elif np.sum(ds) == 0:
		solution = 'no_correct'
		chosen_target_idx, _listener_probs, us = listener_model.infer_from_listener_policy(speaker_message, candidates)		
	
	elif np.sum(ds) > 1:
		solution = 'multiple_correct'
		chosen_target_idx, _listener_probs, us = listener_model.infer_from_listener_policy(speaker_message, candidates)	
	

	reward = self.calculate_reward(chosen_target_idx,target_candidate_idx)
	
	return reward, target_input, speaker_message, chosen_target_idx, solution

#######################################################################################
class BaseAgents(object):
	""" 
	"""
	def __init__(self, config):
		self.config = config
		self.init_dataset()
			
	def training_log(self, metrics):
		if self.config['n_batches'] > 0:
			with open(os.path.join(self.config['modelpath'], 'logs'), 'a+') as ff:
				ff.write("Training\n" + json.dumps(self.config) + "\n###############\n")
				print(" Accuracy: %s %%"%(metrics["accuracy"]))
				ff.write(" Accuracy: %s %%\n"%(metrics["accuracy"]))
				print("Message lexicon size: %s"%(len(metrics["speaker_action_dist"])))
				ff.write("Message lexicon size: %s \n"%(len(metrics["speaker_action_dist"])))
				#ff.write("Speaker action distribution: %s \n"%(metrics["speaker_action_dist"]))
				print("Listener action distribution: %s"%(metrics["listener_action_dist"]))
				ff.write("Listener action distribution: %s \n"%(metrics["listener_action_dist"]))
				print("reward distribution: %s"%(metrics["reward_dist"]))
				ff.write("reward distribution: %s \n"%(metrics["reward_dist"]))
				print("###############\n")
				ff.write("###############\n")
		
	def testing_log(self, path, datas):
		
		for key in datas.keys():
			metrics = obtain_metrics(datas[key]['testing_stats'], self.config)
			
			with open(os.path.join(path, 'logs'), 'a+') as ff:
				
				ff.write("\n###############\n"+key+"\n###############\n")

				ff.write(" Accuracy: %s %%\n"%(metrics["accuracy"]))
				ff.write("Message lexicon size: %s \n"%(len(metrics["speaker_action_dist"])))
				#ff.write("Speaker action distribution: %s \n"%(metrics["speaker_action_dist"]))
				ff.write("Listener action distribution: %s \n"%(metrics["listener_action_dist"]))
				ff.write("reward distribution: %s \n"%(metrics["reward_dist"]))
				#ff.write("###############\n")

				nepoch = len(datas[key]['rds'])
				ce = st.t.ppf((1 + 0.95) / 2., nepoch-1)
				ff.write('reward %.3f %.3f %.3f\n' %(np.mean(datas[key]['rds']), np.std(datas[key]['rds']), st.sem(datas[key]['rds']) * ce))
				ff.write('message_prob %.2f %.2f %.2f\n' %(np.mean(datas[key]['mps']), np.std(datas[key]['mps']), st.sem(datas[key]['mps']) * ce))
				ff.write('choice_prob %.2f %.2f %.2f\n' %(np.mean(datas[key]['css']), np.std(datas[key]['css']), st.sem(datas[key]['css']) * ce))

	def fit(self):
		pbar = ProgressBar()
		for _n in pbar(range(self.config['n_batches'])):
			batch = self.data_generator.training_batch_generator()
			utils_fit(self.speaker_model, self.listener_model, batch)
			if _n%5000==0 or _n==self.config['n_batches']-1:
				metrics = self.evaluate_on_training_set()
				self.training_log(metrics)
				self.speaker_model.save()
				self.speaker_model.save_opt()
				self.listener_model.save()
				self.listener_model.save_opt()

	def evaluate_on_training_set(self):
		"""
		All inputs: Just one instance. No bs dimensize.
		"""
		self.training_eval_stats = []
		#pbar = get_pbar(self.data_generator.n_training_instances)
		for i, train_example in enumerate(self.data_generator.training_set_evaluation_generator()):
			#pbar.update(i)
			reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_test(self.speaker_model, self.listener_model, train_example)
			self.training_eval_stats.append({"reward": reward, "input": target_input, "message": arraytostring(speaker_message), "chosen_target_idx": [chosen_target_idx]})
		#pbar.finish()
		metrics = obtain_metrics(self.training_eval_stats, self.config)
		return metrics

	def predict(self):
		"""
		All inputs: Just one instance. No bs dimensize.
		"""
		nepoch = self.config['predict_nepoch']
		is_rnnconv = type(self).__name__ == 'Agents_rnnconv'
		ce = st.t.ppf((1 + 0.95) / 2., nepoch-1)
		pbar = get_pbar(nepoch * self.data_generator.n_testing_instances)
		resultpath = os.path.join(self.config['modelpath'],'test_'+datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
		os.mkdir(resultpath)
		#
		
		datas = {}
		keys = ['baseline', 'prag_sample_0', 'prag_sample_0.5', 'prag_argmax', 'prag_argmax_virtual', 'rsa_2', 'ibr_2', 'rsa_equi', 'rsa_equi_virtual', 'ibr_equi', 'ibr_equi_virtual', 'table', 'table_virtual', 'table_special', 'table_special_virtual']		
		#keys = ['baseline', 'prag_sample_0', 'prag_sample_0.5', 'prag_argmax', 'rsa_2', 'ibr_2', 'rsa_equi', 'ibr_equi', 'table', 'table_special',]		
		#keys = ['baseline']
		
		for key in keys:
			datas[key] = {'testing_stats':[], 'stat':{}, 'message_im':{}, 'rd':0, 'rds':[], 'mp':0, 'mps':[], 'cs':0, 'css':[], 'pathname':os.path.join(resultpath, 'images_'+key)}	
			if is_rnnconv:
				os.mkdir(datas[key]['pathname'])
			
		def add_data(key, reward, target_input, speaker_message, chosen_target_idx, mp, cs, i):	
			datas[key]['testing_stats'].append({"reward": reward, "input": target_input, "message": arraytostring(speaker_message), "chosen_target_idx": [chosen_target_idx]})
			datas[key]['rd'] += reward
			if reward>0:
				datas[key]['mp'] += mp
				datas[key]['cs'] += cs
			if is_rnnconv:
				message_in_letters = utils_digit2alphabet(speaker_message)
				message_str = os.path.join(datas[key]['pathname'], message_in_letters)
				if not os.path.exists(message_str):
					os.mkdir(message_str)
					datas[key]['message_im'][message_in_letters] = []
					datas[key]['stat'][message_in_letters] = Counter()
				#plt.imsave(os.path.join(message_str, str(i)+'.png'), target_input)
				datas[key]['message_im'][message_in_letters].append(utils_imrmbg(target_input))
				datas[key]['stat'][message_in_letters][test_example[3]] += 1

		for epoch in range(nepoch):
			count_epoch = 0
			for i, test_example in enumerate(self.data_generator.testing_set_generator()):
				i_tot = epoch*self.data_generator.n_testing_instances+i
				pbar.update(i_tot)
				if 'challenge' in self.config.keys() and self.config['challenge']:
					if not utils_judge_sameset(test_example, self.config['challenge_same_set']):
						continue
				count_epoch += 1.0
				
				if 'baseline' in keys:					
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_test(self.speaker_model, self.listener_model, test_example)
					add_data('baseline', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)
				
				if 'prag_sample_0' in keys:
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_sample(self.speaker_model, self.listener_model, test_example, 0)
					add_data('prag_sample_0', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)
					
				if 'prag_sample_0.5' in keys:
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_sample(self.speaker_model, self.listener_model, test_example, 0.5)
					add_data('prag_sample_0.5', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)
				
				if 'prag_argmax' in keys:
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_argmax(self.speaker_model, self.listener_model, test_example)
					add_data('prag_argmax', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)

				if 'rsa_2' in keys:
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_iter(self.speaker_model, self.listener_model, test_example, 2, False, False, False)
					add_data('rsa_2', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)

				if 'ibr_2' in keys:
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_iter(self.speaker_model, self.listener_model, test_example, 2, False, True, True)
					add_data('ibr_2', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)

				if 'rsa_equi' in keys:
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_iter(self.speaker_model, self.listener_model, test_example, 'converge', False, False, False)
					add_data('rsa_equi', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)

				if 'ibr_equi' in keys:
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_iter(self.speaker_model, self.listener_model, test_example, 'converge', False, True, True)
					add_data('ibr_equi', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)
				
				if 'table' in keys:	
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_table(self.speaker_model, self.listener_model, test_example, False)
					add_data('table', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)
					
				if 'table_special' in keys:	
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_table(self.speaker_model, self.listener_model, test_example, True)
					add_data('table_special', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)
					
				if 'prag_argmax_virtual' in keys:
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_argmax_virtual(self.speaker_model, self.listener_model, self.newlistener_a, test_example)
					add_data('prag_argmax_virtual', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)
				
				if 'rsa_equi_virtual' in keys:	
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_iter_virtual(self.speaker_model, self.listener_model, self.newspeaker_a, self.newlistener_a, test_example, 'converge', False, False, False)
					add_data('rsa_equi_virtual', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)

				if 'ibr_equi_virtual' in keys:	
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_prag_iter_virtual(self.speaker_model, self.listener_model, self.newspeaker_a, self.newlistener_a, test_example, 'converge', False, True, True)
					add_data('ibr_equi_virtual', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)
					
				if 'table_virtual' in keys:	
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_table_virtual(self.speaker_model, self.listener_model, self.newspeaker_a, self.newlistener_a, test_example, False)
					add_data('table_virtual', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)

				if 'table_special_virtual' in keys:	
					reward, target_input, speaker_message, chosen_target_idx, mp, cs, _, _ = utils_table_virtual(self.speaker_model, self.listener_model, self.newspeaker_a, self.newlistener_a, test_example, True)
					add_data('table_special_virtual', reward, target_input, speaker_message, chosen_target_idx, mp, cs, i_tot)
					

			for key in keys:
				datas[key]['rds'].append(datas[key]['rd'] / count_epoch)
				datas[key]['mps'].append(datas[key]['mp'] / datas[key]['rd'] )
				datas[key]['css'].append(datas[key]['cs'] / datas[key]['rd'] )
				datas[key]['rd'] = 0
				datas[key]['mp'] = 0
				datas[key]['cs'] = 0
				
		pbar.finish()

		self.testing_log(resultpath, datas)
		for key in keys:
			print('#########################')
			print(key)
			#pprint.pprint(datas[key]['stat'])
			print('reward %.3f %.3f %.3f' %(np.mean(datas[key]['rds']), np.std(datas[key]['rds']), st.sem(datas[key]['rds']) * ce))
			print('message_prob %.2f %.2f %.2f' %(np.mean(datas[key]['mps']), np.std(datas[key]['mps']), st.sem(datas[key]['mps']) * ce))
			print('choice_prob %.2f %.2f %.2f' %(np.mean(datas[key]['css']), np.std(datas[key]['css']), st.sem(datas[key]['css']) * ce))
			
			for mim in datas[key]['message_im'].keys():
				message_str = os.path.join(datas[key]['pathname'], mim)
				plt.imsave(message_str+'/'+key+'_'+mim+'_average.png', utils_immean(datas[key]['message_im'][mim]))
		

class Agents_dense(BaseAgents):
	def __init__(self, config):
		super(Agents_dense, self).__init__(config)
		self.speaker_model = PaperSpeakerNetwork(os.path.join(self.config['modelpath'], 'speaker.h5'),
													os.path.join(self.config['modelpath'],'speaker_opt.pkl'),
													0.001,
													0.01,
													self.config)
		self.listener_model = PaperListenerNetwork(os.path.join(self.config['modelpath'], 'listener.h5'),
													os.path.join(self.config['modelpath'], 'listener_opt.pkl'),
													0.001,
													0.001,
													self.config)

	def init_dataset(self):
		if not os.path.exists(self.config['modelpath']):
			os.mkdir(self.config['modelpath'])
			
		datafile = os.path.join('datadir', 'visa_data_generator.pkl')
		if not os.path.exists(datafile):
			self.data_generator = VisaDatasetWrapper()
			self.data_generator.create_train_test_datasets(self.config)
			with open(datafile, 'wb') as f:
				pickle.dump(self.data_generator, f)
		else:
			with open(datafile, 'rb') as f:
				self.data_generator = pickle.load(f)

class Agents_rnnbasic(Agents_dense):
	def __init__(self, config):
		super(Agents_dense, self).__init__(config)
		self.speaker_model = PaperSpeakerNetwork_rnn(os.path.join(self.config['modelpath'], 'speaker_lstm.h5'),
														os.path.join(self.config['modelpath'], 'speaker_opt.pkl'),
														0.001,
														0.01,
														self.config)
		self.listener_model = PaperListenerNetwork_rnn(os.path.join(self.config['modelpath'], 'listener_lstm.h5'),
														os.path.join(self.config['modelpath'], 'listener_opt.pkl'),
														0.001,
														0.01,
														self.config)

class Agents_rnnconv(Agents_dense):
	def __init__(self, config):
		super(Agents_dense, self).__init__(config)
		self.speaker_model = PaperSpeakerNetwork_rnn_conv(os.path.join(self.config['modelpath'], 'speaker_lstm.h5'),
															os.path.join(self.config['modelpath'], 'speaker_opt.pkl'),
															0.001,#0.0001,
															0.01, #0.01,
															'speaker_conv.h5',
															False,
															self.config)
		self.listener_model = PaperListenerNetwork_rnn_conv(os.path.join(self.config['modelpath'], 'listener_lstm.h5'),
															os.path.join(self.config['modelpath'], 'listener_opt.pkl'),
															0.001,#0.0001,
															0.001, #0.001,
															'listener_conv.h5',
															False,
															self.config)
		
	def init_dataset(self):
		if not os.path.exists(self.config['modelpath']):
			os.mkdir(self.config['modelpath'])

		datafile = os.path.join('datadir', 'bullet_data_generator.pkl')
		if not os.path.exists(datafile):
			self.data_generator = BulletWrapper()
			self.data_generator.create_train_test_datasets(self.config)
			with open(datafile, 'wb') as f:
				pickle.dump(self.data_generator, f)
		else:
			with open(datafile, 'rb') as f:
				self.data_generator = pickle.load(f)

	def pretrain_fit_conv(self):
		print('pretrain_speaker_color_unfixconv')
		pretrain_conv(self.speaker_model.conv_model, self.data_generator, 'color', False, 50000, self.speaker_model.pretrain_convmodel_file)
		print('pretrain_listener_color_unfixconv')
		pretrain_conv(self.listener_model.conv_model, self.data_generator, 'color', False, 50000, self.listener_model.pretrain_convmodel_file)
		
		print('pretrain_speaker_color_fixconv')
		pretrain_conv(self.speaker_model.conv_model, self.data_generator, 'color', True, 50000, None)
		print('pretrain_listener_color_fixconv')
		pretrain_conv(self.listener_model.conv_model, self.data_generator, 'color', True, 50000, None)

		print('pretrain_speaker_shape_fixconv')
		pretrain_conv(self.speaker_model.conv_model, self.data_generator, 'shape', True, 50000, None)
		print('pretrain_listener_shape_fixconv')
		pretrain_conv(self.listener_model.conv_model, self.data_generator, 'shape', True, 50000, None)

	def set_virtual_origin(self):
		self.newspeaker_a = self.speaker_model
		self.newlistener_a = self.listener_model

	def set_virtual_real(self):
		self.newspeaker_a = PaperSpeakerNetwork_rnn_conv(os.path.join(self.config['modelpath'], 'newspeaker_lstm.h5'),
														os.path.join(self.config['modelpath'], 'newspeaker_opt.pkl'),
														0.0001,
														0.01,
														'listener_conv.h5',
														False,
														self.config)

		self.newlistener_a = PaperListenerNetwork_rnn_conv(os.path.join(self.config['modelpath'], 'newlistener_lstm.h5'),
														os.path.join(self.config['modelpath'], 'newlistener_opt.pkl'),
														0.0001,
														0.001,
														'speaker_conv.h5',
														False,
				 										self.config)
		
	def train_virtual_speaker(self):
		pbar = ProgressBar()
		for _n in pbar(range(200000)):
			batch = self.data_generator.training_batch_generator()
			utils_simulate_s_simu(self.speaker_model, self.newspeaker_a, batch)
			
			batch = self.data_generator.training_batch_generator()
			utils_simulate_s_learn(self.listener_model, self.newspeaker_a, batch)
			
			if _n%10000==0 or _n==self.config['n_batches']-1:
				self.check_virtual_speaker()
				self.newspeaker_a.save()
				self.newspeaker_a.save_opt()
				self.newspeaker_a.save()
				self.newspeaker_a.save_opt()
				print(_n)
				#pdb.set_trace()
	
	def train_virtual_listener(self):
		pbar = ProgressBar()
		for _n in pbar(range(100000)):
			batch = self.data_generator.training_batch_generator()
			utils_simulate_l_simu(self.speaker_model, self.listener_model, self.newlistener_a, batch)
			
			batch = self.data_generator.training_batch_generator()
			utils_simulate_l_learn(self.speaker_model, self.newlistener_a, batch)

			if _n%10000==0 or _n==self.config['n_batches']-1:
				self.check_virtual_listener()
				self.newlistener_a.save()
				self.newlistener_a.save_opt()
				self.newlistener_a.save()
				self.newlistener_a.save_opt()
				print(_n)
				#pdb.set_trace()

	def check_virtual_speaker(self):
		ssimtotal = 0
		pbar = get_pbar(self.data_generator.n_testing_instances)

		for i, test_example in enumerate(self.data_generator.testing_set_generator()):
			pbar.update(i)
		#batchnum = 1000
		#for _n in pbar(range(batchnum)):
		#	batch = self.data_generator.training_batch_generator()
			ssim = utils_virtual_check_speaker(self.speaker_model, self.newspeaker_a, test_example)
			ssimtotal += ssim
		print("speaker fidelity:", ssimtotal/i)

	def check_virtual_listener(self):
		lsimtotal = 0
		pbar = get_pbar(self.data_generator.n_testing_instances)

		for i, test_example in enumerate(self.data_generator.testing_set_generator()):
			pbar.update(i)
		#batchnum = 1000
		#for _n in pbar(range(batchnum)):
		#	batch = self.data_generator.training_batch_generator()
			lsim = utils_virtual_check_listener(self.speaker_model, self.listener_model, self.newlistener_a, test_example)
			lsimtotal += lsim
		print("listener fidelity:", lsimtotal/i)

'''
class PaperSymbolicAgents_rnn_conv_crowd(PaperSymbolicAgents_rnn_conv):
	def __init__(self, config):
		super(PaperSymbolicAgents, self).__init__(config)
		self.crowd = []
		for _ in range(self.config['agent_amount']):
			speaker_model = PaperSpeakerNetwork_rnn_conv(os.path.join(self.config['modelpath'], 'speaker_lstm.' + str(_) + '.h5'),
													os.path.join(self.config['modelpath'], 'speaker_opt.' + str(_) + '.pkl'),
													0.001,#0.0001,
													0.01, #0.01,
													#str(_)+'_conv.h5',
													'speaker_conv.h5',
													False,
													self.config)
			listener_model = PaperListenerNetwork_rnn_conv(os.path.join(self.config['modelpath'], 'listener_lstm.' + str(_) + '.h5'),
																os.path.join(self.config['modelpath'], 'listener_opt.' + str(_) + '.pkl'),
																0.001,#0.0001,
																0.001, #0.001,
																#str(_)+'_conv.h5',
																'speaker_conv.h5',
																False,
																self.config)
			self.crowd.append({'speaker':speaker_model, 'listener':listener_model})

	def random_select_agents(self):
		speaker_no = np.random.randint(self.config['agent_amount'])
		listener_no = speaker_no
		while listener_no == speaker_no:
			listener_no = np.random.randint(self.config['agent_amount'])
		ss_model = self.crowd[speaker_no]['speaker']
		sl_model = self.crowd[speaker_no]['listener']
		ls_model = self.crowd[listener_no]['speaker']
		ll_model = self.crowd[listener_no]['listener']
		return ss_model, sl_model, ls_model, ll_model

	def pretrain_fit_conv(self):
		pass
		#pretrain_conv(self.crowd[1]['speaker'].conv_model, self.data_generator, 'color', True, 5000, None)
		#pretrain_conv(self.crowd[1]['listener'].conv_model, self.data_generator, 'color', True, 5000, None)

	def fit(self):
		self.total_training_reward = 0
		pbar = ProgressBar()
		for _n in pbar(range(self.config['n_batches'])):
			batch = self.data_generator.training_batch_generator()
			speaker_model, _sl, _ls, listener_model = self.random_select_agents()
			self.total_training_reward += utils_fit(speaker_model, listener_model, batch)
			if _n%5000==0 or _n==self.config['n_batches']-1:
				self.evaluate_on_training_set()
				for aa in range(self.config['agent_amount']):	
					self.crowd[aa]['speaker'].save()
					self.crowd[aa]['speaker'].save_opt()
					self.crowd[aa]['listener'].save()
					self.crowd[aa]['listener'].save_opt()
				self.training_log()

	def evaluate_on_training_set(self):
		"""
		All inputs: Just one instance. No bs dimensize.
		"""
		self.training_eval_stats = []
		pbar = get_pbar(self.data_generator.n_training_instances)
		for i, train_example in enumerate(self.data_generator.training_set_evaluation_generator()):
			pbar.update(i)
			speaker_model, _sl, _ls, listener_model = self.random_select_agents()
			reward, target_input, speaker_message, chosen_target_idx, sp, lp, _, _ = utils_test(speaker_model, listener_model, train_example)
			self.training_eval_stats.append({"reward": reward, "input": target_input, "message": arraytostring(speaker_message), "chosen_target_idx": [chosen_target_idx]})
		pbar.finish()
		metrics = obtain_metrics(self.training_eval_stats, self.config)
		self.testing_log("Evaluation", metrics)

	def check_virtual(self):
		pbar = ProgressBar()
		ssimtotal = 0
		lsimtotal = 0
		batchnum = 1000
		for _n in pbar(range(batchnum)):
			batch = self.data_generator.training_batch_generator()
			pdb.set_trace()
			ssim, lsim = utils_virtual_check(self.crowd[0]['speaker'], self.crowd[0]['listener'], self.crowd[1]['speaker'], self.crowd[1]['listener'], batch)
			ssimtotal += ssim
			lsimtotal += lsim
		print(ssimtotal/batchnum, lsimtotal/batchnum)

class PaperSymbolicAgents_rnn_conv_color(PaperSymbolicAgents):
	def __init__(self, config):
		super(PaperSymbolicAgents, self).__init__(config)
		self.speaker_model = PaperSpeakerNetwork_rnn_conv_color(os.path.join(self.config['modelpath'], 'speaker_lstm.h5'),
																	os.path.join(self.config['modelpath'], 'speaker_opt.pkl'),
																	0.001,
																	0.01,
																	self.config)
		self.listener_model = PaperListenerNetwork_rnn_conv_color(os.path.join(self.config['modelpath'], 'listener_lstm.h5'),
																	os.path.join(self.config['modelpath'], 'listener_opt.pkl'),
																	0.001,
																	0.001,
																	self.config)

	def init_dataset(self):
		if not os.path.exists(self.config['modelpath']):
			os.mkdir(self.config['modelpath'])
			self.data_generator = BulletWrapper_yieldcolor()
			self.data_generator.create_train_test_datasets(self.config)
			with open(os.path.join(self.config['modelpath'], 'data_generator.pkl'), 'wb') as f:
				pickle.dump(self.data_generator, f)
		else:
			with open(os.path.join(self.config['modelpath'], 'data_generator.pkl'), 'rb') as f:
				self.data_generator = pickle.load(f)

class PaperSymbolicAgents_crowd(BaseAgents):
	def __init__(self, config):
		super(PaperSymbolicAgents_crowd, self).__init__(config)
		self.crowd = []
		for _ in range(self.config['agent_amount']):
			speaker_model = PaperSpeakerNetwork(os.path.join(self.config['modelpath'], 'speaker.' + str(_) + '.h5'),
												os.path.join(self.config['modelpath'], 'speaker_opt.' + str(_) + '.pkl'),
												0.001,
												0.01,
												self.config)
			listener_model = PaperListenerNetwork(os.path.join(self.config['modelpath'], 'listener.' + str(_) + '.h5'),
												os.path.join(self.config['modelpath'], 'listener_opt.' + str(_) + '.pkl'),
												0.001,
												0.001,
												self.config)

			self.crowd.append({'speaker':speaker_model, 'listener':listener_model})

	def random_select_agents(self):
		speaker_no = np.random.randint(self.config['agent_amount'])
		listener_no = speaker_no
		while listener_no == speaker_no:
			listener_no = np.random.randint(self.config['agent_amount'])
		ss_model = self.crowd[speaker_no]['speaker']
		sl_model = self.crowd[speaker_no]['listener']
		ls_model = self.crowd[listener_no]['speaker']
		ll_model = self.crowd[listener_no]['listener']
		return ss_model, sl_model, ls_model, ll_model

	def fit(self):
		self.total_training_reward = 0
		pbar = ProgressBar()
		for _n in pbar(range(self.config['n_batches'])):
			#print("Batch: %s of %s"%(_n, self.config['n_batches']))
			speaker_model, _sl, _ls, listener_model = self.random_select_agents()
			batch = self.data_generator.training_batch_generator()
			self.total_training_reward += utils_fit(speaker_model, listener_model, batch)
		for aa in range(self.config['agent_amount']):	
			self.crowd[aa]['speaker'].save()
			self.crowd[aa]['listener'].save()
		self.training_log()

	def evaluate_on_training_set(self):
		"""
		All inputs: Just one instance. No bs dimensize.
		"""
		self.training_eval_stats = []
		pbar = get_pbar(self.data_generator.n_training_instances)
		for i, train_example in enumerate(self.data_generator.training_set_evaluation_generator()):
			pbar.update(i)
			speaker_model, _sl, _ls, listener_model = self.random_select_agents()
			reward, target_input, speaker_message, chosen_target_idx = utils_test(speaker_model, listener_model, train_example)
			self.training_eval_stats.append({"reward": reward, "input": target_input, "message": arraytostring(speaker_message), "chosen_target_idx": [chosen_target_idx]})
		pbar.finish()
		metrics = obtain_metrics(self.training_eval_stats, self.config)
		self.testing_log("Evaluation", metrics)
	
	def predict(self):
		"""
		All inputs: Just one instance. No bs dimensize.
		"""
		self.testing_stats = []
		pbar = get_pbar(self.data_generator.n_testing_instances)
		for i, test_example in enumerate(self.data_generator.testing_set_generator()):
			pbar.update(i)
			speaker_model, _sl, _ls, listener_model = self.random_select_agents()
			reward, target_input, speaker_message, chosen_target_idx = utils_test(speaker_model, listener_model, test_example)
			self.testing_stats.append({"reward": reward, "input": target_input, "message": arraytostring(speaker_message), "chosen_target_idx": [chosen_target_idx]})
		pbar.finish()
		metrics = obtain_metrics(self.testing_stats, self.config)
		self.testing_log("Test", metrics)

	def predict_game(self):
		self.testing_game_stats = []
		pbar = get_pbar(self.data_generator.n_testing_instances)
		for i, test_example in enumerate(self.data_generator.testing_set_generator()):
			pbar.update(i)
			target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = test_example
			
			ss_model, sl_model, ls_model, ll_model = self.random_select_agents()
			ss_model.save_memory()
			sl_model.save_memory()
			ls_model.save_memory()
			ll_model.save_memory()

			utils_short_term_train(ss_model, sl_model, candidates, 'infer', maxrounds=self.config["max_short_game_round"])
			speaker_message, _speaker_probs = ss_model.infer_from_speaker_policy(target_input)
			utils_short_term_train(ls_model, ll_model, candidates, 'infer', maxrounds=self.config["max_short_game_round"])
			chosen_target_idx, _listener_probs = ll_model.infer_from_listener_policy(speaker_message, candidates)	
			
			ss_model.load_memory()
			sl_model.load_memory()
			ls_model.load_memory()
			ll_model.load_memory()

			reward = utils_calculate_reward(chosen_target_idx,target_candidate_idx)
			self.testing_game_stats.append({"reward": reward, "input": target_input, "message": arraytostring(speaker_message), "chosen_target_idx": [chosen_target_idx]})
		pbar.finish()
		metrics = obtain_metrics(self.testing_game_stats, self.config)
		self.testing_log("Shortgame Selves Infer with " + str(self.config['max_short_game_round']) + " Rounds", metrics)

class PaperSymbolicAgents_direct(PaperSymbolicAgents):
	def __init__(self, config):
		super(PaperSymbolicAgents, self).__init__(config)
		assert self.config['direct']
		self.speaker_model = PaperSpeakerNetwork_direct(os.path.join(self.config['modelpath'], 'speaker_direct.h5'), self.config)
		self.listener_model = PaperListenerNetwork_direct(os.path.join(self.config['modelpath'], 'listener_direct.h5'), self.config)
'''