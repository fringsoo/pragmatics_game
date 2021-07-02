import random
import numpy as np
import scipy
import time
import json
import os
import pdb
import copy
import matplotlib.pyplot as plt
import pickle

from keras.layers import Input, Dense, LSTM, Lambda
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras import regularizers
from keras.utils.np_utils import to_categorical
from utils import convnet_vgg, convnet_mod, convnet_ori, convnet_com

class BaseSpeakerNetwork(object):
	def __init__(self, modelname, optfilename, lr, entropy_coefficient, config_dict):
		assert config_dict, "config_dict does not exist"
		self.modelname = modelname
		self.optfilename = optfilename
		self.lr = lr
		self.entropy_coefficient = entropy_coefficient
		self.config = config_dict
		self.initialize_model()
		self.build_train_fn()

		self.batch_target_input = []
		self.batch_action = []
		self.batch_reward = []

	def rebuild_train_fn(self, entropy_coefficient=None, lr=None):
		if entropy_coefficient:
			self.entropy_coefficient = entropy_coefficient
		if lr:
			self.lr = lr
		self.build_train_fn()

	def load(self):
		self.speaker_model = load_model(self.modelname)

	def load_opt(self):	
		with open(self.optfilename, 'rb') as f:
			weight_values = pickle.load(f)
		self.opt.set_weights(weight_values)

	def save(self):
		self.speaker_model.save(self.modelname)

	def save_opt(self):
		symbolic_weights = self.opt.weights
		weight_values = K.batch_get_value(symbolic_weights)
		with open(self.optfilename, 'wb') as f:
			pickle.dump(weight_values, f)

	def save_memory(self):
		self.memory_model_weights = self.speaker_model.get_weights()

	def load_memory(self):
		self.speaker_model.set_weights(self.memory_model_weights)

class PaperSpeakerNetwork(BaseSpeakerNetwork):
	"""

	"""
	def initialize_model(self):
		"""
		Batch input and output.
		"""
		if not os.path.exists(self.modelname):
			inputs = Input(shape=(self.config['speaker_input_dim'],)) #speaker_model.input[0], shape(bs, speaker_input_dim)
			hidden = Dense(self.config['speaker_dim'], activation='sigmoid')(inputs) #shape(bs, speaker_dim)
			outputs = Dense(self.config['max_message_length'], activation='sigmoid')(hidden)
			outputs = Lambda(lambda x: K.expand_dims(x))(outputs)
			outputs = Lambda(lambda x: K.concatenate([x, 1-x], axis=2))(outputs) #shape(bs, max_message_length, alphabet_size)
			self.speaker_model = Model(inputs=inputs, outputs=outputs)
		else:
			self.load()
			#check!!!
			#assert self.speaker_model.layers[0].output_shape == (None, self.config['speaker_input_dim'])
			#assert self.speaker_model.layers[1].output_shape == (None, self.config['speaker_dim'])
			#assert self.speaker_model.layers[2].output_shape == (None, self.config['max_message_length'])
			#assert self.speaker_model.layers[3].output_shape == (None, self.config['max_message_length'], 1)
			#assert self.speaker_model.layers[4].output_shape == (None, self.config['max_message_length'], 2)

	def build_train_fn(self):
		"""
		Batch input and output.
		"""
		action_prob_placeholder = self.speaker_model.output
		action_onehot_placeholder = K.placeholder(shape=(None, self.config['max_message_length'], self.config['alphabet_size']), name="action_onehot") #shape(bs, None, alphabet_size)
		reward_placeholder = K.placeholder(shape=(None,), name="reward") #shape(bs)

		action_prob_raw = action_prob_placeholder * action_onehot_placeholder #shape(bs, max_message_length, alphabet_size)
		action_prob = K.sum(action_prob_raw, axis=2)
		log_action_prob = K.sum(K.log(action_prob), axis=1)
		
		loss = - log_action_prob * reward_placeholder

		## Add entropy to the loss
		entropy = K.sum(action_prob_placeholder * K.log(action_prob_placeholder + 1e-10), axis=1)
		entropy = K.sum(entropy)

		## TODO: add entropy regularization parameter ...
		loss = loss + self.entropy_coefficient * entropy
		loss = K.mean(loss)
		self.opt = Adam(self.lr)
		self.updates = self.opt.get_updates(params=self.speaker_model.trainable_weights,loss=loss)
		
		if os.path.exists(self.optfilename):
			self.load_opt()
		
		self.train_fn = K.function(
				inputs=[
				self.speaker_model.input, #shape(bs, speaker_input_dim)
				action_onehot_placeholder, #shape(bs, max_message_length, alphabet_size)
				reward_placeholder], #shape(bs)
				outputs=[loss, entropy], updates=self.updates)

	def reshape_target(self, target_input):
		""" Reshape target_input to (1, input_dim) """
		assert len(target_input.shape)==1 and target_input.shape[0]==self.config['speaker_input_dim']
		return np.expand_dims(target_input, axis=0)

	def network_output(self, target_input):
		target_input = self.reshape_target(target_input) #shape(1, speaker_input_dim)
		outputs = self.speaker_model.predict(target_input) #shape(1, max_message_length, 2)
		action_prob = np.squeeze(outputs) #shape(max_message_length, 2)
		return action_prob
	
	def sample_message_from_prob(self, action_prob):
		action = []
		for _ in action_prob:
			action.append(np.random.choice(np.arange(self.config['alphabet_size']), p=_))
		action = np.array(action) #shape(max_message_length)
		return action, action_prob

	def infer_message_from_prob(self, action_prob):
		action = np.argmax(action_prob, axis=1) #shape(max_message_length)
		return action, action_prob

	def sample_from_speaker_policy(self, target_input):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		action_prob = self.network_output(target_input)
		return self.sample_message_from_prob(action_prob)

	def infer_from_speaker_policy(self, target_input):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		action_prob = self.network_output(target_input)
		return self.infer_message_from_prob(action_prob)	

	def train_speaker_policy_on_batch(self):
		"""
		Train as a batch. Loss is an float for a batch
		"""
		action_onehot = to_categorical(self.batch_action, num_classes=self.config['alphabet_size'])
		_loss, _entropy = self.train_fn([
			self.batch_target_input,
			action_onehot,
			self.batch_reward])

		#print("Speaker loss: ", _loss)
		#print("Speaker entropy: ", _entropy)
		
		self.batch_target_input = [] #shape(bs, speaker_input_dim)
		self.batch_action = [] #shape(bs, max_message_length, 2)
		self.batch_reward = [] #shape(bs)

	def remember_speaker_training_details(self, target_input, action, speaker_probs, reward):
		"""
		Inputs are just one instance. No bs dimensize.
		"""
		self.batch_target_input.append(target_input)
		self.batch_action.append(action)
		self.batch_reward.append(reward)

class PaperSpeakerNetwork_rnn(PaperSpeakerNetwork):
	def initialize_model(self):
		"""
		Batch input and output.
		"""
		if not os.path.exists(self.modelname):
			encoder_inputs = Input(shape=(self.config['speaker_input_dim'],)) #speaker_model.input[0], shape(bs, speaker_input_dim)
			e_state_h = Dense(self.config['speaker_dim'], activation=None)(encoder_inputs) #shape(bs, speaker_dim)
			e_state_c = Dense(self.config['speaker_dim'], activation=None)(encoder_inputs) #shape(bs, speaker_dim)
			states = [e_state_h, e_state_c]
			
			decoder_inputs = Input(shape=(1, self.config['alphabet_size']))
			decoder_lstm = LSTM(units=self.config['speaker_dim'], activation='tanh', return_sequences=True, return_state=True)
			decoder_dense = Dense(self.config['alphabet_size'], activation='softmax')
			
			all_outputs = []
			r_inputs = decoder_inputs
			for _ in range(self.config['max_message_length']):
				r_outputs, d_state_h, d_state_c = decoder_lstm(r_inputs, initial_state=states)
				r_outputs = decoder_dense(r_outputs)
				all_outputs.append(r_outputs)
				r_inputs = r_outputs
				states = [d_state_h, d_state_c]
			
			decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
			self.speaker_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

		else:
			self.load()
			#check!!!
			
	def set_updates(self):
		self.opt = Adam(self.lr)
		#adam = RMSprop(self.lr)
		self.updates = self.opt.get_updates(params=self.speaker_model.trainable_weights, loss=self.loss)
		if os.path.exists(self.optfilename):
			self.load_opt()

	def build_train_fn(self):
		"""
		Batch input and output.
		"""
		action_prob_placeholder = self.speaker_model.output
		action_onehot_placeholder = K.placeholder(shape=(None, self.config['max_message_length'], self.config['alphabet_size']), name="action_onehot") #shape(bs, None, alphabet_size)
		reward_placeholder = K.placeholder(shape=(None,), name="reward") #shape(bs)

		action_prob_raw = action_prob_placeholder * action_onehot_placeholder #shape(bs, max_message_length, alphabet_size)
		action_prob = K.sum(action_prob_raw, axis=2)
		log_action_prob = K.sum(K.log(action_prob), axis=1)
		loss = - log_action_prob * reward_placeholder
		## Add entropy to the loss
		entropy = K.sum(action_prob_placeholder * K.log(action_prob_placeholder + 1e-10), axis=1)
		entropy = K.sum(entropy)
		## TODO: add entropy regularization parameter ...
		loss = loss + self.entropy_coefficient * entropy
		self.loss = K.mean(loss)

		self.set_updates()
		self.train_fn = K.function(
				inputs=self.speaker_model.input 
					+[action_onehot_placeholder, #shape(bs, max_message_length, alphabet_size)
					reward_placeholder], #shape(bs)
				outputs=[loss, entropy], updates=self.updates)

	def prepare_decoder_input_data(self, bs):
		""" <START> to lstm to (bs, 1, alphabet_size) """
		decoder_input_data = np.zeros((bs, 1, self.config['alphabet_size'])) #shape(1,1,alphabet_size)
		#decoder_input_data[:, 0, target_token_index['\t']] = 1. 

		return decoder_input_data

	def network_output(self, target_input):
		decoder_input_data = self.prepare_decoder_input_data(1) #shape(1, 1, alphabet_size)
		target_input = self.reshape_target(target_input) #shape(1, speaker_input_dim)
		outputs = self.speaker_model.predict([target_input, decoder_input_data]) #shape(1, max_message_length, 2)

		action_prob = np.squeeze(outputs) #shape(max_message_length, 2)
		return action_prob

	def sample_message_from_prob(self, action_prob):
		action = []
		d = 0
		for _ in action_prob:
			if d == 0 or d == 1:
				md = self.config['alphabet_size'] - 1
				while md == self.config['alphabet_size'] - 1:
					md = np.random.choice(np.arange(self.config['alphabet_size']), p=_)
			else:
				md = np.random.choice(np.arange(self.config['alphabet_size']), p=_)
			
			if md != self.config['alphabet_size'] - 1:
				action.append(md)
				d += 1
			else:
				break 
		action = np.array(action) #shape(max_message_length)
		action_prob = action_prob[:d]
		return action, action_prob

	def sample_from_speaker_policy(self, target_input):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		action_prob = self.network_output(target_input)
		return self.sample_message_from_prob(action_prob)

	def infer_message_from_prob(self, action_prob):
		action = []
		d = 0
		for _ in action_prob:
			md = np.argmax(_)
			if md != self.config['alphabet_size'] - 1:
				action.append(md)
				d += 1
			else:
				if d == 0 or d == 1:
					ap = copy.deepcopy(_)
					ap[md] = 0
					md = np.argmax(ap)
					action.append(md)
					d += 1
				else:
					break 
		action = np.array(action) #shape(max_message_length)
		action_prob = action_prob[:d]
		return action, action_prob

	def infer_from_speaker_policy(self, target_input):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		action_prob = self.network_output(target_input)
		return self.infer_message_from_prob(action_prob)

	def train_speaker_policy_on_batch(self):
		"""
		Train as a batch. Loss is an float for a batch.
		"""
		message_lengths = [len(self.batch_action[bb]) for bb in range(len(self.batch_action))]
		for bb in range(len(self.batch_action)):
			toadd = self.config['max_message_length'] - len(self.batch_action[bb])
			for _ in range(toadd):
				self.batch_action[bb] = np.append(self.batch_action[bb],-1)
		action_onehot = to_categorical(self.batch_action, num_classes=self.config['alphabet_size'])
		action_onehot[:,:,-1] = 0

		for ao in range(action_onehot.shape[0]):
			toadd = self.config['max_message_length'] - message_lengths[ao]
			for d in range(toadd):
				action_onehot[ao][-d-1] += 1
		
		batch_decoder_input_data = self.prepare_decoder_input_data(self.config['batch_size'])

		_loss, _entropy = self.train_fn([
			self.batch_target_input,
			batch_decoder_input_data,
			action_onehot,
			self.batch_reward])

		self.batch_target_input = [] #shape(bs, speaker_input_dim)
		self.batch_action = [] #shape(bs, max_message_length, 2)
		self.batch_reward = [] #shape(bs)

	def sample_from_speaker_policy_fixed_length(self, target_input):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		decoder_input_data = self.prepare_decoder_input_data(1) #shape(1, 1, alphabet_size)
		target_input = self.reshape_target(target_input) #shape(1, speaker_input_dim)
		outputs = self.speaker_model.predict([target_input, decoder_input_data]) #shape(1, max_message_length, 2)

		action_prob = np.squeeze(outputs) #shape(max_message_length, 2)

		action = []
		for _ in action_prob:
			action.append(np.random.choice(np.arange(self.config['alphabet_size']), p=_))
		action = np.array(action) #shape(max_message_length)
		return action, action_prob

	def infer_from_speaker_policy_fixed_length(self, target_input):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		decoder_input_data = self.prepare_decoder_input_data(1) #shape(1, 1, alphabet_size)
		target_input = self.reshape_target(target_input) #shape(1, speaker_input_dim)
		outputs = self.speaker_model.predict([target_input, decoder_input_data]) #shape(1, max_message_length, 2)
		
		action_prob = np.squeeze(outputs) #shape(max_message_length, 2)
		action = np.argmax(action_prob, axis=1) #shape(max_message_length)

		return action, action_prob
	
	def train_speaker_policy_on_batch_fixed_length(self):
		"""
		Train as a batch. Loss is an float for a batch.
		"""
		action_onehot = to_categorical(self.batch_action, num_classes=self.config['alphabet_size'])
		batch_decoder_input_data = self.prepare_decoder_input_data(self.config['batch_size'])

		_loss, _entropy = self.train_fn([
			self.batch_target_input,
			batch_decoder_input_data,
			action_onehot,
			self.batch_reward])

		self.batch_target_input = [] #shape(bs, speaker_input_dim)
		self.batch_action = [] #shape(bs, max_message_length, 2)
		self.batch_reward = [] #shape(bs)

class PaperSpeakerNetwork_rnn_conv(PaperSpeakerNetwork_rnn):
	def __init__(self, modelname, optfilename, lr, entropy_coefficient, pretrain_convmodel_file, traincnn, config):
		self.pretrain_convmodel_file = pretrain_convmodel_file
		self.traincnn = traincnn
		super(PaperSpeakerNetwork_rnn_conv, self).__init__(modelname, optfilename, lr, entropy_coefficient, config)

	def initialize_model(self):
		"""
		Batch input and output.
		"""
		if not os.path.exists(self.modelname):
			self.conv_model = convnet_com(self.config['speaker_input_w'], self.config['speaker_input_h'], 3, preloadfile=self.pretrain_convmodel_file, name='conv_model_s')
			
			encoder_inputs = Input(shape=(self.config['speaker_input_w'], self.config['speaker_input_h'], 3), name='image_s') #speaker_model.input[0], shape(bs, speaker_input_w, speaker_input_h, 3)
			conv_outputs = self.conv_model(encoder_inputs)
		
			e_state_h = Dense(self.config['speaker_dim'], activation=None)(conv_outputs) #shape(bs, speaker_dim)
			e_state_c = Dense(self.config['speaker_dim'], activation=None)(conv_outputs) #shape(bs, speaker_dim)
			states = [e_state_h, e_state_c]
			
			decoder_inputs = Input(shape=(1, self.config['alphabet_size']))
			decoder_lstm = LSTM(units=self.config['speaker_dim'], activation='tanh', return_sequences=True, return_state=True)
			#decoder_dense = Dense(self.config['alphabet_size'], activation='softmax', kernel_regularizer=regularizers.l2(0.01))
			decoder_dense = Dense(self.config['alphabet_size'], activation='softmax')
			
			all_outputs = []
			r_inputs = decoder_inputs
			for _ in range(self.config['max_message_length']):
				r_outputs, d_state_h, d_state_c = decoder_lstm(r_inputs, initial_state=states)
				r_outputs = decoder_dense(r_outputs)
				all_outputs.append(r_outputs)
				r_inputs = r_outputs
				states = [d_state_h, d_state_c]
			
			decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
			self.speaker_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
		else:
			self.load()
			#check!!!
			self.conv_model = [l for l in self.speaker_model.layers if l.name=='conv_model_s'][0]
			#self.speaker_model.layers[-2].kernel_regularizer=None

		self.trainable_weights_others = []
		self.trainable_weights_conv = []
		for layer in self.speaker_model.layers:
			if layer.name!='conv_model_s':
				self.trainable_weights_others.extend(layer.trainable_weights)
			else:
				self.trainable_weights_conv.extend(layer.trainable_weights)
		
	def set_updates(self):
		self.opt = Adam(self.lr)
		#self.opt = RMSprop(self.lr)
		#opt = SGD(self.lr, momentum=0.9, decay=1e-6, nesterov=True)

		if not self.traincnn:
			self.updates = self.opt.get_updates(params=self.trainable_weights_others, loss=self.loss)
		else:
			self.updates = self.opt.get_updates(params=self.speaker_model.trainable_weights, loss=self.loss)
		
		if os.path.exists(self.optfilename):
			self.load_opt()

	def reshape_target(self, target_input):
		""" Reshape target_input to (1, w, h ,c) """
		assert len(target_input.shape)==3
		return np.expand_dims(target_input, axis=0)

'''
class PaperSpeakerNetwork_rnn_conv_color(PaperSpeakerNetwork_rnn):
	def initialize_model(self):
		"""
		Batch input and output.
		"""
		if not os.path.exists(self.modelname):
			encoder_inputs = Input(shape=[8])
			conv_outputs = encoder_inputs
			
			e_state_h = Dense(self.config['speaker_dim'], activation=None)(conv_outputs) #shape(bs, speaker_dim)
			e_state_c = Dense(self.config['speaker_dim'], activation=None)(conv_outputs) #shape(bs, speaker_dim)
			states = [e_state_h, e_state_c]
			
			decoder_inputs = Input(shape=(1, self.config['alphabet_size']))
			decoder_lstm = LSTM(units=self.config['speaker_dim'], activation='tanh', return_sequences=True, return_state=True)
			decoder_dense = Dense(self.config['alphabet_size'], activation='softmax')
			
			all_outputs = []
			r_inputs = decoder_inputs
			for _ in range(self.config['max_message_length']):
				r_outputs, d_state_h, d_state_c = decoder_lstm(r_inputs, initial_state=states)
				r_outputs = decoder_dense(r_outputs)
				all_outputs.append(r_outputs)
				r_inputs = r_outputs
				states = [d_state_h, d_state_c]
			
			decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
			self.speaker_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
		else:
			self.load()
			#check!!!
		self.trainable_weights_rnn = self.speaker_model.trainable_weights[4:7]
		self.trainable_weights_others = self.speaker_model.trainable_weights[:4] + self.speaker_model.trainable_weights[7:]

	def set_updates(self):
		self.opt = Adam(self.lr)
		#adam = RMSprop(self.lr)
		self.updates = self.opt.get_updates(params=self.speaker_model.trainable_weights, loss=self.loss)
		if os.path.exists(self.optfilename):
			self.load_opt()

	def reshape_target(self, target_input):
		assert len(target_input)==8
		return np.expand_dims(target_input, axis=0)

class PaperSpeakerNetwork_direct(BaseSpeakerNetwork):
	"""

	"""
	def initialize_model(self):
		"""
		Batch input and output.
		"""
		if not os.path.exists(self.modelname):
			inputs = Input(shape=(self.config['speaker_input_dim'],)) #speaker_model.input[0], shape(bs, speaker_input_dim)
			#hidden = Dense(self.config['speaker_dim'], activation='sigmoid')(inputs) #shape(bs, speaker_dim)
			hidden = inputs
			outputs = Dense(self.config['max_message_length'], activation='sigmoid')(hidden)
			outputs = Lambda(lambda x: K.expand_dims(x))(outputs)
			outputs = Lambda(lambda x: K.concatenate([x, 1-x], axis=2))(outputs) #shape(bs, max_message_length, alphabet_size)
			self.speaker_model = Model(inputs=inputs, outputs=outputs)
		else:
			self.load()
			#check!!!

	def build_train_fn_bak(self):
		"""
		Batch input and output.
		"""
		action_prob_placeholder = self.speaker_model.output
		action_onehot_placeholder = K.placeholder(shape=(None, self.config['max_message_length'], self.config['alphabet_size']), name="action_onehot") #shape(bs, None, alphabet_size)
		reward_placeholder = K.placeholder(shape=(None,), name="reward") #shape(bs)

		action_prob_raw = action_prob_placeholder * action_onehot_placeholder #shape(bs, max_message_length, alphabet_size)
		action_prob = K.sum(action_prob_raw, axis=2)
		
		#action_prob = K.sum(action_prob_raw)
		#log_action_prob = K.log(action_prob)
		log_action_prob = K.sum(K.log(action_prob), axis=1)
		
		loss = - log_action_prob * reward_placeholder

		## Add entropy to the loss
		entropy = K.sum(action_prob_placeholder * K.log(action_prob_placeholder + 1e-10), axis=1)
		entropy = K.sum(entropy)

		## TODO: add entropy regularization parameter ...
		loss = loss + 0.1*entropy
		loss = K.mean(loss)
		self.opt = Adam(self.config['speaker_lr'])
		self.updates = self.opt.get_updates(params=self.speaker_model.trainable_weights,loss=loss)
		#if os.path.exists(self.optfilename):
		#	self.load_opt()
		self.train_fn = K.function(
				inputs=[
				self.speaker_model.input, #shape(bs, speaker_input_dim)
				action_onehot_placeholder, #shape(bs, max_message_length, alphabet_size)
				reward_placeholder], #shape(bs)
				outputs=[loss, entropy], updates=self.updates)

	def build_train_fn(self):
		"""
		Batch input and output.
		"""
		action_prob_placeholder = self.speaker_model.output
		action_onehot_placeholder = K.placeholder(shape=(None, self.config['max_message_length'], self.config['alphabet_size']), name="action_onehot") #shape(bs, None, alphabet_size)
		reward_placeholder = K.placeholder(shape=(None,), name="reward") #shape(bs)

		action_prob_raw = action_prob_placeholder * action_onehot_placeholder #shape(bs, max_message_length, alphabet_size)
		
		action_prob = K.sum(action_prob_raw)
		loss = - action_prob * reward_placeholder

		#log_action_prob = K.log(action_prob)
		#loss = - log_action_prob * reward_placeholder

		## Add entropy to the loss
		entropy = K.sum(action_prob_placeholder * K.log(action_prob_placeholder + 1e-10), axis=1)
		entropy = K.sum(entropy)

		## TODO: add entropy regularization parameter ...
		loss = loss + 0.1*entropy
		loss = K.mean(loss)
		self.opt = Adam(self.config['speaker_lr'])
		self.updates = self.opt.get_updates(params=self.speaker_model.trainable_weights,loss=loss)
		
		#if os.path.exists(self.optfilename):
		#	self.load_opt()
		self.train_fn = K.function(
				inputs=[
				self.speaker_model.input, #shape(bs, speaker_input_dim)
				action_onehot_placeholder, #shape(bs, max_message_length, alphabet_size)
				reward_placeholder], #shape(bs)
				outputs=[loss, entropy], updates=self.updates)

	def reshape_target(self, target_input):
		""" Reshape target_input to (1, input_dim) """
		
		assert len(target_input.shape)==1 and target_input.shape[0]==self.config['speaker_input_dim']
		return np.expand_dims(target_input, axis=0)

	def sample_from_speaker_policy(self, target_input):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		target_input = self.reshape_target(target_input) #shape(1, speaker_input_dim)
		outputs = self.speaker_model.predict(target_input) #shape(1, max_message_length, 2)
		action_prob = np.squeeze(outputs) #shape(max_message_length, 2)
		action = []
		for _ in action_prob:
			action.append(np.random.choice(np.arange(self.config['alphabet_size']), p=_))
		action = np.array(action) #shape(max_message_length)
		return action, action_prob

	def infer_from_speaker_policy(self, target_input):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		target_input = self.reshape_target(target_input) #shape(1, speaker_input_dim)
		outputs = self.speaker_model.predict(target_input) #shape(1, max_message_length, 2)
		action_prob = np.squeeze(outputs) #shape(max_message_length, 2)
		action = np.argmax(action_prob, axis=1) #shape(max_message_length)
		return action, action_prob

	def train_speaker_policy_on_batch(self):
		"""
		Train as a batch. Loss is an float for a batch
		"""
		action_onehot = to_categorical(self.batch_action, num_classes=self.config['alphabet_size'])
		_loss, _entropy = self.train_fn([
			self.batch_target_input,
			action_onehot,
			self.batch_reward])

		#print("Speaker loss: ", _loss)
		#print("Speaker entropy: ", _entropy)
		
		self.batch_target_input = [] #shape(bs, speaker_input_dim)
		self.batch_action = [] #shape(bs, max_message_length, 2)
		self.batch_reward = [] #shape(bs)

	def remember_speaker_training_details(self, target_input, action, speaker_probs, reward):
		"""
		Inputs are just one instance. No bs dimensize.
		"""
		self.batch_target_input.append(target_input)
		self.batch_action.append(action)
		self.batch_reward.append(reward)
'''