import random
import numpy as np
import scipy
import time
import json
import os
import pdb
import pickle
import pandas
from progressbar import *

from keras.layers import Input, Dense, LSTM, Lambda, concatenate, add, Dot
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras import regularizers
from keras.utils.np_utils import to_categorical
from utils import convnet_vgg, convnet_mod, convnet_ori, convnet_com

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()
	#return x / np.linalg.norm(x)

def makeFunc(x):
	return lambda y:y[:,x]

class BaseListenerNetwork(object):
	def __init__(self, modelname, optfilename, lr, entropy_coefficient, config_dict):
		self.modelname = modelname
		self.optfilename = optfilename
		self.lr = lr
		self.entropy_coefficient = entropy_coefficient
		assert config_dict, "config_dict does not exist"
		self.config = config_dict
		self.initialize_model()
		self.build_train_fn()

	def rebuild_train_fn(self, entropy_coefficient=None, lr=None):
		if entropy_coefficient:
			self.entropy_coefficient = entropy_coefficient
		if lr:
			self.lr = lr
		self.build_train_fn()

	def save(self):
		self.listener_model.save(self.modelname)

	def load(self):
		self.listener_model = load_model(self.modelname)	

	def save_weights(self):
		self.listener_model.save_weights(self.modelname)

	def load_weights(self):
		self.listener_model.load_weights(self.modelname)	

	def save_opt(self):
		symbolic_weights = self.opt.weights
		weight_values = K.batch_get_value(symbolic_weights)
		with open(self.optfilename, 'wb') as f:
			pickle.dump(weight_values, f)

	def load_opt(self):
		with open(self.optfilename, 'rb') as f:
			weight_values = pickle.load(f)
		self.opt.set_weights(weight_values)

	def save_memory(self):
		self.memory_model_weights = self.listener_model.get_weights()

	def load_memory(self):
		self.listener_model.set_weights(self.memory_model_weights)

class PaperListenerNetwork(BaseListenerNetwork):
	def __init__(self, modelname, optfilename, lr, entropy_coefficient, config_dict):
		super(PaperListenerNetwork, self).__init__(modelname, optfilename, lr, entropy_coefficient, config_dict)
		self.batch_speaker_message = []
		self.batch_action = []
		self.batch_candidates = []
		self.batch_reward = []
	
	def initialize_model(self):
		"""
		Batch input and output.
		"""
		if not os.path.exists(self.modelname):
			## Define model
			t_input = Input(shape=(self.config['max_message_length'],)) #Speakers Message, shape(bs, max_message_length)
			c_inputs_all = Input(shape=(self.config['n_classes'], self.config['speaker_input_dim'])) #Candidates, shape(bs, n_class, speaker_input_dim)
			inputs = [t_input, c_inputs_all]
		
			z = Dense(self.config['speaker_input_dim'], activation='sigmoid')(t_input) #shape(bs, speaker_input_dim)
			ts = []
			us = []
			for _ in range(self.config['n_classes']):
				#c_input = Input(shape=(self.config['speaker_input_dim'],)) #shape(bs, speaker_input_dim)
				c_input = Lambda(makeFunc(_))(c_inputs_all) #shape(bs, speaker_input_dim)
				#t = Lambda(lambda x: K.expand_dims(K.sum(-K.square(x), axis=1)))(add([t_trans, Lambda(lambda x: -x)(c_input)])) #shape(bs, 1)
				t = Dot(1, False)([z, c_input]) #shape(bs, 1)
				ts.append(t)
				us.append(c_input)
			
			U = concatenate(ts) #shape(bs, n_classes)
			us = concatenate(us)
			final_output = Lambda(lambda x: K.softmax(x))(U) #shape(bs, n_classes)
			
			#final_output = Dense(self.n_classes, activation='softmax', kernel_initializer='identity')(U)
			#final_output = Dense(self.n_classes, activation='softmax')(U)
			#f1 = Dense(50)(U)
			#f2 = Lambda(lambda x: K.square(x))(f1)
			#final_output = Dense(self.n_classes, activation='softmax')(f2)

			self.listener_model = Model(inputs=inputs, outputs=[final_output, U, z, us])
			#self.listener_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.config['listener_lr']))

		else:
			self.load()
			#check!!!

	def build_train_fn(self):
		"""
		Batch input and output.
		"""
		#direct prob input!!!
		action_prob_placeholder = self.listener_model.output[0] #(bs, n_classes)
		action_onehot_placeholder = K.placeholder(shape=(None, self.config['n_classes']), name="action_onehot") #(bs, n_classes)
		reward_placeholder = K.placeholder(shape=(None,), name="reward") #(?)
		action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
		log_action_prob = K.log(action_prob)
		loss = - log_action_prob * reward_placeholder

		entropy = K.sum(action_prob_placeholder * K.log(action_prob_placeholder + 1e-10), axis=1)
		#entropy = K.sum(entropy)
		loss = loss + self.entropy_coefficient * entropy
		loss = K.mean(loss)

		self.opt = Adam(lr=self.lr)
		self.updates = self.opt.get_updates(params=self.listener_model.trainable_weights, loss=loss)
		if os.path.exists(self.optfilename):
			self.load_opt()
		self.train_fn = K.function(
			inputs = self.listener_model.input + [action_onehot_placeholder, reward_placeholder],
			outputs=[loss, loss], updates=self.updates)
	
	def reshape_message_candidates(self, speaker_message, candidates):
		assert len(speaker_message.shape)==1 and speaker_message.shape[0]==self.config['max_message_length']
		assert len(candidates.shape)==2 and candidates.shape[0]==self.config['n_classes'] and candidates.shape[1]==self.config['speaker_input_dim']
		speaker_message = np.expand_dims(speaker_message, axis=0) #shape(1, max_message_length)
		#X = [speaker_message] + [c.reshape([1,-1]) for c in candidates]
		X = [speaker_message, np.expand_dims(candidates, axis=0)]
		return X

	def sample_from_listener_policy(self, speaker_message, candidates):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		X = self.reshape_message_candidates(speaker_message, candidates)
		listener_output= self.listener_model.predict_on_batch(X)
		y, U, z = listener_output[:3]
		#us = listener_output[3]
		listener_probs = y
		listener_probs = np.squeeze(listener_probs) #shape(n_class)
		listener_action = np.random.choice(np.arange(self.config['n_classes']), p=listener_probs) #int
		U = np.squeeze(U)
		return listener_action, listener_probs, U

	def infer_from_listener_policy(self, speaker_message, candidates):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		X = self.reshape_message_candidates(speaker_message, candidates)
		listener_output= self.listener_model.predict_on_batch(X)
		y, U, z = listener_output[:3]
		#us = listener_output[3]
		listener_probs = y
		listener_probs = np.squeeze(listener_probs) #shape(n_class)
		listener_action = np.argmax(listener_probs) #int
		U = np.squeeze(U)
		return listener_action, listener_probs, U
		
	def train_listener_policy_on_batch(self):
		"""
		Train as a batch. Loss is an float for a batch
		"""
		action_onehot = to_categorical(self.batch_action, num_classes=self.config['n_classes'])
		
		#self.batch_candidates = np.array(self.batch_candidates).transpose([1, 0, 2]).tolist() #shape(num_classes, bs, speaker_input_dim)
		#self.batch_candidates = np.swapaxes(np.array(self.batch_candidates), 0, 1).tolist() #shape(num_classes, bs, speaker_input_dim)
		#self.batch_candidates = np.swapaxes(np.array(self.batch_candidates), 0, 1).astype('float32').tolist() #shape(num_classes, bs, speaker_input_dim)
		#self.batch_candidates = [np.array(_) for _ in self.batch_candidates]

		#_loss, _entropy = self.train_fn([self.batch_speaker_message] + self.batch_candidates + [action_onehot, self.batch_reward] )
		_loss, _entropy = self.train_fn([np.array(self.batch_speaker_message), self.batch_candidates, action_onehot, self.batch_reward] )

		#print("Listener loss: ", _loss)
		self.batch_speaker_message = [] #shape(bs, max_message_length)
		self.batch_action = [] #shape(bs)
		self.batch_candidates = [] #shape(bs, n_classes, speaker_input_dim)
		self.batch_reward = [] #shape(bs)

	def remember_listener_training_details(self, speaker_message, action, action_probs, target, candidates, reward):
		"""
		Inputs are just one instance. No bs dimensize.
		"""
		self.batch_speaker_message.append(speaker_message)
		self.batch_action.append(action)
		self.batch_candidates.append(candidates)
		self.batch_reward.append(reward)

class PaperListenerNetwork_rnn(PaperListenerNetwork):
	def reshape_message_candidates(self, speaker_message, candidates):
		#if not self.config['fixed_length']:
		#	assert len(speaker_message.shape)==1 and speaker_message.shape[0]<=self.config['max_message_length']
		#else:
		#	assert len(speaker_message.shape)==1 and speaker_message.shape[0]==self.config['max_message_length']
		assert len(speaker_message.shape)==1 and speaker_message.shape[0]<=self.config['max_message_length']
		assert len(candidates.shape)==2 and candidates.shape[0]==self.config['n_classes'] and candidates.shape[1]==self.config['speaker_input_dim']
		speaker_message = np.expand_dims(to_categorical(speaker_message, self.config['alphabet_size']), axis=0) #shape(1, message_length, alphabet_size)
		#X = [speaker_message] + [c.reshape([1,-1]) for c in candidates]
		X = [speaker_message, np.expand_dims(candidates, axis=0)]
		return X

	def initialize_model(self):
		"""
		Batch input and output.
		"""
		## Define model
		if not os.path.exists(self.modelname):
			t_input = Input(shape=(None, self.config['alphabet_size'],)) #Speakers Message, shape(bs, message_length, alphabet_size)
			#c_inputs_all = Input(shape=(self.config['n_classes'], self.config['speaker_input_dim'])) #Candidates, shape(bs, n_classes, speaker_input_dim)
			c_inputs_all = Input(shape=(None, self.config['speaker_input_dim'])) #Candidates, shape(bs, n_classes, speaker_input_dim)
			inputs = [t_input, c_inputs_all]
			
			lstm = LSTM(self.config['listener_dim'], activation='tanh', return_sequences=False, return_state=True)
			o, sh, sc = lstm(t_input)
			z = Dense(self.config['listener_dim'], activation='sigmoid')(o) #shape(bs, listener_dim)
			
			ts = []
			us = []
			u = Dense(self.config['listener_dim'], activation='sigmoid')
			for _ in range(self.config['n_classes']):
				#c_input = Input(shape=(self.config['speaker_input_dim'],)) #shape(bs, speaker_input_dim)
				c_input = Lambda(makeFunc(_))(c_inputs_all)
				uc = u(c_input)
				t = Lambda(lambda x: K.expand_dims(K.sum(-K.square(x), axis=1)))(add([z, Lambda(lambda x: -x)(uc)])) #shape(bs, 1)
				#t = Dot(1, False)([z,uc]) #shape(bs, 1)
				ts.append(t)
				us.append(uc)
				
			U = concatenate(ts) #shape(bs, n_classes)
			us = concatenate(us)
			final_output = Lambda(lambda x: K.softmax(x))(U)
			 #shape(bs, n_classes)
			
			self.listener_model = Model(inputs=inputs, outputs=[final_output, U, z, us])
			#self.listener_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.config['listener_lr']))
		else:
			self.load()
			#check!!!

	def set_updates(self):
		self.opt = Adam(lr=self.lr)
		#adam = RMSprop(lr=self.lr)
		self.updates = self.opt.get_updates(params=self.listener_model.trainable_weights, loss=self.loss)
		if os.path.exists(self.optfilename):
			self.load_opt()

	def build_train_fn(self):
		"""
		Batch input and output.
		"""
		#direct prob input!!!
		action_prob_placeholder = self.listener_model.output[0] #(bs, n_classes)
		#action_onehot_placeholder = K.placeholder(shape=(None, self.config['n_classes']), name="action_onehot") #(bs, n_classes)
		action_onehot_placeholder = K.placeholder(shape=(None, None), name="action_onehot") #(bs, n_classes)
		reward_placeholder = K.placeholder(shape=(None,), name="reward") #(?)
		action_prob = K.sum(action_prob_placeholder*action_onehot_placeholder, axis=1)
		log_action_prob = K.log(action_prob)
		loss = - log_action_prob*reward_placeholder

		entropy = K.sum(action_prob_placeholder * K.log(action_prob_placeholder + 1e-10), axis=1)
		#entropy = K.sum(entropy)
		loss = loss + self.entropy_coefficient * entropy

		loss = K.mean(loss)
		self.loss =loss
		self.set_updates()
		
		self.train_fn = K.function(
			inputs = self.listener_model.input + [action_onehot_placeholder, reward_placeholder],
			outputs=[loss, loss], updates=self.updates)
	
	def remember_listener_training_details(self, speaker_message, action, action_probs, target, candidates, reward):
		"""
		Inputs are just one instance. No bs dimensize.
		"""
		#if not self.config['fixed_length']:					
		toadd = self.config['max_message_length'] - len(speaker_message)
		for _ in range(toadd):
			speaker_message = np.append(speaker_message, -1)
		
		speaker_message = to_categorical(speaker_message, self.config['alphabet_size']) #shape(message_length, alphabet_size)
		self.batch_speaker_message.append(speaker_message)
		self.batch_action.append(action)
		self.batch_candidates.append(candidates)
		self.batch_reward.append(reward)

class PaperListenerNetwork_rnn_conv(PaperListenerNetwork_rnn):
	def __init__(self, modelname, optfilename, lr, entropy_coefficient, pretrain_convmodel_file, traincnn, config):
		self.pretrain_convmodel_file = pretrain_convmodel_file
		self.traincnn = traincnn
		super(PaperListenerNetwork_rnn_conv, self).__init__(modelname, optfilename, lr, entropy_coefficient, config)
		
	def initialize_model(self):
		"""
		Batch input and output.
		"""
		if not os.path.exists(self.modelname):
			## Define model
			self.conv_model = convnet_com(self.config['speaker_input_w'], self.config['speaker_input_h'], 3, preloadfile=self.pretrain_convmodel_file, name='conv_model_l')

			t_input = Input(shape=(None, self.config['alphabet_size'],))  #Speakers Message, shape(bs, message_length, alphabet_size)
			c_inputs_all = Input(shape=(self.config['n_classes'], self.config['speaker_input_w'], self.config['speaker_input_h'], 3), name='image_l') #Candidates, shape(bs, speaker_input_w, speaker_input_h, 3)
			inputs = [t_input, c_inputs_all]
			
			lstm = LSTM(self.config['listener_dim'], activation='tanh', return_sequences=False, return_state=True)
			o, sh, sc = lstm(t_input)
			z = Dense(self.config['listener_dim'], activation='sigmoid')(o) #shape(bs, listener_dim)

			#u = Dense(self.config['listener_dim'], activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))
			u = Dense(self.config['listener_dim'], activation='sigmoid')
			ts = []
			us = []

			for _ in range(self.config['n_classes']):
				#c_input = Input(shape=(self.config['speaker_input_w'],self.config['speaker_input_h'],3)) #speaker_model.input[0], shape(bs, speaker_input_w, speaker_input_h, 3)				
				#c_input = Lambda(lambda x: x[:, _])(c_inputs_all)
				c_input = Lambda(makeFunc(_))(c_inputs_all)
				conv_outputs = self.conv_model(c_input)
				
				uc = u(conv_outputs)
				t = Lambda(lambda x: K.expand_dims(K.sum(-K.square(x),axis=1)))(add([z, Lambda(lambda x: -x)(uc)])) #shape(bs, 1)
				#t = Dot(1, False)([z,uc]) #shape(bs, 1)
				ts.append(t)
				us.append(uc)
			
			U = concatenate(ts) #shape(bs, n_classes)
			us = concatenate(us)
			final_output = Lambda(lambda x: K.softmax(x))(U) #shape(bs, n_classes)
			
			self.listener_model = Model(inputs=inputs, outputs=[final_output, U, z, us])
			#self.listener_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.config['listener_lr']))

		else:
			self.load()
			#check!!!
			self.conv_model = [l for l in self.listener_model.layers if l.name=='conv_model_l'][0]
			#self.listener_model.layers[6].kernel_regularizer = None

		#self.internal_model = Model(inputs=self.listener_model.inputs, outputs=[self.listener_model.layers[7].get_output_at(_) for _ in range(2)] + [self.listener_model.layers[6].output, self.listener_model.layers[-2].output]) #dot
		#self.internal_model = Model(inputs=self.listener_model.inputs, outputs=[self.listener_model.layers[6].get_output_at(_) for _ in range(2)] + [self.listener_model.layers[7].output, self.listener_model.layers[-2].output]) #euc

		self.trainable_weights_others = []
		self.trainable_weights_conv = []
		for layer in self.listener_model.layers:
			if layer.name!='conv_model_l':
				self.trainable_weights_others.extend(layer.trainable_weights)
			else:
				self.trainable_weights_conv.extend(layer.trainable_weights)
	
	def set_updates(self):
		self.opt = Adam(lr=self.lr)
		#self.opt = RMSprop(lr=self.lr)
		#opt = SGD(lr=self.lr, momentum=0.9, decay=1e-6, nesterov=True)
		
		if not self.traincnn:
			#self.updates = self.opt.get_updates(params=self.trainable_weights_others+self.trainable_weights_rnn, loss=self.loss)
			self.updates = self.opt.get_updates(params=self.trainable_weights_others, loss=self.loss)
		else:
			self.updates = self.opt.get_updates(params=self.listener_model.trainable_weights, loss=self.loss)

		if os.path.exists(self.optfilename):
			self.load_opt()
	
	def reshape_message_candidates(self, speaker_message, candidates):
		#if not self.config['fixed_length']:
		#	assert len(speaker_message.shape)==1 and speaker_message.shape[0]<=self.config['max_message_length']
		#else:
		#	assert len(speaker_message.shape)==1 and speaker_message.shape[0]==self.config['max_message_length']
		
		assert len(speaker_message.shape)==1 and speaker_message.shape[0]<=self.config['max_message_length']
		assert len(candidates.shape)==4 and candidates.shape[0]==self.config['n_classes'] and candidates.shape[1]==self.config['speaker_input_w'] and candidates.shape[2]==self.config['speaker_input_h']
		speaker_message = np.expand_dims(to_categorical(speaker_message, self.config['alphabet_size']), axis=0) #shape(1, ?, alphabet_size)
		X = [speaker_message, np.expand_dims(candidates, axis=0)]
		return X


'''
class PaperListenerNetwork_rnn_conv_color(PaperListenerNetwork_rnn):
	def initialize_model(self):
		"""
		Batch input and output.
		"""
		if not os.path.exists(self.modelname):
			## Define model
			t_input = Input(shape=(None, self.config['alphabet_size'],))  #Speakers Message, shape(bs, message_length, alphabet_size)
			c_inputs_all = Input(shape=(self.config['n_classes'], 8))
			inputs = [t_input, c_inputs_all]
			
			lstm = LSTM(self.config['listener_dim'], activation='tanh', return_sequences=False, return_state=True)
			o, sh, sc = lstm(t_input)
			z = Dense(self.config['listener_dim'], activation='sigmoid')(o) #shape(bs, listener_dim)

			u = Dense(self.config['listener_dim'], activation='sigmoid')
			ts = []

			for _ in range(self.config['n_classes']):
				#c_input = Input(shape=(self.config['speaker_input_w'],self.config['speaker_input_h'],3)) #speaker_model.input[0], shape(bs, speaker_input_w, speaker_input_h, 3)				
				#c_input = Lambda(lambda x: x[:, _])(c_inputs_all)
				c_input = Lambda(makeFunc(_))(c_inputs_all)
				#conv_outputs = conv_model(c_input)
				#conv_outputs = c_input

				uc = u(c_input)
				
				t = Lambda(lambda x: K.expand_dims(K.sum(-K.square(x),axis=1)))(add([z, Lambda(lambda x: -x)(uc)])) #shape(bs, 1)
				ts.append(t)
			
			U = concatenate(ts) #shape(bs, n_classes)
			final_output = Lambda(lambda x: K.softmax(x))(U) #shape(bs, n_classes)
			
			self.listener_model = Model(inputs=inputs, outputs=[final_output, z, U])
			#self.listener_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.config['listener_lr']))

		else:
			self.load()
			#check!!!
			
		self.trainable_weights_rnn = self.listener_model.trainable_weights[:3]
		self.trainable_weights_others = self.listener_model.trainable_weights[3:]

	def set_updates(self):
		self.opt = Adam(lr=self.lr)
		#opt = RMSprop(lr=self.lr)
		#opt = SGD(lr=self.lr, momentum=0.9, decay=1e-6, nesterov=True)
		
		self.updates = self.opt.get_updates(params=self.listener_model.trainable_weights, loss=self.loss)
		if os.path.exists(self.optfilename):
			self.load_opt()

	def reshape_message_candidates(self, speaker_message, candidates):
		#if not self.config['fixed_length']:
		#	assert len(speaker_message.shape)==1 and speaker_message.shape[0]<=self.config['max_message_length']
		#else:
		#	assert len(speaker_message.shape)==1 and speaker_message.shape[0]==self.config['max_message_length']
		#pdb.set_trace()

		assert len(speaker_message.shape)==1 and speaker_message.shape[0]<=self.config['max_message_length']
		assert len(candidates.shape)==2 and candidates.shape[0]==self.config['n_classes'] and candidates.shape[1]==8
		speaker_message = np.expand_dims(to_categorical(speaker_message, self.config['alphabet_size']), axis=0) #shape(1, ?, alphabet_size)
		X = [speaker_message, np.expand_dims(candidates, axis=0)]
		return X

class PaperListenerNetwork_direct(BaseListenerNetwork):
	def __init__(self, modelname, config_dict):
		assert False #TOMODIFY
		super(PaperListenerNetwork_direct, self).__init__(modelname, config_dict)
		self.batch_speaker_message = []
		self.batch_action = []
		self.batch_candidates = []
		self.batch_reward = []

	def initialize_model(self):
		"""
		Batch input and output.
		"""
		if not os.path.exists(self.modelname):
			## Define model
			## Speakers Message
			t_input = Input(shape=(self.config['max_message_length'],)) #shape(bs, max_message_length)
			t_trans = Dense(self.config['speaker_input_dim'],
							#kernel_initializer=keras.initializers.Identity(gain=1.0),
							#bias_initializer='zeros',
							activation='sigmoid')(t_input) #shape(bs, speaker_input_dim)

			inputs = [t_input]
			ts = []
			for _ in range(self.config['n_classes']):
				c_input = Input(shape=(self.config['speaker_input_dim'],)) #shape(bs, speaker_input_dim)
				t = Lambda(lambda x: K.expand_dims(K.sum(-K.square(x),axis=1)))(add([t_trans, Lambda(lambda x: -x)(c_input)])) #shape(bs, 1)
				inputs.append(c_input)
				ts.append(t)
			
			U = concatenate(ts) #shape(bs, n_classes)
			
			listener_probs = U
			#listener_probs = Lambda(lambda x: K.softmax(x))(U) #shape(bs, n_classes)
			listener_infer_action = Lambda(lambda x: K.argmax(x))(U) #shape(bs)

			target_onehot_placeholder = Input(shape=(self.config['n_classes'],), name="action_onehot") #(bs, n_classes)
			listener_prob_2 = dot([listener_probs, target_onehot_placeholder], axes=1)
			listener_prob_2 = Lambda(lambda x:K.squeeze(x, axis=1))(listener_prob_2)
			self.listener_model = Model(inputs=inputs + [target_onehot_placeholder], outputs=[listener_probs, listener_infer_action, t_trans, listener_prob_2])

		else:
			self.load()
			#check!!!

	def build_train_fn(self):
		"""
		Batch input and output.
		"""
		#direct prob input!!!
		#reward_placeholder = K.placeholder(shape=(None,), name="reward") #(?)
		action_prob = self.listener_model.output[3]
		#loss = K.log(-action_prob)*reward_placeholder
		#loss = - action_prob * reward_placeholder
		loss = - action_prob
		loss = K.mean(loss)
		self.opt = Adam(lr=self.config['listener_lr'])
		self.updates = self.opt.get_updates(params=self.listener_model.trainable_weights,loss=loss)
		#if os.path.exists(self.optfilename):
		#	self.load_opt()
		
		self.train_fn = K.function(
			#inputs = self.listener_model.input + [reward_placeholder],
			inputs = self.listener_model.input,
			outputs=[loss, loss], updates=self.updates)
		
	def sample_from_listener_policy(self, speaker_message, candidates):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		X = self.reshape_message_candidates(speaker_message, candidates) + [np.zeros([1, self.config['n_classes']])]
		listener_probs, listener_infer_action, _t_trans, _lp2 = self.listener_model.predict_on_batch(X)
		listener_probs = np.squeeze(listener_probs) #shape(n_class)
		#listener_probs = scipy.special.softmax(listener_probs)
		listener_probs = softmax(listener_probs)
		#pdb.set_trace() #???norm???
		listener_action = np.random.choice(np.arange(self.config['n_classes']), p=listener_probs) #int
		return listener_action, listener_probs

	def infer_from_listener_policy(self, speaker_message, candidates):
		"""
		Input and output are all just one instance. No bs dimensize.
		"""
		X = self.reshape_message_candidates(speaker_message, candidates) + [np.zeros([1, self.config['n_classes']])]
		listener_probs, listener_infer_action, _t_trans, _lp2 = self.listener_model.predict_on_batch(X)
		listener_probs = np.squeeze(listener_probs) #shape(n_class)
		listener_probs = softmax(listener_probs)
		listener_action = np.squeeze(listener_infer_action).tolist() #int
		return listener_action, listener_probs
		
	def train_listener_policy_on_batch(self):
		"""
		Train as a batch. Loss is an float for a batch
		"""
		self.batch_candidates = np.array(self.batch_candidates).transpose([1, 0, 2]).tolist() #shape(num_classes, bs, speaker_input_dim
		#_loss, _entropy = self.train_fn([self.batch_speaker_message] + self.batch_candidates + [self.batch_action, self.batch_reward] )
		_loss, _entropy = self.train_fn([self.batch_speaker_message] + self.batch_candidates + [self.batch_action] )

		#print("Listener loss: ", _loss)
		self.batch_speaker_message = [] #shape(bs, max_message_length)
		self.batch_action = [] #shape(bs, n_classes)
		self.batch_candidates = [] #shape(bs, n_classes, speaker_input_dim)
		self.batch_reward = [] #shape(bs)

	def remember_listener_training_details(self, speaker_message, action, action_probs, target, candidates, reward):
		"""
		Inputs are just one instance. No bs dimensize.
		"""
		#action_onehot = np.zeros(self.config['n_classes'])
		#action_onehot[action] = 1
		action_onehot = np.ones(self.config['n_classes']) * np.all(target==candidates, axis=1)
		self.batch_action.append(action_onehot)
		self.batch_speaker_message.append(speaker_message)
		self.batch_candidates.append(candidates)
		self.batch_reward.append(reward)
'''