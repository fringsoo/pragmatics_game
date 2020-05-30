import numpy as np
import pdb
import wget
import os
from progressbar import *

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, LSTM, SimpleRNN, Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Activation, Flatten, Lambda, BatchNormalization, merge
from keras.optimizers import RMSprop, Adam, SGD
from keras.applications.vgg16 import VGG16
from keras import backend as K


def convnet_ori(w, h, c, name=None):
	if name:
		model = Sequential(name=name)
	else:
		model = Sequential()
	model.add(Conv2D(32,(3,3),strides=(2,2),input_shape=(w,h,c),padding='valid',activation='relu',kernel_initializer='uniform'))  
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3,3),strides=(1,1)))  
	model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
	model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(BatchNormalization())
	model.add(Conv2D(32,(3,3),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(BatchNormalization())
	model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
	model.add(Flatten())  

	#model.add(Dense(4096,activation='relu'))  
	#model.add(Dropout(0.5))  
	#model.add(Dense(4096,activation='relu'))  
	#model.add(Dropout(0.5))
	return model

def convnet_mod(w, h, c, name=None):
	if name:
		model = Sequential(name=name)
	else:
		model = Sequential()
	model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(w,h,c),padding='valid',activation='relu',kernel_initializer='uniform'))  
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
	model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
	model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(Flatten())  

	#model.add(Dense(4096,activation='relu'))  
	#model.add(Dropout(0.5))  
	#model.add(Dense(4096,activation='relu'))  
	#model.add(Dropout(0.5))  
	return model

def convnet_vgg(w, h, c, name=None):
	toplessvgg = VGG16(include_top=False, input_shape=(w,h,c))
	
	if name:
		model = Sequential(name=name)
	else:
		model = Sequential()
	model.add(toplessvgg)
	model.add(Flatten())
	
	#model.add(Dense(4096,activation='relu'))  
	#model.add(Dropout(0.5))  
	#model.add(Dense(4096,activation='relu'))  
	#model.add(Dropout(0.5))  
	
	return model

def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5, **kwargs):
	"""
	This is the function used for cross channel normalization in the original
	Alexnet
	"""
	def f(X):
		b, ch, r, c = X.shape
		half = n // 2
		square = K.square(X)
		extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1)), ((0,0),(half,half)), data_format='channels_first')
		extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
		scale = k
		for i in range(n):
			scale += alpha * extra_channels[:,i:i+ch,:,:]
		scale = scale ** beta
		return X / scale
	#return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)
	#return Lambda(lambda x:f(x), **kwargs)
	return Lambda(f, **kwargs)

def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
	def f(X):
		div = X.shape[axis] // ratio_split

		if axis == 0:
			output =  X[id_split*div:(id_split+1)*div,:,:,:]
		elif axis == 1:
			output =  X[:, id_split*div:(id_split+1)*div, :, :]
		elif axis == 2:
			output = X[:,:,id_split*div:(id_split+1)*div,:]
		elif axis == 3:
			output = X[:,:,:,id_split*div:(id_split+1)*div]
		else:
			raise ValueError("This axis is not possible")

		return output

	def g(input_shape):
		output_shape=list(input_shape)
		output_shape[axis] = output_shape[axis] // ratio_split
		return tuple(output_shape)

	#return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)
	#return Lambda(lambda x:f(x), **kwargs)
	return Lambda(f, **kwargs)

def convnet_com(w, h, c, preloadfile, name=None):
	'''
	if not os.path.exists('bvlc_alexnet.npy'):
		wget.download("http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy", out="bvlc_alexnet.npy")
	params=np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
	pdb.set_trace()
	'''

	inputs = Input(shape=(w, h, c))
	new_inputs = Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)))(inputs)
	conv_1 = Convolution2D(96, (11,11), strides=(4,4),activation='relu',name='conv_1', data_format='channels_first')(new_inputs)
	conv_2 = MaxPooling2D((3, 3), strides=(2,2), data_format='channels_first')(conv_1)
	conv_2 = crosschannelnormalization()(conv_2)
	conv_2 = ZeroPadding2D((2,2), data_format='channels_first')(conv_2)
	conv_2 = merge([Convolution2D(128,(5,5),activation="relu",name='conv_2_'+str(i+1), data_format='channels_first')(splittensor(ratio_split=2,id_split=i)(conv_2)) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

	conv_3 = MaxPooling2D((3, 3), strides=(2, 2), data_format='channels_first')(conv_2)
	conv_3 = crosschannelnormalization()(conv_3)
	conv_3 = ZeroPadding2D((1,1), data_format='channels_first')(conv_3)
	conv_3 = Convolution2D(384,(3,3),activation='relu',name='conv_3', data_format='channels_first')(conv_3)

	conv_4 = ZeroPadding2D((1,1), data_format='channels_first')(conv_3)
	conv_4 = merge([Convolution2D(192,(3,3),activation="relu",name='conv_4_'+str(i+1), data_format='channels_first')(splittensor(ratio_split=2,id_split=i)(conv_4)) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

	conv_5 = ZeroPadding2D((1,1), data_format='channels_first')(conv_4)
	conv_5 = merge([Convolution2D(128,(3,3),activation="relu",name='conv_5_'+str(i+1), data_format='channels_first')(splittensor(ratio_split=2,id_split=i)(conv_5)) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

	dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5", data_format='channels_first')(conv_5)

	dense_1 = Flatten(name="flatten")(dense_1)
	
	#dense_1 = Dense(4096, activation='relu',name='new_dense_1')(dense_1)
	#dense_2 = Dropout(0.5)(dense_1)
	#dense_2 = Dense(4096, activation='relu',name='new_dense_2')(dense_2)
	#dense_3 = Dropout(0.5)(dense_2)
	#dense_3 = Dense(1000,name='dense_3')(dense_3)
	#prediction = Activation("softmax",name="softmax")(dense_3)

	if name:
		model = Model(input=inputs, output=dense_1, name=name)
	else:
		model = Model(input=inputs, output=dense_1)

	preload_complete = False
	if os.path.exists(os.path.join('model_pretrain', preloadfile)):
		model.load_weights(os.path.join('model_pretrain', preloadfile), True)
		preload_complete = True
		print('preload', preloadfile)
	
	if not preload_complete and preloadfile!=None:
		if not os.path.exists(os.path.join('model_pretrain', 'alexnet_weights.h5')):
			wget.download("http://files.heuritech.com/weights/alexnet_weights.h5", out=os.path.join('model_pretrain', 'alexnet_weights.h5'))
		model.load_weights('model_pretrain/alexnet_weights.h5', True)

	return model

def pretrain_conv(model, data_generator, conv_pretrain_type, fix_convmodel, trainround, savefile):
	if conv_pretrain_type=='shape':
		conv_pretrain_clsnum = 5
	elif conv_pretrain_type=='color':
		conv_pretrain_clsnum = 8
	else:
		assert False
	
	conv_branch_model = Sequential()
	conv_branch_model.add(model)
	conv_branch_model.add(Dense(conv_pretrain_clsnum, activation='softmax'))
	conv_branch_model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True),metrics=['accuracy'])
	
	if fix_convmodel:
		conv_branch_model.layers[0].trainable = False
	
	right_train = 0
	right_test = 0
	while (right_train<2900 or right_test<940):
		right_train = 0
		right_test = 0
		pbar = ProgressBar()
		for _n in pbar(range(trainround)):
			batch = data_generator.training_batch_generator(pretrain_idx_type=conv_pretrain_type)
			for b in batch:
				target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = b
				X = np.array([candidates[target_candidate_idx.argmax()]])
				y = np.array([np.eye(conv_pretrain_clsnum)[sampled_target_idx]])
				conv_branch_model.fit(X, y, verbose=0, epochs=1, batch_size=1)	
		for train_example in data_generator.training_set_evaluation_generator(pretrain_idx_type=conv_pretrain_type):
			target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = train_example
			ans = conv_branch_model.predict(np.array([target_input]))[0].argmax()
			if ans == sampled_target_idx:
				right_train+=1
		for train_example in data_generator.testing_set_generator(pretrain_idx_type=conv_pretrain_type):
			target_input, candidates, target_candidate_idx, sampled_target_idx, candidate_idx_set = train_example
			ans = conv_branch_model.predict(np.array([target_input]))[0].argmax()
			if ans == sampled_target_idx:
				right_test+=1
		print(right_train, right_test)

		
	conv_branch_model.layers[0].trainable = True

	if savefile:
		model.save(os.path.join('model_pretrain', savefile))