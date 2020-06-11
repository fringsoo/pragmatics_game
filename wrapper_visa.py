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

import os
import sys
import glob
import numpy as np
import xml.etree.ElementTree as ET 
from keras.utils.np_utils import to_categorical

class VisaDatasetWrapper(object):
	""" 
	Download from https://drive.google.com/file/d/1xtExj3dHWpLQnUfazzkmsd70hIh5N6Qw/view
	"""
	def __init__(self):
		self.evaltestrepeat = 10
		self.dataset_dir = os.path.join(os.path.dirname(__file__), "datadir", "visa_dataset", "US")
		self.file_extension = ".xml"
		self.attribute_list = []
		self.concept_list = []
		self.concept_dict = {}
		self.category_list = [] #category for every concept
		self.attribute_to_id_dict = {}
		self.id_to_attribute_dict = {}
		self.symbolic_vectors = []
		self.retrieve_xml_file_names()

	def retrieve_xml_file_names(self):
		""" Loop through files in dataset_dir and get names of those with extension """
		self.xml_file_names = []
		for file_name in os.listdir(self.dataset_dir):
			if file_name.endswith(self.file_extension):
				self.xml_file_names.append(os.path.join(self.dataset_dir,file_name))
	
	def get_category_name_from_file(self, file_path):
		""" Get category from xml file name """
		return os.path.basename(file_path).split("_")[0].lower()

	def sample_target_idx(self):
		""" Randomly sample training target idx"""
		return np.random.randint(0, self.n_training_rows)

	def negatively_sample_distractors(self, target_idx, dataset, n_distractors, context):
		""" Negatively sample n_distractors """
		distractors_idx = []
		while len(distractors_idx) < n_distractors:
			if context == 'random':
				if dataset == 'train':
					sampled_idx = np.random.choice(self.training_indices)
				if dataset == 'test':
					sampled_idx = np.random.choice(self.test_indices)
			if context == 'same_category':
				category = self.category_list[target_idx]
				if dataset == 'train':
					sampled_idx = np.random.choice(self.concept_dict_train[category])
				if dataset == 'test':
					sampled_idx = np.random.choice(self.concept_dict_test[category])
			if sampled_idx!=target_idx:
				distractors_idx.append(sampled_idx)
		return distractors_idx

	def categorical_label(self, label):
		""" Convert scalar idx into array """
		l = np.zeros(self.n_distractors+1)
		l[label] = 1.
		return l

	def mean_cosine_similarity(self):
		def mean_cos(vecs):
			cossimsum = 0
			for v0 in range(len(vecs)):
				for v1 in range(len(vecs)):
					if v0 == v1:
						continue
					vec0 = vecs[v0]
					vec1 = vecs[v1]
					cossimsum += np.dot(vec0, vec1)/(np.linalg.norm(vec0) * np.linalg.norm(vec1))
			return cossimsum / (len(vecs)*(len(vecs)-1))

		for category in self.concept_dict:
			print(category)
			print(mean_cos([self.symbolic_vectors[self.concept_to_id_dict[concept]] for concept in self.concept_dict[category]]))
		print('all')
		print(mean_cos(self.symbolic_vectors))

	def create_train_test_datasets(self, config_dict):
		""" Randomly split dataset into train and test sets """
		self.config_dict = config_dict
		self.batch_size = config_dict["batch_size"]
		self.n_distractors = config_dict["n_distractors"]
		
		np.random.seed(0)

		for xml_file_name in self.xml_file_names:
			## Get file name
			file_category_name = self.get_category_name_from_file(xml_file_name)
			print("Reading in XML file for %s"%file_category_name)
			## Parse ElementTree
			tree = ET.parse(xml_file_name)
			## Get root of tree
			root = tree.getroot()
			## Define category's dict
			self.concept_dict[file_category_name] = {} 

			for subcategory in root:
				if subcategory.tag=="concept":
					concept_attributes = []
					for item in subcategory:
						for attribute in item.text.split("\n"):
							if attribute.strip()!="":
								string_attr = attribute.replace("\t","").strip()
								if string_attr not in self.attribute_list:
									self.attribute_list.append(string_attr)
								concept_attributes.append(string_attr)

					## Add list of attributes for concept
					concept_name = subcategory.attrib["name"]
					self.concept_dict[file_category_name][concept_name] = concept_attributes
					#self.concept_list.append(concept_name)
				else:
					for concept in subcategory:
						concept_attributes = []
						for item in concept:
							for attribute in item.text.split("\n"):
								if attribute.strip()!="":
									string_attr = attribute.replace("\t","").strip()
									if string_attr not in self.attribute_list:
										self.attribute_list.append(string_attr)
									concept_attributes.append(string_attr)

						## Add list of attributes for concept
						concept_name = concept.attrib["name"] 
						self.concept_dict[file_category_name][concept_name] = concept_attributes
						#self.concept_list.append(concept_name)


		""" """
		## Attribute to id dict
		self.attribute_to_id_dict = {attribute:key for key, attribute in enumerate(self.attribute_list)}

		## Id to attribute dict
		self.id_to_attribute_dict = {key:attribute for attribute, key in self.attribute_to_id_dict.items()}
		
		n_attributes = len(self.attribute_list)

		for category, subcategory in self.concept_dict.items():
			print("Creating symbolic vectors for %s category"%category)
			for subcategory, items in subcategory.items():
				vect = np.zeros(n_attributes)
				attr_indices = [self.attribute_to_id_dict[item] for item in items]
				vect[attr_indices] = 1.
				self.symbolic_vectors.append(vect)
				self.concept_list.append(subcategory)
				self.category_list.append(category)

		self.concept_to_id_dict = {concept:key for key, concept in enumerate(self.concept_list)}
		self.symbolic_vectors = np.array(self.symbolic_vectors)
		
		## Add test-train split parameter 
		self.train_split_percent = config_dict["train_split_percent"] 
		self.n_dataset_rows = len(self.concept_list)
		self.n_training_rows = int(round(self.n_dataset_rows*self.train_split_percent, 0))
		self.n_testing_rows = self.n_dataset_rows - self.n_training_rows

		## Generate random indixes to partition train and test set
		indices = np.random.permutation(len(self.symbolic_vectors))
		self.training_indices, self.test_indices = indices[:self.n_training_rows], indices[self.n_training_rows:]
		
		self.n_training_instances = self.n_training_rows * self.evaltestrepeat
		self.n_testing_instances = self.n_testing_rows * self.evaltestrepeat


		self.concept_dict_train = {}
		self.concept_dict_test = {}

		for category in self.concept_dict:
			self.concept_dict_train[category] = []
			self.concept_dict_test[category] = []

		count = 0 
		for category in self.concept_dict:
			for concept in self.concept_dict[category]:
				count += 1
				conceptid = self.concept_to_id_dict[concept]
				if conceptid in self.training_indices:
					self.concept_dict_train[category].append(conceptid)
				if conceptid in self.test_indices:
					self.concept_dict_test[category].append(conceptid)
		print(count)
		for category in self.concept_dict:
			#assert self.concept_dict_train[category]
			#assert self.concept_dict_test[category]
			print(category, len(self.concept_dict_train[category]), len(self.concept_dict_test[category]))

		#self.training_set = np.array([self.symbolic_vectors[idx] for idx in training_indices])
		#self.testing_set = np.array([self.symbolic_vectors[idx] for idx in test_indices])

		self.mean_cosine_similarity()

	def add_noise(self, noise_prob, target, candidate_set):
		target = np.logical_xor(target, np.random.choice(2, target.shape, p=noise_prob)).astype(np.float64)
		candidate_set = np.logical_xor(candidate_set, np.random.choice(2, candidate_set.shape, p=noise_prob)).astype(np.float64)
		
		return target, candidate_set

	def training_batch_generator(self):
		""" Generate batches sampled from training set"""
		"""target: 595d rep for target concept"""
		"""candidate_set: 5*595d rep for 5 concept, one for these is the target"""
		"""y_label: one one, other zeros"""
		noise_prob = [1 - self.config_dict['noise'], self.config_dict['noise']]
		for _ in range(self.batch_size):
			sampled_target_idx = np.random.choice(self.training_indices)
			distractors_idx = self.negatively_sample_distractors(sampled_target_idx, 'train', self.n_distractors, self.config_dict['context_type'])

			## Naive shuffling with record. TODO: improve..
			rand_idx = np.random.randint(0, self.config_dict['n_classes'])
			candidate_idx_set = []
			for j,dist_idx in enumerate(distractors_idx):
				if j==rand_idx:
					candidate_idx_set.append(sampled_target_idx)
				candidate_idx_set.append(dist_idx)
			if rand_idx == self.n_distractors:
				candidate_idx_set.append(sampled_target_idx)

			target = self.symbolic_vectors[sampled_target_idx]
			candidate_set = self.symbolic_vectors[candidate_idx_set]
			#target, candidate_set = self.add_noise(noise_prob, target, candidate_set)
			y_label = self.categorical_label(rand_idx)

			#yield target, candidate_set, y_label
			yield target, candidate_set, y_label, sampled_target_idx, candidate_idx_set

	def training_set_evaluation_generator(self):
		noise_prob = [1 - self.config_dict['noise'], self.config_dict['noise']]
		for _repeat in range(self.evaltestrepeat):
			""" Loop once through training set """
			for idx in self.training_indices:
				distractors_idx = self.negatively_sample_distractors(idx, 'train', self.n_distractors, self.config_dict['context_type'])

				## Naive shuffling with record. TODO: improve..
				rand_idx = np.random.randint(0, self.config_dict['n_classes'])
				candidate_idx_set = []
				for j,dist_idx in enumerate(distractors_idx):
					if j==rand_idx:
						candidate_idx_set.append(idx)
					candidate_idx_set.append(dist_idx)
				if rand_idx == self.n_distractors:
					candidate_idx_set.append(idx)

				target = self.symbolic_vectors[idx]
				candidate_set = self.symbolic_vectors[candidate_idx_set]
				#target, candidate_set = self.add_noise(noise_prob, target, candidate_set)
				y_label = self.categorical_label(rand_idx)


				#yield target, candidate_set, y_label
				yield target, candidate_set, y_label, idx, candidate_idx_set

	def testing_set_generator(self):
		noise_prob = [1 - self.config_dict['noise'], self.config_dict['noise']]
		for _repeat in range(self.evaltestrepeat):
			""" Loop once through testing set """
			for idx in self.test_indices:
				distractors_idx = self.negatively_sample_distractors(idx, 'test', self.n_distractors, self.config_dict['context_type'])

				## Naive shuffling with record. TODO: improve..
				rand_idx = np.random.randint(0, self.config_dict['n_classes'])
				candidate_idx_set = []
				for j,dist_idx in  enumerate(distractors_idx):
					if j==rand_idx:
						candidate_idx_set.append(idx)
					candidate_idx_set.append(dist_idx)
				if rand_idx == self.n_distractors:
					candidate_idx_set.append(idx)

				target = self.symbolic_vectors[idx]
				candidate_set = self.symbolic_vectors[candidate_idx_set]
				#target, candidate_set = self.add_noise(noise_prob, target, candidate_set)
				y_label = self.categorical_label(rand_idx)

				#yield target, candidate_set, y_label
				yield target, candidate_set, y_label, idx, candidate_idx_set
				

