import os

agent_config_dict_dense = {	
	###################################fixed for a model###################################
	"modelpath": os.path.join("models", "model_symbolic_dense_maxlength10"),

	#"agent_amount": 10,
	"context_type": 'random',
	
	## Data Generation
	"n_distractors": 4,
	"n_classes": 5, 
	
	## Train-split 
	"train_split_percent": 0.8,

	## Language 
	"max_message_length":10,
	#"alphabet_size": 62,
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYX0123456789'],
	"alphabet_size": 2,
	"alphabet": [c for c in 'ab'],

	## Speaker
	"speaker_dim": 50,
	"speaker_input_dim": 595,

	## Listener
	"listener_dim": 50,

	###################################unfixed for a model###################################
	## Training
	"noise":0.00,

	#"training_epoch": 1000,
	"batch_size": 1,
	"n_batches": 1000000,
	
	"short_game_round": 100,
	'sample_round_explicit': 10,
	"predict_nepoch": 3,
}

agent_config_dict_rnnbasic = {	
	###################################fixed for a model###################################
	"modelpath": os.path.join("models", "model_symbolic_rnnbasic_alpha17_maxlength5"),
	
	
	#"agent_amount": 10,
	"context_type": 'random',

	
	## Data Generation
	"n_distractors": 4,
	"n_classes": 5, 
	
	## Train-split 
	"train_split_percent": 0.8,

	## Language 
	"max_message_length": 5,
	#"alphabet_size": 62,
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYX0123456789'],
	"alphabet_size": 17,
	#"alphabet": [c for c in 'ab'],
	"alphabet": [c for c in 'abcdefghijklmnopq'],
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyz1234567890-_=+'],

	## Speaker
	"speaker_dim": 50,
	"speaker_input_dim": 595,

	## Listener
	"listener_dim": 50,

	###################################unfixed for a model###################################
	## Training
	"noise": 0.00,

	#"training_epoch": 1000,
	"batch_size": 1,
	"n_batches": 1000000,
	
	"max_short_game_round": 1000,
	'sample_round_explicit': 1,
	"predict_nepoch": 3,
}

agent_config_dict_rnnconv = {	
	###################################fixed for a model###################################
	"modelpath": os.path.join("models", "model_pixel_rnnconv_alpha17_maxlength5"),
	#"modelpath": os.path.join("models", "new_model_pixel_rnnconv_alpha17_maxlength5"),

	
	#"agent_amount": 10,
	
	## Data Generation
	"pixel": True,
	"n_distractors": 1,
	"n_classes": 2, 
	
	"n_training_instances": 3000,
	"n_testing_instances": 1000,
	## Train-split 
	#"train_split_percent": 0.8,

	## Language 
	"max_message_length": 5,
	#"alphabet_size": 62,
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYX0123456789'],
	"alphabet_size": 17,
	#"alphabet": [c for c in 'abc'],
	"alphabet": [c for c in 'abcdefghijklmnopq'],
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyz1234567890-_=+'],

	## Speaker
	"speaker_dim": 50,
	"speaker_input_w": 124,
	"speaker_input_h": 124,

	## Listener
	"listener_dim": 50,

	"conv_type": 'common',
	
	###################################unfixed for a model###################################
	## Training
	"noise": 0.00,
	
	#"training_epoch": 1000,
	"batch_size": 1,
	"n_batches": 1000000,
	
	"max_short_game_round": 1000,
	'sample_round_explicit': 1,
	
	"predict_nepoch": 2,

	"challenge": True,
	"challenge_same_set": [[0],[1],[2],[3],[4,5],[6,7]],
	'maskdigit': [3,4],
	'mask': 2,
	'threshold': 0.75,
}



'''
crowd_rnnconv_config_dict = {	
	###################################fixed for a model###################################
	"modelpath": "model_symbolic_rnnconv_alpha17_maxlength5_notsameviewpoint_2candidates_pretrain_euc_amount3",
	#"modelpath": "model_torm",
	
	"agent_amount": 3,
	
	## Data Generation
	"pixel": True,
	"n_distractors": 1,
	"n_classes": 2, 
	
	"n_training_instances": 3000,
	"n_testing_instances": 1000,
	## Train-split 
	#"train_split_percent": 0.8,

	## Language 
	"max_message_length": 5,
	#"alphabet_size": 62,
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYX0123456789'],
	"alphabet_size": 17,
	#"alphabet": [c for c in 'abc'],
	"alphabet": [c for c in 'abcdefghijklmnopq'],
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyz1234567890-_=+'],

	## Speaker
	"speaker_dim": 50,
	"speaker_input_w": 124,
	"speaker_input_h": 124,

	## Listener
	"listener_dim": 50,

	"conv_type": 'common',
	
	###################################unfixed for a model###################################
	## Training
	"noise": 0.00,
	
	#"training_epoch": 1000,
	"batch_size": 1,
	"n_batches": 2000000,
	
	"max_short_game_round": 1000,
	'sample_round_explicit':1,
}

agent_rnnconvcolor_config_dict = {	
	###################################fixed for a model###################################
	"modelpath": "model_symbolic_rnnconv_alpha17_maxlength5_notsameviewpoint_v_trainlstm_2candidates_color",
	#"modelpath": "model_torm",

	#"agent_amount": 10,
	
	## Data Generation
	"pixel": True,
	"n_distractors": 1,
	"n_classes": 2, 
	
	"n_training_instances": 3000,
	"n_testing_instances": 1000,
	## Train-split 
	#"train_split_percent": 0.8,


	## Language 
	"max_message_length": 5,
	#"alphabet_size": 62,
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYX0123456789'],
	"alphabet_size": 17,
	#"alphabet": [c for c in 'abc'],
	"alphabet": [c for c in 'abcdefghijklmnopq'],
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyz1234567890-_=+'],

	## Speaker
	"speaker_dim": 50,
	"speaker_input_w": 124,
	"speaker_input_h": 124,

	## Listener
	"listener_dim": 50,

	"conv_type": 'common',
	
	###################################unfixed for a model###################################
	## Training
	"noise": 0.00,

	
	#"training_epoch": 1000,
	"batch_size": 1,
	"n_batches": 2000000,
	
	"max_short_game_round": 1000,
	'sample_round_explicit':1,
}

crowd_config_dict = {	
	###################################fixed for a model, manually check###################################
	"modelpath": "model_symbolic_crowds_10agents",

	"agent_amount": 10,
	
	## Data Generation
	"n_distractors": 4,
	"n_classes": 5, 
	
	## Train-split 
	"train_split_percent": 0.8,

	## Language 
	"max_message_length": 4,
	#"alphabet_size": 62,
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYX0123456789'],
	"alphabet_size": 2,
	"alphabet": [c for c in 'ab'],

	## Speaker
	"speaker_dim": 595,
	"speaker_input_dim": 595,

	## Listener
	"listener_dim": 50,

	
	###################################unfixed for a model###################################
	## Training
	
	#"training_epoch": 1000,
	"batch_size": 1,
	"n_batches": 10000000,
	
	"short_game_round": 100,
}

agent_direct_config_dict = {	
	###################################fixed for a model###################################
	"modelpath": "model_symbolic_agents_maxlength10_direct",

	#"agent_amount": 10,
	
	## Data Generation
	"n_distractors": 4,
	"n_classes": 5, 
	
	## Train-split 
	"train_split_percent": 0.8,

	## Language 
	"max_message_length": 10,
	#"alphabet_size": 62,
	#"alphabet": [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYX0123456789'],
	"alphabet_size": 2,
	"alphabet": [c for c in 'ab'],

	## Speaker
	"speaker_dim": 595,
	"speaker_input_dim": 595,

	## Listener
	"listener_dim": 50,

	
	###################################unfixed for a model###################################
	## Training
	"speaker_lr": 0.001, #0.0001
	"listener_lr": 0.001, #0.001
	
	#"training_epoch": 1000,
	"batch_size": 1,
	"n_batches": 1000000,
	
	"short_game_round": 100,
}
'''
