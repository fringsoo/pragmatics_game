'''
import tensorflow as tf
import keras.backend.tensorflow_backend
config = tf.ConfigProto(device_count={"CPU": 36},intra_op_parallelism_threads=36,inter_op_parallelism_threads=36)
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
'''


# from agents import Agents_dense as Agents
# from config import agent_config_dict_dense as config_dict

# from agents import Agents_rnnbasic as Agents
# from config import agent_config_dict_rnnbasic as config_dict

from agents import Agents_rnnconv as Agents
from config import agent_config_dict_rnnconv as config_dict


agent = Agents(config_dict)

#agent.pretrain_fit_conv()

#agent.fit()

#agent.set_virtual_origin()
agent.set_virtual_real()
#agent.train_virtual_listener()
#agent.train_virtual_speaker()
agent.check_virtual_listener()
agent.check_virtual_speaker()

agent.predict()

