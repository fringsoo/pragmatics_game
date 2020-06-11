# Pragmatics for Referential Games

## Note
 This codebase accompanies paper "Incorporating Pragmatic Reasoning Communication into Emergent Language", and is based on [Lazaridou et al, 2018](https://github.com/NickLeoMartin/emergent_comm_rl).

## Tested Environment
 AWS t3a.xlarge (4 CPU 16G memory) instance. Ubuntu 16.04.


## Environment Configuration
To config environments and install necessary dependencies, including compiling and rendering tools, run the following script. (You do not have to follow each specific command inside, just make sure you can successfully render pubullet images and store them as numpy arrays.)
```shell
bash env_config.sh
```
Config python venv enviroment:
```shell
source ~/venv/bin/activate
pip install -r ~/pragmatics_game/requirements.txt
```

## Run
To run the experiment:
```shell
DISPLAY=:0 python run_experiments.py
```
In which you can choose to:

### 1. Pretrain CNNs
The pretrain CNN models are already in models/conv_pretrain, you can skip this part. Or to pretrain them by yourself by setting in run_experiments.py:
```shell
agent.pretrain_fit_conv()
```

### 2. Long-term training
To use pretrained emerged language system, config "modelpath" in config.py as:
```shell
os.path.join("models", "model_pixel_rnnconv_alpha17_maxlength5")
```
or train the system by yourself by changing it to a new path and  setting in run_experiments.py:
```shell
agent.fit()
```

### 3. Virtual opponent training
To use the origin models as virtual opponents, in run_experiments.py set:
```shell
agent.set_virtual_origin()
```
or use pretrained virtual oppoents in pretrain language system, in run_experiments.py set:
```shell
agent.set_virtual_real()
```
or train them system by yourself by setting in run_experiments.py:
```shell
agent.set_virtual_real()
agent.train_virtual_listener()
agent.train_virtual_speaker()
```
To check the virtual opponents' fidelity, set:
```shell
agent.check_virtual_listener()
agent.check_virtual_speaker()
```

### 4. For short-term basseline and pragmatics testing:
Define challenge set in config.py:
```shell
"challenge": True, #or False for overall testset
"challenge_same_set": [[0],[1],[2],[3],[4,5],[6,7]], 
# This is an example for the pretrained system. Here the numbers means colors: [[black], [blue], [green], [cyan], [red, magenta], [yellow, white]]
```
Define proposal set:
```shell
'maskdigit': [3,4], 
'mask': 2,
# Masking the last two digit of the message, which are meaningless and always "2" in alphabet in the pretrained model.
'threshold': 0.75, 
# The threshold for proposing highest probability messages.
```

Results in Table 2, 3 in the paper will be printed.

Also in a folder named with test datetime in the modelpath:

Results in Table 2, 3 in the paper are saved in logs file.
Images in Table 4 in the paper are saved. 