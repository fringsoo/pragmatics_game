# Incorporating Pragmatic Reasoning Communication into Emergent Language: Pragmatics for Referential Games

## Note
 This is the codebase for the referential game experiment of NeurIPS2020 spotlight paper [**Incorporating Pragmatic Reasoning Communication into Emergent Language**](https://fringsoo.github.io/pragmatic_in2_emergent_papersite/), authored by Yipeng Kang, [Tonghan Wang](https://tonghanwang.github.io/) and [Gerard de Melo](http://gerard.demelo.org/). It is partially based on [Lazaridou et al, 2018](https://github.com/NickLeoMartin/emergent_comm_rl). Codes of the SCII case study section is in another [independent repo](https://github.com/fringsoo/NDQ).

## Tested Environment
Ubuntu 16.04. Python 3.5.

## Environment Configuration
<!-- To config environments and install necessary dependencies, including compiling and rendering tools, run the following script. (You do not have to follow each specific command inside, just make sure you can successfully render PyBullet images and store them as numpy arrays.)
```shell
bash env_config.sh
``` -->
Config python venv enviroment:
```shell
pip install -r ~/pragmatics_game/requirements.txt
```
You can adjust cpu/gpu version of tensorflow inside.

## Run
To run the experiment:
```shell
python run_experiments.py
```
You can directly execute this script to see the results for the challenge dataset using pretrained model. Or you can choose to config in run_experiments.py and config.py to do the followings:

### 1. Pretrain CNNs
The pretrain CNN models are already in models/conv_pretrain. Or to pretrain them by yourself, set in run_experiments.py:
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
To use the original models as virtual opponents, set in run_experiments.py:
```shell
agent.set_virtual_origin()
```
or use pretrained virtual opponents in pretrain language system, set in run_experiments.py:
```shell
agent.set_virtual_real()
```
or train them system by yourself:
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
# This is specific for "model_pixel_rnnconv_alpha17_maxlength5". Here the numbers refer to same/similiar color candidates that can not be distuiguished by baseline methods: [[black], [blue], [green], [cyan], [red, magenta], [yellow, white]]
```
Define proposal set in config.py:
```shell
'maskdigit': [3,4], 
'mask': 2,
# Masking the last two digits of the message, which are meaningless in  "model_pixel_rnnconv_alpha17_maxlength5" language system.
'threshold': 0.75, 
# The threshold for proposing highest probability messages.
```
Define number of epoches in config.py:
```shell
"predict_nepoch": 3,
```

Results in Table 2, 3 in the paper will be printed. For example, by setting challenge to True, we have:
model | Acc | std
---- | --- | ---
baseline | 53.0 | 2.6
sampleL0 | 54.8 | 2.9
sampleL0.5| 53.9 | 4.5
argmaxL| 56.6 | 2.5
argmaxL virtual| 53.0 | 2.3
RSA2rnd| 55.9 | 1.9
IBR2rnd| 80.6 | 2.8
RSAcnvg| 62.0 | 2.1
RSAcnvg virtual| 54.3 | 1.4
IBRcnvg| 80.6 | 2.8
IBRcnvg virtual| 68.6 | 1.6
GameTable| 74.6 | 2.0
GameTable virtual| 58.1 | 2.2
GameTable-s| 94.0 | 0.6
GameTable-s virtual| 69.9 | 1.8


Also in a folder named with test datetime in the modelpath:

Results in Table 2, 3 in the paper are saved in logs file.
Images in Table 4 in the paper are saved. 