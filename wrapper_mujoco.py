
'''
from mujoco_py import load_model_from_path, MjSim
model = load_model_from_path('mujoco_models.xml')
sim = MjSim(model)
img = sim.render(600, 600)
pyplot.imshow(img)
pyplot.show()
'''

from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pdb
import time
from progressbar import *
import matplotlib.pyplot as plt

class MujocoWrapper(RoboschoolMujocoXmlEnv):
    def __init__(self):
        RoboschoolMujocoXmlEnv.__init__(self, 'mujoco_models.xml', 'body0', action_dim=2, obs_dim=9)
        self.VIDEO_W = 124
        self.VIDEO_H = 124

        self.basic_xml = '\
            <mujoco model="basic">\n\
            <compiler angle="radian" inertiafromgeom="true"/>\n\
            <default>\n\
                <joint armature="1" damping="1" limited="true"/>\n\
            </default>\n\
            <worldbody>\n\
                <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1 1 10" type="plane"/>\n\
                <geom name="target" size=".1 .1 .1" rgba="%s" pos="%s" type="%s"/>\n\
            </worldbody>\n\
            </mujoco>'
        
        self.rgbas = [
            [0.0, 0.0, 1.0, 1], #blue
            [1.0, 0.0, 0.0, 1], #red
            [1.0, 1.0, 1.0, 1], #white
            [0.0, 0.0, 0.0, 1], #black
            [1.0, 1.0, 0.0, 1], #yellow
            [0.0, 1.0, 0.0, 1], #green
            [0.0, 1.0, 1.0, 1], #cyan
            [1.0, 0.0, 1.0, 1], #magenta
        ]

        #self.shapes = ['ellipsoid']
        self.shapes= ['box', 'sphere', 'cylinder', 'capsule']

        self.floor_colors = []

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=0.0, timestep=0.0165, frame_skip=1)

    def camera_adjust(self):
        x, y, z = self.mjcf[0].root_part.pose().xyz()
        fx = np.random.uniform(0,1)
        fy = np.random.uniform(0,1)
        fz = np.random.uniform(0,1)
        self.camera.move_and_look_at(0, 0, 1, 0, 0, -1)

    def generate_params(self):
        rgba = self.rgbas[np.random.randint(len(self.rgbas))]
        rgba = str(rgba).replace(',','')[1:-1]
        shapeno = np.random.randint(len(self.shapes))
        shape = self.shapes[shapeno]
        pos = [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1), np.random.uniform(0,0.1)]
        pos = str(pos).replace(',','')[1:-1] 

        #print(rgba, shape, pos)
        return rgba, shape, pos

    def reset(self, rgba, shape, pos):
        #if self.scene is None:
        self.scene = self.create_single_player_scene()
        if not self.scene.multiplayer:
            self.scene.episode_restart()

        xml = self.basic_xml%(rgba, pos, shape)
        xml_file = os.path.join(os.path.dirname(__file__), 'asset', self.model_xml)
        with open(xml_file, 'w') as f:
            f.write(xml)
        self.mjcf = self.scene.cpp_world.load_mjcf(xml_file)

        for r in self.mjcf:
            r.query_position()
        self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        
    def render(self, mode='human'):
        if mode=="human":
            self.scene.human_render_detected = True
            return self.scene.cpp_world.test_window()
        elif mode=="rgb_array":
            self.camera_adjust()
            rgb, _, _, _, _ = self.camera.render(False, False, False) # render_depth, render_labeling, print_timing)
            rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
            return rendered_rgb
        else:
            assert(0)

    def create_dataset(self, num):
        print('preparing dataset')
        dataset = []
        shapeset = []
        pbar = ProgressBar()
        for _ in pbar(range(num)):
            #pdb.set_trace()
            rgba, shape, pos = self.generate_params()
            self.reset(rgba, shape, pos)
            data = self.render('rgb_array') / 255
            
            #data = np.asarray(data, np.float32)
            dataset.append(data)
            shapeno = self.shapes.index(shape)
            shapeset.append(shapeno)
            plt.imsave('torm.png', data)
            pdb.set_trace()
            
        dataset = np.array(dataset)
        shapeset = np.array(shapeset)
        
        return dataset, shapeset
            
    def create_train_test_datasets(self, config):
        self.config = config
        self.batch_size = config['batch_size']
        self.n_training_instances = 3000
        self.n_testing_instances = 1000
        self.n_distractor_instances = 5000

        self.training_dataset, self.training_shapeset = self.create_dataset(self.n_training_instances)
        self.testing_dataset, self.testing_shapeset = self.create_dataset(self.n_testing_instances)
        self.distractor_dataset, self.distractor_shapeset = self.create_dataset(self.n_distractor_instances)

        del(self.scene, self.mjcf, self.camera)
        
    def training_batch_generator(self):
        for _ in range(self.batch_size):
            t = np.random.randint(self.n_training_instances)
            target = self.training_dataset[t]
            sampled_target_idx = self.training_shapeset[t]
            candidate_idx_set=[0,0,0,0,0]
            y = np.random.randint(self.config['n_classes'])
            y_label = np.eye(self.config['n_classes'])[y]
            candidate_set = [None for c in range(self.config['n_classes'])]
            for c in range(self.config['n_classes']):
                if c!=y:
                    cc = np.random.randint(self.n_distractor_instances)
                    candidate_set[c] = self.distractor_dataset[cc]
                    candidate_idx_set[c] = self.distractor_shapeset[cc]
                else:
                    candidate_set[c] = target
                    candidate_idx_set[c] = sampled_target_idx
            candidate_set = np.array(candidate_set)
            yield target, candidate_set, y_label, sampled_target_idx, candidate_idx_set
    
    def training_set_evaluation_generator(self):
        for idx in range(self.n_training_instances):
            target = self.training_dataset[idx]
            sampled_target_idx = self.training_shapeset[idx]
            candidate_idx_set=[0,0,0,0,0]
            y = np.random.randint(self.config['n_classes'])
            y_label = np.eye(self.config['n_classes'])[y]
            candidate_set = [None for c in range(self.config['n_classes'])]
            for c in range(self.config['n_classes']):
                if c!=y:
                    cc = np.random.randint(self.n_distractor_instances)
                    candidate_set[c] = self.distractor_dataset[cc]
                    candidate_idx_set[c] = self.distractor_shapeset[cc]
                else:
                    candidate_set[c] = target
                    candidate_idx_set[c] = sampled_target_idx
            candidate_set = np.array(candidate_set)
            yield target, candidate_set, y_label, sampled_target_idx, candidate_idx_set

    def testing_set_generator(self):
        for idx in range(self.n_testing_instances):
            target = self.testing_dataset[idx]
            sampled_target_idx = self.testing_shapeset[idx]
            candidate_idx_set=[0,0,0,0,0]
            y = np.random.randint(self.config['n_classes'])
            y_label = np.eye(self.config['n_classes'])[y]
            candidate_set = [None for c in range(self.config['n_classes'])]
            for c in range(self.config['n_classes']):
                if c!=y:
                    cc = np.random.randint(self.n_distractor_instances)
                    candidate_set[c] = self.distractor_dataset[cc]
                    candidate_idx_set[c] = self.distractor_shapeset[cc]
                else:
                    candidate_set[c] = target
                    candidate_idx_set[c] = sampled_target_idx
            candidate_set = np.array(candidate_set)
            yield target, candidate_set, y_label, sampled_target_idx, candidate_idx_set
    
if __name__=='__main__':
    env = MujocoWrapper()
    t0=time.time()
    env.create_dataset(100000)
    t1=time.time()
    print(t1-t0)
