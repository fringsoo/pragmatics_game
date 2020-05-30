import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time
from progressbar import *
#import meshpy.geometry
import pybullet_data
import pybullet as pb
pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())

class BulletWrapper():
    def __init__(self):
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

        self.shapes= ['box', 'sphere', 'cylinder', 'capsule', 'ellipsoid']
        #self.visualShapeId = -1

    def reset(self, rgba, shapeno, floorcolor, pos, ori, shapescale=2, fixed_size=True):
        pb.resetSimulation()
        #pb.loadURDF('assets/plane.urdf')
        if floorcolor==None:
            planeShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName='assets/plane.obj')
        else:
            planeShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName='assets/plane.obj', rgbaColor=floorcolor)
        pb.createMultiBody(baseVisualShapeIndex=planeShapeId, basePosition=[0, 0, 1])

        #if self.visualShapeId>=0:
        #    pb.removeBody(self.visualShapeId)
        shape = self.shapes[shapeno]
        if shape == 'sphere':
            if fixed_size:
                radius = 0.05 * shapescale
            else:
                radius = np.random.uniform(0.04,0.06) * shapescale
            visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_SPHERE, rgbaColor=rgba, visualFramePosition=pos, visualFrameOrientation=ori, radius=radius)
        elif shape == 'cylinder':
            if fixed_size:
                length=0.1 * shapescale
                radius=0.03 * shapescale
            else:
                length=np.random.uniform(0.08,0.12) * shapescale
                radius=np.random.uniform(0.02,0.04) * shapescale
            visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER, rgbaColor=rgba, visualFramePosition=pos, visualFrameOrientation=ori, length=length, radius=radius)
        elif shape == 'box':
            if fixed_size:
                halfExtents=[0.05 * shapescale, 0.05 * shapescale, 0.05 * shapescale]
            else:
                halfExtents=[np.random.uniform(0.04,0.06) * shapescale, np.random.uniform(0.04,0.06) * shapescale, np.random.uniform(0.04,0.06) * shapescale]
            visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_BOX, rgbaColor=rgba, visualFramePosition=pos, visualFrameOrientation=ori, halfExtents=halfExtents)
        elif shape == 'capsule':
            if fixed_size:
                meshScale = [0.07 * shapescale, 0.07 * shapescale, 0.07 * shapescale]
            else:
                meshScale=[np.random.uniform(0.06,0.08) * shapescale, np.random.uniform(0.06,0.08) * shapescale, np.random.uniform(0.06,0.08) * shapescale]
            #self.visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_CAPSULE, rgbaColor=rgba, radius=0.05, length=0.1)
            visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName='assets/capsule-obj/capsule.obj', rgbaColor=rgba, visualFramePosition=pos, visualFrameOrientation=ori, meshScale = meshScale)
        
        elif shape == 'ellipsoid':
            if fixed_size:
                meshScale = [0.1/0.6 * shapescale, 0.05/0.6 * shapescale, 0.05/0.6 * shapescale]
            else:
                meshScale = [np.random.uniform(0.08,0.12)/0.4 * shapescale, np.random.uniform(0.04,0.06)/0.4 * shapescale, np.random.uniform(0.04,0.06)/0.4 * shapescale]
            visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName='assets/spheres-obj/spheres.obj', rgbaColor=rgba, visualFramePosition=pos, visualFrameOrientation=ori, meshScale=meshScale)
        
        else:
            assert False

        '''
        elif shape == 'ellipsoid':
            if fixed_size:
                meshScale = [0.1/4000.0 * shapescale, 0.05/4000.0 * shapescale, 0.05/4000.0 * shapescale]
            else:
                meshScale = [np.random.uniform(0.08,0.12)/3000.0 * shapescale, np.random.uniform(0.04,0.06)/3000.0 * shapescale, np.random.uniform(0.04,0.06)/3000.0 * shapescale]
            visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName='assets/sphere-obj/sphere.obj', rgbaColor=rgba, visualFramePosition=pos, visualFrameOrientation=ori, meshScale=meshScale)
        '''
        
        pb.createMultiBody(baseVisualShapeIndex=visualShapeId, basePosition=[0, 0, 1])
        
    def render(self):
        camTargetPos = [0.5,0,1]
        cameraUp = [0,0,1]
        cameraPos = [1.5,0,2.7]
        pitch = -40.0
        yaw = 60
        roll=0
        upAxisIndex = 2
        camDistance = 0.7
        nearPlane = 0.01
        farPlane = 100
        fov = 60
        #viewMatrix = pb.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
        viewMatrix = pb.computeViewMatrix(cameraEyePosition=cameraPos, cameraTargetPosition=camTargetPos, cameraUpVector=cameraUp)
        aspect = 1
        projectionMatrix = pb.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
        
        img_arr = pb.getCameraImage(self.VIDEO_W, self.VIDEO_H, viewMatrix,projectionMatrix, shadow=1,lightDirection=[1,1,1])
        w=img_arr[0] #width of the image, in pixels
        h=img_arr[1] #height of the image, in pixels
        rgb=np.array(img_arr[2]).reshape(self.VIDEO_W, self.VIDEO_H,4)[:,:,:3] #color data RGB

        return rgb

    def reset_and_render(self, rgba, shapeno, floorcolor, pos, ori):
        self.reset(rgba, shapeno, floorcolor, pos, ori)
        data = self.render() / 255
        #data = np.asarray(data, np.float32)

        '''        
        if shapeno==4:
            print(rgba)
            plt.imsave(os.path.join('torm.png'), data)
            import pdb
            pdb.set_trace()
        '''
        return data

    def generate_params_1(self):
        rgba = self.rgbas[np.random.randint(len(self.rgbas))]
        shapeno = np.random.randint(len(self.shapes))
        floorcolor = [np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1),1]
        #floorcolor = None
        return rgba, shapeno, floorcolor

    def generate_params_2(self):
        pos = [np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(0.1,0.2)]
        ori = [np.random.uniform(0,0.1), np.random.uniform(0,0.1), np.random.uniform(0,0.1), np.random.uniform(0,2*np.pi)]
        return pos, ori

    def create_dataset(self, num, always_one_viewpoint):
        print('preparing dataset')
        dataset = []
        paramset = []
        pbar = ProgressBar()
        for _ in pbar(range(num)):
            rgba, shapeno, floorcolor = self.generate_params_1()
            pos, ori = self.generate_params_2()

            if always_one_viewpoint:
                ori = [0.05, 0.05, 0.05, np.pi]
            data = self.reset_and_render(rgba, shapeno,floorcolor, pos, ori)
            dataset.append(data)
            paramset.append([rgba, shapeno, floorcolor, pos, ori])
        dataset = np.array(dataset)
        return dataset, paramset
                        
    def create_train_test_datasets(self, config):
        self.config = config
        self.batch_size = config['batch_size']
        
        self.VIDEO_W = config["speaker_input_w"]
        self.VIDEO_H = config["speaker_input_h"]

        self.n_training_instances = config['n_training_instances']
        self.n_testing_instances = config['n_testing_instances']

        self.training_dataset, self.training_paramset = self.create_dataset(self.n_training_instances, False)
        self.testing_dataset, self.testing_paramset = self.create_dataset(self.n_testing_instances, False)

    def recreate_data_from_paramset(self, paramset, t):
        rgba = paramset[t][0]
        shapeno = paramset[t][1]
        floorcolor = paramset[t][2]
        pos, ori = self.generate_params_2()
        data = self.reset_and_render(rgba, shapeno, floorcolor, pos, ori)
        return data

    def get_one_instance_from_dataset(self, whichdataset, t, newobjectviewpoint, sameviewpoint, pretrain_idx_type):
        if whichdataset=='train':
            dataset = self.training_dataset
            paramset = self.training_paramset
            n_instances = self.n_training_instances
        elif whichdataset=='test':
            dataset = self.testing_dataset
            paramset = self.testing_paramset
            n_instances = self.n_testing_instances
        else:
            assert False
        
        if newobjectviewpoint:
            target = self.recreate_data_from_paramset(paramset, t)
        else:
            target = dataset[t]


        if pretrain_idx_type=='shape':
            sampled_target_idx = paramset[t][1]
        elif pretrain_idx_type=='color':
            sampled_target_idx = int(4*paramset[t][0][0] + 2*paramset[t][0][1] + paramset[t][0][2])
        elif pretrain_idx_type=='location':
            sampled_target_idx = paramset[t][3]
            #print(sampled_target_idx)
        else:
            assert False
        y = np.random.randint(self.config['n_classes'])
        y_label = np.eye(self.config['n_classes'])[y]
        candidate_set = [None for c in range(self.config['n_classes'])]
        candidate_idx_set=[0 for _c in range(self.config['n_classes'])]
        
        for c in range(self.config['n_classes']):
            if c!=y:
                cc = t
                while cc == t:
                    cc = np.random.randint(n_instances)
                
                if newobjectviewpoint:
                    candidate_set[c] = self.recreate_data_from_paramset(paramset, cc)
                else:
                    candidate_set[c] = dataset[cc]

                
                if pretrain_idx_type=='shape':
                    candidate_idx_set[c] = paramset[cc][1]
                elif pretrain_idx_type=='color':
                    candidate_idx_set[c] = int(4*paramset[cc][0][0] + 2*paramset[cc][0][1] + paramset[cc][0][2])
                elif pretrain_idx_type=='location':
                    candidate_idx_set[c] = paramset[cc][3]
                    #print(candidate_idx_set[c])
                else:
                    assert False
            else:
                if sameviewpoint:
                    candidate_set[c] = target
                else:
                    candidate_set[c] = self.recreate_data_from_paramset(paramset, t)
                candidate_idx_set[c] = sampled_target_idx
        candidate_set = np.array(candidate_set)

        #plt.imsave('torm_target.png', target)
        #plt.imsave('torm_candidate0.png', candidate_set[0])
        #('torm_candidate1.png', candidate_set[1])
        #print(y_label, sampled_target_idx, candidate_idx_set)
        #pdb.set_trace()
        
        return target, candidate_set, y_label, sampled_target_idx, candidate_idx_set
    
    def training_batch_generator(self, pretrain_idx_type='color', newobjectviewpoint=False, sameviewpoint=False):
        for _ in range(self.batch_size):
            t = np.random.randint(self.n_training_instances)
            yield self.get_one_instance_from_dataset('train', t, newobjectviewpoint, sameviewpoint, pretrain_idx_type)
    
    def training_set_evaluation_generator(self, pretrain_idx_type='color', newobjectviewpoint=False, sameviewpoint=False):
        for idx in range(self.n_training_instances):
            yield self.get_one_instance_from_dataset('train', idx, newobjectviewpoint, sameviewpoint, pretrain_idx_type)

    def testing_set_generator(self, pretrain_idx_type='color', newobjectviewpoint=False, sameviewpoint=False):
        for idx in range(self.n_testing_instances):
            yield self.get_one_instance_from_dataset('test', idx, newobjectviewpoint, sameviewpoint, pretrain_idx_type)

class BulletWrapper_yieldloc(BulletWrapper):
    def get_one_instance_from_dataset(self, whichdataset, t, newobjectviewpoint, sameviewpoint):
        t1, t2, t3, t4, t5 = super(BulletWrapper_yieldloc, self).get_one_instance_from_dataset(whichdataset, t, newobjectviewpoint, sameviewpoint)
        return t4, t5, t3, t4, t5
        
class BulletWrapper_yieldcolor(BulletWrapper):
    def get_one_instance_from_dataset(self, whichdataset, t, newobjectviewpoint, sameviewpoint):
        t1, t2, t3, t4, t5 = super(BulletWrapper_yieldcolor, self).get_one_instance_from_dataset(whichdataset, t, newobjectviewpoint, sameviewpoint)
        return np.eye(8)[t4], np.eye(8)[t5], t3, t4, t5

if __name__=='__main__':
    from config import agent_rnnconv_config_dict as config_dict
    env = BulletWrapper_new(124,124)
    t0=time.time()
    env.create_train_test_datasets(config_dict)
    for _ in range(100000):
        if _%100==0:
            print(_)
        t = np.random.randint(3000)
        env.get_one_instance('train', t, False)
    
    t1=time.time()
    print(t1-t0)