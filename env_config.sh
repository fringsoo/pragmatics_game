#ubuntu 16.04

#p2.xlarge (K80) no
#g3 (M60) yes
#g2 (K520) no
#nvidia driver cuda8
#aws gpu offscreen render
# https://towardsdatascience.com/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639
# https://virtualgl.org/Documentation/HeadlessNV
# To solve:
#QXcbIntegration: Cannot create platform OpenGL context, neither GLX nor EGL are enabled,
#QXcbIntegration: Cannot create platform offscreen surface, neither GLX nor EGL are enabled
#Preinstalled deep learning instances are not reliable for gl rendering as it has installed nvidia driver already.
######################################################################################

#! /bin/bash
sudo apt-get -y update
sudo apt-get -y upgrade
######################################################################################
#compile toolbox
sudo apt -y install cmake pkg-config patchelf g++ gcc make linux-generic
#python qt gl
sudo apt-get -y install python3-dev qtbase5-dev libqt5opengl5-dev libassimp-dev libglew-dev
#x server
sudo apt install -y xserver-xorg mesa-utils libgl1-mesa-dev
#blas
#sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
######################################################################################

#pyconfig.h
echo 'export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/include/python3.5' >> ~/.bashrc
source ~/.bashrc
######################################################################################

# sudo echo 'blacklist nouveau'  | sudo tee -a /etc/modprobe.d/blacklist.conf
# sudo echo 'options nouveau modeset=0'  | sudo tee -a /etc/modprobe.d/blacklist.conf
# sudo echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
# sudo update-initramfs -u
# cd ~
######################################################################################

#nvidia driver(for m60 and k80)
#wget http://us.download.nvidia.com/XFree86/Linux-x86_64/384.66/NVIDIA-Linux-x86_64-384.66.run

# wget http://us.download.nvidia.com/tesla/384.183/NVIDIA-Linux-x86_64-384.183.run
# sudo /bin/bash  ./NVIDIA-Linux-x86_64-384.183.run --accept-license --no-questions --ui=none

# #cuda
# #wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
# wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
# chmod +X cuda_9.0.176_384.81_linux-run
# sudo sh cuda_9.0.176_384.81_linux-run --override --no-opengl-libs
# #the no-opengl-libs is very important! otherwise it is same as the preinstalled deep learning instance!
# #(Type n to NOT Install NVIDIA Accelerated Graphics Driver for Linux-x86_64)
# echo 'export PATH=/usr/local/cuda-9.0/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
# source ~/.bashrc
# cd ~/NVIDIA_CUDA-9.0_Samples/
# make
# cd ~/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release/
# ./deviceQuery  # see your graphics card specs
# ./bandwidthTest # check if its operating correctly
# #(Both should state they PASS)
# cd ~

# #cudnn
# #wget http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/cudnn-8.0-linux-x64-v7.tgz
# wget http://developer.download.nvidia.com/compute/redist/cudnn/v7.3.0/cudnn-9.0-linux-x64-v7.3.0.29.tgz
# tar -zxf cudnn-9.0-linux-x64-v7.3.0.29.tgz
# sudo cp cuda/lib64/* /usr/local/cuda/lib64/
# sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
######################################################################################

#rdp or ssh x11 connect
sudo apt install -y ubuntu-desktop xrdp
#after install these, you can have xterm, xinit..
#In the AWS Dashboard edit the Security Group for the EC2 instance and allow inbound TCP connections on port 3389. (port for rdp)
#sudo passwd ubuntu
#rdp require passwd login, however the original aws ubuntu only has key pair login
sudo service xrdp restart
ps aux | grep x  #check the xrdp service process
#then use rdp on windows to login

#or we could use 
#ssh -X ...
netstat -lnp
######################################################################################
#nvidia-smi
#nvidia-xconfig --query-gpu-info
#lspci | grep VGA
#cat /proc/driver/nvidia/version
#nvcc -V

#sudo nvidia-xconfig -a --allow-empty-initial-configuration --virtual=1920x1200 --busid=PCI:0:30:0 #--use-display-device=None 
#if p2 instance
#cd /etc/X11
#sudo cp XF86Config xorg.conf

sudo /usr/bin/X :0 &
#DISPLAY=:0 glxinfo | grep render
#DISPLAY=:0 glxgears
#DISPLAY=:0 python3
######################################################################################

echo "defscrollback 10000" > ~/.screenrc
sudo apt-get -y install python3-pip
sudo apt-get -y install python3-tk
sudo apt-get -y install virtualenv

virtualenv --no-site-packages -p python3 ~/venv
#virtualenv --no-site-packages -p python ~/venv
#conda create -n ~/venv python==3.6.0

#source ~/venv/bin/activate
#conda activate venv
#pip install -r ~/pragmatics_game/requirements.txt
#pip install -r ~/pragmatics_game/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple #china source
#export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

######################################################################################
#sudo pip3 install gym
#sudo pip3 install roboschool

#git clone https://github.com/openai/roboschool.git
#cd roboschool
#./install_boost.sh
#./install_bullet.sh
#source exports.sh
#pkg-config --cflags Qt5OpenGL assimp bullet
#cd roboschool/cpp-household && make clean && make -j4 && cd ../..
#pip install -e .
#sudo pip install roboschool #can not be g2 aws ec2 instance

#mujoco
#wget https://www.roboti.us/download/mjpro150_linux.zip
#mkdir ~/.mujoco
#unzip mjpro150_linux.zip -d ~/.mujoco
#wget https://www.roboti.us/getid/getid_linux
#chmod 755 ./getid_linux
#./getid_linux
#copy mjkey.txt to ~/.mujoco
#echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xdjf/.mujoco/mjpro150/bin' >> ~/.bashrc
#echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384' >> ~/.bashrc
#pip install -U 'mujoco-py<1.50.2,>=1.50.1'
######################################################################################

#cd ~/pragmatics_game
#get visa data set from https://drive.google.com/file/d/1xtExj3dHWpLQnUfazzkmsd70hIh5N6Qw/view?usp=sharing

#git submodule add https://github.com/NickLeoMartin/emergent_comm_rl.git
#git submodule add https://github.com/Russell91/apollocaffe.git
#make all
#export LD_LIBRARY_PATH= $LD_LIBRARY_PATH:APOLLO_ROOT/build/lib (written into .bashrc??)
#another libcaffe.so was called, check if the caffe bulid/lib path is in the LD_LIBRARY_PATH, if so, just put it after the apollocaffe one.
#sudo apt-get -y install libhdf5-serial-dev

######################################################################################

###zoo###
#sudo add-apt-repository ppa:jonathonf/python-3.6
#sudo apt-get update
#sudo apt-get install python3.6
#virtualenv --no-site-packages -p python3.6 ~/venv36
#source ~/venv36/bin/activate
#pip install git+https://github.com/facebookresearch/EGG.git
#python -m egg.zoo.mnist_autoenc.train --vocab=10 --n_epochs=50