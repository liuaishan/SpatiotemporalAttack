# SpatiotemporalAttack
# Attack Process
## Single Angle Attack
With House3D environment and specific dataset, you can simply run attack.py to generate texture changed object (with attack ability). The noise level is controlled by parameter epsilon, which is defined at line 123. 
## Multi Angle Attack
attack_multi_angle.py can also generate texture changed object. Different from attack.py, this file can automaticlly make embodied agent observe target object from various perspective, thanks to data_multi_angle.py. We adopt Das's code to get required information, including the location of agent and all objects in one house, etc., and we changed how the agent observe target object by changing the yaw of agent while considering if there is walls around a target object. (See function havewall and get_frames in data_multi_angle.py)

# Prerequisite
Run pip install -r requirements.txt to install all required packages. 
