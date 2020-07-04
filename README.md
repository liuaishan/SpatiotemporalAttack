# Spatiotemporal Attacks for Embodied Agents

Code for the paper 

[Spatiotemporal Attacks for Embodied Agents](https://arxiv.org/pdf/2005.09161.pdf)
<br>[Aishan Liu](https://liuaishan.github.io/), Tairan Huang, [Xianglong Liu](http://sites.nlsde.buaa.edu.cn/~xlliu/), Yitao Xu, Yuqing Ma, [Xinyun Chen](https://jungyhuk.github.io/), Stephen Maybank, [Dacheng Tao](https://www.sydney.edu.au/engineering/about/our-people/academic-staff/dacheng-tao.html)
<br>European Conference on Computer Vision (ECCV), 2020

<img src="https://github.com/liuaishan/SpatiotemporalAttack/blob/master/frontpage.jpg" width="70%">

<img src="https://github.com/liuaishan/SpatiotemporalAttack/blob/master/demo.gif" width="70%">

In this work, we take the first step to study adversarial attacks for embodied agents. In particular, we generate spatiotemporal perturbations to form 3D adversarial examples, which exploit the interaction history in both the temporal and spatial dimensions. Regarding the temporal dimension, since agents make predictions based on historical observations, we develop a trajectory attention module to explore scene view contributions, which further help localize 3D objects appeared with highest stimuli. By conciliating with clues from the temporal dimension, along the spatial dimension, we adversarially perturb the physical properties (e.g., texture and 3D shape) of the contextual objects that appeared in the most important scene views.

For questions regarding implementation contact [Yitao Xu](xuyitao@buaa.edu.cn)

# To Perform Attacks

## Single View Attack
With [EQA-v1](https://github.com/facebookresearch/EmbodiedQA) dataset, you can simply perform the spatiotemporal attack by running the following code to generate texture changed object (with single view). The noise level is controlled by parameter epsilon.
```python
python attack.py
```

## Multi Views Attack
To further perfomr multi-view attacks, you can simple run:
```python
python attack_multi_angle.py
```
Different from attack.py, this file can automaticlly make embodied agent observe target object from various views, thanks to data_multi_angle.py.

## Postprocess
Once the attack process finishes, you will get many .obj file, each representing a changed object (with strong attack ability). To put these objects into one same house, run
```python
python code_all.py
```
to generate a new .obj file for the same house with changed object in it but all other objects remain the same. 
Now you can use API function in House3D environment to see the effect of our attack algorithm.


# Prerequisite
Run the following code to install all required packages.

```python
pip install -r requirements.txt
```

# Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{Liu2020Spatiotemporal,
    title={Spatiotemporal Attacks for Embodied Agents},
    author={Liu, Aishan and Huang, Tairan and Liu, Xianglong and Xu, Yitao and Ma, Yuqing and Chen, Xinyun and Maybank, Stephen and Tao, Dacheng},
    Booktitle = {European Conference on Computer Vision},
    year={2020}
}
```
