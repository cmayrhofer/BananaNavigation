[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Banana Navigation

In this repository, we provide a code implementation of a [DQN agent](https://www.nature.com/articles/nature14236) which solves a modified version of the [Banana Collector](https://youtu.be/heVMs3t9qSk) [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) environment of [unity](unity3d.com). The [modifications](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) were provided by the [Udacity](https://eu.udacity.com/) team responsible for the [Deep Reinforcment Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). We refer to the modified Banana Collector as Banana Navigation. 

![Trained Agent][image1]

The code solving the Banana Navigation environment is split up in into three parts:
* The `dqn_agent.py`file provides the implementation of the (dqn) `Agent` class and  a `ReplayBuffer` class.
* In `model.py` the deep neural network is defined which we use to learn an approximation of the [action-value function](https://en.wikipedia.org/wiki/Reinforcement_learning) of the agent
* The training of the agent, visuallisation of the agents performance during learning (i.e. learning curve), and the agent playing a test episode after training are all done in the jupyter notebook `Navigation.ipynb`. _All the necessary information on how to run the code are provided within this notebook._

Furthermore, the repository contains also the saved weights of a trained agent. They can be found in the file `checkpoint.pth` and loaded via [pytorch](pytorch.org) to a neural network with the same architecture as the one in `model.py`. In the `Report.md` file you can find further informations on the learning algorithm used to solve this environment and how it is implemented in the above listed files.



**Note** that in addition to these files there are some with names ending on `_visual` and `_Pixels`. These files try to solve the environment for the case in which the states of the environments are provided visually and not in terms of a state vector which has all the frames already "preprocessed" into a 37 dimensional state-vector. For more information see the Report.md file.


## Details of the RL Environment

As the above GIF-animation adumbrates, the agent is confined to a large square surounded by a wall acting as the boundary. The agent is supposed to navigate through the square and collect as many yellow bananas as possible without running into the purple ones. The task is episodic and one episode lasts for 300 steps.

* _`States`_: the state space is 37 dimensional and contains the agent's velocity plus a ray-based perception of objects in front of the agent;
* _`Actions`_: the agent can take four actions: __`0`__ - move forward, __`1`__ - move backward, __`2`__ - turn left, __`3`__ - turn right;
* _`Reward`_: the agent obtains a reward of $+1$ if he collects a yellow banana, a reward of $-1$ for a purple one, and $0$ else.

The environment is to be considered solved if the average score of the agent over the last 100 episodes is greater equal 13.


## Getting Started

To run the code provided in this repository, you must have python 3.6 or higher installed. In addition, you will need to have the packages: [numpy](http://www.numpy.org/), [matplotlib](https://matplotlib.org/), [torch](https://pytorch.org/) (including its dependencies) installed. Then follow the next three steps. Afterwards you should be able to run the `Navigation.ipynb` without any dependencies errrors.

1. Follow the steps in [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) to install the Unity ML-Agents Toolkit.
2. Download the Banana Navigation environment from one of the links below.  Select the environment which matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
3. Place the file in the folder where you also placed the `dqn_agent.py` file , the `model.py` file, and the `Navigation.ipynb` notebook and unzip (or decompress) the file.

If you are still having issues after following this steps regarding the dependencies then please check out the more throughly configuration [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).


### Marginal Comment

As mentioned already above the environment is a modefication of the Banana Collector environment of Unity and was designed as a multiagent test ground. The modifications where done by the [Udacity](https://eu.udacity.com/) team responsible for the [Deep Reinforcment Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program (to provide one of three projects in the DRL Nanodegree program which have to be successfully complited in order to pass the course). The modified version is however a single agent environment, as opposed to the original multiplayer environment, which makes the appearence of the following two lines of code
```
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```
compared to standard single agent environments as you can find them on [OpenAI-Gym](https://gym.openai.com/) obvious.
