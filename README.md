# Reinforcement Learning Project 1: Navigation

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif 'Trained Agent'

## Introduction

In this project, the goal is to teach a reinforcement learning AI agent to navigate a Unity environment and play a simple game - collecting yellow bananas (with a reward of +1) and avoiding blue ones (with a reward of (-1). 

![Trained Agent][image1]

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes. However, to present more of a challenge and get a better sense of the agent's abilities, I've increased the average score threshold to +15 in my own implementation.

## Methods

There are four types of reinforcement learning algorithms used in this project:

- A **deep Q-network**, which is a neural network that approximates Q-values (measures of the 'quality' of a possible action) for each action that can be taken in a particular state.
- A **double deep Q-network**, which is meant to improve on a deep Q-network's tendency to overestimate action-value functions by using two separate sets of value functions: one to select an action, and another to estimate the action's effectiveness.
- A **dueling deep Q-network**, which is a network architecture that separates the estimation of a state value from the effectiveness of each action in that state, and then combines them in an aggregation layer. This allows the Q-network to learn which states are and aren't valuable, without needing to know the effect of each action in each state.
- **Prioritized experience replay**. This is an experience replay buffer that puts a priority on state-action experiences that have large differences between the prediction and the intended target, so the agent is more likely to replay those experiences and learn from its mistakes.

These algorithms are first used separately and then combined, to see how they affect the agent's performance both on their own and together.

## Experiments

The network architecture used in this project has two hidden layers, both of size `64`.

For the first round of trials, the replay buffer has a size of `100000`; a batch size of `64`; a discount factor (gamma) of `0.99`; a soft update factor for target parameters (tau) of `0.001`; a learning rate of `0.0005`; and the network is updated every `4` steps. The epsilon factor for the epsilon-greedy policy is started at `1.0`, then decreased at a decay rate of `0.995` to a minimum of `0.01`.

With these hyperparameters set, here are the results obtained from each round of training the agent:

| | DQN | Double DQN | Dueling DQN | Prioritized Replay | Double/Dueling | Double/PER | Dueling/PER | Double/Dueling/PER |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Episodes to Score **>15** | 740 | 571 | 632 | 659 | 701 | 682 | 689 | 740 |
| Time Taken | 7min 29s | 6min 11s | 7min 2s | 14min 54s | 8min 28s | 15min 51s | 16min 22s | 18min 7s |

![Comparison graph of first-round DQN results](/images/dqn-comparison.png)

According to the plot comparing the learning progress of the agents, there's not much difference in the network architectures, apart from the number of episodes and the length of time it took each agent to reach a mean score of 15 points or higher. On both these counts, the **double Q-network** has a clear advantage over the others, requiring 169 fewer episodes than the vanilla DQN to achieve the target score! However, all of the agents are able to learn to reach that score, and with roughly the same learning curve.

For the second round of trials, the batch size is increased to `128`, the epsilon decay rate is reduced to `.98`, and the epsilon minimum is increased to `0.02`. These three changes produce markedly different results.

| | DQN | Double DQN | Dueling DQN | Prioritized Replay | Double/Dueling | Double/PER | Dueling/PER | Double/Dueling/PER |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Episodes to Score **>15** | 440 | 436 | 472 | 482 | 500 | 755 | 472 | 428 |
| Time Taken | 5min 29s | 5min 45s | 6min 14s | 11min 13s | 7min 56s | 20min 15s | 12min 5s | 10min 31s |

![Comparison graph of second-round DQN results](/images/adjusted-dqn-comparison.png)

When the batch size is doubled, the epsilon decay is decreased, and the minimum epsilon is increased, nearly all of the AI agent architectures are able to reach a mean score higher than 15 points in hundreds of fewer episodes! The exception is the double DQN with prioritized experience replay, which performs much worse - it takes 755 episodes and more than 20 minutes to achieve the intended score, and there's a sharp drop-off in its learning rate close to the 500-episode mark. 

On the other hand, in this situation the top performer is the **dueling double Q-network with prioritized experience replay**, which takes only 428 episodes to solve the environment - although it needs 10.5 minutes to do this, nearly twice as long as it takes for the vanilla DQN to reach the target score. The second-place winner is the **double Q-network** again, which still takes fewer episodes and only slightly longer than the vanilla DQN to solve the environment.

It appears that, for this particular application, a **double Q-network** is the best choice of network architecture overall, for both learning ability and processing speed. Adjusting the hyperparameters has an even bigger impact on the performance of the agent, although some network architectures clearly benefit from certain changes more than others.

## Further Study

I would like to continue to refine the prioritized experience replay buffer; while it clearly requires more processing speed than other architectures, there may be ways to make it more effective.

 The change in the learning rate of each of the network architectures suggests that each type of reinforcement learning algorithm may have very different optimal hyperparameters. It would be worth it to research which ones work best for each type of architecture.

It might be also be interesting to attempt more complex network architectures, like a Noisy Network and even Rainbow DQN. On the other hand, it may be impractical to impose those algorithms on a task this simple - and so they might be better to save for another project.

## Usage

If you want to use this Unity environment on your own system, download the environment from one of the links below.  You need only select the environment that matches your operating system.

For the original banana-gathering game:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

For a more complex implementation, which learns from raw pixel data:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.