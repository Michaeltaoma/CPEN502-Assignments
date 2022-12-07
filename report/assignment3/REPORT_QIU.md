# CPEN502 Assignment 3

## Authors

Xuechun Qiu, 55766737

Tao Ma, 13432885

## Questions

### (4) The use of a neural network to replace the look-up table and approximate the Q-function has some disadvantages and advantages.

#### a) There are 3 options for the architecture of your neural network. Describe and draw all three options and state which you selected and why. (3pts)

<img src="img/4.a.1.png" alt="single Q value output" style="zoom:55%;" />

The first option is straightforward, which is given state-action pair as inputs, and then output a single Q value.

<img src="img/4.a.2.png" alt="single Q value output" style="zoom:55%;" />

The second option is given states as inputs, using a neural network to compute the Q value of all the possible actions. In this way, we consider the neural network as an action selector which will choose the best action based on the state inputs.

<img src="img/4.a.3.png" alt="single Q value output" style="zoom:55%;" />

The third option combines the previous two architectures, give neural networks as inputs, and output single Q value. If we have n actions, this model will use n neural networks to compute the Q value for each action.

Comparing these 3 architectures, there are pros and cons for all three options. Option 1 is an easier implementation, but combining the state and action together may result in issues which will reduce the model accuracy. The option 2 may converges differently for each action, which may lead to unexpected behaviours when combining several actions. The option 3 is a little complicated and time consuming, but it may be more accurate since we use one NN for one action. We implemented three architectures but we chose the second one to generate graphs using in the report.


#### b) Show (as a graph) the results of training your neural network using the contents of the LUT from Part 2. Your answer should describe how you found the hyper-parameters which worked best for you (i.e. momentum, learning rate, number of hidden neurons). Provide graphs to backup your selection process. Compute the RMS error for your best results. (5 pts)


| Learning rates | 0.05  | 0.1   | 0.2   | 0.4   |
|----------------|-------|-------|-------|-------|
| RMS Error      | 0.446 | 0.450 | 0.471 | 0.497 |

<img src="img/4.b.learning.rate.png" alt="compare different learning rates" style="zoom:55%;" />

As we can see from the figure, RMS error will be smaller when learning rate is smaller. The possible reason is that a large learning rate allows the model to learn faster, at the cost of arriving on a sub-optimal final set of weights. A smaller learning rate may allow the model to learn a more optimal or even globally optimal set of weights but may take significantly longer to train. So to balance the tradeoff, we will set learning rate to 0.1.


| Momentum  | 0.0   | 0.3   | 0.6   | 0.9   |
|-----------|-------|-------|-------|-------|
| RMS Error | 0.471 | 0.484 | 0.507 | 1.370 |

<img src="img/4.b.momentum.png" alt="compare different learning rates" style="zoom:55%;" />

As we can see from the figure, RMS error will be smaller when momentum is smaller. The possible reason is that, larger momentum value means more random actions. So based on the figure here, we will choose 0.0 as our momentum.


| Numbers of hidden neurons | 5     | 10    | 15    | 20    |
|---------------------------|-------|-------|-------|-------|
| RMS Error                 | 0.479 | 0.471 | 0.474 | 0.472 |

<img src="img/4.b.hidden.neurons.png" alt="compare different learning rates" style="zoom:55%;" />

As we can see from the figure, RMS error reaches the smallest value when the number of hidden neurons is 10, so we will just go for it.


#### c) Comment on why theoretically a neural network (or any other approach to Q-function approximation) would not necessarily need the same level of state space reduction as a look up table. (2 pts)

In assignment 2, the look-up table which record all states and actions before space reduction, which is very space consuming. However, neural networks and other methods for approximating Q-functions can be more flexible and efficient than lookup tables because they can generalize from examples, rather than needing to have explicit entries for every possible state. This indicates that they can frequently achieve good performance with a significantly smaller collection of states, or even by using continuous states as opposed to discrete ones. Additionally, finding accurate approximations of the Q-function can be made simpler by learning useful representations of the state space using neural networks or any other function approximations. They may be able to function well even in state spaces that are big or complex because to this.


### (5) Hopefully you were able to train your robot to find at least one movement pattern that results in defeat of your chosen enemy tank, most of the time.

#### a) Identify two metrics and use them to measure the performance of your robot with online training. I.e. during battle. Describe how the results were obtained, particularly with regard to exploration? Your answer should provide graphs to support your results. (5 pts)

Two metrics that could be used to measure the performance of a robot with online training are winning rates and RMS errors. Winning rates measure the percentage of battles won by the robot, while RMS errors measure the average difference between the predicted and actual values for a given dataset.

The robot would need to take part in several battles and monitor its performance to get these results. By dividing the total number of victories by the total number of battles, the robot could then determine its winning percentages. The robot would have to forecast the surroundings or its actions in order to calculate RMS errors, then compare these predictions to the actual results. The square root of the average of the squares representing the discrepancies between the expected and actual values is known as the RMS error.

Winning rates and RMS errors can be related to exploration in several ways. A robot that investigates more, for instance, might have a greater winning percentage since it has had the chance to learn more about its surroundings and develop useful techniques. A robot that investigates less, however, might have a smaller RMS error because it is more certain of its predictions and makes less errors as a result. In general, a key component of training a robot with online learning is striking the correct balance between exploration and exploitation.

Metric 1: Winning Rates

<img src="img/5.a.1.png" alt="winning rates" style="zoom:15%;" />


Metric 2: RMS Errors

<img src="img/5.a.2.png" alt="rms errors" style="zoom:20%;" />


#### b) The discount factor can be used to modify influence of future reward. Measure the performance of your robot for different values of  and plot your results. Would you expect higher or lower values to be better and why? (3 pts)

The discount factor is a parameter which determines the relative importance of future rewards. A discount factor of 0 means that future rewards are not considered at all, while a discount factor of 1 means that all future rewards are considered equally.

<img src="img/5.b.png" alt="compare different discount factor" style="zoom:55%;" />

Higher values of the discount factor are often thought to be better since they give future rewards more weight. This can aid the robot in learning longer-term tactics that might not yield results right away but might do so in the long run. Lower values of the discount factor, on the other hand, might be preferable in some situations where rewards are very erratic or the environment is changing quickly since they can assist the robot react to the changing environment more quickly.

#### c) Theory question: With a look-up table, the TD learning algorithm is proven to converge – i.e. will arrive at a stable set of Q-values for all visited states. This is not so when the Q-function is approximated. Explain this convergence in terms of the Bellman equation and also why when using approximation, convergence is no longer guaranteed. (3 pts)

The Bellman equation has a singular solution for the Q-values of each state, demonstrating that the TD learning algorithm converges when employing a look-up table. This indicates that the algorithm will eventually reach the right Q-values for all visited states as it updates the Q-values based on the Bellman equation.

The Bellman equation is a formula that connects a state in a Markov decision process to the actions that can be taken to get to that state and the states that can be reached by doing so. It is defined as follows: 

```math
Q(s,a) = R(s,a) + γ * max(Q(s',a'))
```
where Q(s,a) is the Q-value for state s and action a, R(s,a) is the immediate reward for taking action a in state s, γ is the discount factor, and s' and a' are the next state and action, respectively.

There is only one Q-value that fulfils the Bellman equation for any given state s and action a since each state's Q-values have a unique solution. This is only accurate when a look-up table accurately represents the Q-function.

However, convergence is no longer assured when the Q-function is estimated using a function approximation like a neural network. This is due to the fact that when the Q-function is precisely represented by a look-up table, the Bellman equation only has one unique solution for the Q-values of each state. The Q-function is simply a rough approximation of the real Q-function when approximation is used. This implies that the TD learning algorithm could not always converge to the right Q-values.


#### d) When using a neural network for supervised learning, performance of training is typically measured by computing a total error over the training set. When using the NN for online learning of the -function in robocode this is not possible since there is no a-priori training set to work with. Suggest how you might monitor learning performance of the neural net now. (3 pts)

The possible approaches to monitor the learning performance of neural net may be winning rate, the count of bullet hit, and the learning progress over time.

The winning rate is an important factor when evaluating the performance. If the winning rate increases after some rounds, it will imply that the model performs better than previous. 

The count of bullet hit might also reflect the learning performance. Based on the rule, the robot which can hit more will be more likely to win the game. 

Tracking the learning progress over time may be another strategy for assessing the neural network's performance during learning. To achieve this, the neural network could be operated in real-time while its performance on a collection of held-out episodes was regularly recorded. Then, the performance may be charted over time to demonstrate how it advances as the neural network keeps picking up new skills.


#### e) At each time step, the neural net in your robot performs a back propagation using a single training vector provided by the RL agent. Modify your code so that it keeps an array of the last say n training vectors and at each time step performs n back propagations. Using graphs compare the performance of your robot for different values of n. (4 pts)

<img src="img/5.e.png" alt="rms errors" style="zoom:20%;" />

We chose n = 10 and n = 100 to draw the graph, we can see that n = 10 is not learning, while the winning rate increases gradually when n = 100. Using a larger amount of n will allow the neural network to learn more effectively from a larger amount of training data. 


### (6) Overall Conclusions

#### a) This question is open-ended and offers you an opportunity to reflect on what you have learned overall through this project. For example, what insights are you able to offer with regard to the practical issues surrounding the application of RL & BP to your problem? E.g. What could you do to improve the performance of your robot? How would you suggest convergence problems be addressed? What advice would you give when applying RL with neural network based function approximation to other practical applications? (4 pts)

- What could you do to improve the performance of your robot?

I advise utilising more sophisticated RL algorithms, like deep Q-learning, which are better able to handle big and complex state spaces, to boost the robot's performance. Additionally, I would advise employing more advanced function approximations like deep neural networks, which can develop more potent and adaptable Q-function representations.

- How would you suggest convergence problems be addressed?

I would suggest using regularization techniques, such as weight decay and dropout, to prevent the neural network from overfitting to the training data.

- What advice would you give when applying RL with neural network based function approximation to other practical applications?

I would experimenting with various function approximators, such as shallow neural networks and linear models, to recognise their advantages and disadvantages. The performance and resilience of the learnt policy can be enhanced by combining exploration and regularisation methods with adaptive learning rates and momentum.


#### b) Theory question: Imagine a closed-loop control system for automatically delivering anesthetic to a patient under going surgery. You intend to train the controller using the approach used in this project. Discuss any concerns with this and identify one potential variation that could alleviate those concerns. (3 pts)

The main concern is the ethical issue. It may not be safe to utilise reinforcement learning in this setting, which is one potential worry with employing the method from this project to train a closed-loop control system for autonomously giving anaesthesia to a patient having surgery. This is due to the fact that algorithms used in reinforcement learning are created to maximise a specific reward function, which could not be consistent with the objectives of an anaesthetic delivery system. The training algorithm's reward function, for instance, can put the patient's comfort ahead of other crucial considerations like the patient's vital signs or the dosage of the anaesthetic.

Another concern is that input values can be extremely complicated since you need to consider different factor such as blood pressure, age, allergies, etc. 

One potential variant to address this issue is to employ a hybrid strategy that combines reinforcement learning with other methods like model-based control. Another potential variation is that we can use model to give more information and assist doctors instead of making decisions.

## Appendix
