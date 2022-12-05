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


Comparing these 3 architectures, we chose the third one to implement to get the better result. The reason is that, option 1 is an easier implementation, but combining the state and action together may result in issues which will reduce the model accuracy. The option 2 may converges differently for each action, which may lead to unexpected behaviours when combining several actions. The option 3 is a little complicated, but it will be more accurate since we use one NN for one action.


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

In assignment 2, the look-up table which record all states and actions will need ... space, but if we perform space reduction, it will only take ... memory spaces.

However, neural network set the state and value as input values directly, and we only need to save the weights that connected to neurons. In this case, we can save more memory.


### (5) Hopefully you were able to train your robot to find at least one movement pattern that results in defeat of your chosen enemy tank, most of the time.

#### a) Identify two metrics and use them to measure the performance of your robot with online training. I.e. during battle. Describe how the results were obtained, particularly with regard to exploration? Your answer should provide graphs to support your results. (5 pts)

We chose winning rates and RMS errors as our two metrics. 

Metric 1: Winning Rates


Metric 2: RMS Errors



#### b) The discount factor can be used to modify influence of future reward. Measure the performance of your robot for different values of  and plot your results. Would you expect higher or lower values to be better and why? (3 pts)


#### c) Theory question: With a look-up table, the TD learning algorithm is proven to converge – i.e. will arrive at a stable set of -values for all visited states. This is not so when the -function is approximated. Explain this convergence in terms of the Bellman equation and also why when using approximation, convergence is no longer guaranteed. (3 pts)

```math
\sqrt{3}
```

#### d) When using a neural network for supervised learning, performance of training is typically measured by computing a total error over the training set. When using the NN for online learning of the -function in robocode this is not possible since there is no a-priori training set to work with. Suggest how you might monitor learning performance of the neural net now. (3 pts)

The possible approaches to monitor the learning performance of neural net may be winning rate, and the count of bullet hit.

The winning rate is an important factor when evaluating the performance. If the winning rate increases after some rounds, it will imply that the model performs better than previous. 

The count of bullet hit might also reflect the learning performance. Based on the rule, the robot which can hit more will be more likely to win the game. 


#### e) At each time step, the neural net in your robot performs a back propagation using a single training vector provided by the RL agent. Modify your code so that it keeps an array of the last say n training vectors and at each time step performs n back propagations. Using graphs compare the performance of your robot for different values of n. (4 pts)



### (6) Overall Conclusions

#### a) This question is open-ended and offers you an opportunity to reflect on what you have learned overall through this project. For example, what insights are you able to offer with regard to the practical issues surrounding the application of RL & BP to your problem? E.g. What could you do to improve the performance of your robot? How would you suggest convergence problems be addressed? What advice would you give when applying RL with neural network based function approximation to other practical applications? (4 pts)

- What could you do to improve the performance of your robot?


- How would you suggest convergence problems be addressed?


- What advice would you give when applying RL with neural network based function approximation to other practical applications?




#### b) Theory question: Imagine a closed-loop control system for automatically delivering anesthetic to a patient under going surgery. You intend to train the controller using the approach used in this project. Discuss any concerns with this and identify one potential variation that could alleviate those concerns. (3 pts)

The main concern is the ethical issue. Even for a well-trained model, it still have a chance to make errors during the surgery, which may threaten patient's life. 
Another concern is that input values can be extremely complicated since you need to consider different factor such as blood pressure, age, allergies, etc. 

The potential variation is that we can use model to give more information and assist doctors instead of making decisions.

## Appendix
