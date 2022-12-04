# CPEN502 Assignment 3

## Authors

Xuechun Qiu, 55766737

Tao Ma, 13432885

## Questions

### (4) The use of a neural network to replace the look-up table and approximate the Q-function has some disadvantages and advantages.

#### a) There are 3 options for the architecture of your neural network. Describe and draw all three options and state which you selected and why. (3pts)

### Option 1

<img src="/Users/michaelma/Desktop/Workspace/Screenshots/Snipaste_2022-12-04_08-15-28.png" alt="Snipaste_2022-12-04_08-15-28" style="zoom:30%;" />

The first option is to utilize a straightforward neural network as a form of predictive modelling in order to derive the Q value for the state action pair input. To be more specific, this option takes in an input vector that contains a state vector and an action vector, and it then generates a Q value that is associated with the state-action pair.

### Option 2

<img src="/Users/michaelma/Desktop/Workspace/Screenshots/Snipaste_2022-12-04_08-17-32.png" alt="Snipaste_2022-12-04_08-17-32" style="zoom:30%;" />

The second option is to utilize a simple neural network to compute the corresponding Q value vectors of all the possible actions for the corresponding state input. More specifically, this neural network computes a vector of $Q \ value \in {\Bbb R}^{Num\ of \ Actions}$ from a state vectors consist of  $state\ \in {\Bbb R}^{Num\ of \ States}$. This architecture also enable adapting the neural network into a multi-label classification problem. To be more explicit, rather than considering the neural network as a model that predicts the Q values for each action, we may view the neural network as an action selector that chooses the best action based on the state input. This allows for a more accurate representation of the network's capabilities. However, it is important to take into consideration that the final layer should be changed to a softmax layer in this scenario. 

### Option 3

<img src="/Users/michaelma/Desktop/Workspace/Screenshots/Snipaste_2022-12-04_08-15-28.png" alt="Snipaste_2022-12-04_08-15-28" style="zoom:30%;" />

The third option, combining the previous two options, utilize a straightforward neural network as a form of predictive modelling in order to derive the Q value of an action for the state input. This ensemble method prepare $n$ neural networks for $n$ possible action. When a state vector is being fed into the model, $n$ neural networks work together to compute the $Q \ value \in {\Bbb R}^{Num\ of \ Actions}$ for each action individually.

In the context of this assignment, we have implemented all three of them, and we have decided to report using Option 3 as our model. In this part of the article, we will begin by discussing these three possibilities and explaining the thought process that went into reaching our decision. These three choices each come with their own set of advantages to consider. For instance, Option 1 enables an easier implementation, and the computation is uncomplicated; however, concatenating the state vector and the action vector together could result in an issue due to the fact that they come from distinct distributions. Option 2 is intended to be a standardised approach to the problem of multi-label prediction; but, by choosing this option, you will lose more granular control over each operation. The accompanying graphic demonstrates that each action converges at a different rate, and it is possible that combining several actions will result in unexpected behaviour. In the end, we decided that Option 3 would be the best model to report. Option 3 does have some severe complexity issues, such as the fact that we have to run $n$ times more predictions for $n$ actions, but it does provide clearer illustrations for the analysis of each action and provides for better control over the activities. 

<img src="/Users/michaelma/Downloads/test(1).png" alt="test(1)" style="zoom:30%;" />

#### b) Show (as a graph) the results of training your neural network using the contents of the LUT from Part 2. Your answer should describe how you found the hyper-parameters which worked best for you (i.e. momentum, learning rate, number of hidden neurons). Provide graphs to backup your selection process. Compute the RMS error for your best results. (5 pts)

| Learning rates (Momentum = 0.0, Hidden = 10) | 0.1   | 0.2  | 0.4  | 0.6  |
| -------------------------------------------- | ----- | ---- | ---- | ---- |
| RMS Error                                    | 0.439 |      |      |      |

| Momentum (Learning rate = 0.1, Hidden = 10) | 0.3  | 0.6  | 0.9  | 1.0  |
| :------------------------------------------ | ---- | ---- | ---- | ---- |
| RMS Error                                   |      |      |      |      |

| Numbers of hidden neurons (Learning rate = 0.1, Momentum = 0.0) | 5   | 10  | 15  | 20  |
|---------------------------|-----|-----|-----|-----|
| RMS Error |     |     |     |     |

#### c) Comment on why theoretically a neural network (or any other approach to Q-function approximation) would not necessarily need the same level of state space reduction as a look up table. (2 pts)

In assignment 2, the look-up table which record all states and actions will need ... space, but if we perform space
reduction, it will only take ... memory spaces.

However, neural network set the state and value as input values directly, and we only need to save the weights that
connected to neurons. In this case, we can save more memory.

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

The possible approaches to monitor the learning performance of neural net may be winning rate, and the count of bullet
hit.

The winning rate is an important factor when evaluating the performance. If the winning rate increases after some
rounds, it will imply that the model performs better than previous.

The count of bullet hit might also reflect the learning performance. Based on the rule, the robot which can hit more
will be more likely to win the game.

#### e) At each time step, the neural net in your robot performs a back propagation using a single training vector provided by the RL agent. Modify your code so that it keeps an array of the last say n training vectors and at each time step performs n back propagations. Using graphs compare the performance of your robot for different values of n. (4 pts)

### (6) Overall Conclusions

#### a) This question is open-ended and offers you an opportunity to reflect on what you have learned overall through this project. For example, what insights are you able to offer with regard to the practical issues surrounding the application of RL & BP to your problem? E.g. What could you do to improve the performance of your robot? How would you suggest convergence problems be addressed? What advice would you give when applying RL with neural network based function approximation to other practical applications? (4 pts)

- What could you do to improve the performance of your robot?


- How would you suggest convergence problems be addressed?


- What advice would you give when applying RL with neural network based function approximation to other practical
  applications?

#### b) Theory question: Imagine a closed-loop control system for automatically delivering anesthetic to a patient under going surgery. You intend to train the controller using the approach used in this project. Discuss any concerns with this and identify one potential variation that could alleviate those concerns. (3 pts)

The main concern is the ethical issue. Even for a well-trained model, it still have a chance to make errors during the
surgery, which may threaten patient's life.
Another concern is that input values can be extremely complicated since you need to consider different factor such as
blood pressure, age, allergies, etc.

The potential variation is that we can use model to give more information and assist doctors instead of making
decisions.

## Appendix
