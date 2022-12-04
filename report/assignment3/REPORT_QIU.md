# CPEN502 Assignment 3

## Authors

Xuechun Qiu, 55766737

Tao Ma, 13432885

## Questions

### (4) The use of a neural network to replace the look-up table and approximate the Q-function has some disadvantages and advantages.

#### a) There are 3 options for the architecture of your neural network. Describe and draw all three options and state which you selected and why. (3pts)

The first option is ...

The second option is ...

The third option is ...

We chose ... one to implement. The advantages are ...

#### b) Show (as a graph) the results of training your neural network using the contents of the LUT from Part 2. Your answer should describe how you found the hyper-parameters which worked best for you (i.e. momentum, learning rate, number of hidden neurons). Provide graphs to backup your selection process. Compute the RMS error for your best results. (5 pts)

| Learning rates | 0.1 | 0.2 | 0.4 | 0.6 |
|----------------|-----|-----|-----|-----|
| RMS Error      |     |     |     |     |

| Momentum  | 0.3 | 0.6 | 0.9 | 1.0 |
|-----------|-----|-----|-----|-----|
| RMS Error |     |     |     |     |

| Numbers of hidden neurons | 5   | 10  | 15  | 20  |
|---------------------------|-----|-----|-----|-----|
~~| RMS Error                 |     |     |     |     |~~

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
