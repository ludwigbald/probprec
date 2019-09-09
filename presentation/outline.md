# Investigating Probabilistic Preconditioning on Deep Learning using DeepOBS

## About me (short)
- My name is Ludwig Bald, this is my bachelor's thesis in cognitive science.
- I chose this topic because I wanted to get to know deep learning from the inside. I had never done Deep Learning before and wanted to learn it hands-on.
- In this talk I will talk about the science, but also about the process I used.
- If you have questions, ask them right away!

## Background

### Machine learning
- An algorithm that learns from experience to get better at a task.
- In deep learning, at least what I'm talking about today, that means the algorithm is trying to learn the optimal parameters, or to find the optimal point in parameter space.
- There is a measure called the loss function, which tells us how good a certain point is.
- Image of 2-dimensional problem

### Optimization using GD
- Now, assume we don't know this function, or it's very complicated. Otherwise, we could find the solution analytically.
- A numerical optimizer is Gradient Descent.
  - Evaluate the function and its gradient at the point.
  - Take a step according to the update rule.
    - *Notice the learning rate parameter*
  - On convex problems, this is guaranteed to succeed.
- Image of GD

### SGD
- In real life, we have big data.
- We can evaluate the loss function for a small number of data points, which is much faster, but it introduces sampling noise.
- The general process is still the same, but we have to deal with uncertainty.
- Image of the problem, but noisy

### Preconditioning
- The convergence speed of SGD/GD is upper bounded by the condition number. (What exactly?)
- We can transform the parameter space so that the condition number is better.
- Image of preconditioning in action.
- The hard part is to find that preconditioner. It's even harder in a noisy setting.


### Overview of the Algorithm
- We can set an initial learning rate

### Neural nets
- Does everyone know what a neural net is?
- The important part is that it is a model with many (hundreds of thousands or more) parameters.
- It comes with a loss function and a way to compute gradients.

## Approach

### Evaluating an optimizer
- Benchmarking and comparing optimizers is hard, because everyone uses different testproblems and claims that they are "close to real life"
  - Data sets
  - Batch size
  - Model architecture
  - Hyperparameter tuning method
  - Specific Hardware
- Also, which measure do you use to compare the 'goodness' of an optimizer? Loss, Accuracy, Wallclock time to convergence, time in epochs to convergence.
- DeepOBS proposes standard test problems, a standard protocol for tuning hyperparameters. and provides baselines to compare against.
- Thank you to Aaron and Frank! Next week, Aaron will give his presentation where he will talk about DeepOBS in more Detail

### DeepOBS, what is it in practice?
- It's a library that has versions for tensorflow and pytorch. I used the pytorch version.
- If I want to test the performance of an optimizer, I have to specify the optimizer, its hyperparameters and the testproblem.
- It saves the output in json files
- It provides the analyzer class, which generates matplotlib figures like you're going to see later.

### Implementing the preconditioner
- Overview of its functions and parameters.

### Using the TCML cluster
- Once again, I had never really used a cluster, so it took me a while to get used to this one.
- How to get your code to run on the cluster:
1. Get an account by requesting one per e-mail.
2. If you have any special code requirements, build a Singularity container (kind of like a virtual machine) Otherwise use one of the provided ones.
3. Submit the batch via SSH
4. Get an e-mail when your job starts or finishes.
5. Download the files to your computer. It's easiest to mount the remote folder on your local machine.

## Experiments + Results + Discussion
- I'll tell you which experiments I've done and what I've found.
- If you have questions, please ask them!
- Everything will be explained in more detail in the thesis.

### Preconditioning; Effectiveness of the algorithm
- Show & explain the figure exp_preconditioning, It's created by DeepOBS
  - Different Optimizers on two testproblems. Blue is the PreconditionedSGD and Orange is the AdaptiveSGD.
  - Explain the Optimizers
    - (Explain Testproblems) Only Convolutional nets!
- Both variants perform worse than other widely used algorithms.
  - They converge much slower than the other algorithms, which seem to reach an optimum after 40 epochs, while the two don't converge even after 100 epochs
- Maybe mention the overfitting going on.
- Main conclusion: The algorithm does not perform better than others.

#### Why is it worse than SGD?
- We construct a low-rank Preconditioner matrix. This means out of hundreds of thousands of parameters we can only reduce two eigenvalues. Usually there are other big Eigenvalues, which means the condition number does not really change.
- The other optimizers are tuned to perfection. Some of these hyperparameters are still chosen by guessing.
- The other optimizers use more data for actual parameter updates.

### Performance penalty
- Show figure
  - Running on mnist_mlp with a batch size of 128.
  - Shown Optimizers are
    - SGD at 100% as a baseline.
    - PreconditionedSGD, AdaptiveSGD, OnlyAdaptiveSGD
  - They add a large performance penalty, but that would depend on the batch size.

### Stability depending on initialization
- The authors report unstable behavior if the algorithm was set to estimate the learning rate right at the beginning. Probably they were running at a smaller batch size.
- Either using a different initialization method or using a larger batch size stabilized the algorithm.

### Performance of learning rate
- Seeing that the algorithm is stable for an automatically constructed learning rate, I wanted to see if there are effects on success measures.
- Describing the plot
  - Dropoff of SGD for small learning rates, plateau, cliff for large learning rates.
- Usually, on this testproblem the algorithm would estimate a learning rate of around 0.02 and then slowly decay it over epochs.
- This means we should let the algorithm chose the learning rate automatically

## Conclusion
- DeepOBS is awesome!


## Room for Questions








## the algorithm works in multiple phases
1. Estimate a prior
2. Take a few steps, estimate a posterior
3. Construct a low-rank Preconditioner
4. Every step, apply the preconditioner, then SGD.

## Implementation/Approach
* The class, highlights
  - How to use the class
* How to use DeepOBS
* How to use the TCML cluster

## Experiments
* Experiment, result, what does it mean?

## Conclusion
* Should the class be used for anything?
