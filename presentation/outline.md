# Presentation outline

1. Introduction
  * Plan: structure of this talk
  * Mission of this thesis:
    - Implement the algorithm so that it's easy to use.
    - Use DeepOBS to have a hard look at it.
2. Background
  * Quickly move from no background knowledge
  * Optimization problems look like this:
    - You have many parameters (a vector) and a function to optimize.
    - Example: Simple one-dimensional square function. (Brewing time vs Quality of your coffee)(Plot)
    - High-school: set the first derivative to zero.
  * Now in real life:
    - we don't know the closed form of the function. We have to evaluate that ourselves
    - the data is usually noisy, there might be other factors that pollute the data.
    - These are the kinds of problems we're dealing with in deep learning, only in many dimensions.
    - (Automatic differentiation exists)
  * SGD:
    - Let's say you start with some time, just a guess.
    - The gradient points in the right direction. Take steps
  * Second-order: Notice how the gradient changes. Take one exactly right step.
3. the algorithm works in multiple phases
  1. Estimate a prior
  2. Take a few steps, estimate a posterior
  3. Construct a low-rank Preconditioner
  4. Every step, apply the preconditioner, then SGD.
4. Implementation/Approach
  * The class, highlights
  * How to use DeepOBS
  * How to use the TCML cluster
5. Experiments
  * Experiment, result, what does it mean?
6. Conclusion
  * Should the class be used for anything?
