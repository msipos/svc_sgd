This is my implementation of a simple linear Support Vector Classifier,
in Python, using the Stochastic Gradient Descent algorithm.

### Guide through the code.

For a start, please take a look at the "Playing around" notebook.

This is the notebook I used to test and verify results as I was going along.

Because I wanted to make sure that the SVC is properly coded up, I was visualizing the results of the classification in 2 dimensions using a sample data set.

I then verified that the loss function and the gradient are computed correctly.

### Note about the gradient

The loss functions used are in `loss_functions.py`.

The first function is easier to understand, as it doesn't optimize the w0 term, i.e. the offset from the origin.

The function itself is:

```
  F(w) = \sum_i max(0, 1 - y_i (x_i \cdot w) + \lambda || w ||^2
```

where `y_i` are the targets (-1 or +1) and `x_i` are the vectors of dimension `N`.  `w` is likewise a vector of dimension `N` and represents the normal to the plane separating the two classes.

The gradient is evaluated exactly, by taking the derivative of the above function with respect to `w`.

Interesting point here relates to gradient of the `max` function. In particular, to express the gradient, we must break down the max in the following way:

```
                                       F(w)              F'(w)
  If    1 - y_i (x_i * w) > 0      1-y_i(x_i*w)         -y_i*x_i
  Else                                  0                  0

   +                              lambda ||w||^2       2 lambda w

```

Note that the gradient of `F(w)` is itself a vector of dimension `N`.

The second function, `LossSVM2` also optimizes the offset of the separation plane from the origin.  This offset is encoded inside the `w` vector like this:

```
  w = [ w_0, w_1, w_2, ... w_{N+1} ]
         ^
         |
       offset
```

In the code, we use `w0` for the offset, and `w1` for the remaining `N` coefficients.

The function itself changes to:

```
  F(w) = \sum_i max(0, 1 - y_i (x_i \cdot w1 - w0) + \lambda || w1 ||^2
```

and the gradient is likewise extended.

### Stochastic gradient descent

I implemented SGD using the technique where for each "generation" (there are `N` generations), the `x_i` and `y_i` are shuffled once.  Then all batches are performed for that generation, and after the next shuffle occurs.

### Solution

To see the concrete solution of the test problem, please turn your attention to the "Solution" notebook.

There, 3 instances of mini-batch stochastic gradient solvers are used to solve for the `w`.

To my surprise, all 3 found the same solution and hence have the same precision.

Reading the problem set, I expected that there was an error with my code, so I also compared the solution with `sklearn`'s  implementation of a linear SVC.  That implementation also finds the same solution, with the precisely same precision.

The precision (0.75 or so) can be improved by using a smaller value of `lambda`.

### Some notes

* It is interesting to note that the first term of `F'(w)` does not explicitly depend upon `w`. Rather the dependence is inside the `if` statement, coming from the derivative of the `max` function.

* In the timing measurements, `N` refers to number of full "generations". As such, it refers to **the number of times each data point was used by the training algorithm**.

* My results seem to show that batch_size=1 produces fastest solution, as per measurement of `N`, but again, this measurement does not include the possible benefits of vectorization in the code. (i.e. `N` does not necessarily linearly correspond to actual training time).

* This particular code is not very well vectorized (note the pure Python loops in the loss functions).  This could be improved by expressing the `if` statements in vectorized form in Numpy.  This is usually done by converting an `if` statement into a 0-1 vector in Numpy but I didn't do this in this code.

* I didn't create any unit tests in this case, mostly for the lack of time. Normally I would. However, I think the code is pretty well tested in the "Playing around" notebook.  It is also reassuring that it matches results of `sklearn`'s SGD algorithm.

* In principle the loss function is extracted from the algorithm itself via the the OptimizableFunction interface. Unfortunately, for the stochastic variant of the solver, I had to rethink the interface (hence `LossSVM3`). If this was an implementation for an ML library, it would be worthwhile to think carefully about how to implement the interface.

* I didn't think it was necessary to implement a schedule for the variation of `eta` (the learning rate).  In particular, all 3 batch solutions seemed to locate the minimum quickly and it didn't seem necessary in this case.  If I had deemed it necessary, I would have probably used a "simulated annealing"-type of a schedule where `eta` starts large and is slowly decreased to zero.
