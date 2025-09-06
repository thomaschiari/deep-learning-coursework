# Exercise 1

## 1. Distribution & Overlap

![Exercise 1](images/exercise1.png)

- Class C0 is a compact cluster on the left, with vertical spread. 
- Class C1 is in the upper-middle region with larger variance, overlapping with C0.
- Class C2 is in the lower-middle and more compact, mostly separated from other classes.
- Class C3 is in the far right, tall and narrow, mostly without overlap. 

The only overlap is between C0 and C1. C2 and C3 are well separated from other classes. 

## 2. Can a single linear boundary separate all classes?

![Exercise 1](images/ex1_linear_regions.png)

No, a single straight line can separate only in two regions, but we have four classes. Linear models can work ifwe use a multinomial logistic regression (in the image above), separating in four regions, but still struggle to separate C0 and C1. 

## 3. Neural network decision boundaries

![Exercise 1](images/ex1_mlp_regions.png)

A multi-layer perceptron (MLP) can better separate the boundaries between the classes than a linear model (see image above), bending around the classes and capturing the non-linear boundaries. 