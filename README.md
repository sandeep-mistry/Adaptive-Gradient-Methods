
# Adaptive-Gradient-Methods-With-Dynamic-Bound-Of-Learning-Rate-Reprodicbility-Challenge
- Chosen paper title: ADAPTIVE GRADIENT METHODS WITH DYNAMIC BOUND OF LEARNING RATE
- OpenReview URL of chosen paper: https://openreview.net/pdf?id=Bkg3g2R9FX

# Introduction
New variants of Adam and AMSGrad, called AdaBound and AMSBound, were introduced by Luo et al. (2019) and employed dynamic bounds on learning rates to achieve a smooth and gradual transition from adaptive methods to SGD. We attempted to reproduce their experimental results on various popular tasks and models, however only managed partial reproduction, specifically using feedforward and convolutional neural network models.

# Requirement
see file requirements.txt

# ReadMe
Please see visualisation.py and enter appropriate [model_name] first, then:

```
python [model_name.py] e.g. python MNIST.py
python visualisation.py
```

