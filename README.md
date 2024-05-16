# Hessian_approximation

Delayed compensation using Tailor expasion is commonly used
https://arxiv.org/abs/1609.08326

But full hessian cannot be used because of the computational and storage limits.

We used hessian vector Product to get hessian in less computation.
[Pearlmutter, B. A. (1994). Fast exact multiplication by the Hessian. Neural computation, 6(1), 147-160.]


Tailor expansion does not predict far values well, so if the vector size increases, the hessian compensation is not used.
In addition, if the hessian value grows beyond a certain level, it is not used.
(Experimental confirmation)

We can see hessian compensated gradient can achieve closer to non delayed gradient than non compensated gradient.
(By Euclidean, Cosine norm)
We can also get higher training accuracy than ASGD on average.(0.5%p, Cifar-10 Dataset, Resnet-18)

Since the non-delayed parameters are unknown in the local worker, additional algorithms are needed to use HVP optimizer.

Spectrain can be used to predict weights
https://arxiv.org/abs/1809.02839
