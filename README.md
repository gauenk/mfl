# Median for Federated Learning

## Summary

The median is a robust estimator to measure the central tendancy of data. The coordinate-wise mean and median have been shown to be susceptible to poisoning attacked in Federated Learning (FL). This repository is the first work (to our knowledge) exploring the impact of multi-dimensional median estimates against the current FL attacks. As multi-dimensional medians are not embarassingly parallel like their coordinate-wise counterparts (think numpy.median), we additionally explore the computational and security benefit from randomizing the median computation across randomized blocks of model parameters, splitting the vector of size 10^6 into parallelizable regions.


### Federated Learning Basic

The global model is updated with gradients from local models. The global model may have access to it's own pool of trusted data. The local models all collect data, possibly each data stream following their own data distribution. The goal is to aggregate the gradients from the local models to update the global model. The global model is shared with local models, which is aggregated locally according to their own update scheme.

Attacks in Federated Learning occur when local model gradients are directly controlled by a single, organized party. The goal is to reduce the testing quality of the global model. The attacks are often presented with various levels of information: (i) "full information" means the attacker knows the true direction of the gradient and the gradient of all other local models, while (ii) "partial information" means the attackers only have access to the compromised devices.

### Attacks

Trimmed Mean Attack: The trimmed-mean attack sets the gradient on the compromised devices to be randomly sampled around an extrema. For example, if the direction of the gradient is s = -1 and the maximum value of a parameter coordinate is w_max, w_max > 1, then all the compromised parameters are set of [w_max,w_max*b], for b = 2.

### Medians

This section discusses difference multi-dimensional medians.


### Results

This section discusses difference multi-dimensional medians.
