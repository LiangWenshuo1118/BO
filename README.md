# Bayesian Optimization with Gaussian Process Regression

This repository contains a Python script that demonstrates how to perform Bayesian optimization using Gaussian process regression to find the maximum of a quadratic function. The implementation leverages the BoTorch and GPyTorch libraries, which are built on PyTorch.

## Overview

The script seeks to optimize the function `- (x - 25) ** 2` over the interval `[0, 50]`. Bayesian optimization, which utilizes Gaussian process regression, is employed to efficiently explore the function space and pinpoint the maximum value by proposing new sampling points.

## Prerequisites

Ensure you have Python installed along with the necessary Python libraries:
- `torch`
- `botorch`
- `gpytorch`
