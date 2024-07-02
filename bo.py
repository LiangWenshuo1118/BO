import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

# Define the function
def function(x):
    return - (x - 25) ** 2

# Initial data points
train_x = torch.tensor([
    [11], [21], [31], [41],[51],
], dtype=torch.double)

train_y = complex_function(train_x)  # Compute train_y directly from train_x

# Define bounds for the optimization
bounds = torch.tensor([[0.0], [50.0]], dtype=torch.double)

for iteration in range(10):  # Loop for 10 optimization cycles
    # Initialize the Gaussian process model and marginal log likelihood
    gp = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    # q-Expected Improvement acquisition function
    qEI = qExpectedImprovement(model=gp, best_f=train_y.min())

    # Optimize the acquisition function, requesting 2 candidates
    candidates, acq_values = optimize_acqf(
        qEI,
        bounds,
        q=2,
        num_restarts=10,
        raw_samples=2000,
    )

    # Evaluate the complex function at the new candidates
    new_y = function(candidates)

    # Update the training data
    train_x = torch.cat([train_x, candidates])
    train_y = torch.cat([train_y, new_y])

    # Print the new candidates and their acquisition values
    print(f"Iteration {iteration + 1}: New Candidates:")
    print(candidates)
