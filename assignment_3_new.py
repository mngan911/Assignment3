# -*- coding: utf-8 -*-
"""
Task: Apply Monte Carlo simulations combined with Bootstrap to evaluate quality of inference on B1 using serially correlated data
1.   Simulate data with AR(1) errors
2.   Calculate bootstrap standard errors
3.   Construct a 95% confidence interval for B1, using both the bootstrap and the theoretical standard errors
4.   Perform Monte Carlo simulations T=100, T=500 and assess Empirical coverage of CI

# Simulate data
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

np.random.seed(88) # Set random seed for reproducibility

def generate_ar1_errors(T, rho=0.5, sigma=1):  #Create funtion get AR(1) errors
  """
  Simulate an AR(1) process, taking the errors

  Parameters:
    T (int): Number of observations.
    rho (float): Coefficient of AR(1) process.
    sigma (float): Standard deviation

  Returns:
    u: Simulated AR(1) error terms.
  """
    u = np.zeros(T) #Create numpy array storing all the u
    u[0] = np.random.normal(0, sigma)
    for t in range(1, T):
        u[t] = rho * u[t - 1] + np.random.normal(0, sigma)
    return u

def simulate_regression_with_ar1_errors(T, beta_0=1, beta_1=2, rho=0.5, sigma=1):  #Create funtion to simulate data
  """
  Simulate a regression model with AR(1) error terms, taking x, y, u

  Parameters:
    T (int): Number of observations.
    beta0 (float): Intercept of the regression model.
    beta1 (float): Slope of the regression model.
    rho (float): Coefficient of the AR(1) process in the error term.
    sigma (float): Standard deviation
  Returns:
    tuple: x (independent variable), y (dependent variable))
  """
    x = np.random.normal(0, 1, T)
    u = generate_ar1_errors(T, rho, sigma)
    y = beta_0 + beta_1 * x + u
    return x, y

def ols_estimate(x, y): #Create OLS model, return the parameters of Beta0 and Beta1 & SE
    X = sm.add_constant(x) # Using statsmodel library
    model = sm.OLS(y, X)
    results = model.fit()
    return results.params, results.bse # return parameters and standard errors

"""# Bootstrap SE
Since the data is serially correlated (e.g as in Time Series, data from Jan-Dec should be all taken into consideration, can't omit Feb in between), we will use Moving Block Bootstrap to preserve the correlation.
"""

def bootstrap_standard_errors(x, y, L=12, B=1000):
  """
  Create function calculating Boostrap SE
  L is block length, as in Time series of monthly economic data, L=12 means taking 1 block consist of 12 consecutive months
  B is number of bootstrap samples
  Add a step of calculating Number of Blocks (NB)
  """

  T = len(y)
  NB = T // L + (1 if T % L else 0) # NB: Num of Blocks
  beta_bootstrap = np.zeros((NB, 2)) # Create array storing all Beta0 and Beta1 from Bootstrap

  for i in range(NB):
      indices = np.random.choice(np.arange(NB) * L, size=NB, replace=True) #Take random indices of Block, each time take a Block from the pool
      sample_indices = np.hstack([np.arange(index, min(index + L, T)) for index in indices])
      sample_indices = sample_indices[:T]  # Ensure the bootstrap sample is the same size as the original data

      x_resampled = x[sample_indices]
      y_resampled = y[sample_indices]

      # Refit model on bootstrap model using resampled x and y
      beta_bootstrap[i] = ols_estimate(x_resampled, y_resampled)[0]

  # Calculate the standard deviation for the standard errors
  se_bootstrap = beta_bootstrap.std(axis=0)

  return se_bootstrap

"""# Confidence Interval and Empirical coverage

* We run Monte Carlo simulation, constructing CI using Theoretical SE and Bootstrap SE again and again.
* Then we check if the True Beta1 value fall within those CI (1: fall into, 0: no)
* By taking the mean of all values, we get the Empirical coverage. We expect this coverage to be close to 95% (as in 95% CI).
"""

def monte_carlo_simulation(T, num_simulations=1000):
    np.random.seed(88) # Set random seed for reproducibility
    coverage_theoretical = np.zeros(num_simulations) #Create array to store the Theoretical CI coverage
    coverage_bootstrap = np.zeros(num_simulations) #Create array to store the Bootstrap CI coverage

    for i in range(num_simulations):
        x, y = simulate_regression_with_ar1_errors(T)
        beta_hat, se_theoretical = ols_estimate(x, y) #From OLS estimation: Beta hat taken from result.params, SE theoretical taken from result.bse
        se_bootstrap = bootstrap_standard_errors(x, y)

        # Construct a 95% confidence interval for
        ci_theoretical = [beta_hat - 1.96 * se_theoretical, beta_hat + 1.96 * se_theoretical] #Constructing CI for using Theoretical SE
        ci_bootstrap = [beta_hat - 1.96 * se_bootstrap, beta_hat + 1.96 * se_bootstrap] #Constructing CI for using Bootstrap SE

        # Determine if the true beta_1 falls within the confidence interval
        """
        Count each time the true beta is within the CI equal to 1, true beta fall out of CI equal to 0
        ci_theoretical[0][1] is the lower bound of CI for Beta1, ci_theoretical[1][1] is the upper bound of CI for Beta1
        """
        coverage_theoretical[i] = (ci_theoretical[0][1] <= 2 <= ci_theoretical[1][1])
        coverage_bootstrap[i] = (ci_bootstrap[0][1] <= 2 <= ci_bootstrap[1][1])

    print(f"T = {T}:")
    print(f"Empirical Coverage - Theoretical CI: {coverage_theoretical.mean():.3f}") #Taking the mean to know the % of Coverage
    print(f"Empirical Coverage - Bootstrap CI: {coverage_bootstrap.mean():.3f}")
    print("-" * 40)

monte_carlo_simulation(T=100)

monte_carlo_simulation(T=500)

"""The result of empirical coverage is close to 95%. When we increase T from 100->500, we get even the closer number to 95%, as the estimation of Beta1 is improved due to increased sample size.

# Additional References
1. Confidence Interval using Monte Carlo & Bootstrap: https://stats.stackexchange.com/questions/525063/trying-to-calculate-confidence-intervals-for-a-monte-carlo-estimate-of-pi-what
2. Bootstrap and Monte Carlo Methods - Machine Learning TV: https://www.youtube.com/watch?v=d3mcuJycJfI&t=369s
3. Resampling - Matthew E. Clapham: https://www.youtube.com/watch?v=Kho4VuKmQdE&t=592s
"""