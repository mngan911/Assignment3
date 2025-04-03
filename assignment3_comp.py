
# Final
"""
**1. Simulate Data**
* First, we simulate the AR(1) errors by producing functions in use.
"""
import numpy as np
import statsmodels.api as sm

def simulate_ar1(n, phi, sigma):
  errors = np.zeros(n)
  eta = np.random.normal(0, sigma, n)  # white noise
  for t in range(1, n):
    errors[t] = phi * errors[t - 1] + eta[t]
  return errors

def simulate_regression_with_ar1_errors(n, beta0, beta1, phi_x, phi_u, sigma):
  x = simulate_ar1(n, phi_x, sigma)
  u = simulate_ar1(n, phi_u, sigma)
  y = beta0 + beta1 * x + u
  return x, y, u

"""As in Time Series, the data is serially correlated, therefore the use of white standard errors will not be effective (tested in Appendix sections below). Therefore, we introduce only the HAC standard errors and performe Monte Carlo on it."""

# Substituting data to conduct Monte Carlo
np.random.seed(0) # Set random seed for reproducibility

beta0 = 1.          # Intercept
beta1 = 2           # Slope
phi_x = 0.7         # AR(1) coefficient for x
phi_u = 0.7         # AR(1) coefficient for the errors
sigma = 1           # Standard deviation of the white noise

# Simulating the model
## Do monte carlo

def Monte_Carlo(T):
  t_stats_hac = []
  mean_beta_1 = []
  for i in range(10000):
    x, y, errors = simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma)
    X = sm.add_constant(x)
        ## Use HAC: takes into account serial correlation
    model_1 = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': np.floor(1.3*T**(1/2)).astype(int)})
    t_stats_hac.append(model_1.t_test('x1=2').tvalue)
  # Check we reject the null hypothesis at alpha=0.05 about 5% of the time
  print(f"Monte Carlo T = {T}", f"Empirical size test beta_1=2 using HAC SE: {np.mean(np.abs(np.array(t_stats_hac)) > 1.965)}")
  print(f"Mean Beta_1: {np.mean(model_1.params[[1]])}")

Monte_Carlo(T=100)
Monte_Carlo(T=500)

"""**2. Calculate bootstrap standard errors**

* As we increase T, the result get closer to 5% (at 0.08), also the Mean get closer to True value . Now we use Bootstrap resampling to calculate standard errors.
* Assuming Time Series, the data is dependent (data from Jan-Dec should be all taken into consideration, can't omit Feb in between). Therefore, we use Moving Block Bootstrap (MBB).

* We first define the MBB func by setting the way to calculate the number of blocks, taking bootstrap samples, bootstrap estimates >>> Then conduct the MBB, taking the length of block as 12 (L=12, one block containing data from 12 consecutive months).
"""

def moving_block_bootstrap(x, y, L, num_bootstrap): # L is block length
  np.random.seed(0)
  T = len(y)  # Total number of observations
  num_blocks = T // L + (1 if T % L else 0)

  # Fit the original model
  X = sm.add_constant(x)
  original_model = sm.OLS(y, X)
  original_results = original_model.fit()

  bootstrap_estimates = np.zeros((num_bootstrap, 2))  # Storing estimates for beta_0 and beta_1

  # Perform the bootstrap
  for i in range(num_bootstrap):
    # Create bootstrap sample
    bootstrap_indices = np.random.choice(np.arange(num_blocks) * L, size=num_blocks, replace=True)  # Randomly select block indices from arrays
    bootstrap_sample_indices = np.hstack([np.arange(index, min(index + L, T)) for index in bootstrap_indices])
    bootstrap_sample_indices = bootstrap_sample_indices[:T]  # Ensure the bootstrap sample is the same size as the original data

    x_bootstrap = x[bootstrap_sample_indices]
    y_bootstrap = y[bootstrap_sample_indices]

    # Refit the model on bootstrap sample
    X_bootstrap = sm.add_constant(x_bootstrap)
    bootstrap_model = sm.OLS(y_bootstrap, X_bootstrap)
    bootstrap_results = bootstrap_model.fit()

    # Store the estimates
    bootstrap_estimates[i, :] = bootstrap_results.params

    return bootstrap_estimates
    return np.mean(bootstrap_results.params[1])

# Run moving block bootstrap
block_length = 12 # For example: 1 block containing 12 months
num_bootstrap = 1000 # Set the number of Bootstrap

for T in [100, 500]:
  np.random.seed(0)
  x, y, errors = simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma)
  bootstrap_results = moving_block_bootstrap(x, y, block_length, num_bootstrap)

    # Calculate and print standard errors
  bootstrap_se_T = bootstrap_results.std(axis=0)
  print("Bootstrap Standard Errors,", f"T={T}")
  print("SE(beta_0):", bootstrap_se_T[0])
  print("SE(beta_1):", bootstrap_se_T[1])

    ## Theoretical se from OLS
  X = sm.add_constant(x)
  model = sm.OLS(y, X)
  results = model.fit()

    # Standard errors from statsmodels
  statsmodels_se = results.bse
  print("\nStandard Errors from statsmodels OLS,", f"T={T}")
  print("SE(beta_0):", statsmodels_se[0])
  print("SE(beta_1):", statsmodels_se[1])
  print("--" * 20)

"""The result seems to make sense as:
* Bootstrap standard errors decrease as the number of Bootstrap increase (tested before between 1000 and 10000).
* Also when the number of T in Monte Carlo increase, the standard errors decrease, showing the improve in accuracy.

**3. Construct a 95% confidence interval for Beta_1 using both the bootstrap and the theoretical standard errors.**
"""

def theoretical_se (T):
  np.random.seed(0)
  for i in range(1000):
    x, y, errors = simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma)
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
  return results.bse[1]

# Calculate 95% confidence interval for beta_1
# Using Theoretical se
beta_1_mean = [2.088, 2.003] #Taking beta_1 mean from the first simulation part
ci_lower_100 = beta_1_mean[0] - 1.96 * theoretical_se (100)
ci_upper_100 = beta_1_mean[0] + 1.96 * theoretical_se (100)

ci_lower_500 = beta_1_mean[1] - 1.96 * theoretical_se (500)
ci_upper_500 = beta_1_mean[1] + 1.96 * theoretical_se (500)

print(f"T = 100", "95% Confidence Interval for beta_1:")
print(f"[{ci_lower_100:.4f}, {ci_upper_100:.4f}]\n")

print(f"T = 500", "95% Confidence Interval for beta_1:")
print(f"[{ci_lower_500:.4f}, {ci_upper_500:.4f}]")

# Taking CI and empirical coverage
# Set parameters
np.random.seed(0)
beta_0_true = 1
beta_1_true = 2
sigma = 1
num_simulations = 10000

# Arrays to store the estimates from each simulation
def CI_and_coverage(T):
  beta_0_estimates = np.zeros(num_simulations)
  beta_1_estimates = np.zeros(num_simulations)
  beta_0_in = np.zeros(num_simulations)
  beta_1_in = np.zeros(num_simulations)

# Run simulations
  for i in range(num_simulations):
    x = np.random.standard_cauchy(T)
    u = np.random.standard_cauchy(T)
    y = beta_0_true + beta_1_true * x + u
    # OLS estimation
    X = np.vstack([np.ones(T), x]).T
    XXinv = np.linalg.inv(X.T @ X)
    beta_hat = XXinv @ X.T @ y
    beta_0_estimates[i] = beta_hat[0]
    beta_1_estimates[i] = beta_hat[1]
    u_hat = y - beta_hat[0] - beta_hat[1] * x
    sigma2_hat = np.dot(u_hat, u_hat)/(T-2)
    variance_hat = sigma2_hat*XXinv
    se_0 = np.sqrt(variance_hat[0,0])
    se_1 = np.sqrt(variance_hat[1,1])
      ## Check whether beta_1 in CI 95%
    beta_1_in[i] = beta_hat[1] - 1.965*se_1 < beta_1_true < beta_hat[1] + 1.965*se_1
  print(f"Monte Carlo T = {T}")
  print("95% Confidence Interval for beta_1:", f"[{beta_hat[1] - 1.965*se_1:.4f}, {beta_hat[1] + 1.965*se_1:.4f}]")
  print(f"The empirical 95% CI coverage for beta_1: {np.mean(beta_1_in)}\n") #Taking the proportion of True value, or beta_1_true is in the confidence interval

# Output the results
CI_and_coverage(T=100)
CI_and_coverage(T=500)

"""The result seems to match the concept, as increasing simulation from 100 to 500 will increase coverage for Beta_1, showing the improvement.

The team will try to improve the assignment more ♥

# Appendix.
"""

import numpy as np
import statsmodels.api as sm

def simulate_ar1(n, phi, sigma):
  """
  Simulate an AR(1) process.

  Parameters:
  n (int): Number of observations.
  phi (float): Coefficient of AR(1) process.
  sigma (float): Standard deviation of the innovation term.

  Returns:
  np.array: Simulated AR(1) error terms.
  """
  errors = np.zeros(n)
  eta = np.random.normal(0, sigma, n)  # white noise
  for t in range(1, n):
    errors[t] = phi * errors[t - 1] + eta[t]
  return errors

def simulate_regression_with_ar1_errors(n, beta0, beta1, phi_x, phi_u, sigma):
  """
  Simulate a regression model with AR(1) error terms.
  Parameters:
    n (int): Number of observations.
    beta0 (float): Intercept of the regression model.
    beta1 (float): Slope of the regression model.
    phi (float): Coefficient of the AR(1) process in the error term.
    sigma (float): Standard deviation of the innovation term in the AR(1) process.
    Returns:
  tuple: x (independent variable), y (dependent variable), errors (AR(1) process)
  """
  x = simulate_ar1(n, phi_x, sigma)
  u = simulate_ar1(n, phi_u, sigma)
  y = beta0 + beta1 * x + u
  return x, y, u

# Substituing data
T = 500              # Number of observations
beta0 = 1.           # Intercept
beta1 = 2           # Slope
phi_x = 0.7             # AR(1) coefficient for x
phi_u = 0.7             # AR(1) coefficient for the errors
sigma = 1             # Standard deviation of the white noise

# Simulating the model

## Do monte carlo to find both white se and HAC se
t_stats_hc = []
t_stats_hac = []

for i in range(1000):
  x, y, errors = simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma)
  X = sm.add_constant(x)
  model = sm.OLS(y, X).fit(cov_type='HC1')
  t_stats_hc.append(model.t_test('x1=2').tvalue)
     ## Use HAC: takes into account serial correlation
  model2 = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': np.floor(1.3*T**(1/2)).astype(int)})
  t_stats_hac.append(model2.t_test('x1=2').tvalue)

## Check we reject the null hypothesis at alpha=0.05 about 5% of the time

print(f"Empirical size test beta_1=2 using White SE: {np.mean(np.abs(np.array(t_stats_hc)) > 1.965)}")
print(f"Empirical size test beta_1=2 using HAC SE: {np.mean(np.abs(np.array(t_stats_hac)) > 1.965)}")

"""This would suggest that the White standard errors are not adequately accounting for the autocorrelation in the errors.

 The HAC standard errors, though being 0.09 - almost double the 5%, is performing better, as it takes into account the serial correlation. By increasing the Bootstrap sample size, we might get a closer number to 5%.

**Other notes:**
* Monte Carlo: Math tools - simulate and describe exactly the probability distribution of the output Z. When knows exactly how the data is generated, given the parameters of the model.

  Only use when you can draw many sample of size T.

* Bootstrap: Simulate on sample instead of population, resampling repetitively (draw n times with replacement). Using empirical distribution F^ as a replacement for the true distribution.

  Use when you can't access to the whole population.
"""