# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# define simulation function
def simulate_path(s0, mu, sigma, horizon, timesteps, n_sims):

    # set the random seed for reproducibility
    np.random.seed(10000)

    # read parameters
    S0 = s0                 # initial spot price
    r = mu                  # mu = rf in risk neutral framework
    T = horizon             # time horizon
    t = timesteps           # number of time steps
    n = n_sims              # number of simulation

    # define dt
    dt = T/t                # length of time interval

    # simulate 'n' asset price path with 't' timesteps
    S = np.zeros((t,n))
    S[0] = S0

    for i in range(0, t-1):
        w = np.random.standard_normal(n)
        S[i+1] = S[i] * (1 + r*dt + sigma*np.sqrt(dt)*w)

    return S


# Assign simulated price path to dataframe for analysis and plotting
price_path = pd.DataFrame(simulate_path(100,0.05,0.2,1,252,10000))

# Verify the generated price paths
price_path


# Plot the histogram of the simulated price path at maturity
price_path.iloc[-1].hist(bins=100, color='orange', label='End-Period Price in Histogram')
plt.legend()
plt.title('Simulated Geometric Brownian Motion at Maturity')


# Plot initial 100 simulated path using matplotlib
plt.plot(price_path.iloc[:,:1000])
plt.xlabel('Time Steps')
plt.xlim(0,252)
plt.ylabel('Index Levels')
plt.title('Monte Carlo Simulated Asset Prices');


# Call the simulation function
S = simulate_path(100,0.05,0.2,1,252,100000)

# Define parameters
K=100; r=0.05; T=1

# Calculate the discounted value of the expeced payoff
C0 = np.exp(-r*T) * np.mean(np.maximum(0, S[-1]-K))
P0 = np.exp(-r*T) * np.mean(np.maximum(0, K-S[-1]))

# Print the values
print(f"European Call Option Value is {C0:0.4f}")
print(f"European Put Option Value is {P0:0.4f}")

K


# range of spot prices
sT= np.linspace(50,150,100)

# visualize call and put price for range of spot prices
figure, axes = plt.subplots(1,2, figsize=(20,6), sharey=True)
title = ['Call Payoff', 'Put Payoff']
payoff = [np.maximum(sT-K, 0), np.maximum(K-sT, 0)]
color = ['green', 'red']
label = ['Call', 'Put']

# plot payoff
for i in range(2):
    axes[i].plot(sT, payoff[i], color=color[i], label=label[i])
    axes[i].set_title(title[i])
    axes[i].legend()

figure.suptitle('Option Payoff at Maturity');


# Call the simulation function
S = simulate_path(100,0.05,0.2,1,252,100000)

# Define parameters
K=100; r=0.05; T=1

# Average price
A = S.mean(axis=0)

C0 = np.exp(-r*T) * np.mean(np.maximum(0, A-K))
P0 = np.exp(-r*T) * np.mean(np.maximum(0, K-A))

# Print the values
print(f"Asian Call Option Value is {C0:0.4f}")
print(f"Asian Put Option Value is {P0:0.4f}")


# Call the simulation function
S = simulate_path(100,0.05,0.2,1,252,100000)

# Define parameters
K=100; B=150; r=0.05; sigma=0.20; T=3; t=756; dt=T/t; n=100000; rebate = 30; value=0

# Barrier shift - continuity correction for discrete monitoring
B_shift = B*np.exp(0.5826*sigma*np.sqrt(dt))
print(B_shift)

# Calculate the discounted value of the expeced payoff
for i in range(n):
    if S[:,i].max() < B_shift:
        value += np.maximum(0, S[-1,i]-K)
    else:
        value += rebate

C0 = np.exp(-r*T) * value/n

# Print the values
print(f"Up-and-Out Barrier Call Option Value is {C0:0.4f}")


figure, axes = plt.subplots(1,3, figsize=(20,6), constrained_layout=True)
title = ['Visualising the Barrier Condition (Shows All Lines)', \
         'Spot Touched Barrier (For those simulates who can touch the Dash Line)', \
         'Spot Below Barrier (For those who cannot touch the Dash Line)']

axes[0].plot(S[:,:200])
for i in range(200):
    if S[:,i].max() > B_shift:
        axes[1].plot(S[:,i])
    else:
        axes[2].plot(S[:,i])

for i in range(3):
    axes[i].set_title(title[i])
    axes[i].hlines(B_shift, 0, 252, colors='k', linestyles='dashed')

figure.supxlabel('time steps')
figure.supylabel('index levels')

plt.show()
