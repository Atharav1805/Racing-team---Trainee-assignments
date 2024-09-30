import matplotlib.pyplot as plt
import numpy as np

# Parameters
dt = 1  # Time step
process_noise_variance = 1  # Process noise covariance
measurement_noise_variance = 2  # Measurement noise covariance
C = 1  # Measurement matrix (assuming direct measurement)
hostel_positions = [5, 15, 25]  # Known hostel positions
mean = 0  # Initial mean (starting position)
covar = 1  # Initial covariance (initial uncertainty)
x = np.arange(0, 30, 0.01)  # Position space for plotting

# Gaussian function for plotting distributions
def gauss(mean, covar, x):
    return np.exp(-(x - mean)**2 / (2 * covar)) / np.sqrt(2 * np.pi * covar)

# Kalman Filter function
def kalman_filter(mean, covar, u, z):
    # Prediction step
    mean_pred = mean + dt * u
    covar_pred = covar + process_noise_variance
    
    # Update step (if measurement is available)
    K = covar_pred * C / (C * covar_pred * C + measurement_noise_variance)
    mean = mean_pred + K * (z - C * mean_pred)
    covar = (1 - K * C) * covar_pred
    
    return mean, covar

# Set up plot
plt.ion()  # Interactive mode for real-time plotting
fig, ax = plt.subplots()
line_pred, = ax.plot([], [], 'b-', label='Predicted')
line_corr, = ax.plot([], [], 'g-', label='Corrected')
line_obs, = ax.plot([], [], 'm-', label='Observed')
ax.set_xlim(0, 30)
ax.set_ylim(0, 0.5)
ax.set_xlabel('Position')
ax.set_ylabel('Probability Density')
ax.legend()
ax.grid(True)

# Simulation loop
for i in range(100):
    u = 1  # Constant control input (speed)
    
    # Prediction
    mean = mean + dt * u
    covar = covar + process_noise_variance
    y_pred = gauss(mean, covar, x)
    
    # Check if the robot is near a hostel
    if any(abs(mean - h) < 0.5 for h in hostel_positions):
        z = mean + np.random.normal(0, np.sqrt(measurement_noise_variance))  # Noisy measurement
        mean, covar = kalman_filter(mean, covar, u, z)
        y_corr = gauss(mean, covar, x)
        y_obs = gauss(z, measurement_noise_variance, x)
        
        # Update plots
        line_pred.set_data(x, y_pred)
        line_corr.set_data(x, y_corr)
        line_obs.set_data(x, y_obs)
        ax.set_title(f'Step {i+1}')
        plt.pause(0.1)  # Pause to simulate real-time update
    else:
        # If no measurement is available, update only prediction
        line_pred.set_data(x, y_pred)
        ax.set_title(f'Step {i+1} (No Measurement)')
        plt.pause(0.1)

# Keep the plot open after the simulation
plt.ioff()
plt.show()

