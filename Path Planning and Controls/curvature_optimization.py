import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize  # For optimization-based autotuning


# Define constants (initial guess for PID parameters)
Kp = 0.2
Ki = 0.05
Kd = 0.5
dt = 0.1  # Simulation time step (seconds)

# Initial conditions
panel_position = 0.0  # Degrees
set_sun_position = 45.0  # Degrees (desired sun position)
error_integral = 0.0
previous_error = 0.0

# Simulation data (pre-allocate for efficiency)
time_data = [0]  # Initialize with starting time (0)
panel_position_data = [panel_position]
error_data = []
sun_data = [set_sun_position]  # Add initial sun position


def control_loop(Kp, Ki, Kd):
    global panel_position, error_integral, previous_error

    # Calculate error
    error = set_sun_position - panel_position

    # Update integral term
    error_integral += error

    # Update derivative term
    derivative = error - previous_error
    previous_error = error

    # Calculate PID control output
    control_output = Kp * error + Ki * error_integral + Kd * derivative

    # Limit control output (optional)
    control_output = max(min(control_output, 10.0), -10.0)  # Example limits

    # Update panel position (simulation only)
    panel_position += control_output

    # Record data (append for efficiency)
    time_data.append(time_data[-1] + dt)  # Update time based on previous time
    panel_position_data.append(panel_position)
    error_data.append(error)
    sun_data.append(set_sun_position)

    return abs(error)  # Return absolute error for cost function


def autotune(dt, set_sun_position):
    """
    Performs optimization-based PID autotuning.

    Args:
        dt (float): Simulation time step.
        set_sun_position (float): Desired sun position.

    Returns:
        tuple: (Kp, Ki, Kd): Optimized PID parameters.
    """

    def cost_function(params):
        """
        Cost function to minimize: sum of squared errors during simulation.

        Args:
            params (list): List of PID parameters (Kp, Ki, Kd).

        Returns:
            float: Sum of squared errors.
        """
        Kp, Ki, Kd = params
        error_sum = 0
        panel_position = 0.0
        error_integral = 0.0
        previous_error = 0.0

        for _ in range(100):  # Simulate for a fixed duration
            error = set_sun_position - panel_position
            error_integral += error
            derivative = error - previous_error
            previous_error = error

            control_output = Kp * error + Ki * error_integral + Kd * derivative
            panel_position += control_output * dt
            error_sum += error**2

        return error_sum

    # Perform optimization to minimize cost function
    bounds = ((0, 10), (0, 1), (0, 10))  # Define reasonable bounds for PID parameters
    result = minimize(cost_function, x0=[Kp, Ki, Kd], method='SLSQP', bounds=bounds)
    return result.x  # Return optimized parameters


# Autotune PID parameters
Kp, Ki, Kd = autotune(dt, set_sun_position)
print(f"Optimized PID parameters: Kp={Kp:.2f}, Ki={Ki:.4f}, Kd={Kd:.2f}")

# Simulation loop
Kp = 0.2
Ki = 0.05
Kd = 0.5
precision = 0.1
start_time = time.perf_counter()

while control_loop(Kp, Ki, Kd) > precision:
    pass

end_time = time.perf_counter()

# Calculate total simulation time
simulation_time = end_time - start_time


# Print results
print(f"Panel reached set-point in {simulation_time} seconds.")

# Plotting (unchanged)
plt.plot(time_data, panel_position_data, label="Panel Position")
plt.plot(time_data, sun_data, label="Desired Position")
plt.show()
