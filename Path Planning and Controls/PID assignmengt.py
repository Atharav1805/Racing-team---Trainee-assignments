import matplotlib.pyplot as plt
import time

# Define constants
Kp = 10  # Proportional gain
Ki = 0.05  # Integral gain
Kd = 0.5  # Derivative gain
dt = 0.1  # Simulation time step (seconds)

# Initial conditions
panel_position = 0.0  # Degrees
set_sun_position = 45.0  # Degrees (desired sun position)
error_integral = 0.0
previous_error = 0.0

# Simulation data storage
time_data = []
panel_position_data = []
error_data = []
sun_data = []

def control_loop():
  global panel_position, error_integral, previous_error

  # Calculate error
  error = set_sun_position - panel_position
  error_data.append(error)

  # Update integral term
  error_integral += error * dt

  # Update derivative term
  derivative = (error - previous_error) / dt
  previous_error = error

  # Calculate PID control output
  control_output = Kp * error + Ki * error_integral + Kd * derivative

  # Limit control output (optional)
  control_output = max(min(control_output, 10.0), -10.0)  # Example limits

  # Update panel position (simulation only)
  panel_position += control_output * dt

  # Record data (using time steps)
  time_data.append(len(time_data) * dt)  # Generate time steps based on loop iterations
  panel_position_data.append(panel_position)
  sun_data.append(45)

  # Terminate loop if target reached within a small tolerance
  return  # Exit the loop when error is close to zero


precision = 0.00000001
start_time = time.perf_counter()

# Simulate until the panel reaches the setpoint
while True:
    control_loop()
    if abs(previous_error) <= precision: break

end_time = time.perf_counter()



# Calculate total simulation time
simulation_time = end_time - start_time

# Print results
print(f"Panel reached set-point in {simulation_time} seconds.")

# Plot the results
plt.plot(time_data, panel_position_data, label="Panel Position")
plt.plot(time_data, sun_data, label="Desired Position")
plt.xlabel("Time (s)")
plt.ylabel("Degrees")
plt.title("Panel Position Control Simulation (PID) - Terminate at Setpoint")
plt.legend()
plt.grid(True)
plt.show()


