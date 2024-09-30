from scipy import optimize, interpolate
import matplotlib.pyplot as plt
import numpy as np

global x_data, y_data, x_new

def Cubic_Spline(optimizers):
    derivative1_start, derivative1_end = optimizers
    cs = interpolate.CubicSpline(x_data, y_data, bc_type=((1, derivative1_start), (1, derivative1_end)))
    der1 = cs.derivative(1)
    der2 = cs.derivative(2)
    der1value = der1(x_new)
    der2value = der2(x_new)
    sum = 0
    for i in range(len(x_new)):
        R = abs((1 + (der1value[i]) ** 2) ** 1.5 / der2value[i])
        sum += (1 / R) ** 2

    return sum


x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([2, 3, 6, 5, 6, 3, 7, 11, 8, 6, 7])
x_new = np.linspace(min(x_data), max(x_data), 100)  # New X coordinates for plotting

dydx_start = 0.5
dydx_end = 0.5

optimizers = optimize.minimize(Cubic_Spline, (dydx_start, dydx_end), method='CG')

optimized_spline = interpolate.CubicSpline(x_data, y_data, bc_type = ((1, optimizers.x[0]), (1, optimizers.x[1])))
max_curvature = Cubic_Spline(optimizers.x)

print(f"The optimised Curvature is: {max_curvature}"
      f"The optimizing clamping conditions: {optimizers.x[0], optimizers.x[1]}")

# Plot the original data points
plt.plot(x_data, y_data, 'o', label='Data Points', color = '#00A1D8')

# Plot the clamped spline
plt.plot(x_new, optimized_spline(x_new), label='Cubic Spline Interpolation - Clamped', color = '#F01D7F')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clamped Cubic Spline')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
