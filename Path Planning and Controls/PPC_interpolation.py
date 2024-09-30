import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

#--------------------------------------------------------------------------------------------------------------------------
def cubic_spline_interpolation(x, y, first_deriv_start=None, first_deriv_end=None):

    n = len(x)
    h = np.diff(x)
    # Coeff a is just equal to y
    a = y[:]

    # A matrix will be modified based on clamped conditions
    A = np.zeros((n, n))
    B = np.zeros(n)

    # Boundary conditions for clamped splines
    if first_deriv_start is not None:
        A[0, 0] = 2
        A[0, 1] = 1
        B[0] = 3 * first_deriv_start
    else:
        A[0, 0] = 1
        A[0, 1] = 2

    if first_deriv_end is not None:
        A[n-1, n-2] = 1
        A[n-1, n-1] = 2
        B[n-1] = 3 * first_deriv_end
    else:
        A[n-1, n-2] = 2
        A[n-1, n-1] = 1

    # Fill the rest of the tridiagonal matrix as before
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]

    # Construct the right-hand side vector (B)
    for i in range(1, n-1):
        B[i] = 3 * (((y[i+1] - y[i]) / h[i]) - ((y[i] - y[i-1]) / h[i-1]))

    # Solve the tridiagonal linear system
    c = np.linalg.solve(A, B)

    # Calculate the remaining coefficients
    b = (np.diff(y) / h) - (h * (2 * c[:-1] + c[1:])) / 3
    d = (c[1:] - c[:-1]) / (3 * h)

    return a, b, c[:-1], d
#--------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------
def objective_function(x_data, y_data, first_deriv_start=None, first_deriv_end=None):

    n = len(x_data)
    a_coeffs, b_coeffs, c_coeffs, d_coeffs = cubic_spline_interpolation(x_data, y_data, first_deriv_start, first_deriv_end)

    # Define points for plotting the interpolated curve
    x_interp = np.linspace(min(x_data), max(x_data), 100)  # generates equally spaces points in given range

    # define empty arrays to store 1st and 2nd derivatives
    der_1 = np.zeros_like(x_interp)
    der_2 = np.zeros_like(x_interp)
    sum = 0

    # Evaluate the interpolated curve at each x-value
    for i, x in enumerate(x_interp):
        for j in range(n-1):
            # 2]range(n-1): The range function generates a sequence of numbers starting from 0 (inclusive) and ending at the specified value (exclusive). In this case, it creates a sequence from 0 to 9 (10 numbers).
            if x_data[j] <= x <= x_data[j + 1]:
                dx = x_data[j + 1] - x_data[j]
                t = (x - x_data[j]) / dx  # curve is taken as dt^3 + ct^2 + bt + a where t = (x-xj) kinda
                                          # divide by dx to normalize t between 0 to 1

                der_1[i] = b_coeffs[j] + t * (2 * c_coeffs[j] + t * (3 * d_coeffs[j]))
                der_2[i] = 2 * c_coeffs[j] + t * (6 * d_coeffs[j])
                R = abs((1 + (der_1[i]) ** 2) ** 1.5 / der_2[i])  # Calcualte the radius at discrete x
                sum = sum + (1 / R) ** 2  # Calcualte the sum of squares of curvatures

    return sum  # Negate for minimization (maximize actual sum) when using scipy
#--------------------------------------------------------------------------------------------------------------------------

#define a wrapper function:
def objective_wrapper(vars, x_data, y_data):
    first_deriv_start, first_deriv_end = vars
    return -objective_function(x_data, y_data, first_deriv_start, first_deriv_end)

#--------------------------------------------------------------------------------------------------------------------------


'''
--------------------------------------------------------------------------------------------------------------------------
Main Program Starts
--------------------------------------------------------------------------------------------------------------------------
'''

# Define data points
x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([2, 3, 6, 5, 6, 3, 7, 11, 8, 6, 7])
n = (len(x_data)-1)      # 1]len(x_data) - 1: This part calculates the length of x_data and subtracts 1 from it. If x_data has 11 elements, len(x_data) would be 11, and subtracting 1 from that gives us 10, which would be the last index.
first_deriv_start, first_deriv_end = 0,0


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
min_value = 5000
for i in np.linspace(-9,10, 1000):                            # i is counter variable for variation of first_deriv_start
    for j in np.linspace(-9,10, 100):                        # j is counter variable for variation of first_deriv_start
        value = objective_function(x_data, y_data, i, j)
        if value < min_value:
            min_value = value
            first_deriv_start, first_deriv_end = i, j

print(f"Optimal clamping conditions (First Derivatives at end point): {first_deriv_start} , {first_deriv_end} \nMaximum Curvature: {min_value}")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

''''
# Minimize the objective function using SCIPY

result = minimize(objective_wrapper, x0=(0, 0), args=(x_data, y_data), method='Nelder-Mead')

# Print the optimal values and function value (negative of sum)
print("Optimal first_deriv_start:", result.x[0])
print("Optimal first_deriv_end:", result.x[1])
print("Function value at minimum (negative of sum):", -result.fun)
first_deriv_start = result.x[0]
first_deriv_end = result.x[1]
'''
# Perform cubic spline interpolation (calculate coefficients for each spline)
a_coeffs, b_coeffs, c_coeffs, d_coeffs = cubic_spline_interpolation(x_data, y_data,first_deriv_start,first_deriv_end)


# Define points for plotting the interpolated curve
x_interp = np.linspace(min(x_data), max(x_data), 100)  #generates equally spaces points in given range
y_interp = np.zeros_like(x_interp)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Evaluate the interpolated curve at each x-value
for i, x in enumerate(x_interp):
    for j in range(n):
                                                            # 2]range(len(x_data) - 1): The range function generates a sequence of numbers starting from 0 (inclusive) and ending at the specified value (exclusive). In this case, it creates a sequence from 0 to 9 (10 numbers).
        if x_data[j] <= x <= x_data[j+1]:
            dx = x_data[j+1] - x_data[j]
            t = (x - x_data[j]) / dx                        # curve is taken as dt^3 + ct^2 + bt + a where t = (x-xj) kinda
                                                            # divide by dx to normalize t between 0 to 1
            y_interp[i] = a_coeffs[j] + t * (b_coeffs[j] + t * (c_coeffs[j] + d_coeffs[j] * t))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------
# Plot the original data points and the interpolated curve
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'o', label='Data Points', color = '#00A1D8')
plt.plot(x_interp, y_interp, label='Cubic Spline Interpolation', color = '#F01D7F')
plt.title('Cubic Spline Interpolation without scipy')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
#-------------------------------------------------------------------------------------------------------------------------

'''
# To calculate interpolated y value for a given x value
x_new = float(input("Enter the value of x: "))
for i in range(n):
    if ((x_new > x_data[i]) and (x_new < x_data[i+1])):
        dx = x_data[i + 1] - x_data[i]
        t = (x_new - x_data[i]) / dx
        y_new = a_coeffs[i] + t * (b_coeffs[i] + t * (c_coeffs[i] + d_coeffs[i] * t))
    elif x_new == x_data[i]:
        y_new = y_data[i]
    elif x_new == x_data[i+1]:
        y_new = y_data[i+1]

print(f"The interpolated y value for x = {x_new} is: {y_new}")
'''





