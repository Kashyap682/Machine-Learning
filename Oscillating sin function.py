import numpy as np
import matplotlib.pyplot as plt

# Generate the data
np.random.seed(123)
x = np.linspace(0, 10, 200)
y = np.sin(x) + np.random.normal(0, 0.5, 200)

# Plot the data
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Oscillating Sinusoidal Function with Noise')
plt.show()

# Create the regression model
for degree in range(1,10):  # the degree of the polynomial to fit
    p = np.polyfit(x, y, degree)
    y_pred = np.polyval(p, x)
    
    # Plot the regression model
    plt.plot(x, y_pred, color='red')
    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Regression Model')
    plt.show()
