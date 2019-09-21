# Quiz 3 - Logistic Function

import numpy as np
import matplotlib.pyplot as plt

#sample data
data = np.array([[-1, 0], [-2, 0], [-1.5, 0], [1, 1], [2, 1], [1.2, 0], [-0.2, 1], [0, 1]])

#error function for logistic equation: y = 1.0 / (1 + np.e **(-(a * x + b))
def J(a, b):
    temp = 0
    for i in data:
            temp += (1.0 / (1 + np.e **(-(a * i[0] + b))) - i[1]) ** 2
    return temp / (2 * len(data))

#partial derivatives with respect to each of the variables
def partial_a_J(a, b):
    partial = J(a, b) * (1 - J(a, b)) * a
    J /=  len(data)
    return J
    
def partial_b_J(a, b):
    partial = J(a, b) * (1 - J(a, b))
    partial /=  len(data)
    return partial

def main():
    # range of values for a and b in logistic equation
    a = np.arange(-5, 5, 0.1)
    b = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(a, b)
    a = 0
    b = 0
    alpha = 0.1
    min_dif = 0.01

    Z = J(a, b)
    prev_J = J(a, b)
    temp = a
    a = a - alpha * partial_a_J(a, b)
    b = b - alpha * partial_b_J(temp, b)
    current_J = J(a, b)

    surf = plt.contour(X, Y, Z, 40)

    while (current_J - prev_J) > min_dif:
      prev_J = current_J
      temp = a
      a = a - alpha * partial_a_J(a, b)
      b = b - alpha * partial_b_J(temp, b)
      current_J = J(a, b)
      plt.plot(a, b)

    plt.show()
    print(a)
    print(b)

if __name__ == "__main__":
        main()
