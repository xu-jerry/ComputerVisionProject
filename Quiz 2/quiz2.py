# Quiz 2 - Gradient Descent Algorithm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# error function
def J(a, b):
        temp = 0
        for i in data:
                temp += (a + b * i[0] - i[1]) ** 2
        return temp / (2 * len(data))

# partial derivative of J with respect to theta_0
def partial_0_J(a, b):
    J = 0
    for i in data:
        J += (a + b * i[0] - i[1])
    J /=  len(data)
    return J

# partial derivative of J with respect to theta_1
def partial_1_J(a, b):
    J = 0
    for i in data:
        J += (a + b * i[0] - i[1]) * i[0]
    J /=  len(data)
    return J

def main():
    data = np.array([[0.5, 0.9], [0.3, 0.7], [1.1, 2.3], [2.0, 4.3], [3.5, 6.8], [4.1, 8], [0, 0.1], [5.8, 11]])

    # initialize variables at reasonable values
    theta_0 = np.arange(-3, 5, 0.1)
    theta_1 = np.arange(0, 3, 0.1)
    X, Y = np.meshgrid(theta_0, theta_1)
    theta_0 = 0
    theta_1 = 0
    alpha = 0.1
    min_dif = 0.01

    Z= J(X, Y)
    prev_J = J(theta_0, theta_1)
    temp = theta_0
    theta_0 = theta_0 - alpha * partial_0_J(theta_0, theta_1)
    theta_1 = theta_1 - alpha * partial_1_J(temp, theta_1)
    current_J = J(theta_0, theta_1)
    
    # plot the loss curve
    surf = plt.contour(X, Y, Z, 40)

    # find lowest loss
    while (current_J - prev_J) > min_dif:
        prev_J = current_J
        temp = theta_0
        theta_0 = theta_0 - alpha * partial_0_J(theta_0, theta_1)
        theta_1 = theta_1 - alpha * partial_1_J(temp, theta_1)
        current_J = J(theta_0, theta_1)
        plt.plot(theta_0, theta_1)

    plt.show()
    print(theta_0)
    print(theta_1)

if __name__ == "__main__":
        main()
