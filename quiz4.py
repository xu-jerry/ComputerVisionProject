# Quiz 4 - Model Complexity

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#test data
data = np.array([[0.5, 0.9], [0.3, 0.7], [1.1, 2.3], [2.0, 4.3], [3.5, 6.8], [4.1, 8], [0, 0.1], [5.8, 11]])

#error and partial derivatives for linear approximation
def partial_0_J(a, b):
    J = 0
    for i in data:
        J += (a + b * i[0] - i[1])
    J /=  len(data)
    return J

def partial_1_J(a, b):
    J = 0
    for i in data:
        J += (a + b * i[0] - i[1]) * i[0]
    J /=  len(data)
    return J

def J(a, b):
        temp = 0
        for i in data:
                temp += (a + b * i[0] - i[1]) ** 2
        return temp / (2 * len(data))

#error and partial derivatives for quadratic approximation
def partial_0_J(a, b, c):
    J = 0
    for i in data:
        J += (a + b * i[0] + c * i[0] ** 2 - i[1])
    J /=  len(data)
    return J

def partial_1_J(a, b, c):
    J = 0
    for i in data:
        J += (a + b * i[0] + c * i[0] ** 2 - i[1]) * i[0]
    J /=  len(data)
    return J

def partial_2_J(a, b, c):
    J = 0
    for i in data:
        J += (a + b * i[0] + c * i[0] ** 2 - i[1]) * i[0] ** 2
    J /=  len(data)
    return J

def J(a, b, c):
        temp = 0
        for i in data:
                temp += (a + b * i[0] + c * i[0] ** 2 - i[1]) ** 2
        return temp / (2 * len(data))

def main():
    for i in data:
        plt.scatter(i[0], i[1])

    # for linear
    theta_0 = 0
    theta_1 = 0
    alpha = 0.1
    min_dif = 0.01
    
    prev_J = J(theta_0, theta_1)
    temp = theta_0
    theta_0 = theta_0 - alpha * partial_0_J(theta_0, theta_1)
    theta_1 = theta_1 - alpha * partial_1_J(temp, theta_1)
    current_J = J(theta_0, theta_1)


    while (current_J - prev_J) > min_dif:
        prev_J = current_J
        temp = theta_0
        theta_0 = theta_0 - alpha * partial_0_J(theta_0, theta_1)
        theta_1 = theta_1 - alpha * partial_1_J(temp, theta_1)
        current_J = J(theta_0, theta_1)

    X = np.arange(0, 6, 0.1)
    Y = theta_0 + theta_1 * X
    plt.plot(X, Y)
    
    # quadratic

    theta_0 = 0
    theta_1 = 0
    theta_2 = 0
    alpha = 0.1
    min_dif = 0.01
    
    prev_J = J(theta_0, theta_1, theta_2)
    temp0 = theta_0
    temp1 = theta_1
    theta_0 = theta_0 - alpha * partial_0_J(theta_0, theta_1, theta_2)
    theta_1 = theta_1 - alpha * partial_1_J(temp0, theta_1, theta_2)
    theta_2 = theta_2 - alpha * partial_2_J(temp0, temp1, theta_2)
    current_J = J(theta_0, theta_1, theta_2)


    while (current_J - prev_J) > min_dif:
        prev_J = current_J
        temp0 = theta_0
        temp1 = theta_1
        theta_0 = theta_0 - alpha * partial_0_J(theta_0, theta_1, theta_2)
        theta_1 = theta_1 - alpha * partial_1_J(temp0, theta_1, theta_2)
        theta_2 = theta_2 - alpha * partial_2_J(temp0, temp1, theta_2)
        current_J = J(theta_0, theta_1, theta_2)

    X = np.arange(0, 6, 0.1)
    Y = theta_0 + theta_1 * X + theta_2 * X ** 2
    plt.plot(X, Y)
    plt.show()

if __name__ == "__main__":
        main()
