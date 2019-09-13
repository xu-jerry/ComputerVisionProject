# Quiz 1 - Plot J(theta_0, theta_1)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# the error function
def J(a, b):
        # test data
        data = np.array([[0.5, 0.9], [0.3, 0.7], [1.1, 2.3], [2.0, 4.3], [3.5, 6.8], [4.1, 8], [0, 0.1], [5.8, 11]])
        
        temp = 0
        for i in data:
                temp += (a + b * i[0] - i[1]) ** 2
        return temp / (2 * len(data))

def main():
        # make a 3-D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        #create range of x, y, and z
        theta_0 = np.linspace(-3, 5, 30)
        theta_1 = np.linspace(0, 3, 30)
        X, Y = np.meshgrid(theta_0, theta_1)
        Z = J(X, Y)
        
        # plot and label the graph
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax.set_xlabel('theta_0')
        ax.set_ylabel('theta_1')
        ax.set_zlabel('J')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

if __name__ == "__main__":
        main()
