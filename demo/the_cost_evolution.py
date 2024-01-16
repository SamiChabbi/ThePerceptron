import numpy as np 
import matplotlib.pyplot as plt 

#we create an array with different values that can take y_hat
y_hat = np.linspace(0.001, 0.999, 1000, dtype="float")

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.title("the evolution of the cost when y = 0" , fontsize=20)
plt.ylabel("the cost", fontsize=15)
plt.xlabel("value that could take y_hat ", fontsize=15)
#let's plot the cost function when y = 0
plt.plot(y_hat, -np.log(1 - y_hat))
plt.subplot(1, 2, 2)
plt.title("the evolution of the cost when y = 1", fontsize=20)
plt.ylabel("the cost", fontsize=15)
plt.xlabel("value that could take y_hat ", fontsize=15)
#let's plot the cost function when y = 1
plt.plot(y_hat, -np.log(y_hat))
plt.show()
