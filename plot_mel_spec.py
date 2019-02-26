import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(9,5))
X = np.arange(0., 8000., 10)
inside = 1 + X/700
Y  = 2595*np.log10(inside)
plt.plot(X, Y)
plt.grid()
plt.xlabel('Escala Hertz')
plt.ylabel('Escala Mel')
plt.show()
