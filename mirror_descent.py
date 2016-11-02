import numpy as np
import matplotlib.pyplot as plt

# Cost function
f = np.ones((50,50))
f *= np.arange(50)

plt.imshow(f, cmap='afmhot')
plt.plot(25, 25, "o", markersize=150, markeredgewidth=2,
        markeredgecolor='k', markerfacecolor='None')
plt.savefig('mirror_background.png')
