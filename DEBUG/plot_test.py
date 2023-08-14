import matplotlib.pyplot as plt
import numpy as np
# a = np.loadtxt(r"../wrong_action.txt").astype("uint32")
a = np.load(r"../logs/(8,210,160,3)_state.txt.npy")
print(a.shape)
plt.imshow(a[0])
plt.show()