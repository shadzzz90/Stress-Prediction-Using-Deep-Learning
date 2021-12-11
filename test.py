import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
x = np.arange(0, 300, 1)
y = np.arange(0, 300, 1)

x, y = np.meshgrid(x, y)

print(x.shape, y.shape)

trainInput2 = np.load('trainInput2_1.npy')

z = trainInput2[0][:][:]
#
# scale = StandardScaler()
#
# # scale.fit(z)
#
# z = scale.fit_transform(z)
# print(scale.mean_.shape)
#
# z  = scale.inverse_transform(z)

plt.contourf(x,y,z, cmap='rainbow')
plt.colorbar()
plt.show()



