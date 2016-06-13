import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm

time = df_train['time']
x = df_train['x']
y = df_train['y']
accuracy = df_train['accuracy']

n, bins, patches = plt.hist(time, 50, normed=1, facecolor='green', alpha=0.75)
plt.title('Data density at different time dimension')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

n, bins, patches = plt.hist(accuracy, 50, normed=1, facecolor='green', alpha=0.75)
plt.title('Histogram of location accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

bins = 20
while bins <=160:
    plt.hist2d(x, y, bins=bins, norm=LogNorm())
    plt.colorbar()
    plt.title('x and y location histogram - ' + str(bins) + ' bins')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    bins = bins * 2
