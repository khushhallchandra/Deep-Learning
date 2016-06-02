import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

train_data = pd.read_csv('../input/train.csv')
mean_train_data = train_data.groupby('place_id').mean()
std_train_data = train_data.groupby('place_id').std()

acc_df = pd.concat([mean_train_data['accuracy'],std_train_data['x'],std_train_data['y']], axis=1)
acc_df.rename(columns={'accuracy':'mean_accuracy','x':'std_x','y':'std_y'}, inplace=True)

acc_df.fillna(0, inplace=True)

p = plt.hist(acc_df.mean_accuracy, bins=np.arange(min(acc_df.mean_accuracy), max(acc_df.mean_accuracy) + 1, 1))
plt.xlabel('Mean Accuracy')
plt.ylabel('Count')
plt.title('Counts of mean accuracy for places')

plt.show()


p = plt.hist(acc_df.std_x, bins=np.arange(min(acc_df.std_x), max(acc_df.std_x) + .01, .01))
plt.xlabel('Std(y)')
plt.ylabel('Count')
plt.title('Counts of std(x) for places')

plt.show()


p = plt.hist(acc_df.std_y, bins=np.arange(min(acc_df.std_y), max(acc_df.std_y) + .001, .001))
plt.xlabel('Std(y)')
plt.ylabel('Count')
plt.title('Counts of std(y) for places')

plt.show()

p = plt.scatter(acc_df.mean_accuracy,acc_df.std_x)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(x)')
plt.title('std(x) vs Mean accuracy for each place')

plt.show()


p = plt.scatter(acc_df.mean_accuracy, acc_df.std_y)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(y)')
plt.title('std(y) vs mean accuracy for each place')

plt.show()


