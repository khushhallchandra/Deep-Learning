df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head()

# Sample them for quicker visualisations
df_train_sample = df_train.sample(n=1000000)
df_test_sample = df_test.sample(n=1000000)

counts1, bins1 = np.histogram(df_train["accuracy"], bins=50)
binsc1 = bins1[:-1] + np.diff(bins1)/2.

counts2, bins2 = np.histogram(df_test["accuracy"], bins=50)
binsc2 = bins2[:-1] + np.diff(bins2)/2.

plt.figure(0, figsize=(14,4))

plt.subplot(121)
plt.bar(binsc1, counts1/(counts1.sum()*1.0), width=np.diff(bins1)[0])
plt.grid(True)
plt.xlabel("Accuracy")
plt.ylabel("Fraction")
plt.title("Train")

plt.subplot(122)
plt.bar(binsc2, counts2/(counts2.sum()*1.0), width=np.diff(bins2)[0])
plt.grid(True)
plt.xlabel("Accuracy")
plt.ylabel("Fraction")
plt.title("Test")

plt.show()



current_palette = sns.color_palette()

counts1, bins1 = np.histogram(df_train["time"], bins=50)
binsc1 = bins1[:-1] + np.diff(bins1)/2.

counts2, bins2 = np.histogram(df_test["time"], bins=50)
binsc2 = bins2[:-1] + np.diff(bins2)/2.

plt.figure(1, figsize=(12,3))

plt.subplot(121)
plt.bar(binsc1, counts1/(counts1.sum()*1.0), width=np.diff(bins1)[0], color=current_palette[0])
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Fraction")
plt.title("Train")

plt.subplot(122)
plt.bar(binsc2, counts2/(counts2.sum()*1.0), width=np.diff(bins2)[0], color=current_palette[1])
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Fraction")
plt.title("Test")

plt.show()

plt.figure(2, figsize=(12,3))
plt.bar(binsc1, counts1/(counts1.sum()*1.0), width=np.diff(bins1)[0], color=current_palette[0], label="Train")
plt.bar(binsc2, counts2/(counts2.sum()*1.0), width=np.diff(bins2)[0], color=current_palette[1], label="Test")
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Fraction")
plt.title("Test")
plt.legend()
plt.show()




# Check how how frequently different locations appear
df_placecounts = df_train["place_id"].value_counts()

counts, bins = np.histogram(df_placecounts.values, bins=50)
binsc = bins[:-1] + np.diff(bins)/2.

plt.figure(3, figsize=(12,6))
plt.bar(binsc, counts/(counts.sum()*1.0), width=np.diff(bins)[0])
plt.grid(True)
plt.xlabel("Number of place occurances")
plt.ylabel("Fraction")
plt.title("Train")
plt.show()



# Check if accuracy of signal corresponds with time
plt.figure(4, figsize=(12,10))

plt.subplot(211)
plt.scatter(df_train_sample["time"], df_train_sample["accuracy"], s=1, c='k', lw=0, alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.xlim(df_train_sample["time"].min(), df_train_sample["time"].max())
plt.ylim(df_train_sample["accuracy"].min(), df_train_sample["accuracy"].max())
plt.title("Train")

plt.subplot(212)
plt.scatter(df_test_sample["time"], df_test_sample["accuracy"], s=1, c='k', lw=0, alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.xlim(df_test_sample["time"].min(), df_test_sample["time"].max())
plt.ylim(df_test_sample["accuracy"].min(), df_test_sample["accuracy"].max())
plt.title("Test")

plt.show()



# Does the accuracy vary with location?  Check within 100x100m spots
df_train_sample["xround"] = df_train_sample["x"].round(decimals=1)
df_train_sample["yround"] = df_train_sample["y"].round(decimals=1)
df_groupxy = df_train_sample.groupby(["xround", "yround"]).agg({"accuracy":[np.mean, np.std]})
df_groupxy.head()



idx = np.asarray(list(df_groupxy.index.values))
plt.figure(5, figsize=(14,6))

plt.subplot(121)
plt.scatter(idx[:,0], idx[:,1], s=20, c=df_groupxy["accuracy", "mean"], marker='s', lw=0, cmap=plt.cm.viridis)
plt.colorbar().set_label("Mean accuracy")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(0,10)
plt.ylim(0,10)

plt.subplot(122)
plt.scatter(idx[:,0], idx[:,1], s=20, c=df_groupxy["accuracy", "std"], marker='s', lw=0, cmap=plt.cm.viridis)
plt.colorbar().set_label("Std accuracy")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(0,10)
plt.ylim(0,10)

plt.tight_layout()
plt.show()




