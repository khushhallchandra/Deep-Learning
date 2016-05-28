df_train["hour"]       = (df_train["time"]%(60*24))//60.
df_train["dayofweek"]  = np.ceil((df_train["time"]%(60*24*7))//(60.*24))
df_train["day"]  = np.ceil((df_train["time"]/(60*24)))
df_train["week"] = np.ceil((df_train["time"]/(60*24*7)))
df_test["hour"]        = (df_test["time"]%(60*24))//60.
df_test["dayofweek"]   = np.ceil((df_test["time"]%(60*24*7))//(60.*24))
df_test["day"]   = np.ceil((df_test["time"]/(60*24)))
df_test["week"]  = np.ceil((df_test["time"]/(60*24*7)))
