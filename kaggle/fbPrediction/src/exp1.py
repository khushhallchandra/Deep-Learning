import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv('../data/train.csv')

time = df_train['time']

def process_one_cell(df_train, df_test, x_min, x_max, y_min, y_max):
    x_border_augment = 0.02
    y_border_augment = 0.02

    #Working on df_train
    df_cell_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment) &
                               (df_train['y'] >= y_min-y_border_augment) & (df_train['y'] < y_max+y_border_augment)]
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]
    
    #Working on df_test
    # to be delete: df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    df_cell_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max) &
                               (df_test['y'] >= y_min) & (df_test['y'] < y_max)]
    row_ids = df_cell_test.index

    #Feature engineering on x and y
    df_cell_train.loc[:,'x'] *= fw[0]
    df_cell_train.loc[:,'y'] *= fw[1]
    df_cell_test.loc[:,'x'] *= fw[0]
    df_cell_test.loc[:,'y'] *= fw[1]
    
    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values.astype(float)
    X_test = df_cell_test.values.astype(float)

    #Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=26, weights='distance', 
                               metric='manhattan', n_jobs=2)
    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3]) 
    
    return pred_labels, row_ids
