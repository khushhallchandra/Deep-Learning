import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier


def prepare_data(df_train, df_test, n_cell_x, n_cell_y):
    """
    Some feature engineering (mainly with the time feature) + normalization 
    of all features (substracting the mean and dividing by std) +  
    computation of a grid (size = n_cell_x * n_cell_y), which is included
    as a new column (grid_cell) in the dataframes.
    
    Parameters:
    ----------    
    df_train: pandas DataFrame
              Training data
    df_test : pandas DataFrame
              Test data
    n_cell_x: int
              Number of grid cells on the x axis
    n_cell_y: int
              Number of grid cells on the y axis
    
    Returns:
    -------    
    df_train, df_test: pandas DataFrame
                       Modified training and test datasets.
    """  
    print('Feature engineering...')
    print('    Computing some features from x and y ...')
    ##x, y, and accuracy remain the same
        ##New feature x/y
    eps = 0.00001  #required to avoid some divisions by zero.
    df_train['x_d_y'] = df_train.x.values / (df_train.y.values + eps) 
    df_test['x_d_y'] = df_test.x.values / (df_test.y.values + eps) 
        ##New feature x*y
    df_train['x_t_y'] = df_train.x.values * df_train.y.values  
    df_test['x_t_y'] = df_test.x.values * df_test.y.values
    
    print('    Creating datetime features ...')
    ##time related features (assuming the time = minutes)
    initial_date = np.datetime64('2014-01-01T01:01',   #Arbitrary decision
                                 dtype='datetime64[m]') 
        #working on df_train  
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df_train.time.values)    
    df_train['hour'] = d_times.hour
    df_train['weekday'] = d_times.weekday
    df_train['day'] = d_times.day
    df_train['month'] = d_times.month
    df_train['year'] = d_times.year
    df_train = df_train.drop(['time'], axis=1)
        #working on df_test    
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df_test.time.values)    
    df_test['hour'] = d_times.hour
    df_test['weekday'] = d_times.weekday
    df_test['day'] = d_times.day
    df_test['month'] = d_times.month
    df_test['year'] = d_times.year
    df_test = df_test.drop(['time'], axis=1)
    
    print('Computing the grid ...')
    #Creating a new colum with grid_cell id  (there will be 
    #n = (n_cell_x * n_cell_y) cells enumerated from 0 to n-1)
    size_x = 10. / n_cell_x
    size_y = 10. / n_cell_y
        #df_train
    xs = np.where(df_train.x.values < eps, 0, df_train.x.values - eps)
    ys = np.where(df_train.y.values < eps, 0, df_train.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    df_train['grid_cell'] = pos_y * n_cell_x + pos_x
            #df_test
    xs = np.where(df_test.x.values < eps, 0, df_test.x.values - eps)
    ys = np.where(df_test.y.values < eps, 0, df_test.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    df_test['grid_cell'] = pos_y * n_cell_x + pos_x 
    
    ##Normalization
    print('Normalizing the data: (X - mean(X)) / std(X) ...')
    cols = ['x', 'y', 'accuracy', 'x_d_y', 'x_t_y', 'hour', 
            'weekday', 'day', 'month', 'year']
    for cl in cols:
        ave = df_train[cl].mean()
        std = df_train[cl].std()
        df_train[cl] = (df_train[cl].values - ave ) / std
        df_test[cl] = (df_test[cl].values - ave ) / std
        
    #Returning the modified dataframes
    return df_train, df_test
