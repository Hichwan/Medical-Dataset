import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree

def dataset():
    dataset1 = pd.read_csv('82c67289-d5be-46a2-929a-e05f34fb3cb5.csv')
    dataset2 = pd.read_csv('f5fc3b16-5bc8-4b22-98de-c836bf2cd687.csv')

    print(dataset1.head())
    print(dataset1.columns)
    print(dataset1.info())

    columns_to_keep = ['FAC_NAME','COUNTY_NAME', 'BEG_DATE', 'END_DATE', 'OP_STATUS', 'TYPE_HOSP',
                   'TEACH_RURL','AVL_BEDS', 'STF_BEDS', 'VIS_TOT', 'GRIP_TOT',
                   'DAY_TOT', 'GROP_TOT', 'DED_TOT', 'CAP_TOT',
                   'NET_TOT', 'TOT_OP_EXP', 'EXP_PIPS', 'EXP_POPS',
                   'PHY_COMP', 'LOG_VIS_TOT', 'LOG_GRIP_TOT', 'LOG_DAY_TOT', 'LOG_GROP_TOT', 'LOG_DED_TOT', 'LOG_CAP_TOT', 'LOG_DED_TOT', 'LOG_NET_TOT'
                   , 'LOG_TOT_OP_EXP', 'LOG_EXP_PIPS', 'LOG_EXP_POPS', 'LOG_PHY_COMP']

    columns_to_convert=['VIS_TOT', 'GRIP_TOT',
                   'DAY_TOT', 'GROP_TOT', 'DED_TOT', 'CAP_TOT',
                   'NET_TOT', 'TOT_OP_EXP', 'EXP_PIPS', 'EXP_POPS',
                   'PHY_COMP']


    dataset1[columns_to_convert] = dataset1[columns_to_convert].replace('###########', pd.NA).fillna(0)
    dataset2[columns_to_convert] = dataset2[columns_to_convert].replace('###########', pd.NA).fillna(0)

    for columns in columns_to_convert: #convert to float numbers
        dataset1[columns] = dataset1[columns].astype(float)
        dataset2[columns] = dataset2[columns].astype(float)

    dataset1['TEACH_RURL'].fillna('N/A', inplace=True)
    dataset2['TEACH_RURL'].fillna('N/A', inplace=True)

    dataset1['BEG_DATE'] = pd.to_datetime(dataset1['BEG_DATE'])
    dataset1['END_DATE'] = pd.to_datetime(dataset1['END_DATE'])
    dataset2['BEG_DATE'] = pd.to_datetime(dataset2['BEG_DATE'])
    dataset2['END_DATE'] = pd.to_datetime(dataset2['END_DATE'])

    sns.boxplot(x=dataset1['NET_TOT'])
    plt.show()
    Q1 = dataset1['NET_TOT'].quantile(0.25)
    Q3 = dataset1['NET_TOT'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    outliers = dataset1[dataset1['NET_TOT'] > upper_bound]
    print(outliers)

    skewness = dataset1['NET_TOT'].skew()
    print(f"Skewness of NET_TOT: {skewness}")

    #log of net total as skewness is too high (too far right)
    #+1 to avoid 0s, min_value/shift to prevent negatives
    for column in columns_to_convert:
        min_value = dataset1[column].min()
        if min_value < 0:
            dataset1[f'LOG_{column}'] = np.log(dataset1[column] - min_value + 1)
        else:
            dataset1[f'LOG_{column}'] = np.log(dataset1[column] + 1)

    for column in columns_to_convert:
        min_value = dataset2[column].min()
        if min_value < 0:
            dataset2[f'LOG_{column}'] = np.log(dataset2[column] - min_value + 1)
        else:
            dataset2[f'LOG_{column}'] = np.log(dataset2[column] + 1)

    sns.boxplot(x=dataset1['LOG_NET_TOT'])
    plt.show()
    skewness = dataset1['LOG_NET_TOT'].skew()
    print(f"Skewness of LOG_NET_TOT: {skewness}")

    decisiontree(dataset1)
    decisiontree(dataset2)
    logdecisiontree(dataset1)
    logdecisiontree(dataset2)
    dataset = pd.merge(dataset1[columns_to_keep], dataset2[columns_to_keep], on =['FAC_NAME','COUNTY_NAME'], how = 'inner', suffixes=('_q1','_q2'))


    dataset.dropna(subset = ['FAC_NAME'], inplace = True)
    dataset.drop_duplicates(inplace = True)



    dataset.to_csv('Dataset_Hospital_2024.csv', index = False)

def decisiontree(dataset):
    X = dataset[['VIS_TOT', 'DAY_TOT', 'AVL_BEDS', 'STF_BEDS']]
    y = dataset['NET_TOT']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 88)

    model = DecisionTreeRegressor(random_state = 88)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Err: {mse}')
    print(f'R-squared: {r2}')

    # Visualize the decision tree
    plt.figure(figsize=(20,10))
    tree.plot_tree(model, filled=True, feature_names=['VIS_TOT', 'DAY_TOT', 'AVL_BEDS', 'STF_BEDS'], fontsize=12)
    plt.show()

    df_predictions = X_test.copy()
    df_predictions['predictions'] = y_pred

    df_predictions.to_csv('predictions.csv', index= False)
def logdecisiontree(dataset):
    X = dataset[['VIS_TOT', 'DAY_TOT', 'AVL_BEDS', 'STF_BEDS']]
    y = dataset['LOG_NET_TOT']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 88)

    model = DecisionTreeRegressor(random_state = 88)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Err: {mse}')
    print(f'R-squared: {r2}')

    # Visualize the decision tree
    plt.figure(figsize=(20,10))
    tree.plot_tree(model, filled=True, feature_names=['VIS_TOT', 'DAY_TOT', 'AVL_BEDS', 'STF_BEDS'], fontsize=12)
    plt.show()

    df_predictions = X_test.copy()
    df_predictions['predictions'] = y_pred

    df_predictions.to_csv('predictionslog.csv', index= False)

def main():
    dataset()

if __name__== "__main__":
    main()