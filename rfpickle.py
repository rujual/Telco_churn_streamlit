import pickle
#imported in randomforest
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


data_path=('Data.csv')
df_churn1 = pd.read_csv(data_path)

def convert_data(df_churn):
    empty_cols=['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
           'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
    for i in empty_cols:
        df_churn[i]=df_churn[i].replace(" ",np.nan)

    df_churn.drop('cluster_number', axis=1, inplace=True)
    df_churn.drop('customerID', axis=1, inplace=True)
    df_churn=df_churn.dropna()
    binary_cols=['Partner','Dependents','PhoneService','PaperlessBilling']

    for i in binary_cols:
        df_churn[i]=df_churn[i].replace({"Yes":1,"No":0})

    #Encoding column 'gender'
    df_churn['gender']=df_churn['gender'].replace({"Male":1,"Female":0})


    category_cols=['PaymentMethod','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract']

    for cc in category_cols:
        dummies = pd.get_dummies(df_churn[cc], drop_first=False)
        dummies = dummies.add_prefix("{}#".format(cc))
        df_churn.drop(cc, axis=1, inplace=True)
        df_churn = df_churn.join(dummies)
    df_churn['Churn']=df_churn['Churn'].replace({"Yes":1,"No":0})
    return df_churn


df_churn2 = convert_data(df_churn1)
df_churn2.to_csv('Data_modified.csv', index=False)
print('Converting done!')

#random forest code


print('process started')

data_path1 = ('Data_modified.csv')
df = pd.read_csv(data_path1)

X = df.loc[:, df.columns != 'Churn']
y = df.loc[:, df.columns == 'Churn']


sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X, y)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2,4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


rfc=RandomForestClassifier(random_state=42)
gsv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
rfc.fit(X_train_res, y_train_res)


#rfc_best = gsv_rfc.best_estimator_
rfc_best=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 50, max_depth=8,
                                criterion='gini')

rfc_best.fit(X_train_res, y_train_res)

filename = 'finalized_model.sav'
pickle.dump(rfc_best, open(filename, 'wb'))
