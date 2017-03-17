
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.chdir("G:\\titanic")#切换到你自己放数据的路径
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[2]:

test_PassengerId = test["PassengerId"] #取出测试集的PassengerId
#填充港口
train.Embarked = train.Embarked.fillna(train.Embarked.mode()[0] )
print(train.Embarked.isnull().sum())
#填充票价
test.Fare = test.Fare.fillna(test.Fare.mean() )
print(test.Fare.isnull().sum())


# In[3]:

from sklearn.neighbors import KNeighborsRegressor #KNN
income_imputer = KNeighborsRegressor(n_neighbors=1)
# 数据分为两部分，有缺失的和无缺失的，用无缺失的数据建立模型来判断缺失数据的可能取值
train_age = train[train.Age.isnull()==False]  #年龄非空
train_null_age = train[train.Age.isnull()==True]  #年龄为空


# In[4]:

cols = ['Fare', 'Parch', 'SibSp', 'Pclass']
income_imputer.fit(train_age[cols], train_age.Age)
train_values = income_imputer.predict(train_null_age[cols])
train_null_age.ix[:,'Age']=train_values
newtrain = train_age.append(train_null_age)
newtrain.Embarked = newtrain.Embarked.fillna(newtrain.Embarked.mode()[0] )
newtrain.info()


# In[5]:

test_age = test[test.Age.isnull()==False]  #年龄非空
test_null_age = test[test.Age.isnull()==True]  #年龄为空
cols = ['Fare', 'Parch', 'SibSp', 'Pclass']
income_imputer.fit(test_age[cols], test_age.Age)
# 用房产贷款次数以及未结束贷款次数两个特征来训练月收入
test_values = income_imputer.predict(test_null_age[cols])
# 再用模型预测缺失值中的月收入
test_null_age.ix[:,'Age']=test_values
newtest = test_age.append(test_null_age)
newtest.info()


# In[6]:

# all_data = newtrain.append( newtest , ignore_index = True )
# df = all_data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)


# In[7]:

# mybins = list(range(0,90,10)) 
# df['age_bin'] = pd.cut(df.Age, bins=mybins)
# pd.value_counts(df['age_bin'])


# In[8]:

newtrain_drop = newtrain.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
newtest_drop = newtest.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)


# In[9]:

mybins = list(range(0,90,10)) 
newtrain_drop['age_bin'] = pd.cut(newtrain_drop.Age, bins=mybins)
newtest_drop['age_bin'] = pd.cut(newtest_drop.Age, bins=mybins)


# In[10]:

imputed_train =  pd.get_dummies(newtrain_drop, columns=['Embarked', 'Sex','age_bin'])
imputed_test =  pd.get_dummies(newtest_drop, columns=['Embarked', 'Sex','age_bin'])


# In[11]:

# newdf = pd.get_dummies(df, columns=['Embarked', 'Sex','age_bin']) #亚变量变换
# imputed_train = newdf.ix[0:890]
# imputed_test = newdf.ix[891:1309]

y = imputed_train['Survived']
X = imputed_train.ix[:,imputed_train.columns != 'Survived'] #去除标签列
imputed_test = imputed_test.ix[:,imputed_test.columns != 'Survived']
print (X.columns.tolist())
print (X.shape)


# In[10]:

from sklearn.cross_validation import train_test_split 
train_X, test_X, train_y, test_y = train_test_split(X, y ,train_size=0.7,random_state=1)
#train_X  训练集中切分的训练子集
#test_X  训练集中切分的测试子集
#train_y  训练集中切分的训练子集标签列
#test_y  训练集中切分的测试子集标签列
print (train_X.shape)
print (test_X.shape)


# In[12]:

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV


# In[12]:

rf_clf = RandomForestClassifier(max_depth = 7, n_estimators=70, min_samples_split =  100,min_samples_leaf=20,max_features='sqrt')
rf_clf.fit(train_X,train_y)
rf_preds = rf_clf.predict(test_X)
rf_pre = rf_clf.predict_proba(test_X) 
roc_auc_score(test_y,rf_pre[:,1]) 


# In[19]:

rf_clf = RandomForestClassifier(max_depth = 7, n_estimators=70, min_samples_split =  100,min_samples_leaf=20,max_features='sqrt')
rf_clf.fit(X,y)
rf_preds = rf_clf.predict(imputed_test)
#rf_preds = rf_preds.astype(int)
#np.savetxt("rf_preds.csv", rf_preds, delimiter=",")


# In[72]:

#np.savetxt("imputed_test.csv", imputed_test, delimiter=",")


# In[20]:

rf_preds


# In[17]:

submission = pd.DataFrame({
        "PassengerId": test_PassengerId,
        "Survived": rf_preds
    })
submission.to_csv('rf_preds2.csv', index=False)


# In[14]:

xgb_clf =  XGBClassifier( learning_rate =0.1,                          n_estimators=30,                         max_depth=7,                          min_child_weight=5,                          gamma=0,                         subsample=0.8,                         colsample_bytree=0.8,                         objective= 'binary:logistic',                         nthread=4,                          scale_pos_weight=1,                          seed=27)
xgb_train_X = train_X.as_matrix()
xgb_train_y = train_y.as_matrix()
xgb_test_X = test_X.as_matrix()
xgb_test_y = test_y.as_matrix()
xgb_clf.fit(xgb_train_X,xgb_train_y)
xgb_preds = xgb_clf.predict(xgb_test_X)
xgb_pre = xgb_clf.predict_proba(xgb_test_X) 
roc_auc_score(xgb_test_y,xgb_pre[:,1]) 


# In[15]:

GBDT_clf = GradientBoostingClassifier(n_estimators = 70,max_depth = 5, min_samples_split = 190,
                                  min_samples_leaf=20,max_features='sqrt',learning_rate = 0.1)
GBDT_clf.fit(train_X,train_y)
GBDT_preds = GBDT_clf.predict(test_X)
GBDT_pre = GBDT_clf.predict_proba(test_X) 
roc_auc_score(test_y,GBDT_pre[:,1]) 


# In[16]:

adb_clf = AdaBoostClassifier(n_estimators=40, learning_rate=0.3)
adb_clf.fit(train_X,train_y)
adb_preds = adb_clf.predict(test_X)
adb_pre = adb_clf.predict_proba(test_X) 
roc_auc_score(test_y,adb_pre[:,1]) 


# In[17]:

from sklearn.cross_validation import KFold
ntrain = train_X.shape[0] #train_X训练集中切分的训练子集行数
ntest = test_X.shape[0]  #test_X训练集中切分的测试子集行数
SEED = 0 
NFOLDS = 5  
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

'''
训练单模型分别得到下一层训练集和测试集的一列
'''
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))  
    oof_test = np.zeros((ntest,)) 
    oof_test_skf = np.empty((NFOLDS, ntest))  #NFOLDS行，ntest列的二维array

    for i, (train_index, test_index) in enumerate(kf): #循环NFOLDS次
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr) #调用自定义的SklearnHelper方法中的train方法，本质也是fit训练

        oof_train[test_index] = clf.predict(x_te) 
        oof_test_skf[i, :] = clf.predict(x_test)  #固定行填充，循环一次，填充一行

    oof_test[:] = oof_test_skf.mean(axis=0)  #axis=0,按列求平均，最后保留一行
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1  #转置，从一行变为一列


# In[18]:

rf_params = {
    'n_jobs': -1,
    'n_estimators': 70,
    'max_depth': 7,
    'max_features' : 'sqrt',
    'min_samples_split' : 100
}

ada_params = {
    'n_estimators': 40,
    'learning_rate' : 0.3
}

gb_params = {
    'n_estimators': 110,
    'learning_rate' : 0.1,
    'min_samples_leaf': 20
}

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
knn
svm
logit = 


# In[56]:

#train_X, train_y, test_X, , test_y 
#x_train,  y_train, x_test, y_test
x_train = train_X.values  #获得训练子集的特征列 (623 ,18)  
y_train = train_y.values # 将训练子集标签列记为y_train (623 ,1)  用ravel()函数可以去除索引，也可以就用values,只保留标签列的值,因为train_y是pd.Series类型
x_test = test_X.values   #获得测试子集的特征列  (268, 18) 同样这样得到一个二维array，而不是pd.DataFrame
y_test = test_y.values # 将测试子集标签列记为y_train (268, 1)


# In[57]:

# 用别的模型来得到新的train and test 预测结果. 把这些结果作为新的特征
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)  #rf_oof_train是(623 ,1)， rf_oof_test (268,1)
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test)


# In[20]:

#得到了第二层模型的训练集特征集合x_train，训练集标签列y_train，测试集特征集合x_test,测试集标签列y_test
x_train = np.concatenate(( rf_oof_train, ada_oof_train, gb_oof_train), axis=1) #axis=1表示按列合并，三个(623 ,1)合并成(623 ,3)
x_test = np.concatenate(( rf_oof_test, ada_oof_test, gb_oof_test), axis=1)   #(268,3)
print("{},{}".format(x_train.shape, x_test.shape))


# In[26]:

import xgboost as xgb 
from xgboost.sklearn import XGBClassifier
gbm = xgb.XGBClassifier(learning_rate = 0.95, n_estimators= 16000, max_depth= 4,min_child_weight= 2,gamma=1,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread= -1,scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# In[35]:

logit_clf = LogisticRegression()
logit_clf.fit(x_train, y_train)
logit_pre = logit_clf.predict_proba(x_test) 
roc_auc_score(y_test,logit_pre[:,1]) 


