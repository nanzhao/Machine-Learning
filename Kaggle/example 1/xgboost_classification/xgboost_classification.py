import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('Train.csv', encoding="ISO-8859-1") # add encoding method ISO here
test = pd.read_csv('Test.csv',  encoding="ISO-8859-1") # add encoding method ISO here

#合成一个总的data
train['source']= 'train'
test['source'] = 'test'
data=pd.concat([train, test],ignore_index=True)

#print(data.shape)
#print(data.describe())
#print(data.apply(lambda x: sum(x.isnull()))) # 看看少了那些数据

"""
var = ['Gender','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','Source']
for v in var:
    print('\n%s这一列数据的不同取值和出现的次数\n'%v)
    print(data[v].value_counts())
# 要是泰坦尼克号的话，这里就开始画图了
"""
#print(len(data['City'].unique()))
data.drop('City',axis=1,inplace=True) # 类型太多，直接丢掉了

#创建一个年龄的字段Age
#print(data['DOB'].head())
data['Age'] = data['DOB'].apply(lambda x: 117 - int(x[-2:]))
#print(data['Age'].head())

#把原始的DOB字段去掉:
data.drop('DOB',axis=1,inplace=True)
#data.boxplot(column=['EMI_Loan_Submitted'], return_type='axes')

#好像缺失值比较多，干脆就开一个新的字段，表明是缺失值还是不是缺失值
data['EMI_Loan_Submitted_Missing'] = data['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
#data[['EMI_Loan_Submitted','EMI_Loan_Submitted_Missing']].head(10)
#原始那一列就可以不要了
data.drop('EMI_Loan_Submitted',axis=1,inplace=True)

#len(data['Employer_Name'].value_counts())
#丢掉
data.drop('Employer_Name',axis=1,inplace=True)

#data.boxplot(column='Existing_EMI',return_type='axes')
data['Existing_EMI'].describe()
#缺省值不多，用均值代替
data['Existing_EMI'].fillna(3.636342e+03, inplace=True)

#data.boxplot(column=['Interest_Rate'],return_type='axes')
#缺省值太多，也造一个字段，表示有无
data['Interest_Rate_Missing'] = data['Interest_Rate'].apply(lambda x: 1 if pd.isnull(x) else 0)
#print(data[['Interest_Rate','Interest_Rate_Missing']].head(10))
data.drop('Interest_Rate',axis=1,inplace=True)

#不！要！了！，是的，不要了！！！
data.drop('Lead_Creation_Date',axis=1,inplace=True)

#找中位数去填补缺省值（因为缺省的不多）
data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(),inplace=True)
data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(),inplace=True)

# 缺省值太多。。。是否缺省。。。
data['Loan_Amount_Submitted_Missing'] = data['Loan_Amount_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data['Loan_Tenure_Submitted_Missing'] = data['Loan_Tenure_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
#原来的字段就没用了
data.drop(['Loan_Amount_Submitted','Loan_Tenure_Submitted'],axis=1,inplace=True)

#没想好怎么用。。。不要了。。。
data.drop('LoggedIn',axis=1,inplace=True)

# 可能对接多个银行，所以也不要了
data.drop('Salary_Account',axis=1,inplace=True)

#和之前一样的处理，有或者没有
data['Processing_Fee_Missing'] = data['Processing_Fee'].apply(lambda x: 1 if pd.isnull(x) else 0)
#旧的字段不要了
data.drop('Processing_Fee',axis=1,inplace=True)

data['Source'] = data['Source'].apply(lambda x: 'others' if x not in ['S122','S133'] else x)
#data['Source'].value_counts()
print(data.apply(lambda x: sum(x.isnull()))) # 按列统计缺省值

# 把非数字化的列，用数字来表示， 原类型是object的
#print(data.dtypes)
#print(data.head())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['Device_Type','Filled_Form','Gender','Var1','Var2','Mobile_Verified','Source']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
#one-hot编码
data = pd.get_dummies(data, columns=var_to_encode)
print(data.columns)

# 区分训练和测试数据
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']
train.drop('source',axis=1,inplace=True)
test.drop(['source','Disbursed'],axis=1,inplace=True)

train.to_csv('train_modified2.csv',index=False)
test.to_csv('test_modified2.csv',index=False)



#plt.show()



