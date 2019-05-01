import pandas as pd
import numpy as np
import os

#getting file paths for test and train data sets
parent_dir=os.path.join(os.path.pardir, 'data_science_projects', 'datasets')
train_data=os.path.join(parent_dir, 'train.csv')
test_data=os.path.join(parent_dir, 'test.csv')

#printing the file paths of data sets
print(parent_dir)
print(train_data)
print(test_data)

#creating dataframes for train and test data sets using pandas
train_df=pd.read_csv(train_data, index_col="PassengerId")
test_df=pd.read_csv(test_data, index_col="PassengerId")

#printing the type of dataframes
print(type(train_df))
print(type(test_df))

#printing the info about the test and train data sets
print(train_df.info())
print(test_df.info())

#info about train_dataframe :

"""
Train_data insights:
Int64Index: 891 entries, 1 to 891
Data columns (total 11 columns):
Survived    891 non-null int64
Pclass      891 non-null int64
Name        891 non-null object
Sex         891 non-null object
Age         714 non-null float64
SibSp       891 non-null int64
Parch       891 non-null int64
Ticket      891 non-null object
Fare        891 non-null float64
Cabin       204 non-null object
Embarked    889 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 83.5+ KB
"""

#info about test_dataframe:


"""
Test_data insights:

<class 'pandas.core.frame.DataFrame'>
Int64Index: 418 entries, 892 to 1309
Data columns (total 10 columns):
Pclass      418 non-null int64
Name        418 non-null object
Sex         418 non-null object
Age         332 non-null float64
SibSp       418 non-null int64
Parch       418 non-null int64
Ticket      418 non-null object
Fare        417 non-null float64
Cabin       91 non-null object
Embarked    418 non-null object
dtypes: float64(2), int64(3), object(5)
memory usage: 35.9+ KB
"""

#in order to work with the whole dataset we need to concatinate test and train datasets. But we can gain insight
#that test dataset does not have survived col, so we need to add that col manually so that the entire dataset is
#similar in structure

#adding survived col in test dataset with a default value
test_df['Survived']=-888

#checking the new test dataframe
print(test_df.info())

#concatinating the test and train datasets, if axis=0 then rows are  appended one below another, else if axis=1, cols are 
#appended one besides other
df=pd.concat((train_df, test_df), axis=0)
print(df.info())

#whole datasets:
"""
Whole dataset(train+test) insight:

<class 'pandas.core.frame.DataFrame'>
Int64Index: 1309 entries, 1 to 1309
Data columns (total 11 columns):
Age         1046 non-null float64
Cabin       295 non-null object
Embarked    1307 non-null object
Fare        1308 non-null float64
Name        1309 non-null object
Parch       1309 non-null int64
Pclass      1309 non-null int64
Sex         1309 non-null object
SibSp       1309 non-null int64
Survived    1309 non-null int64
Ticket      1309 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 122.7+ KB
"""

#Printing head, be default top 5 rows:
print(df.head())

"""
Top 5 rows of the dataframe:

                                                          Name  Parch  Pclass  \
PassengerId
1                                      Braund, Mr. Owen Harris      0       3
2            Cumings, Mrs. John Bradley (Florence Briggs Th...      0       1
3                                       Heikkinen, Miss. Laina      0       3
4                 Futrelle, Mrs. Jacques Heath (Lily May Peel)      0       1
5                                     Allen, Mr. William Henry      0       3

                Sex  SibSp  Survived            Ticket
PassengerId
1              male      1         0         A/5 21171
2            female      1         1          PC 17599
3            female      0         1  STON/O2. 3101282
4            female      1         1            113803
5              male      0         0            373450
"""
#printing the top 10 rows
print(df.head(10))

"""
Top 10 rowms from the dataframe:

                                                          Name  Parch  Pclass  \
PassengerId
1                                      Braund, Mr. Owen Harris      0       3
2            Cumings, Mrs. John Bradley (Florence Briggs Th...      0       1
3                                       Heikkinen, Miss. Laina      0       3
4                 Futrelle, Mrs. Jacques Heath (Lily May Peel)      0       1
5                                     Allen, Mr. William Henry      0       3
6                                             Moran, Mr. James      0       3
7                                      McCarthy, Mr. Timothy J      0       1
8                               Palsson, Master. Gosta Leonard      1       3
9            Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)      2       3
10                         Nasser, Mrs. Nicholas (Adele Achem)      0       2

                Sex  SibSp  Survived            Ticket
PassengerId
1              male      1         0         A/5 21171
2            female      1         1          PC 17599
3            female      0         1  STON/O2. 3101282
4            female      1         1            113803
5              male      0         0            373450
6              male      0         0            330877
7              male      0         0             17463
8              male      3         0            349909
9            female      0         1            347742
10           female      1         1            237736
"""

#printing the last rows, 5 by default
print(df.tail())

"""
Last 5 rows:
             Age Cabin Embarked      Fare                          Name  \
PassengerId
1305          NaN   NaN        S    8.0500            Spector, Mr. Woolf
1306         39.0  C105        C  108.9000  Oliva y Ocana, Dona. Fermina
1307         38.5   NaN        S    7.2500  Saether, Mr. Simon Sivertsen
1308          NaN   NaN        S    8.0500           Ware, Mr. Frederick
1309          NaN   NaN        C   22.3583      Peter, Master. Michael J

             Parch  Pclass     Sex  SibSp  Survived              Ticket
PassengerId
1305             0       3    male      0      -888           A.5. 3236
1306             0       1  female      0      -888            PC 17758
1307             0       3    male      0      -888  SOTON/O.Q. 3101262
1308             0       3    male      0      -888              359309
1309             1       3    male      1      -888                2668
"""

#printing last 10 rows:
print(df.tail(10))

"""
Last 10 rows:
                                                        Name  Parch  Pclass  \
PassengerId
1300                         Riordan, Miss. Johanna Hannah""      0       3
1301                               Peacock, Miss. Treasteall      1       3
1302                                  Naughton, Miss. Hannah      0       3
1303         Minahan, Mrs. William Edward (Lillian E Thorpe)      0       1
1304                          Henriksson, Miss. Jenny Lovisa      0       3
1305                                      Spector, Mr. Woolf      0       3
1306                            Oliva y Ocana, Dona. Fermina      0       1
1307                            Saether, Mr. Simon Sivertsen      0       3
1308                                     Ware, Mr. Frederick      0       3
1309                                Peter, Master. Michael J      1       3

                Sex  SibSp  Survived              Ticket
PassengerId
1300         female      0      -888              334915
1301         female      1      -888  SOTON/O.Q. 3101315
1302         female      0      -888              365237
1303         female      1      -888               19928
1304         female      0      -888              347086
1305           male      0      -888           A.5. 3236
1306         female      0      -888            PC 17758
1307           male      0      -888  SOTON/O.Q. 3101262
1308           male      0      -888              359309
1309           male      1      -888                2668
"""
