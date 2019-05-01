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

#getting particular col details
print(df.Name)

"""
Names of Passengers:

PassengerId
1                                 Braund, Mr. Owen Harris
2       Cumings, Mrs. John Bradley (Florence Briggs Th...
3                                  Heikkinen, Miss. Laina
4            Futrelle, Mrs. Jacques Heath (Lily May Peel)
5                                Allen, Mr. William Henry
6                                        Moran, Mr. James
7                                 McCarthy, Mr. Timothy J
8                          Palsson, Master. Gosta Leonard
9       Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)
10                    Nasser, Mrs. Nicholas (Adele Achem)
11                        Sandstrom, Miss. Marguerite Rut
12                               Bonnell, Miss. Elizabeth
13                         Saundercock, Mr. William Henry
14                            Andersson, Mr. Anders Johan
15                   Vestrom, Miss. Hulda Amanda Adolfina
16                       Hewlett, Mrs. (Mary D Kingcome)
17                                   Rice, Master. Eugene
18                           Williams, Mr. Charles Eugene
19      Vander Planke, Mrs. Julius (Emelia Maria Vande...
20                                Masselmani, Mrs. Fatima
21                                   Fynney, Mr. Joseph J
22                                  Beesley, Mr. Lawrence
23                            McGowan, Miss. Anna "Annie"
24                           Sloper, Mr. William Thompson
25                          Palsson, Miss. Torborg Danira
26      Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...
27                                Emir, Mr. Farred Chehab
28                         Fortune, Mr. Charles Alexander
29                          O'Dwyer, Miss. Ellen "Nellie"
30                                    Todoroff, Mr. Lalio
                              ...
1280                                 Canavan, Mr. Patrick
1281                          Palsson, Master. Paul Folke
1282                           Payne, Mr. Vivian Ponsonby
1283       Lines, Mrs. Ernest H (Elizabeth Lindsey James)
1284                        Abbott, Master. Eugene Joseph
1285                                 Gilbert, Mr. William
1286                             Kink-Heilmann, Mr. Anton
1287       Smith, Mrs. Lucien Philip (Mary Eloise Hughes)
1288                                 Colbert, Mr. Patrick
1289    Frolicher-Stehli, Mrs. Maxmillian (Margaretha ...
1290                       Larsson-Rondberg, Mr. Edvard A
1291                             Conlon, Mr. Thomas Henry
1292                              Bonnell, Miss. Caroline
1293                                      Gale, Mr. Harry
1294                       Gibson, Miss. Dorothy Winifred
1295                               Carrau, Mr. Jose Pedro
1296                         Frauenthal, Mr. Isaac Gerald
1297         Nourney, Mr. Alfred (Baron von Drachstedt")"
1298                            Ware, Mr. William Jeffery
1299                           Widener, Mr. George Dunton
1300                      Riordan, Miss. Johanna Hannah""
1301                            Peacock, Miss. Treasteall
1302                               Naughton, Miss. Hannah
1303      Minahan, Mrs. William Edward (Lillian E Thorpe)
1304                       Henriksson, Miss. Jenny Lovisa
1305                                   Spector, Mr. Woolf
1306                         Oliva y Ocana, Dona. Fermina
1307                         Saether, Mr. Simon Sivertsen
1308                                  Ware, Mr. Frederick
1309                             Peter, Master. Michael J
Name: Name, dtype: object
"""

#we can access different col values in a dataframe by passing the different feature name within []
print(df['Name'])

#we can use multiple features together inside []i.e. df[['feature1','feature2',...'featureN']]
print(df[['Name','Age','Sex']])
"""
                                                          Name   Age     Sex
PassengerId
1                                      Braund, Mr. Owen Harris  22.0    male
2            Cumings, Mrs. John Bradley (Florence Briggs Th...  38.0  female
3                                       Heikkinen, Miss. Laina  26.0  female
4                 Futrelle, Mrs. Jacques Heath (Lily May Peel)  35.0  female
5                                     Allen, Mr. William Henry  35.0    male
6                                             Moran, Mr. James   NaN    male
7                                      McCarthy, Mr. Timothy J  54.0    male
8                               Palsson, Master. Gosta Leonard   2.0    male
9            Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  27.0  female
10                         Nasser, Mrs. Nicholas (Adele Achem)  14.0  female
11                             Sandstrom, Miss. Marguerite Rut   4.0  female
12                                    Bonnell, Miss. Elizabeth  58.0  female
13                              Saundercock, Mr. William Henry  20.0    male
14                                 Andersson, Mr. Anders Johan  39.0    male
15                        Vestrom, Miss. Hulda Amanda Adolfina  14.0  female
16                            Hewlett, Mrs. (Mary D Kingcome)   55.0  female
17                                        Rice, Master. Eugene   2.0    male
18                                Williams, Mr. Charles Eugene   NaN    male
19           Vander Planke, Mrs. Julius (Emelia Maria Vande...  31.0  female
20                                     Masselmani, Mrs. Fatima   NaN  female
21                                        Fynney, Mr. Joseph J  35.0    male
22                                       Beesley, Mr. Lawrence  34.0    male
23                                 McGowan, Miss. Anna "Annie"  15.0  female
24                                Sloper, Mr. William Thompson  28.0    male
25                               Palsson, Miss. Torborg Danira   8.0  female
26           Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...  38.0  female
27                                     Emir, Mr. Farred Chehab   NaN    male
28                              Fortune, Mr. Charles Alexander  19.0    male
29                               O'Dwyer, Miss. Ellen "Nellie"   NaN  female
30                                         Todoroff, Mr. Lalio   NaN    male
...                                                        ...   ...     ...
1280                                      Canavan, Mr. Patrick  21.0    male
1281                               Palsson, Master. Paul Folke   6.0    male
1282                                Payne, Mr. Vivian Ponsonby  23.0    male
1283            Lines, Mrs. Ernest H (Elizabeth Lindsey James)  51.0  female
1284                             Abbott, Master. Eugene Joseph  13.0    male
1285                                      Gilbert, Mr. William  47.0    male
1286                                  Kink-Heilmann, Mr. Anton  29.0    male
1287            Smith, Mrs. Lucien Philip (Mary Eloise Hughes)  18.0  female
1288                                      Colbert, Mr. Patrick  24.0    male
1289         Frolicher-Stehli, Mrs. Maxmillian (Margaretha ...  48.0  female
1290                            Larsson-Rondberg, Mr. Edvard A  22.0    male
1291                                  Conlon, Mr. Thomas Henry  31.0    male
1292                                   Bonnell, Miss. Caroline  30.0  female
1293                                           Gale, Mr. Harry  38.0    male
1294                            Gibson, Miss. Dorothy Winifred  22.0  female
1295                                    Carrau, Mr. Jose Pedro  17.0    male
1296                              Frauenthal, Mr. Isaac Gerald  43.0    male
1297              Nourney, Mr. Alfred (Baron von Drachstedt")"  20.0    male
1298                                 Ware, Mr. William Jeffery  23.0    male
1299                                Widener, Mr. George Dunton  50.0    male
1300                           Riordan, Miss. Johanna Hannah""   NaN  female
1301                                 Peacock, Miss. Treasteall   3.0  female
1302                                    Naughton, Miss. Hannah   NaN  female
1303           Minahan, Mrs. William Edward (Lillian E Thorpe)  37.0  female
1304                            Henriksson, Miss. Jenny Lovisa  28.0  female
1305                                        Spector, Mr. Woolf   NaN    male
1306                              Oliva y Ocana, Dona. Fermina  39.0  female
1307                              Saether, Mr. Simon Sivertsen  38.5    male
1308                                       Ware, Mr. Frederick   NaN    male
1309                                  Peter, Master. Michael J   NaN    male
"""

#we can access different data in data sets based on indexing: label-based indexing & position-based indexing

#-----Label-based indexing---------
#getting specific data based on indexing using df.loc[row_indexing, coloumn_indexing]
#below code will print all the cols in between rows with passenger id 20-30:
print(df.loc[20:30,]) 
"""
All the feature values for rows inclusive range [20,30]:
                                                          Name  Parch  Pclass  \
PassengerId
20                                     Masselmani, Mrs. Fatima      0       3
21                                        Fynney, Mr. Joseph J      0       2
22                                       Beesley, Mr. Lawrence      0       2
23                                 McGowan, Miss. Anna "Annie"      0       3
24                                Sloper, Mr. William Thompson      0       1
25                               Palsson, Miss. Torborg Danira      1       3
26           Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...      5       3
27                                     Emir, Mr. Farred Chehab      0       3
28                              Fortune, Mr. Charles Alexander      2       1
29                               O'Dwyer, Miss. Ellen "Nellie"      0       3
30                                         Todoroff, Mr. Lalio      0       3

                Sex  SibSp  Survived  Ticket
PassengerId
20           female      0         1    2649
21             male      0         0  239865
22             male      0         1  248698
23           female      0         1  330923
24             male      0         1  113788
25           female      3         0  349909
26           female      1         1  347077
27             male      0         0    2631
28             male      3         0   19950
29           female      0         1  330959
30             male      0         0  349216
"""

#we can provide col indexing as well:
print(df.loc[1000:1010, 'Name':'Sex'])

"""
Features(PassengerId, Name, Parch, Pclass, Sex) from rows 1000-1010:
                                               Name  Parch  Pclass     Sex
PassengerId
1000               Willer, Mr. Aaron (Abi Weller")"      0       3    male
1001                              Swane, Mr. George      0       2    male
1002                       Stanton, Mr. Samuel Ward      0       2    male
1003                     Shine, Miss. Ellen Natalia      0       3  female
1004                       Evans, Miss. Edith Corse      0       1  female
1005                       Buckley, Miss. Katherine      0       3  female
1006         Straus, Mrs. Isidor (Rosalie Ida Blun)      0       1  female
1007                    Chronopoulos, Mr. Demetrios      0       3    male
1008                               Thomas, Mr. John      0       3    male
1009                Sandstrom, Miss. Beatrice Irene      1       3  female
1010                           Beattie, Mr. Thomson      0       1    male
"""
#accessing specific feature values within a range of rows:
print(df.loc[500:520, ['Name', 'Sex', 'Ticket']])

"""
features ['Name', 'Sex', 'Ticket'] values from passenger id range[500:520]:

                                                          Name     Sex  \
PassengerId
500                                         Svensson, Mr. Olof    male
501                                           Calic, Mr. Petar    male
502                                        Canavan, Miss. Mary  female
503                             O'Sullivan, Miss. Bridget Mary  female
504                             Laitinen, Miss. Kristina Sofia  female
505                                      Maioni, Miss. Roberta  female
506                 Penasco y Castellana, Mr. Victor de Satode    male
507              Quick, Mrs. Frederick Charles (Jane Richards)  female
508              Bradley, Mr. George ("George Arthur Brayton")    male
509                                   Olsen, Mr. Henry Margido    male
510                                             Lang, Mr. Fang    male
511                                   Daly, Mr. Eugene Patrick    male
512                                          Webber, Mr. James    male
513                                  McGough, Mr. James Robert    male
514             Rothschild, Mrs. Martin (Elizabeth L. Barrett)  female
515                                          Coleff, Mr. Satio    male
516                               Walker, Mr. William Anderson    male
517                               Lemore, Mrs. (Amelia Milley)  female
518                                          Ryan, Mr. Patrick    male
519          Angle, Mrs. William A (Florence "Mary" Agnes H...  female
520                                        Pavlovic, Mr. Stefo    male

                       Ticket
PassengerId
500                    350035
501                    315086
502                    364846
503                    330909
504                      4135
505                    110152
506                  PC 17758
507                     26360
508                    111427
509                    C 4001
510                      1601
511                    382651
512          SOTON/OQ 3101316
513                  PC 17473
514                  PC 17603
515                    349209
516                     36967
517                C.A. 34260
518                    371110
519                    226875
520                    349242
"""
#------position-based indexing---------
#index in python is 0 based, so using iloc for accesing data from dataframe for position based indexing:
print(df.iloc[10:15, 3:6])

"""
Data from passengerId range[10:15] and feature range[3:6]:
                Fare                                  Name  Parch
PassengerId
11           16.7000       Sandstrom, Miss. Marguerite Rut      1
12           26.5500              Bonnell, Miss. Elizabeth      0
13            8.0500        Saundercock, Mr. William Henry      0
14           31.2750           Andersson, Mr. Anders Johan      5
15            7.8542  Vestrom, Miss. Hulda Amanda Adolfina      0
"""

#we can use selection logic to get specific data from our dataframe
#below code selects the number of male passengers
male_passengers=df.loc[df.Sex=='male', :]
print('Number of male passengers: {0}'.format(len(male_passengers)))

"""
Number of male passengers: 843
"""
#finding our the number of male passenger in first class
first_class_males=df.loc[((df.Sex=='male') & (df.Pclass==1)), :]
print('Number of male passengers travelling first class : {0}'.format(len(first_class_males)))

"""
Number of male passengers travelling first class : 179
"""
#printing the names of passengers who all are male
male_passengers_name=df.loc[df.Sex=='male', ['Name']]
print(male_passengers_name)

"""
                                                     Name
PassengerId
1                                 Braund, Mr. Owen Harris
5                                Allen, Mr. William Henry
6                                        Moran, Mr. James
7                                 McCarthy, Mr. Timothy J
8                          Palsson, Master. Gosta Leonard
13                         Saundercock, Mr. William Henry
14                            Andersson, Mr. Anders Johan
17                                   Rice, Master. Eugene
18                           Williams, Mr. Charles Eugene
21                                   Fynney, Mr. Joseph J
22                                  Beesley, Mr. Lawrence
24                           Sloper, Mr. William Thompson
27                                Emir, Mr. Farred Chehab
28                         Fortune, Mr. Charles Alexander
30                                    Todoroff, Mr. Lalio
31                               Uruchurtu, Don. Manuel E
34                                  Wheadon, Mr. Edward H
35                                Meyer, Mr. Edgar Joseph
36                         Holverson, Mr. Alexander Oskar
37                                       Mamee, Mr. Hanna
38                               Cann, Mr. Ernest Charles
43                                    Kraeff, Mr. Theodor
46                               Rogers, Mr. William John
47                                      Lennon, Mr. Denis
49                                    Samaan, Mr. Youssef
51                             Panula, Master. Juha Niilo
52                           Nosworthy, Mr. Richard Cater
55                         Ostby, Mr. Engelhart Cornelius
56                                      Woolner, Mr. Hugh
58                                    Novel, Mr. Mansouer
...                                                   ...
1262                                     Giles, Mr. Edgar
1264                              Ismay, Mr. Joseph Bruce
1265                               Harbeck, Mr. William H
1269                         Cotterill, Mr. Henry Harry""
1270                          Hipkins, Mr. William Edward
1271                          Asplund, Master. Carl Edgar
1272                                O'Connor, Mr. Patrick
1273                                    Foley, Mr. Joseph
1276                       Wheeler, Mr. Edwin Frederick""
1278                       Aronsson, Mr. Ernst Axel Algot
1279                                      Ashby, Mr. John
1280                                 Canavan, Mr. Patrick
1281                          Palsson, Master. Paul Folke
1282                           Payne, Mr. Vivian Ponsonby
1284                        Abbott, Master. Eugene Joseph
1285                                 Gilbert, Mr. William
1286                             Kink-Heilmann, Mr. Anton
1288                                 Colbert, Mr. Patrick
1290                       Larsson-Rondberg, Mr. Edvard A
1291                             Conlon, Mr. Thomas Henry
1293                                      Gale, Mr. Harry
1295                               Carrau, Mr. Jose Pedro
1296                         Frauenthal, Mr. Isaac Gerald
1297         Nourney, Mr. Alfred (Baron von Drachstedt")"
1298                            Ware, Mr. William Jeffery
1299                           Widener, Mr. George Dunton
1305                                   Spector, Mr. Woolf
1307                         Saether, Mr. Simon Sivertsen
1308                                  Ware, Mr. Frederick
1309                             Peter, Master. Michael J

[843 rows x 1 columns]
"""