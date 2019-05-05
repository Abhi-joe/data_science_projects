import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as pl

parent_path = os.path.join(os.path.pardir, 'data_science_projects')
test_data_path = os.path.join(parent_path, 'datasets','test.csv')
train_data_path = os.path.join(parent_path, 'datasets', 'train.csv')

print(parent_path)
print(train_data_path)
print(test_data_path)

train_data= pd.read_csv(train_data_path, index_col='PassengerId')
test_data= pd.read_csv(test_data_path, index_col='PassengerId')

print(train_data.info())
print(test_data.info())

test_data['Survived']= -888

print(test_data.info())

data= pd.concat((train_data, test_data), axis=0)
print(data.info())

print(data['Name'])
print(data.loc[0:1309,:])

"""
              Age        Cabin Embarked      Fare  \
PassengerId
1            22.0          NaN        S    7.2500
2            38.0          C85        C   71.2833
3            26.0          NaN        S    7.9250
4            35.0         C123        S   53.1000
5            35.0          NaN        S    8.0500
6             NaN          NaN        Q    8.4583
7            54.0          E46        S   51.8625
8             2.0          NaN        S   21.0750
9            27.0          NaN        S   11.1333
10           14.0          NaN        C   30.0708
11            4.0           G6        S   16.7000
12           58.0         C103        S   26.5500
13           20.0          NaN        S    8.0500
14           39.0          NaN        S   31.2750
15           14.0          NaN        S    7.8542
16           55.0          NaN        S   16.0000
17            2.0          NaN        Q   29.1250
18            NaN          NaN        S   13.0000
19           31.0          NaN        S   18.0000
20            NaN          NaN        C    7.2250
21           35.0          NaN        S   26.0000
22           34.0          D56        S   13.0000
23           15.0          NaN        Q    8.0292
24           28.0           A6        S   35.5000
25            8.0          NaN        S   21.0750
26           38.0          NaN        S   31.3875
27            NaN          NaN        C    7.2250
28           19.0  C23 C25 C27        S  263.0000
29            NaN          NaN        Q    7.8792
30            NaN          NaN        S    7.8958
...           ...          ...      ...       ...
1280         21.0          NaN        Q    7.7500
1281          6.0          NaN        S   21.0750
1282         23.0          B24        S   93.5000
1283         51.0          D28        S   39.4000
1284         13.0          NaN        S   20.2500
1285         47.0          NaN        S   10.5000
1286         29.0          NaN        S   22.0250
1287         18.0          C31        S   60.0000
1288         24.0          NaN        Q    7.2500
1289         48.0          B41        C   79.2000
1290         22.0          NaN        S    7.7750
1291         31.0          NaN        Q    7.7333
1292         30.0           C7        S  164.8667
1293         38.0          NaN        S   21.0000
1294         22.0          NaN        C   59.4000
1295         17.0          NaN        S   47.1000
1296         43.0          D40        C   27.7208
1297         20.0          D38        C   13.8625
1298         23.0          NaN        S   10.5000
1299         50.0          C80        C  211.5000
1300          NaN          NaN        Q    7.7208
1301          3.0          NaN        S   13.7750
1302          NaN          NaN        Q    7.7500
1303         37.0          C78        Q   90.0000
1304         28.0          NaN        S    7.7750
1305          NaN          NaN        S    8.0500
1306         39.0         C105        C  108.9000
1307         38.5          NaN        S    7.2500
1308          NaN          NaN        S    8.0500
1309          NaN          NaN        C   22.3583

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
11                             Sandstrom, Miss. Marguerite Rut      1       3
12                                    Bonnell, Miss. Elizabeth      0       1
13                              Saundercock, Mr. William Henry      0       3
14                                 Andersson, Mr. Anders Johan      5       3
15                        Vestrom, Miss. Hulda Amanda Adolfina      0       3
16                            Hewlett, Mrs. (Mary D Kingcome)       0       2
17                                        Rice, Master. Eugene      1       3
18                                Williams, Mr. Charles Eugene      0       2
19           Vander Planke, Mrs. Julius (Emelia Maria Vande...      0       3
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
...                                                        ...    ...     ...
1280                                      Canavan, Mr. Patrick      0       3
1281                               Palsson, Master. Paul Folke      1       3
1282                                Payne, Mr. Vivian Ponsonby      0       1
1283            Lines, Mrs. Ernest H (Elizabeth Lindsey James)      1       1
1284                             Abbott, Master. Eugene Joseph      2       3
1285                                      Gilbert, Mr. William      0       2
1286                                  Kink-Heilmann, Mr. Anton      1       3
1287            Smith, Mrs. Lucien Philip (Mary Eloise Hughes)      0       1
1288                                      Colbert, Mr. Patrick      0       3
1289         Frolicher-Stehli, Mrs. Maxmillian (Margaretha ...      1       1
1290                            Larsson-Rondberg, Mr. Edvard A      0       3
1291                                  Conlon, Mr. Thomas Henry      0       3
1292                                   Bonnell, Miss. Caroline      0       1
1293                                           Gale, Mr. Harry      0       2
1294                            Gibson, Miss. Dorothy Winifred      1       1
1295                                    Carrau, Mr. Jose Pedro      0       1
1296                              Frauenthal, Mr. Isaac Gerald      0       1
1297              Nourney, Mr. Alfred (Baron von Drachstedt")"      0       2
1298                                 Ware, Mr. William Jeffery      0       2
1299                                Widener, Mr. George Dunton      1       1
1300                           Riordan, Miss. Johanna Hannah""      0       3
1301                                 Peacock, Miss. Treasteall      1       3
1302                                    Naughton, Miss. Hannah      0       3
1303           Minahan, Mrs. William Edward (Lillian E Thorpe)      0       1
1304                            Henriksson, Miss. Jenny Lovisa      0       3
1305                                        Spector, Mr. Woolf      0       3
1306                              Oliva y Ocana, Dona. Fermina      0       1
1307                              Saether, Mr. Simon Sivertsen      0       3
1308                                       Ware, Mr. Frederick      0       3
1309                                  Peter, Master. Michael J      1       3

                Sex  SibSp  Survived              Ticket
PassengerId
1              male      1         0           A/5 21171
2            female      1         1            PC 17599
3            female      0         1    STON/O2. 3101282
4            female      1         1              113803
5              male      0         0              373450
6              male      0         0              330877
7              male      0         0               17463
8              male      3         0              349909
9            female      0         1              347742
10           female      1         1              237736
11           female      1         1             PP 9549
12           female      0         1              113783
13             male      0         0           A/5. 2151
14             male      1         0              347082
15           female      0         0              350406
16           female      0         1              248706
17             male      4         0              382652
18             male      0         1              244373
19           female      1         0              345763
20           female      0         1                2649
21             male      0         0              239865
22             male      0         1              248698
23           female      0         1              330923
24             male      0         1              113788
25           female      3         0              349909
26           female      1         1              347077
27             male      0         0                2631
28             male      3         0               19950
29           female      0         1              330959
30             male      0         0              349216
...             ...    ...       ...                 ...
1280           male      0      -888              364858
1281           male      3      -888              349909
1282           male      0      -888               12749
1283         female      0      -888            PC 17592
1284           male      0      -888           C.A. 2673
1285           male      0      -888          C.A. 30769
1286           male      3      -888              315153
1287         female      1      -888               13695
1288           male      0      -888              371109
1289         female      1      -888               13567
1290           male      0      -888              347065
1291           male      0      -888               21332
1292         female      0      -888               36928
1293           male      1      -888               28664
1294         female      0      -888              112378
1295           male      0      -888              113059
1296           male      1      -888               17765
1297           male      0      -888       SC/PARIS 2166
1298           male      1      -888               28666
1299           male      1      -888              113503
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
#getting summary statistics through pandas in one go
print(data.describe())
"""
Summary statistics of thhe whole dataframe:

               Age         Fare        Parch       Pclass        SibSp  \
count  1046.000000  1308.000000  1309.000000  1309.000000  1309.000000
mean     29.881138    33.295479     0.385027     2.294882     0.498854
std      14.413493    51.758668     0.865560     0.837836     1.041658
min       0.170000     0.000000     0.000000     1.000000     0.000000
25%            NaN          NaN     0.000000     2.000000     0.000000
50%            NaN          NaN     0.000000     3.000000     0.000000
75%            NaN          NaN     0.000000     3.000000     1.000000
max      80.000000   512.329200     9.000000     3.000000     8.000000

          Survived
count  1309.000000
mean   -283.301757
std     414.337413
min    -888.000000
25%    -888.000000
50%       0.000000
75%       1.000000
max       1.000000
"""

#Centrality measures:
#calculating mean:
print('Mean of Fare: {0}'.format(data.Fare.mean()))

#calculating median:
print('Median of Fare: {0}'.format(data.Fare.median()))

"""
Mean of Fare: 33.2954792813
Median of Fare: 14.4542
"""

#Dispersion Measures:
print('Minimum fare: {0}'.format(data.Fare.min()))
print('Maximum fare: {0}'.format(data.Fare.max()))
print('Fare range: {0}'.format((data.Fare.max()-data.Fare.min())))
print('Fare variance: {0}'.format(data.Fare.var()))
print('Fare standard deviation: {0}'.format(data.Fare.std()))
print('Fare 25th Percentile: {0}'.format(data.Fare.quantile(.25)))
print('Fare 50th Percentile: {0}'.format(data.Fare.quantile(.5)))
print('Fare 75th Percentile: {0}'.format(data.Fare.quantile(.75)))

"""
Minimum fare: 0.0
Maximum fare: 512.3292
Fare range: 512.3292
Fare variance: 2678.95973789
Fare standard deviation: 51.75866
Fare 25th Percentile: nan
Fare 50th Percentile: nan
Fare 75th Percentile: nan
"""

#plotting a box-whisker for percentiles:
print(data.Fare.plot(kind='box'))


#Summary stats for categorical data;
print(data.describe(include="all"))

"""
Summary statistics for both numerical and categorical data:

Axes(0.125,0.1;0.775x0.8)
                Age        Cabin Embarked         Fare                  Name  \
count   1046.000000          295     1307  1308.000000                  1309
unique          NaN          186        3          NaN                  1307
top             NaN  C23 C25 C27        S          NaN  Connolly, Miss. Kate
freq            NaN            6      914          NaN                     2
mean      29.881138          NaN      NaN    33.295479                   NaN
std       14.413493          NaN      NaN    51.758668                   NaN
min        0.170000          NaN      NaN     0.000000                   NaN
25%             NaN          NaN      NaN          NaN                   NaN
50%             NaN          NaN      NaN          NaN                   NaN
75%             NaN          NaN      NaN          NaN                   NaN
max       80.000000          NaN      NaN   512.329200                   NaN

              Parch       Pclass   Sex        SibSp     Survived    Ticket
count   1309.000000  1309.000000  1309  1309.000000  1309.000000      1309
unique          NaN          NaN     2          NaN          NaN       929
top             NaN          NaN  male          NaN          NaN  CA. 2343
freq            NaN          NaN   843          NaN          NaN        11
mean       0.385027     2.294882   NaN     0.498854  -283.301757       NaN
std        0.865560     0.837836   NaN     1.041658   414.337413       NaN
min        0.000000     1.000000   NaN     0.000000  -888.000000       NaN
25%        0.000000     2.000000   NaN     0.000000  -888.000000       NaN
50%        0.000000     3.000000   NaN     0.000000     0.000000       NaN
75%        0.000000     3.000000   NaN     1.000000     1.000000       NaN
max        9.000000     3.000000   NaN     8.000000     1.000000       NaN
"""
#feature specific categorical data statistics:
print(data.Sex.value_counts())
"""
male      843
female    466
Name: Sex, dtype: int64
"""
#printing the proportion of males and females:
print(data.Sex.value_counts(normalize=True))
"""
male      0.644003
female    0.355997
Name: Sex, dtype: float64
""" 
#printing the counts of survived vs not survived:
print(data[data.Survived!=-888].Survived.value_counts())
"""
0    549
1    342
Name: Survived, dtype: int64
"""

#proportion of survival:
print(data[data.Survived!=-888].Survived.value_counts(normalize=True))
"""
0    0.616162
1    0.383838
Name: Survived, dtype: float64
"""
#to plot a bar graph for categorical data
data[data.Survived!=-888].Survived.value_counts().plot(kind='bar', rot=0, title='Survival bar graph', color='c')