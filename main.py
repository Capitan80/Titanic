# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# load Dataset
titanic = sns.load_dataset('titanic')

# get the number of survivors on the titanic
survivors_count = titanic['survived'].value_counts()

# visualize the number of survivors on the titanic
number_survived_visualize = sns.countplot(titanic['survived'])

# visualize the number of survivors on the titanic for 'who' 'sex' 'pclass' 'sibsp' 'parch' 'embarked'
columns = ['who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked']
rows_number = 2
columns_number = 3
figure, axs = plt.subplots(rows_number, columns_number, figsize=(rows_number * 3.2, columns_number * 3.2))
for _ in range(rows_number):
    for __ in range(columns_number):
        index = _ * columns_number + __
        ax = axs[_][__]
        sns.countplot(titanic[columns[index]], hue=titanic['survived'], ax=ax)
        ax.set_title(columns[index])
        ax.legend(title='survived', loc='upper right')

# look at survival rate by sex
survived_by_sex = titanic.groupby('sex')[['survived']].mean()

# look at survival rate by sex and class visually
survived_by_sex_pclass = titanic.pivot_table('survived', index='sex', columns='pclass').plot()

# look at survival rate by class visually
survived_by_class = sns.countplot(titanic['pclass'], hue=titanic['survived'])

# look at survival rate by sex and age and class visually
age = pd.cut(['age'], [0, 18, 80])
survived_by_sex_age_class = titanic.pivot_table('survived', ['sex', 'age'], 'class')

# drop useless columns
titanic.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'alone', 'adult_male'], axis=1)
titanic = titanic.dropna(subset=['embarked', 'age'])

labelEncoder = LabelEncoder()

# change data type to numerical
titanic.iloc[:, 2] = labelEncoder.fit_transform(titanic.iloc[:, 2].values)
titanic.iloc[:, 7] = labelEncoder.fit_transform(titanic.iloc[:, 7].values)

# create x, y
x = titanic.iloc[:, 1:7].values
y = titanic.iloc[:, 0].values

# create test, train for x and y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# create models for prediction
def models(x_train, y_train):
    # use logistic regression
    log = LogisticRegression(random_state=0)
    log.fit(x_train, y_train)

    # use Kneighbors
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(x_train, y_train)

    # use SVC
    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(x_train, y_train)

    # use GaussianNB
    gauss = GaussianNB()
    gauss.fit(x_train, y_train)

    # use Decision Tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(x_train, y_train)

    # use RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)

    # print('[0] logistic regression training accuracy', log.score(x_train, y_train))
    # print('[0] KNeighbors training accuracy', knn.score(x_train, y_train))
    # print('[0] SVC training accuracy', svc_lin.score(x_train, y_train))
    # print('[0] GaussianNB training accuracy', gauss.score(x_train, y_train))
    # print('[0] Decision Tree training accuracy', tree.score(x_train, y_train))
    # print('[0] Random forest training accuracy', forest.score(x_train, y_train))

    return log, knn, svc_lin, gauss, tree, forest


# create a class for Illustration
print("Enter : Person-class, Sex, Age, sibship-number, Parch, Embarked")
info_list = list()


class show:
    def __init__(self, args):
        self.my_dict_sex = {
            "male": 1,
            "female": 0
        }
        self.my_dict_pclass = {
            "Third": 3,
            "Second": 2,
            "First": 1
        }
        self.args = args

    def change_Person_class(self):
        if self.args[0] in self.my_dict_pclass:
            info_list.append(self.my_dict_pclass.get(self.args[0]))

    def change_Sex(self):
        if self.args[1] in self.my_dict_sex:
            info_list.append(self.my_dict_sex.get(self.args[1]))

    def add_other(self):
        for _ in self.args[2:]:
            info_list.append(_)


input_ = list()
for item in range(6):
    input_.append(input())

x = show(input_)
x.change_Person_class()
x.change_Sex()
x.add_other()

my_survival = list()
my_survival.append(info_list)

sc = StandardScaler()
my_survival_scaled = sc.fit_transform(my_survival)
model = models(x_train, y_train)
predict = model[5].predict(my_survival_scaled)
if predict == 0:
    print('Oh No  ^_^ . You died')
else:
    print("Nice :) .  You Survived")
