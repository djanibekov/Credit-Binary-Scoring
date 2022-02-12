import os
import dotenv
import pickle
from datetime import date
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def categorize_df(data_f):
    return pd.get_dummies(data_f, prefix='dum')


class CreditDataCleaning:
    def __init__(self, path_to_file, cat_col=None):
        self.cat_col = cat_col
        self.df = pd.read_csv(path_to_file)
        self.filtered = []

    def _info(self):
        return self.df.info()

    def filter_y(self):
        # bad group
        self.df['credit_status_name'] = np.where(self.df['credit_status_name'] == 'проблемный',
                                                 0, self.df['credit_status_name'])
        self.df['credit_status_name'] = np.where(self.df['credit_status_name'] == 'задерживается',
                                                 0.5, self.df['credit_status_name'])
        self.df['credit_status_name'] = np.where(self.df['credit_status_name'] == 'на паузе',
                                                 0.3, self.df['credit_status_name'])
        self.df['credit_status_name'] = np.where(self.df['credit_status_name'] == 'удаленный',
                                                 0.3, self.df['credit_status_name'])
        # good group
        self.df['credit_status_name'] = np.where(self.df['credit_status_name'] == 'активный',
                                                 0.8, self.df['credit_status_name'])
        self.df['credit_status_name'] = np.where(self.df['credit_status_name'] == 'закрытый',
                                                 1, self.df['credit_status_name'])
        # closed, but caused problems
        self.df['credit_status_name'] = np.where(
            (self.df['credit_status_name'] == 1) & (self.df['penalty_amount'].notnull()),
            0.7, self.df['credit_status_name'])
        # active, but almost closed
        self.df['credit_status_name'] = np.where((self.df['credit_status_name'] == 0.8) & (self.df['completion'] >= 75),
                                                 1, self.df['credit_status_name'])

        self.df['credit_status_name'] = self.df['credit_status_name'].astype('float64')

    def categorize_x(self):
        data_f = self.df[self.cat_col]
        return pd.get_dummies(data_f, prefix='dum')

    def filter_x(self):
        X_features = pd.DataFrame()

        def calculate_age_by_days(born_date):
            born_date = datetime.strptime(born_date, "%m/%d/%Y").date()
            today = date.today()
            birth_date = date(born_date.year, born_date.month, born_date.day)
            today_date = date(today.year, today.month, today.day)

            time_difference = today_date - birth_date
            return time_difference.days

        def standardize(arr):
            dotenv_file = dotenv.find_dotenv()
            dotenv.load_dotenv(dotenv_file)

            mean = arr.mean()
            x = abs(arr - mean) ** 2
            std = np.sqrt(np.mean(x))

            print(mean, std)
            os.environ['MEAN'] = str(mean)
            os.environ['STD'] = str(std)
            dotenv.set_key(dotenv_file, "MEAN", os.environ["MEAN"])
            dotenv.set_key(dotenv_file, "STD", os.environ["STD"])
            new_arr = (arr - mean) / std

            return new_arr

        # '/MM/DD/YY' to overall days
        self.df['birth_date'] = self.df['birth_date'].apply(calculate_age_by_days)  # rewrites values in data frame
        self.df['birth_date'] = standardize(self.df['birth_date'].values)  # rewrites values in data frame by
        # standardizing each element
        if self.cat_col:
            X_features = self.categorize_x()

        X_features['birth_date'] = self.df['birth_date']
        return X_features

    def sieve_x_y(self, df_y):
        positive_val_index = df_y.index[self.df['credit_status_name'] == 1].tolist()
        negative_val_index = df_y.index[self.df['credit_status_name'] == 0].tolist()
        # no need to sort, later may be removed for optimization
        self.filtered = sorted(positive_val_index + negative_val_index)

    def get_x_y(self):
        self.filter_y()  # no var since the changes occur in dataframe directly, to identify good credits
        X_features = self.filter_x()

        self.sieve_x_y(self.df)

        y_target = self.df.loc[self.filtered, 'credit_status_name']
        X_features = X_features.iloc[self.filtered]

        return X_features, y_target

    def plot_hist(self, column_name):  # support function
        sns.countplot(x=column_name, data=self.df, palette='hls')
        plt.show()

    def get_ratio_y(self):  # support function
        credit_status_name = self.df['credit_status_name']
        good_closed = len(self.df[credit_status_name == 1])
        trouble_maker = len(self.df[credit_status_name == 0])

        active = len(self.df[credit_status_name == 0.8])
        bad_closed = len(self.df[credit_status_name == 0.7])
        paused = len(self.df[credit_status_name == 0.5])
        suspend = len(self.df[credit_status_name == 0.3])

        total_absolute = self.df['credit_status_name'].count()
        total_relative = good_closed + trouble_maker

        good = good_closed / total_absolute
        bad = trouble_maker / total_absolute
        print('+-----------------------------------+')
        print('% good credits absolute', good * 100)
        print('% bad credits absolute', bad * 100)

        print('-------------------------------------')

        good_r = good_closed / total_relative
        bad_r = trouble_maker / total_relative
        print('% good credits relative', good_r * 100)
        print('% bad credits relative', bad_r * 100)
        print('+-----------------------------------+')


data = CreditDataCleaning('credits_clean.csv', cat_col=['gender', 'marital_status'])
X, y = data.get_x_y()
print(X)
print('--------------------------------------------------------')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg_model = LogisticRegression()

log_reg_model.fit(X_train, y_train)


# another training method
def stats_model_training(y, X):
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary2())


y_pred = log_reg_model.predict(X_test)
print('Accuracy : {:.2f}'.format(log_reg_model.score(X_test, y_test)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print('Confusion_matrix: \n', confusion_matrix)
print('--------------------------------------------------------')
print(classification_report(y_test, y_pred))

# ---Python Version.---
# Take note of the python version. You almost certainly require the same major (and maybe
# minor) version of Python used to serialize the model when you later load it and deserialize it.

# ---Library Versions.---
# The version of all major libraries used in your machine learning project almost certainly need to be
# the same when deserializing a saved model. This is not limited to the version of NumPy and the version of
# scikit-learn.

# ---Manual Serialization.---
# You might like to manually output the parameters of your learned model so
# that you can use them directly in scikit-learn or another platform in the future. Often the algorithms used by
# machine learning algorithms to make predictions are a lot simpler than those used to learn the parameters can may
# be easy to implement in custom code that you have control over.

filename_path = f'../finalized_model.sav'
pickle.dump(log_reg_model, open(filename_path, 'wb'))
