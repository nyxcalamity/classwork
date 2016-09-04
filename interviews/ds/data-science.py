#!/usr/bin/env python
"""
Licencing terms here
"""
import matmix as mt
import utilmix as ut
import logging

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV


__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"


# setting up constants
outlier_share = 0.1
validation_share = 0.1

# operational settings
ut.set_logging()
parser = ut.get_cli_parser()
parser.add_option("-f", "--file", dest="filename", default=ut.join_paths(ut.get_script_path(), 'data', 'day.csv'),
                  help="data file with urls")
opts = ut.parse_cli_options(parser)
data_path = opts.filename
is_hour_data = 'hour' in data_path

# loading data
data = pd.read_csv(data_path, parse_dates=[1], index_col=1, date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))

logging.info('Cleaning and normalizing the feature space')
# we have no need in an index feature
features = data.drop(['instant'], axis=1)
fields_to_encode = ['season', 'yr', 'mnth', 'weekday', 'weathersit']
if is_hour_data:
    fields_to_encode.append('hr')
mt.hot_encode(features, fields_to_encode)
fields_to_normalize = ['casual', 'registered', 'cnt']
for field in fields_to_normalize:
    features[field] = mt.normalize(features[field])

logging.info('Checking covariances and correlations')
labels = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
# labels = ['atemp', 'casual', 'registered', 'cnt']
cov_df = features.loc[:, labels]
# check revenue shape
plt.plot(cov_df.cnt)
plt.title('Normalized revenue shape')
plt.show()
# check covariances
print('Covariance matrix: \n{}'.format(cov_df.cov()))
print('Correlation matrix: \n{}'.format(cov_df.corr()))
n = len(labels)
f, splt = plt.subplots(n, n)
for i in range(0, n):
    for j in range(0, n):
        splt[i, j].scatter(cov_df[labels[i]], cov_df[labels[j]])
        splt[i, j].set_title('{0} vs {1}'.format(labels[i], labels[j]), fontsize=10)
plt.show()

logging.info('Removing insignificant features')
# casual and registered users info is what we need to predict and we're not building separate models for them
features = features.drop(['casual', 'registered'], axis=1)
# temp, hum and windspeed are already implicitly present in atemp
features = features.drop(['temp', 'hum', 'windspeed'], axis=1)

logging.info('Removing {0:.2f}% of outliers'.format(outlier_share*100))
x_train = features.drop(['cnt'], axis=1)
y_train = features['cnt']
clf = linear_model.RidgeCV(alphas=np.arange(0, 10, .2), cv=10)
y_predict = pd.Series(clf.fit(x_train, y_train).predict(x_train), index=x_train.index)
errors = (y_train - y_predict).abs()
eps = errors.nlargest(round(outlier_share * features.shape[0])).min()
features = features[errors < eps]

logging.info('Partitioning the data')
n_validation = round(validation_share*data.shape[0])
idx = set(np.random.randint(0, features.shape[0], size=n_validation))
while len(idx) != n_validation:
    idx.add(np.random.randint(0, features.shape[0]))
idx = list(idx)
validation = features.loc[features.index[idx]]
features.drop(features.index[idx], inplace=True)

logging.info('Fitting a model')
x_train = features.drop(['cnt'], axis=1)
y_train = features['cnt']
clf.set_params(cv=5)
y_predict = pd.Series(clf.fit(x_train, y_train).predict(x_train), index=x_train.index)
print('Model parameters: {}'.format(clf.best_params_ if isinstance(clf, GridSearchCV) else {'alpha': clf.alpha_}))
if not is_hour_data:
    mt.show_prediction(y_train, y_predict)

logging.info('Validating results on {0:.2f}% of unused data'.format(validation_share*100))
x_valid = validation.drop(['cnt'], axis=1)
y_valid = validation['cnt']
y_predict = pd.Series(clf.predict(x_valid), index=x_valid.index)
error_measures = {
    'RMSE': mt.rmse(y_valid, y_predict),
    'MAE':  mt.mae(y_valid, y_predict),
    'MPE (in %)':  100*mt.mpe(y_valid, y_predict),
    'MAPE (in %)': 100*mt.mape(y_valid, y_predict)
}
for key, val in error_measures.items():
    if not np.isnan(val) and not np.isinf(val):
        print('{0}: {1:.4f}'.format(key, val))

# TODO:scale back the values for production reuse
