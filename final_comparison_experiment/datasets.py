import os
import os.path
import urllib

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tools import RANDOM_SEED

dirname = os.path.dirname(__file__)


def read_dataset(name, label=None, sensitive_attribute=None, fold=None, **kwargs):
    if name == 'crimes':
        y_name = label if label is not None else 'ViolentCrimesPerPop'
        z_name = sensitive_attribute if sensitive_attribute is not None else 'racepctblack'
        fold_id = fold if fold is not None else 1
        return read_crimes(label=y_name, sensitive_attribute=z_name, fold=fold_id)
    elif name == 'adult':
        return read_adult(**kwargs)
    elif name == "uscensus":
        return read_uscensus()
    elif name == "synthetic":
        return read_synthetic(**kwargs)
    else:
        raise NotImplemented('Dataset {} does not exists'.format(name))


def read_uscensus():
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)
    group = group.astype(float)
    group /= np.max(group)
    x_train, x_test, y_train, y_test, a_train, a_test = train_test_split(features, label, group, test_size=0.2)
    return x_train, y_train, a_train, x_test, y_test, a_test


def read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1):
    if not os.path.isfile('./data/communities.data'):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
            "./data/communities.data")
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            "./data/communities.names")

    # create names
    names = []
    with open('./data/communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv('./data/communities.data', names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)

    folds = data['fold'].astype(int)

    y = data[label].values
    to_drop += [label]

    z = data[sensitive_attribute].values
    to_drop += [sensitive_attribute]

    data.drop(to_drop + [label], axis=1, inplace=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    x = np.array(data.values)
    return x[folds != fold], y[folds != fold], z[folds != fold], x[folds == fold], y[folds == fold], z[folds == fold]


# This function is a minor modification from https://github.com/jmikko/fair_ERM
def read_adult(nTrain=None, scaler=True, shuffle=False, portion_kept=0.3, permute_rows=False, **kwargs):
    if shuffle:
        print('Warning: I wont shuffle because adult has fixed test set')
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    if not os.path.isfile('./data/adult.data'):
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "./data/adult.data")
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", "./data/adult.test")
    data = pd.read_csv(
        "./data/adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
    )
    data = data.iloc[:int(len(data) * portion_kept)]
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        "./data/adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
        skiprows=1, header=None
    )
    data_test = data_test.iloc[:int(len(data_test) * portion_kept)]
    data = pd.concat([data, data_test])
    if permute_rows:
        data = data.sample(frac=1).reset_index(drop=True)
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = " ?" # data.loc[4, "workclass"].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    # Care there is a final dot in the class only in test set which creates 4 different classes
    target = np.array([-1.0 if (val == 0 or val == 1) else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if nTrain is None:
        nTrain = len_train
    target = (target + 1) / 2
    to_protect = 1. * (datamat[:, 9] != datamat[:, 9][0])
    data = np.delete(datamat, 9, axis=1)  # TODO: discuss if A should be dropped from X or not
    return data[:nTrain, :], target[:nTrain], to_protect[:nTrain], data[nTrain:, :], target[nTrain:], to_protect[
                                                                                                      nTrain:]


def read_synthetic_general(etas, gammas, informations, feature_sizes, train_size, test_size, **kwargs):
    """
    eta: P(A=1)
    gamma_0: P(Y=1|A=0)
    gamma_1: P(Y=1|A)
    """
    assert abs(sum(etas) - 1.0) < 0.001
    size = train_size + test_size
    num_categories = len(etas)
    A = np.random.choice(np.arange(num_categories), size=size, replace=True, p=etas)
    Y_options = np.vstack([np.random.choice([0, 1], size=size, replace=True, p=[gammas[i], 1 - gammas[i]]) for i in
                           range(num_categories)])
    Y = np.select([A == i for i in range(num_categories)], Y_options)
    X_base = np.vstack([np.where(A == i, Y[i], -10) for i in range(num_categories)])
    X_list = [A, A]
    for i in range(num_categories):
        X_list += [informations[i] * X_base[i] - 1 + 2 * np.random.rand(size) for _ in range(feature_sizes[i])]
    X = np.stack(X_list, axis=-1)
    A = A / (num_categories - 1)
    return X[:train_size], Y[:train_size], A[:train_size], X[train_size:], Y[train_size:], A[train_size:]


def read_synthetic(eta, gamma_0, gamma_1, information_0, information_1, feature_size_0, feature_size_1, train_size, test_size, seed=RANDOM_SEED):
    """
    eta: P(A=1)
    gamma_0: P(Y=1|A=0)
    gamma_1: P(Y=1|A)
    """
    np.random.seed(seed)
    size = train_size + test_size
    A = np.random.choice([0,1], size=size, replace=True, p=[1-eta, eta]) # generates the A values
    Y_0 = np.random.choice([0,1], size=size, replace=True, p=[1-gamma_0, gamma_0]) # generates Y values given A=0
    Y_1 = np.random.choice([0,1], size=size, replace=True, p=[1-gamma_1, gamma_1]) # generates Y values given A=1
    Y = np.where(A, Y_1, Y_0) # choose Y_a for every sample

    X_0 = np.where(A, 0, 2 * Y_0 - 1)
    X_1 = np.where(A, 2 * Y_1 - 1, -1)
    X = np.stack([A, A] + [information_0 * X_0 - 1 + 2 * np.random.rand(*X_0.shape) for _ in range(feature_size_0)] + [information_1 * X_1 - 1 + 2 * np.random.rand(*X_1.shape) for
                                                                                                 _ in range(feature_size_1)], axis=-1)
    # A[0] = A[-1] = Y[0] = Y[-1] = 1
    return X[:train_size], Y[:train_size], A[:train_size], X[train_size:], Y[train_size:], A[train_size:]