import numpy as np
import pandas as pd

TRAIN_DATA = "train.csv"
TEST_DATA = "test.csv"
OUT_DATA = "gender_submission.csv"
AGE_LEVEL = [15, 25, 35, 45, 55, 65, 70, 200]
FARE_LEVEL = [10, 20, 30, 60, 100, 200, 1000]

row_num = 0
survived_num = 0
none_survived_num = 0


def get_rid_of_nan(arr):
    np.nan_to_num(arr.values, copy=False)


def get_rid_of_nan_str(arr):
    values = arr.values
    for i in range(0, len(values)):
        if not isinstance(values[i], str):
            values[i] = ''


def put_numbers_into_buckets(arr, buckets):
    values = arr.values
    for i in range(0, len(values)):
        for j, fence in enumerate(buckets):
            if values[i] <= fence:
                values[i] = j
                break


def count_if(arr, expected):
    result = 0
    for item in arr.values:
        if item == expected:
            result += 1
    return result


def calc_probability_num_with_buckets(dataframe, attr_name, search_value, buckets, survived=True):
    level = 0
    count = 0
    search_values = dataframe[attr_name].values
    survived_values = dataframe['Survived'].values
    for j, fence in enumerate(buckets):
        if search_value <= fence:
            level = j
            break
    for i in range(0, len(search_values)):
        if search_values[i] == level:
            if survived:
                if survived_values[i] == 1:
                    count += 1
            elif survived_values[i] == 0:
                count += 1
    survived_count = survived_num if survived else none_survived_num
    # for Laplace smoothing count += 1 and denominator add buckets length
    count += 1
    return count * 1.0 / (len(buckets) + survived_count)


def calc_probability_num(dataframe, attr_name, search_value, survived=True):
    temp_dict = {}
    count = 0
    search_values = dataframe[attr_name].values
    survived_values = dataframe['Survived'].values
    for item in search_values:
        temp_dict[str(item)] = item
    condition_count = len(temp_dict.keys())
    for i in range(0, len(search_values)):
        if search_values[i] == search_value:
            if survived:
                if survived_values[i] == 1:
                    count += 1
            elif survived_values[i] == 0:
                count += 1
    survived_count = survived_num if survived else none_survived_num
    # for Laplace smoothing count += 1 and denominator add condition_count
    count += 1
    return count * 1.0 / (condition_count + survived_count)


def calc_probability_str(dataframe, attr_name, search_value, survived=True):
    temp_dict = {}
    count = 0
    search_values = dataframe[attr_name].values
    survived_values = dataframe['Survived'].values
    for item in search_values:
        temp_dict[item] = item
    condition_count = len(temp_dict.keys())
    search_value_tokens = search_value.split(" ")
    for i in range(0, len(search_values)):
        for search_item in search_value_tokens:
            search_item = search_item.strip(',.()\"')
            if search_values[i].find(search_item) >= 0:
                if survived:
                    if survived_values[i] == 1:
                        count += 1
                elif survived_values[i] == 0:
                    count += 1
    survived_count = survived_num if survived else none_survived_num
    # for Laplace smoothing count += 1 and denominator add condition_count
    count += 1
    return count * 1.0 / (condition_count + survived_count)


def fetch_data_at(dataframe, row, column):
    values = dataframe.loc[row, [column]].values
    return values[0]


def collect_for_num(old_pair, train_data, test_data, row, column):
    survive = calc_probability_num(train_data, column, fetch_data_at(test_data, row, column), survived=True)
    none_survive = calc_probability_num(train_data, column, fetch_data_at(test_data, row, column), survived=False)
    return survive * old_pair[0], none_survive * old_pair[1]


def collect_with_buckets(old_pair, train_data, test_data, row, column, buckets):
    survive = calc_probability_num_with_buckets(train_data, column, fetch_data_at(test_data, row, column), buckets, survived=True)
    none_survive = calc_probability_num_with_buckets(train_data, column, fetch_data_at(test_data, row, column), buckets, survived=False)
    return survive * old_pair[0], none_survive * old_pair[1]


def collect_for_str(old_pair, train_data, test_data, row, column):
    survive = calc_probability_str(train_data, column, fetch_data_at(test_data, row, column), survived=True)
    none_survive = calc_probability_str(train_data, column, fetch_data_at(test_data, row, column), survived=False)
    return survive * old_pair[0], none_survive * old_pair[1]


def collect_for_name(old_pair, train_data, test_data, row):
    column = 'Name'
    data = fetch_data_at(test_data, row, column)
    index = data.find("Mr. ")
    if index < 0:
        index = data.find("Mrs. ")
        if index < 0:
            index = data.find("Miss. ")
            if index < 0 :
                index = data.find("Master. ")
                if index >= 0:
                    index += 8
            else:
                index += 6
        else:
            index += 5
    else:
        index += 4
    if index > 0:
        data = data[index:]
    survive = calc_probability_str(train_data, column, data, survived=True)
    none_survive = calc_probability_str(train_data, column, data, survived=False)
    return survive * old_pair[0], none_survive * old_pair[1]


alldata = pd.read_csv(TRAIN_DATA)
subdata = alldata.loc[:,['PassengerId','Survived', 'Name', 'Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
get_rid_of_nan(subdata['Age'])
get_rid_of_nan_str(subdata['Cabin'])
get_rid_of_nan_str(subdata['Name'])
put_numbers_into_buckets(subdata['Age'], AGE_LEVEL)
put_numbers_into_buckets(subdata['Fare'], FARE_LEVEL)

row_num = subdata.iloc[:,0].size
survived_num = count_if(subdata['Survived'], 1)
none_survived_num = count_if(subdata['Survived'], 0)

# for Laplace smoothing, K = 2, lambda = 1
survived_probability = (survived_num + 1.0) / (row_num + 2.0)
# for Laplace smoothing, K = 2, lambda = 1
none_survived_probability = (none_survived_num + 1.0) / (row_num + 2.0)

test_data = pd.read_csv(TEST_DATA).loc[:, ['PassengerId', 'Name', 'Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
get_rid_of_nan(test_data['Age'])
get_rid_of_nan_str(test_data['Cabin'])
get_rid_of_nan_str(test_data['Name'])
put_numbers_into_buckets(test_data['Age'], AGE_LEVEL)
put_numbers_into_buckets(test_data['Fare'], FARE_LEVEL)

test_rows = len(test_data.iloc[:, [0]])
result = np.zeros([test_rows, 1])

for i in range(test_rows):
    sum = 0.0
    pair = (1.0, 1.0)
    pair = collect_for_num(pair, subdata, test_data, i, 'Pclass')
    pair = collect_for_str(pair, subdata, test_data, i, 'Sex')
    pair = collect_with_buckets(pair, subdata, test_data, i, 'Age', AGE_LEVEL)
    pair = collect_for_num(pair, subdata, test_data, i, 'SibSp')
    pair = collect_for_num(pair, subdata, test_data, i, 'Parch')
    pair = collect_with_buckets(pair, subdata, test_data, i, 'Fare', FARE_LEVEL)
    pair = collect_for_str(pair, subdata, test_data, i, 'Cabin')
    # pair = collect_for_num(pair, subdata, test_data, i, 'Embarked')
    pair = collect_for_name(pair, subdata, test_data, i)

    survive = pair[0] * survived_probability
    none_survive = pair[1] * none_survived_probability

    result[i][0] = int(1 if survive >= none_survive else 0)
    print("----> row (", i, ") survive ", survive, "; none_survive ", none_survive)

test_data['Survived'] = result
test_data = test_data.loc[:, ['PassengerId', 'Survived']]
test_data.to_csv(OUT_DATA)
