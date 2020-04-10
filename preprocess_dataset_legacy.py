# Importing the libararies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
from datetime import datetime
import math

# Importing the dataset
dataset = pd.read_csv("Dataset/test_dataset.csv", header=None)
print(dataset.head())

# Extracting the DateTime Column
dataset_time = dataset.iloc[:, [5]]
print(dataset_time)
dataset_time.columns = range(dataset_time.shape[1])
print(dataset_time.columns)
print(dataset_time.head())

# Dropping the DateTime Column from orignal Dataset
dataset.drop(dataset.columns[[5, 6]], axis=1, inplace=True)
dataset.columns = range(dataset.shape[1])
dataset.head()

# Dependent and Independent Varibales
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Get columns that contain categorical variables
cols = dataset.columns
num_cols = dataset._get_numeric_data().columns
categorical_index = list(set(cols) - set(num_cols))
print("Column indexs with Categorical Variables: {}".format(categorical_index))

# Uncomment the below lines to remove Categorical Varibales with classes more than 5
# thresh = [number for number in categorical_index if number > 5]
# for index in thresh:
#     categorical_index.remove(index)
# X = np.delete(X, thresh, axis=1)

# Encoding
for index in categorical_index:
    print(index)
    # If number of classes in Categorical Column less than 5 (Use OneHotEncoding)
    if len(set(X[:, index])) <= 5:
        label_encoder_X = LabelEncoder()
        print(X[:, index])
        X[:, index] = label_encoder_X.fit_transform(X[:, index])
        print(index)
        onehotencoder = OneHotEncoder(categorical_features=[index])
        X = onehotencoder.fit_transform(X).toarray()
        print(X)

    # If greater than 5 (Use simple Label Encoding)
    elif len(set(X[:, index])) > 5:
        label_encoder_X = LabelEncoder()
        X[:, index] = label_encoder_X.fit_transform(X[:, index])
        print(X)

    else:
        print("Value {} is not understood.".format(index))


# Generate a temporary dataframe to process data for padding
tempDataframe = pd.concat(
    [dataset_time, pd.DataFrame(X), pd.DataFrame(y)], axis=1)
tempDataframe.columns = range(tempDataframe.shape[1])
tempDataframe.head()

# Generate Unique time
uniqueTime = list(set(tempDataframe[0]))
uniqueTime.sort(key=lambda date: datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
uniqueTime[:5]

# Padding

for time in uniqueTime:

    # Number of Rows corresponding to the time
    numberOfRows = len(tempDataframe.loc[tempDataframe[0] == time])

    # Indexes of those rows associated to corresponding time
    rowIndexes = tempDataframe.loc[tempDataframe[0] == time].index

    if numberOfRows < 120:

        # Append zero rows for difference to 120
        for i in range(120-numberOfRows):
            tempDataframe = tempDataframe.append(pd.Series([str(time)]+[zero*0 for zero in range(
                tempDataframe.shape[1]-1)], index=tempDataframe.columns), ignore_index=True)

        print("Number of rows at time {}: {} Padded to --> {}".format(
            time, numberOfRows, len(tempDataframe.loc[tempDataframe[0] == time])))

    elif numberOfRows > 120:

        totalDifference = numberOfRows - 120

        # Drop Rows for difference to 120
        tempDataframe.drop(rowIndexes[-totalDifference:], inplace=True)
        tempDataframe.reset_index()

        print("Number of rows at time {}: {} Dropped to --> {}".format(
            time, numberOfRows, len(tempDataframe.loc[tempDataframe[0] == time])))

# Sort according to time
tempDataframe = tempDataframe.sort_values(by=0, ascending=True)
tempDataframe.reset_index(inplace=True)
tempDataframe = tempDataframe.iloc[:, 1:]

# Scale the data
sc_tempDataframe = StandardScaler()
tempDataframe.iloc[:, 1:-
                   1] = sc_tempDataframe.fit_transform(tempDataframe.iloc[:, 1:-1])

# Extracting data to dataframe
tensorDataframe = pd.DataFrame(columns=['X', 'y'])

for idx, time in enumerate(uniqueTime):
    data = tempDataframe.loc[tempDataframe[0] == time]
    X = data.iloc[:, 1:-1].values
    y = math.ceil(sum(data.iloc[:, -1].values)/len(data.iloc[:, -1].values))
    tensorDataframe.loc[idx] = [X, y]

# Final Variables
X = tensorDataframe.iloc[:, 0].values
y = tensorDataframe.iloc[:, 1].values

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Saving them to feed model
np.save("Dataset/Processed/X_train", X_train)
np.save("Dataset/Processed/X_test", X_test)
np.save("Dataset/Processed/y_train", y_train)
np.save("Dataset/Processed/y_test", y_test)
