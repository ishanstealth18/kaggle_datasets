import numpy as np
import pandas as pd
import torch
import torchmetrics.classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import shap

pd.set_option('future.no_silent_downcasting', True)


# function to read file
def read_file():
    input_src_file = pd.read_csv('accident_prediction_india.csv')
    return input_src_file


# function to remove unwanted columns, for me I consider below columns to be unwanted as they dont add much value to
# model
def remove_columns(input_file):
    columns_names = list(input_file.columns)
    for i in columns_names:
        if (i == 'Year' or i == 'Month' or i == 'Day of Week' or i == 'Time of Day' or i == 'Driver Gender'
                or i == 'City Name' or i == 'State Name' or i == 'Number of Casualties'):
            input_file.drop([i], axis=1, inplace=True)
    print("After removing columns: ")
    print(input_file.info())

    return input_file


# function to remove null values
def remove_null_val(input_file):
    # check null values
    print(input_file.isna().sum())

    input_file = input_file.dropna()
    print('After removing null values')
    print(input_file.isna().sum())

    return input_file


# function to convert categorical vealues to numeric values, as model accept only numeric values. I used pd.get_dummies()
def categorical_to_numeric_val(input_file):

    accident_severity_dummy = pd.get_dummies(input_file['Accident Severity'], dtype=int, prefix='Accident_Severity', drop_first=True)
    vehicle_type_dummy = pd.get_dummies(input_file['Vehicle Type Involved'], dtype=int, prefix='Vehicle_Type', drop_first=True)
    weather_condition_dummy = pd.get_dummies(input_file['Weather Conditions'], dtype=int, prefix='Weather_conditions', drop_first=True)
    road_type_dummy = pd.get_dummies(input_file['Road Type'], dtype=int, prefix='Road_Type', drop_first=True)
    road_condition_dummy = pd.get_dummies(input_file['Road Condition'], dtype=int, prefix='Road_Condition', drop_first=True)
    lighting_condition_dummy = pd.get_dummies(input_file['Lighting Conditions'], dtype=int, prefix='Lighting_Conditions', drop_first=True)
    traffic_control_dummy = pd.get_dummies(input_file['Traffic Control Presence'], dtype=int, prefix='Traffic_Control_Presence', drop_first=True)
    driver_license_dummy = pd.get_dummies(input_file['Driver License Status'], dtype=int, prefix='Driver_License_Status', drop_first=True)
    alcohol_dummy = pd.get_dummies(input_file['Alcohol Involvement'], dtype=int, prefix='Alcohol_Involvement', drop_first=True)

    combined_input_file = pd.concat([input_file, accident_severity_dummy, vehicle_type_dummy, weather_condition_dummy,
                                    road_type_dummy, road_condition_dummy, lighting_condition_dummy, traffic_control_dummy,
                                    driver_license_dummy, alcohol_dummy], axis=1)

    combined_input_file.drop(['Accident Severity', 'Vehicle Type Involved', 'Weather Conditions', 'Road Type','Road Condition',
                     'Lighting Conditions', 'Traffic Control Presence', 'Driver License Status', 'Alcohol Involvement'],
                    axis=1, inplace=True)

    return combined_input_file


# function to check outliers if any
def check_outliers(input_file):
    # as we have only 4 columns with data type int64, we will check for them only
    column_names = ['Number of Vehicles Involved', 'Number of Fatalities', 'Speed Limit (km/h)', 'Driver Age']

    for i in range(len(column_names)):
        plt.subplot(2, 2, i + 1)
        plt.boxplot(input_file[column_names[i]])
        plt.xlabel(column_names[i])
        plt.plot()

    plt.tight_layout()
    plt.show()


# function to replace target values with numeric ordeal values(0,1,2,3) for classification
def replace_target_values(input_file):
    input_file = input_file.replace({'Accident Location Details': {'Bridge': 0, 'Curve': 1, 'Intersection': 2,
                                                                   'Straight Road': 3}})
    input_file['Accident Location Details'] = pd.to_numeric(input_file['Accident Location Details'])
    #print(input_file.head(5))

    return input_file


# function to normalize values to get all the features on 1 scale. Did for both training and test data
def normalize(x_tr, x_te):
    scalar = StandardScaler()
    x_tr = scalar.fit_transform(x_tr)
    x_te = scalar.transform(x_te)
    return x_tr, x_te


# function to create KNN model
def knn_model(x_tr, x_te, y_tr, y_te):
    scores = []
    for i in range(1,31):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_tr, y_tr)
        knn.predict(x_te)
        score = knn.score(x_te, y_te)
        scores.append(np.mean(score))

    plt.plot(range(1, 31), scores, marker='o')
    plt.xlabel('K neighbors')
    plt.ylabel('Mean accuracy')
    plt.title('KNN Accuracy')
    plt.show()


# function to check importance of features contribution to the accuracy
def shap_val(x_tr,x_te,y_tr, features):
    knn = KNeighborsClassifier()
    knn.fit(x_tr, y_tr)
    explainer = shap.KernelExplainer(knn.predict, x_tr)
    shap_values = explainer.shap_values(x_te, nsamples=200)
    shap.summary_plot(shap_values, x_te, feature_names=features, plot_type='bar')


# function to create nural network model
def neural_network(x_tr, x_te, y_tr, y_te):

    # creating and load training and validation dataset (training data: 70%, test data: 30%)
    dataset = TensorDataset(torch.Tensor(x_tr), torch.Tensor(y_tr).to(torch.LongTensor()))
    validation_dataset = TensorDataset(torch.Tensor(x_te), torch.Tensor(y_te).to(torch.LongTensor()))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True)
    feature_count = len(x_tr[0])
    num_of_classes = 4

    # create sequential layers, took hidden layers count as (num of features + 1= 30). Using Relu() activation
    # function to reduce vanishing gradient. You can play with different activation functions, I got better result
    # with ReLu(). You can try different count of hidden layers and nurons.
    model = nn.Sequential(nn.Linear(feature_count, 30),
                          nn.Linear(30, 20),
                          nn.Linear(20,10),
                          nn.Linear(10, 5),
                          nn.ReLU(),
                          #nn.Dropout(p=0.5),
                          nn.Linear(5, num_of_classes),
    )

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer with learning rate and momentum, momentum will help not getting model stuck at some point in learning.
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.85)
    # metric for accuracy
    metric = torchmetrics.Accuracy(task='multiclass', num_classes=4)

    training_epoch_accuracy_list = []
    loss_list = []
    validation_loss_list = []
    validation_epoch_accuracy = []

    # I got better result with epoch count 50. You can try some another count.
    for epoch in range(50):
        training_loss = 0.0
        validation_loss = 0.0

        for data in dataloader:
            # making gradients zero before starting
            optimizer.zero_grad()
            # load data
            feature, target = data
            # predict value on training data
            predicted_val = model(feature)
            # calculate loss
            loss = criterion(predicted_val, target)
            # compute gradients
            loss.backward()
            # update gradients for all the layers
            optimizer.step()
            # again making gradients 0
            optimizer.zero_grad()
            training_loss += loss.item()
            # update metric
            metric.update(predicted_val, target)
        # calculate mean training loss per epoch
        mean_training_loss = training_loss/len(dataloader)
        loss_list.append(mean_training_loss)
        # calculate accuracy
        accuracy = metric.compute()
        training_epoch_accuracy_list.append(accuracy)

        # do metric reset
        metric.reset()
        # put model into evaluate mode
        model.eval()
        # again make gradients 0
        with torch.no_grad():
            # repeat same steps as training model
            for data in validation_dataloader:
                v_feature, v_target = data
                output = model(v_feature)
                v_loss = criterion(output, v_target)
                validation_loss += v_loss
                metric.update(output, v_target)
        mean_validation_loss = validation_loss/len(validation_dataloader)
        validation_loss_list.append(mean_validation_loss)
        validation_accuracy = metric.compute()
        validation_epoch_accuracy.append(validation_accuracy)
        #put model into training mode again
        model.train()
        metric.reset()


    # plot loss and accuracy
    plt.plot(range(50), loss_list, marker='o', label='Training Loss')
    plt.plot(range(50), validation_loss_list, marker='o', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Values')
    plt.title('Training Loss v/s Validation Loss')
    plt.legend()
    plt.show()

    plt.plot(range(50), training_epoch_accuracy_list, marker='o', label='Training Accuracy')
    plt.plot(range(50), validation_epoch_accuracy, marker='o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()


def data_process():
    input_file = read_file()

    # remove unnecessary columns
    input_after_col_removed = remove_columns(input_file)

    # check and remove null values
    input_file_null_removed = remove_null_val(input_after_col_removed)

    # replace target values as numeric encoding (optional). In this case its possible so I am doing it.
    input_file_replaced_val= replace_target_values(input_file_null_removed)

    # check outliers
    check_outliers(input_file_replaced_val)

    # do one hot encoding for categorical values
    numeric_val_input_file = categorical_to_numeric_val(input_file_replaced_val)

    # assign feature and target variables
    x = numeric_val_input_file.drop(['Accident Location Details'], axis=1).values
    y = numeric_val_input_file['Accident Location Details'].values
    feature_names = list(numeric_val_input_file.drop(['Accident Location Details'], axis=1).columns)

    # divide into training and test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)

    # normalize the data
    normalized_x_train, normalized_X_test = normalize(X_train, X_test)

    # knn model
    #knn_model(x, y)
    knn_model(normalized_x_train, normalized_X_test, y_train, y_test)

    # SHAP values for KNN model
    shap_val(normalized_x_train, normalized_X_test, y_train, feature_names)

    # Neural network
    neural_network(normalized_x_train, normalized_X_test, y_train, y_test)

data_process()
