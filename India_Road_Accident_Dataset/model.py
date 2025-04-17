import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('future.no_silent_downcasting', True)
def read_file():
    input_src_file = pd.read_csv('accident_prediction_india.csv')
    return input_src_file


def remove_columns(input_file):
    columns_names = list(input_file.columns)
    for i in columns_names:
        if (i == 'Year' or i == 'Month' or i == 'Day of Week' or i == 'Time of Day' or i == 'Driver Gender'
                or i == 'City Name' or i == 'State Name' or i == 'Number of Casualties'):
            input_file.drop([i], axis=1, inplace=True)
    print("After removing columns: ")
    print(input_file.info())

    return input_file


def remove_null_val(input_file):
    # check null values
    print(input_file.isna().sum())

    input_file = input_file.dropna()
    print('After removing null values')
    print(input_file.isna().sum())

    return input_file


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
    #print(combined_input_file.info())

    return combined_input_file


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


def replace_target_values(input_file):
    input_file = input_file.replace({'Accident Location Details': {'Bridge': 1, 'Curve': 2, 'Intersection': 3,
                                                                   'Straight Road': 4}})
    input_file['Accident Location Details'] = pd.to_numeric(input_file['Accident Location Details'])
    #print(input_file.head(5))

    return input_file


def normalize(x_tr, x_te):
    scalar = StandardScaler()
    scalar.fit_transform(x_tr)
    scalar.transform(x_te)

    return x_tr, x_te


def data_process():
    input_file = read_file()
    print(input_file.info())

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

    # divide into training and test set
    X_train, y_train, X_test, y_test = train_test_split(x, y, test_size=0.3, random_state=21, stratify=True)

    # normalize the data
    normalized_x_train, normalized_X_test = normalize(X_train, X_test)






data_process()
