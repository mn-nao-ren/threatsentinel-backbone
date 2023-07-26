# intrusion_detection_model.py

import pandas as pd
import zipfile
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def train_intrusion_detection_model():
    # Load the data
    url = "http://icsdweb.aegean.gr/awid/features.html"
    features = pd.read_html(url)[0]["Field Name"].tolist()

    with zipfile.ZipFile("awid_training_data.zip", "r") as zip_ref:
        with zip_ref.open("awid_training_data.csv") as file:
            awid = pd.read_csv(file, header=None, names=features)

    # Preprocess the data
    awid.replace({"?": None}, inplace=True)
    awid.dropna(inplace=True)
    columns_with_mostly_null_data = awid.columns[awid.isnull().mean() >= 0.5]
    awid.drop(columns_with_mostly_null_data, axis=1, inplace=True)
    X, y = awid.select_dtypes(['number']), awid['class']

    # Encode the target variable
    encoder = LabelEncoder()
    binarizer = LabelBinarizer()
    encoded_y = encoder.fit_transform(y)
    binarized_y = binarizer.fit_transform(encoded_y)

    # Build the neural network model
    def create_network_model(hidden_dimensions, input_dim):
        model = Sequential()
        model.add(Dense(hidden_dimensions[0], input_dim=input_dim, activation='relu'))
        for dimension in hidden_dimensions[1:]:
            model.add(Dense(dimension, activation='relu'))
        model.add(Dense(4, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Create the pipeline with scaling and classification
    preprocessing = Pipeline([("scale", StandardScaler())])
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", KerasClassifier(build_fn=create_network_model, epochs=10, batch_size=128,
                                       verbose=0, hidden_dimensions=(30, 30, 30, 10), input_dim=74))
    ])

    # Fit the pipeline to the data
    pipeline.fit(X, binarized_y)

    # Load the testing dataset
    with zipfile.ZipFile("awid_test_data.zip", "r") as zip_ref:
        with zip_ref.open("awid_test_data.csv") as file:
            awid_test = pd.read_csv(file, header=None, names=features)

    
    # Preprocess the testing data
    awid_test.replace({"?": None}, inplace=True)
    awid_test.dropna(inplace=True)
    awid_test.drop(columns_with_mostly_null_data, axis=1, inplace=True)
    X_test, y_test = awid_test.select_dtypes(['number']), awid_test['class']


    # Encode the target variable for the testing dataset
    encoded_y_test = encoder.transform(y_test)
    binarized_y_test = binarizer.transform(encoded_y_test)

    # Get predictions from the neural network model on the testing dataset
    predictions_intrudetector_nn_test = pipeline.predict(X_test)

    return predictions_intrudetector_nn_test

def main():
    predictions_test_result = train_intrusion_detection_model()
    
    
     # Print the predictions for the testing dataset
    print("Predictions for the testing dataset:")
    for prediction in predictions_test_result:
        print(prediction)

    
    return predictions_test_result

   
if __name__ == "__main__":
    # Call the main function to obtain the predictions
    main()