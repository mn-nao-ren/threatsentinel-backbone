# generate-accuracy-vs-epochs.py

import numpy as np
import pandas as pd
import zipfile
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

# Function to build the neural network model
def network_builder(hidden_dimensions, input_dim):
    model = Sequential()
    model.add(Dense(hidden_dimensions[0], input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    for dimension in hidden_dimensions[1:]:
        model.add(Dense(dimension, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main function to train the model and plot accuracy versus epoch
def main():
    # Load the data (Note: Update the URL with the actual data source if needed)
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

    # Initialize the model with the desired hidden layer dimensions and other parameters
    model = network_builder(hidden_dimensions=(60, 30, 10), input_dim=74)

    # Train the model for the specified number of epochs and record accuracy after each epoch
    epochs = 10
    accuracy_scores = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.fit(X, binarized_y, epochs=1, batch_size=128, verbose=1)
        _, accuracy = model.evaluate(X, binarized_y)
        accuracy_scores.append(accuracy)

    # Print the accuracy scores for each epoch
    for epoch, accuracy in enumerate(accuracy_scores, start=1):
        print(f"Epoch {epoch}: {accuracy:.4f}")

    # Plot accuracy scores versus epoch number
    plt.plot(range(1, epochs + 1), accuracy_scores, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Scores vs. Epoch Number')
    plt.grid(True)
    plt.show()

    # Return accuracy scores and the plot
    return accuracy_scores, plt

if __name__ == "__main__":
    accuracy_scores, plot = main()