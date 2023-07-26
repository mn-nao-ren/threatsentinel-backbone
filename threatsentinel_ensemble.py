# threatsentinel_ensemble.py

import numpy as np
from vulnfind_model import main as vulnfind_main
from intrusion_detection_model import main as intrusion_detection_main
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
import pandas as pd

import zipfile


# Function to extract ground truth labels from a code gadget
def extract_ground_truth_labels(code_gadget):
    # Split the code gadget into lines
    lines = code_gadget.split('\n')
    
    # Find the last line that is a digit and return it
    for line in reversed(lines):
        if line.strip().isdigit():
            return int(line.strip())
    
        

# Function to extract ground truth labels from the entire dataset
def get_ground_truth_labels(dataset):
    # Split the dataset into separate code gadgets using the delimiter "---------------------------------"
    code_gadgets = dataset.split('---------------------------------')
                                  
    # Extract ground truth labels for each code gadget
    ground_truth_labels = [extract_ground_truth_labels(gadget) for gadget in code_gadgets if gadget.strip()]
    
    return ground_truth_labels

def prepare_for_credential_attack(intrusion_type, has_vulnerabilities):
    
    if intrusion_type == 'impersonation' and has_vulnerabilities == 'yes':
        return True
    


def main():
    # Train and validate each learner using their respective training datasets and validation datasets
    vulnfind_predictions = vulnfind_main() # predictions are generated from vulnfind's TESTING dataset, not the TRAINING dataset
    intrusion_detection_predictions = intrusion_detection_main() #  predictions are generated from intrusion detector's TESTING dataset, not the TRAINING dataset

    # Get the ground truth labels from the testing dataset for VulnFind
    vulnfind_dataset_filename = "cwe_cgd_testing_dataset.txt"
    with open(vulnfind_dataset_filename, 'r') as file:
        vulnfind_dataset = file.read()

    vulnfind_ground_truth = get_ground_truth_labels(vulnfind_dataset)

    # Get ground truth labels from intrusion detection testing dataset
    url = "http://icsdweb.aegean.gr/awid/features.html"
    features = pd.read_html(url)[0]["Field Name"].tolist()

    with zipfile.ZipFile("awid_test_data.zip", "r") as zip_ref:
        with zip_ref.open("awid_test_data.csv") as file:
            awid_test = pd.read_csv(file, header=None, names=features)

    # Preprocess the testing data 
    awid_test.replace({"?": None}, inplace=True)
    awid_test.dropna(inplace=True)
    columns_with_mostly_null_data = awid_test.columns[awid_test.isnull().mean() >= 0.5]
    awid_test.drop(columns_with_mostly_null_data, axis=1, inplace=True)
    X_test, y_test = awid_test.select_dtypes(['number']), awid_test['class']

    # Encode the ground truth labels for the testing dataset
    encoder = LabelEncoder()
    binarizer = LabelBinarizer()
    encoded_y_test = encoder.transform(y_test)
    binarized_y_test = binarizer.transform(encoded_y_test)




    # Calculate the accuracy of the vulnfind predictions
    vulnfind_accuracy = accuracy_score(vulnfind_ground_truth, vulnfind_predictions)
    print(f'VulnFind accuracy: {vulnfind_accuracy}')

    # Calculate the accuracy of the intrusion detection model on the testing dataset
    accuracy = accuracy_score(binarized_y_test, intrusion_detection_predictions)
    print(f"Intrusion detection Accuracy on the testing dataset: {accuracy}")

if __name__ == "__main__":
    main()