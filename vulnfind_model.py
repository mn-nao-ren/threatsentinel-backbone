# vulnfind_model.py


import subprocess
import os
import numpy as np

def train_model(dataset_path):
    train_script_path = "vulnfind_train.py"
    command = ["python", train_script_path, dataset_path]
    subprocess.run(command, check=True)

def diagnose_model(dataset_path, model_path):
    diagnose_script_path = "vulnfind_diagnose.py"
    command = ["python", diagnose_script_path, dataset_path, model_path]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()

def main():
    # Replace with the path to your training dataset
    training_dataset = "cwe399_cgd_dataset.txt"

    # Train the model
    train_model(training_dataset)

    # Replace with the path where you want to save the trained model
    model_save_path = "cwe399_cgd_dataset_model.h5"

    

    # Test dataset
    testing_dataset = "cwe_cgd_testing_dataset.txt"

    
    vulnfind_output_diagnosis = diagnose_model(testing_dataset, model_save_path)

    # Print the evaluation result
    print("VulnFind Model Output Diagnosis (can be treated as an evaluation result for the VulnFind component): ")
    print(vulnfind_output_diagnosis)

    # Convert the diagnoses/predictions back into an array
    predictions = vulnfind_output_diagnosis.replace('\n', '').replace('[', '').replace(']', '').replace('.', '').split(' ')
    predictions = np.array(predictions, dtype=int)

    return predictions
    

if __name__ == "__main__":
    main()
