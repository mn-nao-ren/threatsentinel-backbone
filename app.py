from flask import Flask, request, jsonify
from threading import Thread  # Import Thread from the 'threading' module

app = Flask(__name__)

# Import the machine learning functionality from threatsentinel_ensemble.py
from threatsentinel_ensemble import main as run_machine_learning

# API endpoint for handling incoming requests from the GUI
@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected datasets from the request (you may need to adapt this part based on your GUI)
    selected_network_dataset = request.form['network_traffic_dataset']
    selected_code_samples_dataset = request.form['source_code_samples_dataset']

    # Check if both dataset options are selected
    if selected_network_dataset and selected_code_samples_dataset:
        # Run the machine learning script and get the results
        results = run_machine_learning()

        # Return the results as JSON to the GUI
        return jsonify(results)
    else:
        # Return an error message if the user has not selected both dataset options
        return jsonify({"error": "Please select both Network Traffic Dataset and Software Source Code."})

if __name__ == "__main__":
    app.run(debug=True)
