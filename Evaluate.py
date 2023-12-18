import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path to the folder containing the JSON files
folder_path = 'output_folder'
output_folder_path = 'model_tests'
output_file_path = os.path.join(output_folder_path, 'final_outputs.json')

# Function to calculate metrics
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return accuracy, precision, recall, f1

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Iterate through JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.startswith('answers-problem') and filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        # Read the JSON data
        with open(file_path, 'r') as file:
            results = json.load(file)

        # Check if the results list contains information about the model run
        if "num_init_impost" in results[-1]:
            # Information about the model run
            model_info = results.pop()

            # Metrics for each task
            true_labels = [entry["true-author"] for entry in results]
            predicted_labels = [entry["predicted-author"] for entry in results]
            metrics = {
                "accuracy": accuracy_score(true_labels, predicted_labels),
                "precision": precision_score(true_labels, predicted_labels, average='weighted'),
                "recall": recall_score(true_labels, predicted_labels, average='weighted'),
                "f1": f1_score(true_labels, predicted_labels, average='weighted'),
                **model_info
            }

            # Save metrics as a JSON file
            with open(output_file_path, 'a') as output_file:
                json.dump(metrics, output_file, indent=4)
                output_file.write('\n')

            # Print or store the metrics for each task
            print(f"Metrics for {filename}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
            print("\n")
