from river import linear_model, metrics, preprocessing
import csv

# Initialize the model pipeline: scale features + logistic regression
model = preprocessing.StandardScaler() | linear_model.LogisticRegression()

# Initialize accuracy metric to track performance over time
metric = metrics.Accuracy()

# Store accuracy after each update
accuracy_log = []

def learn_stream(x, y):
    """
    Update the online model with one sample at a time.
    """
    y_pred = model.predict_one(x)  # Predict current label
    model.learn_one(x, y)          # Update model with true label
    metric.update(y, y_pred)       # Update accuracy metric
    
    accuracy_log.append(metric.get())  # Log accuracy
    
    return y_pred

def get_accuracy():
    """
    Returns the current accuracy score of the model.
    """
    return metric.get()

def reset_model():
    """
    Resets the model and metric to start fresh learning.
    """
    global model, metric, accuracy_log
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    metric = metrics.Accuracy()
    accuracy_log = []

def save_accuracy_log(filename="online_accuracy_log.csv"):
    """
    Saves the accuracy log to a CSV file.
    """
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Accuracy"])
        for i, acc in enumerate(accuracy_log, 1):
            writer.writerow([i, acc])
