import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from ultralytics import YOLO

# Set constants and paths
EXPERIMENT_NAME = "TurtleBot_ObjectDetection"
RUN_NAME = "Run_3"
DATASET_NAME = "TurtleBot_Dataset_SW1_v1"
EPOCHS = 20
BATCH_SIZE = 16
data_path = './yolo_data/data.yaml'
mlflow.set_tracking_uri("./res/mlruns/")  # Set the mlflow artifacts directory to the root directory of the project

def smooth(y, box_pts=10):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Specify the save directory for training runs
save_dir = './res/'
os.makedirs(save_dir, exist_ok=True)

def main():
    # Check if dataset exists and print absolute path
    abs_data_path = os.path.abspath(data_path)
    print(f"Absolute path of data.yaml: {abs_data_path}")
    
    # Print the contents of the directory
    data_dir = os.path.dirname(abs_data_path)
    print(f"Contents of the directory {data_dir}:")
    for item in os.listdir(data_dir):
        print(item)
    
    if not os.path.exists(abs_data_path):
        raise FileNotFoundError(f"Dataset '{abs_data_path}' does not exist.")
    
    # Initialize and load the YOLO model
    model = YOLO("yolov8n.pt")
    dict_classes = model.model.names

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as mlflow_run:
        # Log the model details
        mlflow.set_tag("base_model", "YOLOv8")
        mlflow.set_tag("optimizer", "Adam")  # Assuming Adam optimizer is used
        mlflow.set_tag("loss", "custom_loss")  # Assuming a custom loss function is used

        # Log the training parameters
        mlflow.log_param("num_epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        # Train the model
        results = model.train(data=abs_data_path, 
                              epochs=EPOCHS, 
                              batch=BATCH_SIZE, 
                              optimizer='AdamW',
                              lr0=0.01,
                              lrf=0.01,
                              plots=True, 
                              visualize=True, 
                              show=True, 
                              save_dir=save_dir,
                              patience=20,
                               save=True,
                               exist_ok=False
                                 )


        # Extract training metrics
        metrics = results.box
        mlflow.log_metric("precision", metrics.p[-1])
        mlflow.log_metric("recall", metrics.r[-1])
        mlflow.log_metric("mAP50", float(metrics.map50))
        mlflow.log_metric("mAP50_95", float(metrics.map))
        mlflow.log_metric("mAP75", float(metrics.map75))

        # Log the model
        mlflow.pytorch.log_model(model.model, "model")

        mlflow_run_id = mlflow_run.info.run_id
        print("MLFlow Run ID: ", mlflow_run_id)

if __name__ == "__main__":
    main()
