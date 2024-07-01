import os
import torch
from ultralytics import YOLO
import mlflow
import mlflow.pytorch

# Set constants and paths
EXPERIMENT_NAME = "TurtleBot_ObjectDetection_SW_V2"
RUN_NAME = "Run_1"
EPOCHS = 20
BATCH_SIZE = 32
data_path = './yolo_data_v2/data.yaml'  # Updated to new dataset path
pretrained_model_path = 'runs/detect/train3/weights/best.pt'  # Path to the previously trained model
mlflow.set_tracking_uri("./res/mlruns/")  # Set the mlflow artifacts directory to the root directory of the project

def freeze_backbone_layers(model):
    # Assuming the backbone consists of the initial layers before the detection layers
    # This will vary based on the specific architecture of YOLOv8
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

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
    model = YOLO(pretrained_model_path)
    model.model.names = ['cone', 'obstacle', 'turtlebot']  # Update the class names
    model.model.nc = 3  # Update the number of classes

    # Freeze the backbone layers
    freeze_backbone_layers(model.model)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as mlflow_run:
        # Log the model details
        mlflow.set_tag("base_model", "YOLOv8")
        mlflow.set_tag("optimizer", "AdamW")  # Assuming AdamW optimizer is used
        mlflow.set_tag("loss", "custom_loss")  # Assuming a custom loss function is used

        # Log the training parameters
        mlflow.log_param("num_epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("learning_rate_factor", 0.01)

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

        # Evaluate the model on the test set
        test_results = model.val(data=abs_data_path)
        mlflow.log_metric("test_precision", test_results.p[-1])
        mlflow.log_metric("test_recall", test_results.r[-1])
        mlflow.log_metric("test_mAP50", float(test_results.map50))
        mlflow.log_metric("test_mAP50_95", float(test_results.map))
        mlflow.log_metric("test_mAP75", float(test_results.map75))

        # Log the model
        mlflow.pytorch.log_model(model.model, "model")

        mlflow_run_id = mlflow_run.info.run_id
        print("MLFlow Run ID: ", mlflow_run_id)

if __name__ == "__main__":
    main()
