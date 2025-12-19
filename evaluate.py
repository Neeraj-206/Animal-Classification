"""
Evaluation script for Animal Classification Model
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from data_loader import load_data_from_directory, get_dataset_info


def evaluate_model(model_path, dataset_path, batch_size=32, img_size=(224, 224)):
    """
    Evaluate the trained model on test data
    
    Args:
        model_path: Path to trained model
        dataset_path: Path to dataset folder
        batch_size: Batch size for evaluation
        img_size: Image size (height, width)
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Load class names
    class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.json')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    else:
        print("Warning: class_names.json not found. Getting class names from dataset...")
        dataset_info = get_dataset_info(dataset_path)
        class_names = dataset_info['classes']
    
    print(f"\nNumber of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Load validation data (we'll use validation set as test set for evaluation)
    print("\nLoading validation data...")
    train_gen, val_gen, _ = load_data_from_directory(
        dataset_path,
        img_size=img_size,
        batch_size=batch_size
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(val_gen, verbose=1)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    if len(results) > 2:
        print(f"Top-3 Accuracy: {results[2]:.4f} ({results[2]*100:.2f}%)")
    print("=" * 60)
    
    # Get predictions for confusion matrix
    print("\nGenerating predictions for detailed analysis...")
    val_gen.reset()
    y_true = []
    y_pred = []
    
    num_batches = len(val_gen)
    for i in range(num_batches):
        if i % 10 == 0:
            print(f"Processing batch {i+1}/{num_batches}...")
        X_batch, y_batch = val_gen[i]
        predictions = model.predict(X_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved to confusion_matrix.png")
    plt.show()
    
    # Per-class accuracy
    print("\n" + "=" * 60)
    print("PER-CLASS ACCURACY")
    print("=" * 60)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_accuracies[i]*100:.2f}%")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/animal_classifier.keras"
    DATASET_PATH = "dataset"
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        print("Please train the model first using train.py")
    else:
        evaluate_model(MODEL_PATH, DATASET_PATH, BATCH_SIZE, IMG_SIZE)

