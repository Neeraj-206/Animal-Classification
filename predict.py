"""
Prediction script for Animal Classification
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import argparse
from data_loader import preprocess_image
import matplotlib.pyplot as plt
from PIL import Image


def load_model_and_classes(model_path):
    """
    Load trained model and class names
    
    Args:
        model_path: Path to saved model
        
    Returns:
        model, class_names
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load class names
    class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.json')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    else:
        print("Warning: class_names.json not found. Using default class names.")
        class_names = None
    
    return model, class_names


def predict_image(model, image_path, class_names=None, top_k=3):
    """
    Predict animal class for a single image
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        predictions: List of (class_name, probability) tuples
    """
    # Preprocess image
    img = preprocess_image(image_path)
    if img is None:
        return None
    
    # Make prediction
    predictions = model.predict(img, verbose=0)[0]
    
    # Get top k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        class_name = class_names[idx] if class_names else f"Class {idx}"
        probability = float(predictions[idx])
        results.append((class_name, probability))
    
    return results


def predict_batch(model, image_paths, class_names=None, top_k=3):
    """
    Predict animal classes for multiple images
    
    Args:
        model: Trained Keras model
        image_paths: List of image file paths
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        List of prediction results
    """
    results = []
    for image_path in image_paths:
        pred = predict_image(model, image_path, class_names, top_k)
        results.append((image_path, pred))
    return results


def visualize_prediction(image_path, predictions, save_path=None):
    """
    Visualize prediction results
    
    Args:
        image_path: Path to image
        predictions: List of (class_name, probability) tuples
        save_path: Optional path to save visualization
    """
    # Load and display image
    img = Image.open(image_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Display predictions
    classes = [pred[0] for pred in predictions]
    probs = [pred[1] * 100 for pred in predictions]
    
    y_pos = np.arange(len(classes))
    axes[1].barh(y_pos, probs, align='center')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(classes)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Probability (%)', fontsize=12)
    axes[1].set_title('Predictions', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 100)
    
    # Add probability labels
    for i, prob in enumerate(probs):
        axes[1].text(prob + 1, i, f'{prob:.2f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predict animal class from image')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/animal_classifier.keras', 
                    help='Path to trained model')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--save_viz', type=str, default=None, help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using train.py")
        return
    
    # Load model and classes
    print(f"Loading model from {args.model}...")
    model, class_names = load_model_and_classes(args.model)
    print("Model loaded successfully!")
    
    # Make prediction
    print(f"\nPredicting animal class for: {args.image}")
    predictions = predict_image(model, args.image, class_names, args.top_k)
    
    if predictions:
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        for i, (class_name, probability) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {probability * 100:.2f}%")
        print("=" * 60)
        
        # Visualize if requested
        if args.visualize or args.save_viz:
            visualize_prediction(args.image, predictions, args.save_viz)
    else:
        print("Error: Could not make prediction")


if __name__ == "__main__":
    main()

