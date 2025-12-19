"""
Data loading and preprocessing utilities for animal classification
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image


def get_class_names(dataset_path):
    """
    Get list of class names from dataset directory structure
    
    Args:
        dataset_path: Path to dataset folder containing animal folders
        
    Returns:
        List of class names (animal names)
    """
    class_names = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            class_names.append(item)
    return sorted(class_names)


def load_data_from_directory(dataset_path, img_size=(224, 224), test_size=0.2, validation_size=0.1, batch_size=32):
    """
    Load and preprocess images from directory structure
    
    Args:
        dataset_path: Path to dataset folder
        img_size: Target image size (height, width)
        test_size: Proportion of data for testing
        validation_size: Proportion of training data for validation
        batch_size: Batch size for data generators
        
    Returns:
        train_generator, validation_generator, test_generator, class_names
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=validation_size
    )
    
    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_size
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = val_test_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class names
    class_names = sorted(train_generator.class_indices.keys())
    
    return train_generator, validation_generator, class_names


def preprocess_image(image_path, img_size=(224, 224)):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to image file
        img_size: Target image size (height, width)
        
    Returns:
        Preprocessed image array
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, img_size)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def get_dataset_info(dataset_path):
    """
    Get information about the dataset
    
    Args:
        dataset_path: Path to dataset folder
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'classes': [],
        'class_counts': {},
        'total_images': 0
    }
    
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            # Count images in this class
            image_count = len([f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            info['classes'].append(class_name)
            info['class_counts'][class_name] = image_count
            info['total_images'] += image_count
    
    return info

