"""
Training script for Animal Classification Model
Uses Transfer Learning with ResNet50
"""
import os
import sys
import io

# Fix Windows encoding issues
if sys.platform == 'win32':
    # Set environment variable first
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Reconfigure stdout/stderr if they are not already wrapped
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import matplotlib.pyplot as plt
from data_loader import load_data_from_directory, get_dataset_info

# Set random seeds for reproducibility
tf.random.set_seed(42)


def create_model(num_classes, img_size=(224, 224), learning_rate=0.001):
    """
    Create a transfer learning model based on ResNet50
    
    Args:
        num_classes: Number of animal classes
        img_size: Input image size
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50 model (without top layers)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model, base_model


def train_model(dataset_path, epochs=50, batch_size=32, img_size=(224, 224), 
                model_save_path='models/animal_classifier.keras'):
    """
    Train the animal classification model
    
    Args:
        dataset_path: Path to dataset folder
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size (height, width)
        model_save_path: Path to save the trained model
    """
    print("=" * 60)
    print("Animal Classification Model Training")
    print("=" * 60)
    
    # Get dataset info
    print("\nLoading dataset information...")
    dataset_info = get_dataset_info(dataset_path)
    print(f"Number of classes: {len(dataset_info['classes'])}")
    print(f"Total images: {dataset_info['total_images']}")
    print("\nClass distribution:")
    for class_name, count in dataset_info['class_counts'].items():
        print(f"  {class_name}: {count} images")
    
    # Load data generators
    print("\nLoading data generators...")
    train_gen, val_gen, class_names = load_data_from_directory(
        dataset_path, 
        img_size=img_size, 
        batch_size=batch_size
    )
    
    num_classes = len(class_names)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Create model
    print("\nCreating model...")
    model, base_model = create_model(num_classes, img_size)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model (Phase 1: Frozen base model)
    print("\n" + "=" * 60)
    print("Phase 1: Training with frozen base model")
    print("=" * 60)
    
    history1 = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=2  # Use verbose=2 to avoid Unicode issues on Windows
    )
    
    # Fine-tuning (Phase 2: Unfreeze some layers)
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning (unfreezing top layers)")
    print("=" * 60)
    
    # Unfreeze top layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze bottom layers
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate * 0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    # Continue training
    history2 = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        initial_epoch=len(history1.history['loss']),
        verbose=2  # Use verbose=2 to avoid Unicode issues on Windows
    )
    
    # Combine histories
    history = {
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
    }
    
    # Save final model
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save class names
    import json
    class_names_path = os.path.join(os.path.dirname(model_save_path), 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")
    
    return model, history


def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved to training_history.png")
    plt.show()


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "dataset"  # Relative to script location
    EPOCHS = 50
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    MODEL_SAVE_PATH = "models/animal_classifier.keras"
    LEARNING_RATE = 0.001
    
    # Train the model
    model, history = train_model(
        dataset_path=DATASET_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        model_save_path=MODEL_SAVE_PATH
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

