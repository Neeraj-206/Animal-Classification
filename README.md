# Animal Classification AI Project

A deep learning project for classifying 15 different animal species using transfer learning with ResNet50.

## 🐾 Animals Classified

The model can classify the following 15 animals:
- Bear
- Bird
- Cat
- Cow
- Deer
- Dog
- Dolphin
- Elephant
- Giraffe
- Horse
- Kangaroo
- Lion
- Panda
- Tiger
- Zebra

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow 2.13.0 or higher
- See `requirements.txt` for full list of dependencies

## 🚀 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Animal Classification"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
Animal Classification/
├── dataset/              # Dataset folder with animal subfolders
│   ├── Bear/
│   ├── Bird/
│   ├── Cat/
│   └── ...
├── models/               # Saved models (created after training)
├── data_loader.py        # Data loading and preprocessing utilities
├── train.py             # Training script
├── predict.py           # Prediction script
├── evaluate.py          # Model evaluation script
├── gui_predict.py       # GUI for predictions (optional)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🎯 Usage

### 1. Training the Model

Train the model on your dataset:

```bash
python train.py
```

**Configuration (in `train.py`):**
- `DATASET_PATH`: Path to dataset folder (default: "dataset")
- `EPOCHS`: Number of training epochs (default: 50)
- `BATCH_SIZE`: Batch size (default: 32)
- `IMG_SIZE`: Image size (default: (224, 224))
- `MODEL_SAVE_PATH`: Where to save the model (default: "models/animal_classifier.h5")

The training process includes:
- **Phase 1**: Training with frozen ResNet50 base model
- **Phase 2**: Fine-tuning with unfrozen top layers
- Automatic model checkpointing
- Early stopping to prevent overfitting
- Learning rate reduction on plateau

After training, you'll get:
- Saved model: `models/animal_classifier.h5`
- Class names: `models/class_names.json`
- Training history plot: `training_history.png`

### 2. Making Predictions

#### Command Line Interface

Predict a single image:

```bash
python predict.py --image path/to/image.jpg
```

**Options:**
- `--image`: Path to image file (required)
- `--model`: Path to trained model (default: "models/animal_classifier.h5")
- `--top_k`: Number of top predictions to show (default: 3)
- `--visualize`: Show visualization of predictions
- `--save_viz`: Save visualization to file

**Example:**
```bash
python predict.py --image dataset/Bear/Bear_1.jpg --visualize --top_k 5
```

#### GUI Interface

For a user-friendly interface:

```bash
python gui_predict.py
```

This opens a graphical interface where you can:
- Select an image file
- View the image
- See top predictions with probabilities
- Save results

### 3. Evaluating the Model

Evaluate model performance on validation data:

```bash
python evaluate.py
```

This will generate:
- Overall accuracy and loss
- Per-class accuracy
- Classification report
- Confusion matrix (saved as `confusion_matrix.png`)

## 📊 Model Architecture

The model uses **Transfer Learning** with ResNet50:

1. **Base Model**: Pre-trained ResNet50 (ImageNet weights)
2. **Custom Head**:
   - Global Average Pooling
   - Dense layer (512 units, ReLU)
   - Dropout (0.5)
   - Dense layer (256 units, ReLU)
   - Dropout (0.3)
   - Output layer (15 units, Softmax)

## 🔧 Customization

### Changing Image Size

Edit `IMG_SIZE` in the scripts:
```python
IMG_SIZE = (224, 224)  # Change to (256, 256) or (299, 299) for example
```

### Adjusting Training Parameters

Modify in `train.py`:
```python
EPOCHS = 50           # Increase for more training
BATCH_SIZE = 32       # Adjust based on GPU memory
LEARNING_RATE = 0.001 # Lower for fine-tuning
```

### Using Different Base Models

You can modify `train.py` to use other pre-trained models:
- EfficientNetB0/B1/B2
- VGG16/VGG19
- InceptionV3
- MobileNetV2

Example:
```python
from tensorflow.keras.applications import EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, ...)
```

## 📈 Training Tips

1. **GPU Recommended**: Training is much faster on GPU. Install TensorFlow GPU version if available.

2. **Data Augmentation**: Already included in the training pipeline to improve generalization.

3. **Early Stopping**: The model automatically stops if validation accuracy doesn't improve for 10 epochs.

4. **Model Checkpointing**: Best model is automatically saved based on validation accuracy.

5. **Learning Rate**: Automatically reduced if validation loss plateaus.

 Troubleshooting
Out of Memory Error
- Reduce `BATCH_SIZE` in `train.py`
- Reduce `IMG_SIZE`
- Use a smaller base model (e.g., MobileNetV2)

Model Not Found Error
- Make sure you've trained the model first using `train.py`
- Check that `models/animal_classifier.h5` exists

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## 📝 Notes

- The dataset should be organized with each animal class in its own folder
- Images can be in JPG, JPEG, or PNG format
- The model automatically splits data into training (80%) and validation (20%)
- Validation set is used for both validation during training and final evaluation

## 🎓 Learning Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

