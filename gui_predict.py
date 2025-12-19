"""
GUI Application for Animal Classification Predictions
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageTk
import json
from data_loader import preprocess_image


class AnimalClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Classification AI")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        self.model = None
        self.class_names = None
        self.current_image_path = None
        self.model_path = "models/animal_classifier.keras"
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            messagebox.showwarning(
                "Model Not Found",
                f"Model file not found at {self.model_path}\n\n"
                "Please train the model first using train.py"
            )
        else:
            self.load_model()
        
        self.create_widgets()
    
    def load_model(self):
        """Load the trained model and class names"""
        try:
            self.model = keras.models.load_model(self.model_path)
            
            # Load class names
            class_names_path = os.path.join(os.path.dirname(self.model_path), 'class_names.json')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
            else:
                self.class_names = None
            
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model = None
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Title
        title_label = tk.Label(
            self.root,
            text="🐾 Animal Classification AI",
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Model status
        status_frame = tk.Frame(self.root, bg='#f0f0f0')
        status_frame.pack(pady=10)
        
        if self.model:
            status_label = tk.Label(
                status_frame,
                text="✓ Model Loaded",
                font=("Arial", 10),
                bg='#d4edda',
                fg='#155724',
                padx=10,
                pady=5
            )
        else:
            status_label = tk.Label(
                status_frame,
                text="✗ Model Not Found",
                font=("Arial", 10),
                bg='#f8d7da',
                fg='#721c24',
                padx=10,
                pady=5
            )
        status_label.pack()
        
        # Image selection button
        select_btn = tk.Button(
            self.root,
            text="Select Image",
            command=self.select_image,
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        select_btn.pack(pady=20)
        
        # Image display frame
        image_frame = tk.Frame(self.root, bg='white', relief=tk.SUNKEN, bd=2)
        image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(
            image_frame,
            text="No image selected",
            bg='white',
            font=("Arial", 12),
            fg='gray'
        )
        self.image_label.pack(expand=True)
        
        # Predict button
        predict_btn = tk.Button(
            self.root,
            text="Predict Animal",
            command=self.predict,
            font=("Arial", 12, "bold"),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3,
            state=tk.DISABLED
        )
        predict_btn.pack(pady=10)
        self.predict_btn = predict_btn
        
        # Results frame
        results_frame = tk.LabelFrame(
            self.root,
            text="Predictions",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        results_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Results text
        self.results_text = tk.Text(
            results_frame,
            height=8,
            font=("Arial", 11),
            bg='white',
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
    
    def select_image(self):
        """Open file dialog to select image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.config(state=tk.DISABLED)
    
    def display_image(self, image_path):
        """Display selected image in GUI"""
        try:
            # Open and resize image
            img = Image.open(image_path)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def predict(self):
        """Make prediction on selected image"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please train the model first.")
            return
        
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        
        try:
            # Preprocess image
            img = preprocess_image(self.current_image_path)
            if img is None:
                messagebox.showerror("Error", "Failed to preprocess image.")
                return
            
            # Make prediction
            predictions = self.model.predict(img, verbose=0)[0]
            
            # Get top 5 predictions
            top_indices = np.argsort(predictions)[-5:][::-1]
            
            # Display results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            self.results_text.insert(tk.END, "Top 5 Predictions:\n\n", "title")
            self.results_text.tag_config("title", font=("Arial", 12, "bold"))
            
            for i, idx in enumerate(top_indices, 1):
                class_name = self.class_names[idx] if self.class_names else f"Class {idx}"
                probability = predictions[idx] * 100
                
                # Create progress bar representation
                bar_length = int(probability / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                result_text = f"{i}. {class_name:15s} {probability:6.2f}% {bar}\n"
                self.results_text.insert(tk.END, result_text)
            
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")


def main():
    root = tk.Tk()
    app = AnimalClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

