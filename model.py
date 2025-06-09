import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Configure system for optimal CPU performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.get_logger().setLevel('ERROR')

# Enable mixed precision for better performance
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

class DepressionDetectionModel:
    """
    Optimized CNN-based model for detecting depression through facial micro-expressions
    with CPU-efficient training implementation.
    """
    
    def __init__(self, img_size=(128, 128), batch_size=32, epochs=30, use_pretrained=True):
        # Reduced image size for faster processing
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_pretrained = use_pretrained
        self.model = None
        self.history = None
        self.class_names = ['No Depression', 'Depression']
        
        # Initialize paths (update these with your actual paths)
        self.ckplus_path = 'path/to/CK+'
        self.casme_path = 'path/to/CASME_II'
        
    def load_and_preprocess_data(self):
        """
        Optimized data loading and preprocessing with caching
        """
        print("Loading and preprocessing data...")
        
        # Load datasets
        posed_images, posed_labels = self._load_ckplus()
        spontaneous_images, spontaneous_labels = self._load_spontaneous_datasets()
        
        # Combine datasets
        all_images = np.concatenate([posed_images, spontaneous_images])
        all_labels = np.concatenate([posed_labels, spontaneous_labels])
        
        # Convert to float32 and normalize once
        all_images = all_images.astype('float32') / 255.0
        
        # Split into train/test with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
        )
        
        # Further split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
        )
        
        # Add channel dimension
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _load_ckplus(self, csv_path='ckextended.csv'):
        """Optimized CK+ dataset loading with pre-allocation"""
        data = pd.read_csv(csv_path)
        
        # Pre-allocate arrays
        X = np.empty((len(data), *self.img_size), dtype=np.uint8)
        y = np.empty(len(data), dtype=np.uint8)
        
        depression_emotions = [0, 2, 4]  # Angry, Fear, Sad = Depressed
        non_depression_emotions = [1, 3, 5, 6]  # Disgust, Happy, Surprise, Neutral
        
        valid_idx = 0
        for _, row in data.iterrows():
            label = row['emotion']
            if label in depression_emotions:
                y[valid_idx] = 1
            elif label in non_depression_emotions:
                y[valid_idx] = 0
            else:
                continue  # skip unclassified
                
            pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
            img = pixels.reshape((48, 48))
            img = cv2.resize(img, self.img_size)
            X[valid_idx] = img
            valid_idx += 1
        
        # Trim to actual size
        return X[:valid_idx], y[:valid_idx]
    
    def _load_spontaneous_datasets(self):
        """Simplified spontaneous datasets loading"""
        print("Loading spontaneous micro-expression datasets...")
        
        # Placeholder implementation - replace with your actual dataset loading
        # For demonstration, we'll create some dummy data
        num_samples = 1000
        images = np.random.randint(0, 256, (num_samples, *self.img_size), dtype=np.uint8)
        labels = np.random.randint(0, 2, num_samples, dtype=np.uint8)
        
        return images, labels
    
    def build_model(self, input_shape=(128, 128, 1)):
        """Optimized model architecture for CPU training"""
        print("Building optimized model...")
        
        inputs = layers.Input(shape=input_shape)
        
        if self.use_pretrained:
            # Efficient grayscale to RGB conversion
            x = layers.Concatenate()([inputs, inputs, inputs])
            
            # Use smaller base model
            base_model = applications.MobileNetV3Small(
                include_top=False,
                weights='imagenet',
                input_shape=(input_shape[0], input_shape[1], 3),
                pooling='avg'  # Direct global pooling
            )
            base_model.trainable = False
            x = base_model(x)
        else:
            # Lightweight custom architecture
            x = layers.Conv2D(32, (5, 5), activation='relu')(inputs)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Flatten()(x)
        
        # Reduced complexity head
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs, outputs)
        
        # Use larger initial learning rate with decay
        optimizer = optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print(self.model.summary())
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Optimized training process with efficient augmentation"""
        print("Starting optimized training...")
        
        # Simplified callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Use Keras's built-in ImageDataGenerator for efficient CPU augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='constant',
            cval=0
        )
        
        # Train with validation every 2 epochs to reduce overhead
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1,
            validation_freq=2
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Efficient model evaluation"""
        print("Evaluating model...")
        
        # Predict in batches to reduce memory usage
        y_pred = (self.model.predict(X_test, batch_size=self.batch_size) > 0.5).astype(int)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        # ROC Curve
        y_prob = self.model.predict(X_test, batch_size=self.batch_size)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc
        }

# Example usage
if __name__ == "__main__":
    # Initialize optimized model
    depression_detector = DepressionDetectionModel(
        img_size=(128, 128),  # Reduced image size
        batch_size=32,
        epochs=30,
        use_pretrained=True
    )
    
    # Load and preprocess data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = depression_detector.load_and_preprocess_data()
    
    # Build optimized model
    depression_detector.build_model()
    
    # Train with efficient augmentation
    history = depression_detector.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    metrics = depression_detector.evaluate(X_test, y_test)
    
    # Save the trained model
    depression_detector.model.save('optimized_depression_model.h5')
    print("Model saved successfully.")