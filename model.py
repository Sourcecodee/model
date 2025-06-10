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
from tqdm import tqdm
import albumentations as A

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class DepressionDetectionModel:
    """
    A comprehensive CNN-based model for detecting concealed depression through facial micro-expressions.
    Incorporates ethical considerations, uses pre-trained models, and provides evaluation metrics.
    """
    
    def __init__(self, img_size=(224, 224), batch_size=32, epochs=50, use_pretrained=True):
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_pretrained = use_pretrained
        self.model = None
        self.history = None
        self.class_names = ['No Depression', 'Depression']
        
        # Initialize ethical considerations
        self.fairness_metrics = {}
        self.demographic_data = {}  # To track performance across groups
        
        # Data augmentation
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.CoarseDropout(max_holes=8, p=0.2)
        ])
        
        # Initialize paths (you'll need to set these)
        self.ckplus_path = 'path/to/CK+'
        self.casme_path = 'path/to/CASME_II'
        
    def load_and_preprocess_data(self):
        """
        Load CK+, CASME II, and SAMM datasets, preprocess them, and create train/test splits.
        Handles both posed (CK+) and spontaneous (CASME II/SAMM) expressions.
        """
        print("Loading and preprocessing data...")
        
        # Placeholder for dataset loading - you'll need to implement actual loading
        # This is a simplified version - real implementation would handle each dataset's structure
        
        # Simulated data loading (replace with actual dataset loading)
        posed_images, posed_labels = self._load_ckplus()
        spontaneous_images, spontaneous_labels = self._load_spontaneous_datasets()
        
        # Combine datasets
        all_images = np.concatenate([posed_images, spontaneous_images])
        all_labels = np.concatenate([posed_labels, spontaneous_labels])
        
        # Split into train/test with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
        )
        
        # Further split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    


    def _load_ckplus(self, csv_path='ckextended.csv'):
        """Load and preprocess CK+ dataset for posed micro-expressions"""
        
        data = pd.read_csv(csv_path)

        # Define emotion mapping (customize as needed for depression detection)
        # Example: group emotions to binary class (0 = non-depressed, 1 = depressed)
        depression_emotions = [0, 2, 4]  # Angry, Fear, Sad = Depressed
        non_depression_emotions = [1, 3, 5, 6]  # Disgust, Happy, Surprise, Neutral

        def map_emotion(label):
            if label in depression_emotions:
                return 1
            elif label in non_depression_emotions:
                return 0
            else:
                return -1  # optional, skip unclassified

        X = []
        y = []

        for _, row in data.iterrows():
            label = map_emotion(row['emotion'])
            if label == -1:
                continue  # skip if not mapped

            pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
            img = pixels.reshape((48, 48))  # FER2013 images are 48x48 grayscale
            img = cv2.resize(img, self.img_size)
            X.append(img)
            y.append(label)
        print(data)

        return X, y
    
    def _load_spontaneous_datasets(self):
        """Load and preprocess CASME II and SAMM datasets (spontaneous micro-expressions)"""
        # Implement actual loading of spontaneous datasets
        print("Loading spontaneous micro-expression datasets...")
        images = []
        labels = []
        
        # Example for CASME II (adapt to your actual dataset)
        casme_data = pd.read_excel(os.path.join(self.casme_path, 'CASME II_label_ver.xlsx'))
        
        for idx, row in casme_data.iterrows():
            subject = row['Subject']
            onset_frame = row['OnsetFrame']
            apex_frame = row['ApexFrame']
            emotion = row['Emotion']
            
            # Determine if this micro-expression indicates depression
            # This is simplified - you'd need clinical validation
            label = 1 if emotion in ['sad', 'fear', 'disgust'] else 0
            
            # Load the image sequence
            subject_path = os.path.join(self.casme_path, f'sub{subject:02d}')
            if os.path.exists(subject_path):
                img_files = sorted([f for f in os.listdir(subject_path) if f.endswith('.jpg')])
                for frame_idx in range(onset_frame, apex_frame + 1):
                    if frame_idx <= len(img_files):
                        img_path = os.path.join(subject_path, img_files[frame_idx - 1])
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype('float32') / 255.0
                        images.append(img)
                        labels.append(label)
        
        # Similar loading for SAMM dataset...
        
        return np.array(images), np.array(labels)
    
    def build_model(self, input_shape=(224, 224, 1)):
        """Build the CNN model, optionally using transfer learning"""
        print("Building model...")
        
        inputs = layers.Input(shape=input_shape)
        
        if self.use_pretrained:
            # Use EfficientNetV2B0 as feature extractor (pretrained on ImageNet)
            # Note: We need to adapt for grayscale input
            base_model = applications.EfficientNetV2B0(
                include_top=False,
                weights='imagenet',
                input_shape=(input_shape[0], input_shape[1], 3)
            )
            
            # Freeze the base model
            base_model.trainable = False
            
            # Convert grayscale to RGB by repeating channels
            x = layers.Conv2D(3, (3, 3), padding='same')(inputs)
            x = base_model(x)
        else:
            # Build from scratch
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
        
        # Common head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs, outputs)
        
        optimizer = optimizers.Adam(learning_rate=0.0001)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print(self.model.summary())
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model with callbacks and data augmentation"""
        print("Training model...")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Data augmentation generator
        def train_generator(X, y, batch_size):
            while True:
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i + batch_size]
                    batch_y = y[i:i + batch_size]
                    
                    # Apply augmentation
                    augmented_X = []
                    for img in batch_X:
                        # Convert to 3 channels for albumentations
                        img_3ch = np.stack([img.squeeze()]*3, axis=-1)
                        augmented = self.augmentation(image=img_3ch)['image']
                        # Convert back to grayscale
                        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
                        augmented_X.append(augmented)
                    
                    augmented_X = np.array(augmented_X)
                    augmented_X = np.expand_dims(augmented_X, axis=-1)  # Add channel dimension
                    
                    yield augmented_X, batch_y
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // self.batch_size
        
        # Train
        self.history = self.model.fit(
            train_generator(X_train, y_train, self.batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=(np.expand_dims(X_val, axis=-1), y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance with comprehensive metrics"""
        print("Evaluating model...")
        
        # Predict
        X_test = np.expand_dims(X_test, axis=-1)
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        
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
        y_prob = self.model.predict(X_test)
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
    
    def analyze_fairness(self, X_test, y_test, demographic_data):
        """
        Analyze model performance across different demographic groups
        to identify potential biases.
        """
        print("Analyzing model fairness...")
        
        X_test = np.expand_dims(X_test, axis=-1)
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        
        # Example demographic analysis (adapt based on your metadata)
        for group_name, group_indices in demographic_data.items():
            group_y_test = y_test[group_indices]
            group_y_pred = y_pred[group_indices]
            
            accuracy = accuracy_score(group_y_test, group_y_pred)
            precision = precision_score(group_y_test, group_y_pred)
            recall = recall_score(group_y_test, group_y_pred)
            f1 = f1_score(group_y_test, group_y_pred)
            
            self.fairness_metrics[group_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': len(group_indices)
            }
        
        print("\nFairness Analysis:")
        for group, metrics in self.fairness_metrics.items():
            print(f"\nGroup: {group}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1: {metrics['f1']:.3f}")
            print(f"  Samples: {metrics['support']}")
        
        return self.fairness_metrics
    
    def visualize_activations(self, sample_image):
        """Visualize what the model is focusing on using Grad-CAM"""
        print("Generating activation visualizations...")
        
        # Expand dimensions if needed
        if len(sample_image.shape) == 2:
            sample_image = np.expand_dims(sample_image, axis=(0, -1))
        elif len(sample_image.shape) == 3:
            sample_image = np.expand_dims(sample_image, axis=0)
        
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer = layer.name
                break
        
        if last_conv_layer is None:
            print("No convolutional layer found in model")
            return
        
        # Create Grad-CAM model
        grad_model = models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(last_conv_layer).output, self.model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(sample_image)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (self.img_size[1], self.img_size[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        sample_img_color = cv2.cvtColor(np.uint8(255 * sample_image[0].squeeze()), cv2.COLOR_GRAY2BGR)
        superimposed_img = cv2.addWeighted(sample_img_color, 0.6, heatmap, 0.4, 0)
        
        # Display
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_image[0].squeeze(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title('Activation Heatmap')
        plt.axis('off')
        plt.show()
        
        return superimposed_img

# Example usage
if __name__ == "__main__":
    # Initialize model
    depression_detector = DepressionDetectionModel(
        img_size=(224, 224),
        batch_size=32,
        epochs=10,
        use_pretrained=True
    )
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = depression_detector.load_and_preprocess_data()
    
    # Build model
    depression_detector.build_model()
    
    # Train model
    history = depression_detector.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    metrics = depression_detector.evaluate(X_test, y_test)
    
    # Fairness analysis (requires demographic metadata)
    # demographic_data = {'gender': {'male': male_indices, 'female': female_indices}}
    # fairness_metrics = depression_detector.analyze_fairness(X_test, y_test, demographic_data)
    
    # Visualize activations for a sample
    sample_idx = np.random.randint(0, len(X_test))
    depression_detector.visualize_activations(X_test[sample_idx])