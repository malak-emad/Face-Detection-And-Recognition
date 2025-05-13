import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle

import numpy as np

class MyPCA:
    def __init__(self, n_components=0.95): #keeping 95% of principal components
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None

    def fit(self, X): #fitting PCA model to dataset x
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0) #calc mean for each feature (column) to center the data
        X_centered = X - self.mean_ #data centering around the mean 

        # COV matrix of centerd data (X_bar:X_centered)
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # eigenvalues, vectors calc
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sorting in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Explained variance ratio
        total_variance = np.sum(eigenvalues) #total var
        explained_variance_ratio = eigenvalues / total_variance #var by each component

        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            cumulative_variance = np.cumsum(explained_variance_ratio) #cumulated sum of var of each component
            k = np.searchsorted(cumulative_variance, self.n_components) + 1 #least number of PC needed to reach desired threshold 
        else:
            k = int(self.n_components)

        self.components_ = eigenvectors[:, :k].T #store K eigenvalues as PCs
        self.explained_variance_ratio_ = explained_variance_ratio[:k]
        self.n_components_ = k

    def transform(self, X): #project data on PCs (dot product)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)




class FaceRecognition:
    def __init__(self):
        self.training_images = [] #storing images
        self.training_labels = [] #storing labels
        self.subject_names = {} #mapping laabels to subject name
        self.first_images = {}  #storing first image for each subject
        
        #initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # PCA components init
        self.pca = None
        self.mean_face = None
        self.eigenfaces = None
        self.weights = None 
        
        # status flags
        self.model_trained = False
        self.eigenfaces_computed = False
        
    def load_training_data(self, folder_path):
        self.training_images = []
        self.training_labels = []
        self.subject_names = {}
        self.first_images = {}  #reset first images
        
        subject_count = 0
        total_images = 0
        images_data = {}
        
        subject_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))] # each subdirectory is a subject 
        
        for subject_dir in sorted(subject_dirs): #iterating over each subject folder, saving labels
            subject_path = os.path.join(folder_path, subject_dir)
            subject_count += 1
            subject_label = subject_count
            self.subject_names[subject_label] = subject_dir
            
            image_files = [f for f in os.listdir(subject_path) 
                        if f.lower().endswith(('.pgm', '.jpg', '.jpeg', '.png'))]
            
            subject_images = []
            
            for i, img_file in enumerate(image_files): # for each image file of each subject, read image and detect faces
                img_path = os.path.join(subject_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)                
                faces = self.face_cascade.detectMultiScale(img, 1.1, 5)
                
                if len(faces) > 0: #crop face region and resize to 100x100, then add it to training images and labels
                    x, y, w, h = faces[0]
                    face_img = img[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (100, 100))
                    self.training_images.append(face_img)
                    self.training_labels.append(subject_label)
                    subject_images.append(face_img)
                    
                    #store first training image
                    if subject_label not in self.first_images:
                        self.first_images[subject_label] = face_img.copy()
                else:
                    img_resized = cv2.resize(img, (100, 100))
                    self.training_images.append(img_resized)
                    self.training_labels.append(subject_label)
                    subject_images.append(img_resized)

                    #store first training image
                    if subject_label not in self.first_images:
                        self.first_images[subject_label] = img_resized.copy()
                total_images += 1
            
            images_data[subject_dir] = subject_images
        
        if self.training_images:
            self.training_images = np.array(self.training_images)
            self.training_labels = np.array(self.training_labels)
            self.model_trained = True
            
        status_msg = f"Successfully loaded {total_images} images from {subject_count} subjects"
        return True, status_msg, images_data
                

        
    def recognize_face(self, image):  # returns recognized image, results,  best match image, called when matching!!!!

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        result_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        results = []
        best_match_image = None

        for (x, y, w, h) in faces: #for face in test data
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            face_vector = face_roi.flatten().reshape(1, -1) #100x100 array flatenning to 10,000
            face_weights = self.pca.transform(face_vector) #get weights after projection on PCs 

            #prepare detected face to be compared
            distances = np.sqrt(((self.weights - face_weights) ** 2).sum(axis=1))
            min_idx = np.argmin(distances) #take least distance
            eigen_label = self.training_labels[min_idx] #lableing

            max_distance = np.max(distances) if np.max(distances) > 0 else 1
            eigen_confidence = (max_distance - distances[min_idx]) / max_distance #confidence calc

            #draw box and label on result_image
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            subject = self.subject_names.get(eigen_label, f"Unknown (Label {eigen_label})")
            text = f"{subject} ({eigen_confidence:.2f})"
            cv2.putText(result_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 2)

            #store corresponding training image for display
            best_match_image = self.first_images.get(eigen_label, None)

            results.append((eigen_label, eigen_confidence))

            break 

        return result_image, results, best_match_image


    def compute_eigenfaces(self, n_components=0.95):
        h, w = self.training_images[0].shape
        X = np.array([img.flatten() for img in self.training_images])  # precompute flattening for all images

        self.pca = PCA(n_components=n_components)  #apply PCA to training data 
        self.pca.fit(X)

        self.mean_face = self.pca.mean_.reshape(h, w)
        self.eigenfaces = self.pca.components_.reshape(self.pca.n_components_, h, w)
        self.weights = self.pca.transform(X) #get weights of training dataa to be compared later and calc dist
        self.eigenfaces_computed = True
        
        self.visualize_eigenfaces()
        
        return True, f"Computed {self.pca.n_components_} eigenfaces capturing {self.pca.explained_variance_ratio_.sum()*100:.2f}% of variance"
        


    def visualize_eigenfaces(self, num_to_show=49):
        if not self.eigenfaces_computed:
            print("Eigenfaces not yet computed")
            return
            
        num_eigenfaces = min(num_to_show, len(self.eigenfaces))
        
        # Create a 10x3 grid to fit 30 images (1 mean face + 29 eigenfaces)
        plt.figure(figsize=(15, 18))
        
        # Plot mean face in the first position
        plt.subplot(5, 10, 1)
        plt.imshow(self.mean_face, cmap='gray')
        plt.title('Mean Face')
        plt.axis('off')
        
        # Plot eigenfaces in the remaining positions
        for i in range(num_eigenfaces):
            plt.subplot(5, 10, i+2)
            plt.imshow(self.eigenfaces[i], cmap='gray')
            plt.title(f'Eigenface {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        

    def visualize_training_data(self, images_data):
        num_subjects = len(images_data)
        fig, axes = plt.subplots(num_subjects, 5, figsize=(15, 3*num_subjects))
        
        if num_subjects == 1:
            axes = [axes]
            
        for i, (subject, images) in enumerate(images_data.items()):
            if num_subjects > 1:
                axes[i][0].text(-0.5, 0.5, subject, fontsize=12, ha='right', va='center')
            else:
                axes[0].text(-0.5, 0.5, subject, fontsize=12, ha='right', va='center')
            
            for j in range(min(5, len(images))):
                if num_subjects > 1:
                    axes[i][j].imshow(images[j], cmap='gray')
                    axes[i][j].axis('off')
                else:
                    axes[j].imshow(images[j], cmap='gray')
                    axes[j].axis('off')
                    
            for j in range(len(images), 5):
                if num_subjects > 1 and j < len(axes[i]):
                    axes[i][j].axis('off')
                elif j < len(axes):
                    axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()

    
    def generate_confusion_matrix(self, test_data_folder=None):
        """
        Generate and display a confusion matrix from test data folder.
        
        Args:
            test_data_folder (str, optional): Path to folder containing test subject folders.
                                            If None, will use the test folder from the current directory structure.
        """
        if test_data_folder is None:
            # Try to find the test folder in the current directory structure
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Look for standard test data path
            possible_test_paths = [
                os.path.join(current_dir, 'dataset', 'MIT-CBCL-facerec-database', 'SOME DATA', 'test'),
                os.path.join(current_dir, 'test')
            ]
            
            for path in possible_test_paths:
                if os.path.exists(path) and os.path.isdir(path):
                    test_data_folder = path
                    break
                    
            if test_data_folder is None:
                print("Error: Could not find test data folder")
                return None, None
        
        y_true = []  # True labels
        y_pred = []  # Predicted labels
        
        # Process each subject folder in the test directory
        for subject_dir in os.listdir(test_data_folder):
            subject_path = os.path.join(test_data_folder, subject_dir)
            
            # Skip if not a directory
            if not os.path.isdir(subject_path):
                continue
                
            # Process each image in the subject folder
            image_files = [f for f in os.listdir(subject_path) 
                        if f.lower().endswith(('.pgm', '.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                # Use the folder name as the true label
                true_label = subject_dir
                
                img_path = os.path.join(subject_path, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                    
                y_true.append(true_label)
                
                # Get prediction for this image
                _, results, _ = self.recognize_face(img)
                
                if results:
                    pred_label_num = results[0][0]  # Get first result's label
                    # Map the numeric label to the subject folder name
                    pred_subject = self.subject_names.get(pred_label_num, "Unknown")
                    y_pred.append(pred_subject)
                else:
                    y_pred.append("Unknown")
        
        # Get all unique sorted labels
        all_labels = sorted(list(set(y_true + y_pred)))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        
        # Create and display the confusion matrix plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues')
        
        # Set labels
        ax.set_xticks(np.arange(len(all_labels)))
        ax.set_yticks(np.arange(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.set_yticklabels(all_labels)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add text annotations
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                ax.text(j, i, cm[i, j], 
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
        
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
        return cm, all_labels

    def generate_roc_curve(self, test_face_label, selected_eigenfaces=None):

        if not self.model_trained or not self.eigenfaces_computed:
            print("Model not trained or eigenfaces not computed.")
            return

        # Binary labels: 1 if match test label, 0 otherwise
        binary_labels = [1 if self.subject_names[label] == test_face_label else 0 for label in self.training_labels]

        if selected_eigenfaces is None:
            selected_eigenfaces = list(range(min(10, self.pca.n_components_)))  # Default: first 10 eigenfaces

        # Compute ROC per selected eigenface
        plt.figure(figsize=(14, 8))
        roc_auc_values = []

        for idx in selected_eigenfaces:
            scores = self.weights[:, idx]  # projection values as "confidence scores"
            fpr, tpr, _ = roc_curve(binary_labels, scores)
            auc_val = roc_auc_score(binary_labels, scores)
            roc_auc_values.append(auc_val)

            plt.plot(fpr, tpr, label=f'Eigenface {idx+1} (AUC = {auc_val:.2f})', linewidth=2)

        # Plot reference line
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves using PCA Projections for Class {test_face_label}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        # Print Mean AUC
        mean_auc = np.mean(roc_auc_values)
        print(f"Mean AUC over {len(selected_eigenfaces)} eigenfaces: {mean_auc:.3f}")
