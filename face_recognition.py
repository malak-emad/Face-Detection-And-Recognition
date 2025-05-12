import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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