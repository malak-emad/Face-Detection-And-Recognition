import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class FaceRecognition:
    def __init__(self):
        # Store training data
        self.training_images = []
        self.training_labels = []
        self.subject_names = {}
        self.first_images = {}  # Store the first image for each subject
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # PCA components
        self.pca = None
        self.mean_face = None
        self.eigenfaces = None
        self.weights = None
        
        # Status flags
        self.model_trained = False
        self.eigenfaces_computed = False
        
    def load_training_data(self, folder_path):
            if not os.path.isdir(folder_path):
                return False, f"Not a valid directory: {folder_path}", None
                
            try:
                self.training_images = []
                self.training_labels = []
                self.subject_names = {}
                self.first_images = {}  # Reset first images
                
                subject_count = 0
                total_images = 0
                images_data = {}
                
                subject_dirs = [d for d in os.listdir(folder_path) 
                            if os.path.isdir(os.path.join(folder_path, d))]
                
                for subject_dir in sorted(subject_dirs):
                    subject_path = os.path.join(folder_path, subject_dir)
                    subject_count += 1
                    subject_label = subject_count
                    self.subject_names[subject_label] = subject_dir
                    
                    image_files = [f for f in os.listdir(subject_path) 
                                if f.lower().endswith(('.pgm', '.jpg', '.jpeg', '.png'))]
                    
                    subject_images = []
                    
                    for i, img_file in enumerate(image_files):
                        img_path = os.path.join(subject_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            print(f"Warning: Could not read {img_file}")
                            continue
                        
                        faces = self.face_cascade.detectMultiScale(img, 1.1, 5)
                        
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_img = img[y:y+h, x:x+w]
                            face_img = cv2.resize(face_img, (100, 100))
                            self.training_images.append(face_img)
                            self.training_labels.append(subject_label)
                            subject_images.append(face_img)
                            
                            # Store the first training image
                            if subject_label not in self.first_images:
                                self.first_images[subject_label] = face_img.copy()
                        else:
                            img_resized = cv2.resize(img, (100, 100))
                            self.training_images.append(img_resized)
                            self.training_labels.append(subject_label)
                            subject_images.append(img_resized)

                            # Store the first training image
                            if subject_label not in self.first_images:
                                self.first_images[subject_label] = img_resized.copy()

                        total_images += 1
                    
                    images_data[subject_dir] = subject_images
                
                if total_images == 0:
                    return False, "No valid images found in the selected folder", None
                    
                if self.training_images:
                    self.training_images = np.array(self.training_images)
                    self.training_labels = np.array(self.training_labels)
                    self.model_trained = True
                    
                status_msg = f"Successfully loaded {total_images} images from {subject_count} subjects"
                return True, status_msg, images_data
                
            except Exception as e:
                error_msg = f"Error loading training data: {str(e)}"
                return False, error_msg, None
        
    def recognize_face(self, image):
        if not self.model_trained or not self.eigenfaces_computed:
            return None, [], None  # now returns a third value for best match image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        result_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        results = []
        best_match_image = None

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            face_vector = face_roi.flatten().reshape(1, -1)
            face_weights = self.pca.transform(face_vector)

            distances = np.sqrt(((self.weights - face_weights) ** 2).sum(axis=1))
            min_idx = np.argmin(distances)
            eigen_label = self.training_labels[min_idx]

            max_distance = np.max(distances) if np.max(distances) > 0 else 1
            eigen_confidence = (max_distance - distances[min_idx]) / max_distance

            # Draw box and label on result_image
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            subject = self.subject_names.get(eigen_label, f"Unknown (Label {eigen_label})")
            text = f"{subject} ({eigen_confidence:.2f})"
            cv2.putText(result_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Store corresponding training image for display
            best_match_image = self.first_images.get(eigen_label, None)

            results.append((eigen_label, eigen_confidence))

            break  # Handle only the first detected face

        return result_image, results, best_match_image


    
    def compute_eigenfaces(self, n_components=0.95):
        if len(self.training_images) == 0:
            return False, "No training data available"
        
        try:
            num_images = len(self.training_images)
            h, w = self.training_images[0].shape
            X = np.array([img.flatten() for img in self.training_images])  # Precompute flattening

            # Use PCA without whitening if appropriate
            self.pca = PCA(n_components=n_components)  # Consider removing whiten=True
            self.pca.fit(X)

            self.mean_face = self.pca.mean_.reshape(h, w)
            self.eigenfaces = self.pca.components_.reshape(self.pca.n_components_, h, w)
            self.weights = self.pca.transform(X)
            self.eigenfaces_computed = True
            
            self.visualize_eigenfaces()
            
            return True, f"Computed {self.pca.n_components_} eigenfaces capturing {self.pca.explained_variance_ratio_.sum()*100:.2f}% of variance"
            
        except Exception as e:
            return False, f"Error computing eigenfaces: {str(e)}"
            
            self.visualize_eigenfaces()
            
            return True, f"Computed {self.pca.n_components_} eigenfaces capturing {self.pca.explained_variance_ratio_.sum()*100:.2f}% of variance"
            
        except Exception as e:
            return False, f"Error computing eigenfaces: {str(e)}"
    
    def visualize_eigenfaces(self, num_to_show=16):
        if not self.eigenfaces_computed:
            print("Eigenfaces not yet computed")
            return
            
        num_eigenfaces = min(num_to_show, len(self.eigenfaces))
        
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 5, 1)
        plt.imshow(self.mean_face, cmap='gray')
        plt.title('Mean Face')
        plt.axis('off')
        
        for i in range(num_eigenfaces):
            plt.subplot(4, 5, i+2)
            plt.imshow(self.eigenfaces[i], cmap='gray')
            plt.title(f'Eigenface {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        


    
    def visualize_training_data(self, images_data):
        if not images_data:
            print("No data to visualize")
            return
                
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