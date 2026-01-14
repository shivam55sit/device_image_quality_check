import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import timm
from PIL import Image
import cv2
import numpy as np
import os
import time
import json
import joblib
import xgboost as xgb
import pandas as pd
import pywt
from scipy import ndimage
from skimage import feature
from skimage.feature import graycomatrix, graycoprops
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from enum import Enum


class QualityState(Enum):
    YES = 'Y'      # Good quality
    NO = 'N'       # Bad quality  
    PARTIAL = 'P'  # Partially acceptable

class OverallQuality(Enum):
    BAD = 'Bad Quality'
    USABLE = 'Usable Quality'
    GOOD = 'Good Quality'


class EyeDetector:
    """Eye presence detection using ensemble of ResNet18 models"""
    
    def __init__(self, model_dir='./models/peakmodels', use_ensemble=True, device=None):
        self.model_dir = model_dir
        self.use_ensemble = use_ensemble
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.models = self._load_models()
        
    def _load_models(self):
        models = []
        num_models = 5 if self.use_ensemble else 1
        
        for seed in range(num_models):
            model = torchvision.models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 2)
            
            model_path = os.path.join(self.model_dir, f'Classeye_10k_full_L0.0001_M0.99_{seed}.pth')
            
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"Loaded eye detection model {seed + 1}/{num_models}")
            except FileNotFoundError:
                print(f"Warning: Eye detection model file not found: {model_path}")
                if seed == 0:
                    raise FileNotFoundError(f"No eye detection model files found in {self.model_dir}")
                
        return models
    
    def predict(self, image_path: str) -> Dict:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            predictions = []
            confidences = []
            
            with torch.no_grad():
                for model in self.models:
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    predictions.append(predicted.item())
                    confidences.append(probabilities[0, 1].item())
            
            if self.use_ensemble:
                vote_sum = sum(predictions)
                has_eye = vote_sum > len(self.models) / 2
                avg_confidence = np.mean(confidences)
            else:
                has_eye = bool(predictions[0])
                avg_confidence = confidences[0]
            
            return {
                'status': 'success',
                'has_eye': has_eye,
                'confidence': float(avg_confidence)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'has_eye': False,
                'confidence': 0.0
            }

class FocusDetector:
    """Focus/Blur detection model"""
    
    def __init__(self, model_path: str, scaler_path: str, feature_names_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(feature_names_path, 'r') as f:
            self.feature_names = f.read().strip().split(',')
    
    def extract_focus_features(self, img):
        img_norm = img.astype(float) / 255.0
        features = {}
        
        # 1. Laplacian
        laplacian = cv2.Laplacian(img_norm, cv2.CV_64F)
        features['laplacian_var'] = np.var(laplacian)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))
        
        # 2. Sobel
        sobelx = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features['sobel_magnitude_mean'] = np.mean(sobel_magnitude)
        features['sobel_magnitude_var'] = np.var(sobel_magnitude)
        
        # 3. Wavelet
        coeffs = pywt.dwt2(img_norm, 'haar')
        LL, (LH, HL, HH) = coeffs
        features['wavelet_LH_energy'] = np.mean(LH**2)
        features['wavelet_HL_energy'] = np.mean(HL**2)
        features['wavelet_HH_energy'] = np.mean(HH**2)
        features['wavelet_detail_to_approx_ratio'] = (np.mean(LH**2) + np.mean(HL**2) + np.mean(HH**2)) / (np.mean(LL**2) + 1e-10)
        
        # 4. Canny Edge
        edges = feature.canny(img_norm)
        features['edge_density'] = np.sum(edges) / edges.size
        
        # 5. GLCM
        glcm = graycomatrix((img_norm * 255).astype(np.uint8), distances=[1], 
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast')[0])
        features['glcm_dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity')[0])
        features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity')[0])
        features['glcm_energy'] = np.mean(graycoprops(glcm, 'energy')[0])
        features['glcm_correlation'] = np.mean(graycoprops(glcm, 'correlation')[0])
        
        # 6. Frequency
        f_transform = np.fft.fft2(img_norm)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
        h, w = magnitude_spectrum.shape
        center_y, center_x = h//2, w//2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        high_freq_mask = dist_from_center > min(h, w) // 4
        low_freq_mask = dist_from_center <= min(h, w) // 8
        features['high_freq_energy'] = np.sum(magnitude_spectrum * high_freq_mask) / np.sum(high_freq_mask)
        features['low_freq_energy'] = np.sum(magnitude_spectrum * low_freq_mask) / np.sum(low_freq_mask)
        features['freq_energy_ratio'] = features['high_freq_energy'] / (features['low_freq_energy'] + 1e-10)
        
        # 7. Local Contrast
        local_contrast = ndimage.generic_filter(img_norm, np.std, size=5)
        features['local_contrast_mean'] = np.mean(local_contrast)
        features['local_contrast_var'] = np.var(local_contrast)
        
        return features
    
    def predict(self, image_path: str) -> Dict:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img = cv2.resize(img, (224, 224))
            features = self.extract_focus_features(img)
            features_df = pd.DataFrame([features])[self.feature_names]
            
            features_scaled = self.scaler.transform(features_df)
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            confidence = float(max(probability))
            # RELAXED LOGIC: Only reject if confident it's Bad (Blurry)
            if prediction == 0 and confidence > 0.8:
                quality_state = QualityState.NO
            else:
                quality_state = QualityState.YES
            
            return {
                'status': 'success',
                'prediction': 'Focused' if prediction == 1 else 'Blurry',
                'quality_state': quality_state.value,
                'confidence': confidence
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'quality_state': QualityState.NO.value
            }

class LightweightDilatedCNN(nn.Module):
    """Lightweight CNN for illumination detection"""
    def __init__(self, in_channels=3, feature_dim=256):
        super(LightweightDilatedCNN, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.dilated_block1 = self._make_dilated_block(64, 128, dilation=1)
        self.dilated_block2 = self._make_dilated_block(128, 128, dilation=2)
        self.dilated_block3 = self._make_dilated_block(128, 128, dilation=4)
        self.dilated_block4 = self._make_dilated_block(128, 128, dilation=8)
        
        self.attention = nn.Sequential(
            nn.Conv2d(128 * 4, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128 * 4, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(128 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.feature_projection = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def _make_dilated_block(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.initial_conv(x)
        
        d1 = self.dilated_block1(x)
        d2 = self.dilated_block2(d1)
        d3 = self.dilated_block3(d2)
        d4 = self.dilated_block4(d3)
        
        multi_scale = torch.cat([d1, d2, d3, d4], dim=1)
        
        attention_weights = self.attention(multi_scale)
        attended_features = multi_scale * attention_weights
        
        features = self.feature_fusion(attended_features)
        features = features.view(features.size(0), -1)
        features = self.feature_projection(features)
        
        return features

class HandcraftedFeatureExtractor:
    """Extract handcrafted features for illumination"""
    def __init__(self):
        self.feature_names = []
        
    def extract_histogram_features(self, image):
        features = []
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        for i, channel_name in enumerate(['L', 'A', 'B']):
            channel = lab[:, :, i]
            hist, _ = np.histogram(channel, bins=32, range=(0, 256))
            hist = hist.astype(float) / hist.sum()
            
            mean = np.mean(channel)
            std = np.std(channel)
            skewness = np.mean(((channel - mean) / (std + 1e-7)) ** 3)
            kurtosis = np.mean(((channel - mean) / (std + 1e-7)) ** 4) - 3
            cv = std / (mean + 1e-7)
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            percentiles = np.percentile(channel, [10, 25, 50, 75, 90])
            
            features.extend([mean, std, skewness, kurtosis, cv, entropy])
            features.extend(percentiles.tolist())
            
            if not self.feature_names or len(self.feature_names) < 33:
                self.feature_names.extend([
                    f'{channel_name}_mean', f'{channel_name}_std',
                    f'{channel_name}_skewness', f'{channel_name}_kurtosis',
                    f'{channel_name}_cv', f'{channel_name}_entropy',
                    f'{channel_name}_p10', f'{channel_name}_p25',
                    f'{channel_name}_p50', f'{channel_name}_p75',
                    f'{channel_name}_p90'
                ])
        
        return features
    
    def extract_gradient_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        features = []
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)
        
        features.extend([
            np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag),
            np.percentile(grad_mag, 90),
            np.sum(grad_mag > np.mean(grad_mag)) / grad_mag.size,
            np.std(grad_dir)
        ])
        
        if 'grad_mean' not in self.feature_names:
            self.feature_names.extend([
                'grad_mean', 'grad_std', 'grad_max', 'grad_p90',
                'edge_density', 'grad_dir_std'
            ])
        
        return features
    
    def extract_local_contrast_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        features = []
        
        for window_size in [3, 7, 15]:
            kernel = np.ones((window_size, window_size)) / (window_size ** 2)
            local_mean = cv2.filter2D(gray, -1, kernel)
            local_contrast = np.abs(gray - local_mean)
            
            features.extend([
                np.mean(local_contrast), np.std(local_contrast), np.max(local_contrast)
            ])
            
            if f'contrast_{window_size}_mean' not in self.feature_names:
                self.feature_names.extend([
                    f'contrast_{window_size}_mean',
                    f'contrast_{window_size}_std',
                    f'contrast_{window_size}_max'
                ])
        
        return features
    
    def extract_illumination_uniformity_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        features = []
        
        h, w = gray.shape
        block_size = 16
        block_means = []
        block_stds = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_means.append(np.mean(block))
                block_stds.append(np.std(block))
        
        block_means = np.array(block_means)
        block_stds = np.array(block_stds)
        
        features.extend([
            np.std(block_means),
            np.max(block_means) - np.min(block_means),
            np.std(block_means) / (np.mean(block_means) + 1e-7),
            np.mean(block_stds),
            np.std(block_stds)
        ])
        
        if 'block_mean_std' not in self.feature_names:
            self.feature_names.extend([
                'block_mean_std', 'block_range', 'block_cv',
                'block_std_mean', 'block_std_std'
            ])
        
        return features
    
    def extract_saturation_intensity_features(self, image):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        height, width = image_gray.shape
        
        saturation_thresh = 250
        raw_saturation_region = cv2.threshold(image_gray, saturation_thresh, 255, cv2.THRESH_BINARY)[1]
        
        num_raw_saturation_regions, raw_saturation_regions, stats, _ = cv2.connectedComponentsWithStats(raw_saturation_region)
        
        if num_raw_saturation_regions > 1:
            area_raw_saturation_regions = stats[1:, 4]
            max_saturation_area = np.max(area_raw_saturation_regions)
        else:
            area_raw_saturation_regions = np.array([0])
            max_saturation_area = 0
        
        mean_intensity = np.mean(image_gray)
        
        hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        
        low_threshold = np.count_nonzero(v < 50)
        high_intensity_pixels = np.count_nonzero(image_gray > 200)
        total_pixels = height * width
        percent_low = (low_threshold / total_pixels) * 100
        percent_high = (high_intensity_pixels / total_pixels) * 100
        
        num_saturation_regions = max(0, num_raw_saturation_regions - 1)
        total_saturation_area = np.sum(area_raw_saturation_regions) if num_raw_saturation_regions > 1 else 0
        saturation_area_ratio = total_saturation_area / total_pixels
        
        intensity_std = np.std(image_gray)
        intensity_range = np.max(image_gray) - np.min(image_gray)
        intensity_iqr = np.percentile(image_gray, 75) - np.percentile(image_gray, 25)
        
        saturation_mean = np.mean(s)
        saturation_std = np.std(s)
        value_mean = np.mean(v)
        value_std = np.std(v)
        value_cv = value_std / (value_mean + 1e-7)
        
        features = [
            max_saturation_area, mean_intensity, percent_low, percent_high,
            num_saturation_regions, total_saturation_area, saturation_area_ratio,
            intensity_std, intensity_range, intensity_iqr,
            saturation_mean, saturation_std, value_mean, value_std, value_cv
        ]
        
        if 'max_saturation_area' not in self.feature_names:
            self.feature_names.extend([
                'max_saturation_area', 'mean_intensity', 'percent_low_illumination',
                'percent_high_illumination', 'num_saturation_regions', 'total_saturation_area',
                'saturation_area_ratio', 'intensity_std', 'intensity_range', 'intensity_iqr',
                'hsv_saturation_mean', 'hsv_saturation_std', 'hsv_value_mean', 
                'hsv_value_std', 'hsv_value_cv'
            ])
        
        return features
    
    def extract_all_features(self, image):
        features = []
        features.extend(self.extract_histogram_features(image))
        features.extend(self.extract_gradient_features(image))
        features.extend(self.extract_local_contrast_features(image))
        features.extend(self.extract_illumination_uniformity_features(image))
        features.extend(self.extract_saturation_intensity_features(image))
        return np.array(features)

class IlluminationDetector:
    """Illumination uniformity detection model"""
    
    def __init__(self, model_dir: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model_dir = model_dir
        
        self.cnn_extractor = LightweightDilatedCNN(feature_dim=256).to(self.device)
        cnn_checkpoint = torch.load(
            os.path.join(model_dir, 'lightweight_cnn_extractor.pth'), 
            map_location=self.device
        )
        self.cnn_extractor.load_state_dict(cnn_checkpoint['model_state_dict'])
        self.cnn_extractor.eval()
        
        self.classifier = joblib.load(os.path.join(model_dir, 'svm_classifier.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
        
        self.handcrafted_extractor = HandcraftedFeatureExtractor()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def predict(self, image_path: str) -> Dict:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(image_rgb, (128, 128))
            
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            image_tensor = self.transform(image_clahe).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cnn_features = self.cnn_extractor(image_tensor).cpu().numpy()
            
            handcrafted_features = self.handcrafted_extractor.extract_all_features(image_rgb)
            
            combined_features = np.hstack([cnn_features, handcrafted_features.reshape(1, -1)])
            
            features_scaled = self.scaler.transform(combined_features)
            prediction = self.classifier.predict(features_scaled)[0]
            
            return {
                'status': 'success',
                'lighting_correct': prediction == 0
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'lighting_correct': False
            }

class MobileViTReflectionDetector(nn.Module):
    """MobileViT model for corneal reflection detection"""
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        
        self.backbone = timm.create_model('mobilevit_s', pretrained=pretrained, num_classes=0)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        attention_weights = self.attention(features)
        features = features * attention_weights
        output = self.classifier(features)
        return output

class ReflectionDetector:
    """Corneal reflection detection model"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = MobileViTReflectionDetector(num_classes=2, pretrained=False)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str) -> Dict:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            pred_class = predicted.item()
            pred_confidence = confidence.item()
            
            # RELAXED LOGIC: Only reject if confident it's Bad (Reflection present)
            if pred_class == 1 and pred_confidence > 0.8:
                quality_state = QualityState.NO
            else:
                quality_state = QualityState.YES
            
            return {
                'status': 'success',
                'quality_state': quality_state.value,
                'confidence': pred_confidence
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'quality_state': QualityState.NO.value
            }

class CompletenessDetector:
    """Corneal completeness detection model"""
    
    def __init__(self, model_path: str, xgb_model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        self.resnet.load_state_dict(torch.load(model_path, map_location=device))
        self.resnet = self.resnet.to(self.device)
        self.resnet.eval()
        
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.xgb_classifier = xgb.XGBClassifier()
        self.xgb_classifier.load_model(xgb_model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str) -> Dict:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(image_tensor).view(1, -1).cpu().numpy()
            
            prediction = self.xgb_classifier.predict(features)[0]
            probabilities = self.xgb_classifier.predict_proba(features)[0]
            
            confidence = float(max(probabilities))
            
            # RELAXED LOGIC: Only reject if confident it's Bad (Incomplete)
            if prediction == 0 and confidence > 0.8:
                quality_state = QualityState.NO
            else:
                quality_state = QualityState.YES
            
            return {
                'status': 'success',
                'quality_state': quality_state.value,
                'confidence': confidence
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'quality_state': QualityState.NO.value
            }

class ResolutionDetector:
    """Image resolution quality detection model"""
    
    def __init__(self, model_path: str, xgb_model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.resnet.load_state_dict(torch.load(model_path, map_location=device))
        self.resnet = self.resnet.to(self.device)
        self.resnet.eval()
        
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.xgb_model = joblib.load(xgb_model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str) -> Dict:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(image_tensor).view(1, -1).cpu().numpy()
            
            prediction = self.xgb_model.predict(features)[0]
            
            if hasattr(self.xgb_model, 'predict_proba'):
                probabilities = self.xgb_model.predict_proba(features)[0]
                confidence = float(max(probabilities))
            else:
                pred_class = 1 if prediction >= 0.5 else 0
                confidence = float(abs(prediction - 0.5) * 2)
                prediction = pred_class
            
            # RELAXED LOGIC: Only reject if confident it's Bad (Low Res)
            if prediction == 0 and confidence > 0.8:
                quality_state = QualityState.NO
            else:
                quality_state = QualityState.YES
            
            return {
                'status': 'success',
                'quality_state': quality_state.value,
                'confidence': confidence
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'quality_state': QualityState.NO.value
            }


class EyeQualityAssessment:
    """Streamlined eye quality assessment pipeline"""
    
    QUALITY_MAPPING = {
        'PPPP': OverallQuality.BAD,
        'YPPP': OverallQuality.BAD,
        'PYPP': OverallQuality.BAD,
        'PPYP': OverallQuality.BAD,
        'PPPY': OverallQuality.BAD,
        'YYPP': OverallQuality.BAD,
        'PYPY': OverallQuality.BAD,
        'PPYY': OverallQuality.USABLE,
        'PYYY': OverallQuality.USABLE,
        'YYPY': OverallQuality.USABLE,
        'YYYP': OverallQuality.USABLE,
        'YPYY': OverallQuality.USABLE,
        'YPYP': OverallQuality.USABLE,
        'YPPY': OverallQuality.USABLE,
        'PYYP': OverallQuality.USABLE,
        'YNYY': OverallQuality.USABLE,
        'YYYN': OverallQuality.USABLE,
        'YNYP': OverallQuality.USABLE,
        'YPYN': OverallQuality.USABLE,
        'YNYN': OverallQuality.USABLE,
        'PPYN': OverallQuality.USABLE,
        'YYYY': OverallQuality.GOOD
    }
    
    def __init__(self):
        self.models_loaded = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_models(self, config: Dict):
        """Load all required models"""
        print("Loading models...")
        try:
            # Eye detector
            self.eye_detector = EyeDetector(
                model_dir=config['eye_model_dir'],
                use_ensemble=config['use_eye_ensemble'],
                device=self.device
            )
            
            # Focus detector
            self.focus_detector = FocusDetector(
                config['focus_model_path'],
                config['focus_scaler_path'],
                config['focus_feature_names_path']
            )
            
            # Illumination detector
            self.illumination_detector = IlluminationDetector(
                config['illumination_model_dir'],
                self.device
            )
            
            # Reflection detector
            self.reflection_detector = ReflectionDetector(
                config['reflection_model_path'],
                self.device
            )
            
            # Completeness detector
            self.completeness_detector = CompletenessDetector(
                config['completeness_model_path'],
                config['completeness_xgb_model_path'],
                self.device
            )
            
            # Resolution detector
            self.resolution_detector = ResolutionDetector(
                config['resolution_model_path'],
                config['resolution_xgb_model_path'],
                self.device
            )
            
            self.models_loaded = True
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
            raise
    
    def _determine_overall_quality(self, quality_results: Dict) -> Tuple[str, OverallQuality]:
        """Determine overall quality based on pattern"""
        pattern = ''
        pattern += quality_results.get('resolution', {}).get('quality_state', 'N')
        pattern += quality_results.get('completeness', {}).get('quality_state', 'N')
        pattern += quality_results.get('focus', {}).get('quality_state', 'N')
        pattern += quality_results.get('reflection', {}).get('quality_state', 'N')
        
        
        # LOGIC UPDATE (User Request): Focus (Blur) and Reflection (Glare) are CRITICAL.
        # If either is Bad ('N'), the image is Bad, regardless of the '2 Ys' count.
        focus_state = quality_results.get('focus', {}).get('quality_state', 'N')
        reflection_state = quality_results.get('reflection', {}).get('quality_state', 'N')

        if focus_state == 'N' or reflection_state == 'N':
            overall_quality = OverallQuality.BAD
        elif pattern.count('Y') >= 2:
            # If critical checks pass (Focus=Y, Reflection=Y), we basically have 2 Ys.
            # But checking >= 2 ensures we stick to the permissive logic for the other params if needed 
            # (though Focus=Y + Reflection=Y is already 2).
            overall_quality = OverallQuality.GOOD
        else:
            overall_quality = OverallQuality.BAD
            
        return pattern, overall_quality
    
    def _generate_recommendations(self, quality_results: Dict) -> List[str]:
        """Generate recommendations based on quality results"""
        recommendations = []
        
        if quality_results.get('resolution', {}).get('quality_state') != 'Y':
            recommendations.append("Use higher resolution camera or move closer")
            
        if quality_results.get('completeness', {}).get('quality_state') != 'Y':
            recommendations.append("Ensure entire cornea is visible in frame")
            
        if quality_results.get('focus', {}).get('quality_state') != 'Y':
            recommendations.append("Ensure camera is properly focused on the eye")
            
        if quality_results.get('reflection', {}).get('quality_state') != 'Y':
            recommendations.append("Adjust lighting angle to minimize reflections")
            
        return recommendations
    
    def mainqualitycheck(self, image_path: str) -> Dict:
        """Main function to assess image quality"""
        if not self.models_loaded:
            return {
                'status': 'error',
                'error': 'Models not loaded. Call load_models() first.',
                'overall_quality': None
            }
        
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'error': f'Image file not found: {image_path}',
                'overall_quality': None
            }
        
        start_time = time.time()
        
        # Initialize result
        result = {
            'status': 'success',
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'overall_quality': None,
            'quality_pattern': None,
            'recommendations': []
        }
        
        # Step 1: Eye Detection
        eye_result = self.eye_detector.predict(image_path)
        
        if not eye_result['has_eye']:
            result['overall_quality'] = 'Not Assessed - No Eye'
            result['recommendations'] = ['Ensure eye is visible and centered in the image']
            result['processing_time'] = time.time() - start_time
            return result
        
        # Step 2: Lighting Check
        lighting_result = self.illumination_detector.predict(image_path)
        
        if not lighting_result['lighting_correct']:
            # RELAXED LOGIC: Don't fail immediately on bad lighting. Continue to check other parameters.
            # result['overall_quality'] = 'Not Assessed - Bad Lighting'
            # result['recommendations'] = ['Ensure uniform lighting across the entire eye area']
            # result['processing_time'] = time.time() - start_time
            # return result
            
            # Just add the recommendation but continue
            result['recommendations'].append('Ensure uniform lighting across the entire eye area')
        
        # Step 3: Parallel Quality Assessment
        quality_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_metric = {
                executor.submit(self.resolution_detector.predict, image_path): 'resolution',
                executor.submit(self.completeness_detector.predict, image_path): 'completeness',
                executor.submit(self.focus_detector.predict, image_path): 'focus',
                executor.submit(self.reflection_detector.predict, image_path): 'reflection'
            }
            
            for future in as_completed(future_to_metric):
                metric = future_to_metric[future]
                try:
                    quality_results[metric] = future.result()
                except Exception as e:
                    quality_results[metric] = {
                        'status': 'error',
                        'quality_state': 'N'
                    }
        
        # Determine overall quality
        pattern, overall_quality = self._determine_overall_quality(quality_results)
        result['quality_pattern'] = pattern
        result['overall_quality'] = overall_quality.value
        
        # Generate recommendations
        result['recommendations'].extend(self._generate_recommendations(quality_results))
        
        result['processing_time'] = time.time() - start_time
        
        return result


def mainqualitycheck(image_path: str) -> Dict:
    """Main function to assess eye image quality"""
    
    # Configuration
    config = {
        'eye_model_dir': './models/peakmodels',
        'use_eye_ensemble': True,
        'focus_model_path': './models/focus_svm_model.joblib',
        'focus_scaler_path': './models/focus_scaler.joblib',
        'focus_feature_names_path': './models/focus_feature_names.txt',
        'illumination_model_dir': 'models',
        'reflection_model_path': './models/best_mobilevit_model.pth',
        'completeness_model_path': './models/resnet_completeness2.pth',
        'completeness_xgb_model_path': './models/xgboost_completeness2.json',
        'resolution_model_path': './models/resnet_resolution.pth',
        'resolution_xgb_model_path': './models/xgboost_resolution_model.pkl'
    }
    
    # Initialize assessment pipeline
    assessment = EyeQualityAssessment()
    
    # Load models
    try:
        assessment.load_models(config)
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Failed to load models: {str(e)}',
            'overall_quality': None
        }
    
    # Run quality check
    result = assessment.mainqualitycheck(image_path)
    
    # Print results
    print("\n" + "="*50)
    print("EYE IMAGE QUALITY ASSESSMENT RESULTS")
    print("="*50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Overall Quality: {result['overall_quality']}")
    
    if result['quality_pattern']:
        print(f"Quality Pattern: {result['quality_pattern']}")
    
    if result['recommendations']:
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  • {rec}")
    
    print(f"\nProcessing Time: {result.get('processing_time', 0):.3f}s")
    print("="*50)
    
    # Save results to JSON
    output_filename = f"quality_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_filename}")
    
    return result

# Example 
if __name__ == "__main__":
    image_path = r"C:\Users\satyam.tripathi\Downloads\02.49.36_07.JPG"
    result = mainqualitycheck(image_path)