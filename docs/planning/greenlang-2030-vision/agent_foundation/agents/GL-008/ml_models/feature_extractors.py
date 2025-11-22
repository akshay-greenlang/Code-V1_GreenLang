# -*- coding: utf-8 -*-
"""
Feature extraction utilities for GL-008 ML models.

This module provides deterministic feature extraction from acoustic signals
and thermal images for ML model training and inference.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import librosa
import cv2
from scipy import signal, stats
from scipy.fft import fft, fftfreq


class AcousticFeatureExtractor:
    """
    Extract features from acoustic signals for ML model input.

    Features extracted:
    - Spectral: centroid, rolloff, bandwidth, contrast
    - Temporal: zero-crossing rate, RMS energy
    - Frequency: FFT peaks, power spectral density
    - MFCCs: Mel-frequency cepstral coefficients
    """

    def __init__(self, sampling_rate: int = 250000):
        """
        Initialize acoustic feature extractor.

        Args:
            sampling_rate: Signal sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512

    def extract_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive feature set from acoustic signal.

        Args:
            signal_data: Time-domain audio signal array

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Temporal features
        features['rms_energy'] = float(np.sqrt(np.mean(signal_data ** 2)))
        features['zero_crossing_rate'] = self._calculate_zcr(signal_data)
        features['signal_variance'] = float(np.var(signal_data))
        features['signal_skewness'] = float(stats.skew(signal_data))
        features['signal_kurtosis'] = float(stats.kurtosis(signal_data))

        # Frequency domain features
        fft_features = self._extract_fft_features(signal_data)
        features.update(fft_features)

        # Spectral features (using librosa)
        spectral_features = self._extract_spectral_features(signal_data)
        features.update(spectral_features)

        # MFCCs
        mfcc_features = self._extract_mfcc_features(signal_data)
        features.update(mfcc_features)

        # Power spectral density features
        psd_features = self._extract_psd_features(signal_data)
        features.update(psd_features)

        return features

    def _calculate_zcr(self, signal_data: np.ndarray) -> float:
        """Calculate zero-crossing rate."""
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        return float(len(zero_crossings) / len(signal_data))

    def _extract_fft_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract FFT-based features."""
        # Compute FFT
        fft_values = fft(signal_data)
        fft_magnitude = np.abs(fft_values)
        fft_freqs = fftfreq(len(signal_data), 1/self.sampling_rate)

        # Positive frequencies only
        positive_freqs = fft_freqs[:len(fft_freqs)//2]
        positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]

        # Find peak frequency
        peak_idx = np.argmax(positive_magnitude)
        peak_freq = positive_freqs[peak_idx]
        peak_magnitude = positive_magnitude[peak_idx]

        # Frequency bands energy
        band_20_40khz = self._band_energy(positive_freqs, positive_magnitude, 20000, 40000)
        band_40_60khz = self._band_energy(positive_freqs, positive_magnitude, 40000, 60000)
        band_60_80khz = self._band_energy(positive_freqs, positive_magnitude, 60000, 80000)
        band_80_100khz = self._band_energy(positive_freqs, positive_magnitude, 80000, 100000)

        return {
            'fft_peak_frequency_hz': float(peak_freq),
            'fft_peak_magnitude': float(peak_magnitude),
            'fft_band_20_40khz_energy': band_20_40khz,
            'fft_band_40_60khz_energy': band_40_60khz,
            'fft_band_60_80khz_energy': band_60_80khz,
            'fft_band_80_100khz_energy': band_80_100khz,
            'fft_total_energy': float(np.sum(positive_magnitude ** 2))
        }

    def _band_energy(
        self, freqs: np.ndarray, magnitude: np.ndarray,
        low_hz: float, high_hz: float
    ) -> float:
        """Calculate energy in frequency band."""
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        band_magnitude = magnitude[band_mask]
        return float(np.sum(band_magnitude ** 2))

    def _extract_spectral_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract spectral features using librosa."""
        # Compute spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=signal_data, sr=self.sampling_rate, n_fft=self.n_fft
        )[0]

        # Compute spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=signal_data, sr=self.sampling_rate, n_fft=self.n_fft
        )[0]

        # Compute spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=signal_data, sr=self.sampling_rate, n_fft=self.n_fft
        )[0]

        # Compute spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=signal_data, sr=self.sampling_rate, n_fft=self.n_fft
        )

        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
            'spectral_contrast_mean': float(np.mean(spectral_contrast)),
            'spectral_contrast_std': float(np.std(spectral_contrast))
        }

    def _extract_mfcc_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract MFCC features."""
        mfccs = librosa.feature.mfcc(
            y=signal_data, sr=self.sampling_rate,
            n_mfcc=self.n_mfcc, n_fft=self.n_fft
        )

        features = {}
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))

        return features

    def _extract_psd_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract power spectral density features."""
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=1024)

        # Total power
        total_power = np.sum(psd)

        # Power in specific bands
        band_20_40khz_power = self._band_energy(freqs, psd, 20000, 40000)
        band_40_60khz_power = self._band_energy(freqs, psd, 40000, 60000)

        return {
            'psd_total_power': float(total_power),
            'psd_band_20_40khz_power': band_20_40khz_power,
            'psd_band_40_60khz_power': band_40_60khz_power,
            'psd_band_ratio': band_40_60khz_power / (band_20_40khz_power + 1e-10)
        }


class ThermalFeatureExtractor:
    """
    Extract features from thermal images for ML model input.

    Features extracted:
    - Statistical: mean, std, min, max, percentiles
    - Spatial: gradients, edges, hot/cold spots
    - Texture: contrast, homogeneity, energy
    - Temperature distribution: histogram features
    """

    def __init__(self):
        """Initialize thermal feature extractor."""
        self.image_size = (64, 64)  # Resize for CNN input

    def extract_features(self, thermal_image: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive feature set from thermal image.

        Args:
            thermal_image: 2D array of temperature values

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Statistical features
        stat_features = self._extract_statistical_features(thermal_image)
        features.update(stat_features)

        # Gradient features
        gradient_features = self._extract_gradient_features(thermal_image)
        features.update(gradient_features)

        # Edge features
        edge_features = self._extract_edge_features(thermal_image)
        features.update(edge_features)

        # Texture features
        texture_features = self._extract_texture_features(thermal_image)
        features.update(texture_features)

        # Temperature distribution features
        dist_features = self._extract_distribution_features(thermal_image)
        features.update(dist_features)

        # Hot/cold spot features
        spot_features = self._extract_spot_features(thermal_image)
        features.update(spot_features)

        return features

    def _extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features."""
        return {
            'temp_mean': float(np.mean(image)),
            'temp_std': float(np.std(image)),
            'temp_min': float(np.min(image)),
            'temp_max': float(np.max(image)),
            'temp_range': float(np.max(image) - np.min(image)),
            'temp_median': float(np.median(image)),
            'temp_p25': float(np.percentile(image, 25)),
            'temp_p75': float(np.percentile(image, 75)),
            'temp_iqr': float(np.percentile(image, 75) - np.percentile(image, 25)),
            'temp_skewness': float(stats.skew(image.flatten())),
            'temp_kurtosis': float(stats.kurtosis(image.flatten()))
        }

    def _extract_gradient_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract gradient-based features."""
        # Calculate gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return {
            'gradient_mean': float(np.mean(grad_magnitude)),
            'gradient_std': float(np.std(grad_magnitude)),
            'gradient_max': float(np.max(grad_magnitude)),
            'gradient_x_mean': float(np.mean(np.abs(grad_x))),
            'gradient_y_mean': float(np.mean(np.abs(grad_y)))
        }

    def _extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract edge detection features."""
        # Normalize image for edge detection
        image_norm = cv2.normalize(image, None, 0, 255, cv2.CV_8U)

        # Canny edge detection
        edges = cv2.Canny(image_norm, 100, 200)

        return {
            'edge_density': float(np.sum(edges > 0) / edges.size),
            'edge_count': float(np.sum(edges > 0))
        }

    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture features using GLCM."""
        # Normalize to 8-bit for GLCM
        image_norm = cv2.normalize(image, None, 0, 255, cv2.CV_8U)

        # Calculate GLCM (simplified)
        # For production, use skimage.feature.graycomatrix

        # Local binary patterns (simplified)
        contrast = float(np.std(image_norm))
        homogeneity = float(1.0 / (1.0 + contrast))

        return {
            'texture_contrast': contrast,
            'texture_homogeneity': homogeneity,
            'texture_energy': float(np.sum(image_norm ** 2))
        }

    def _extract_distribution_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract temperature distribution features."""
        # Histogram features
        hist, bin_edges = np.histogram(image.flatten(), bins=20)
        hist_normalized = hist / np.sum(hist)

        # Entropy
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))

        return {
            'histogram_entropy': float(entropy),
            'histogram_peak_value': float(bin_edges[np.argmax(hist)]),
            'histogram_peak_count': float(np.max(hist))
        }

    def _extract_spot_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract hot and cold spot features."""
        mean_temp = np.mean(image)
        std_temp = np.std(image)

        # Hot spots (> mean + 2*std)
        hot_threshold = mean_temp + 2 * std_temp
        hot_spots = image > hot_threshold
        hot_spot_count = np.sum(hot_spots)

        # Cold spots (< mean - 2*std)
        cold_threshold = mean_temp - 2 * std_temp
        cold_spots = image < cold_threshold
        cold_spot_count = np.sum(cold_spots)

        return {
            'hot_spot_count': float(hot_spot_count),
            'hot_spot_density': float(hot_spot_count / image.size),
            'hot_spot_max_temp': float(np.max(image[hot_spots])) if hot_spot_count > 0 else 0.0,
            'cold_spot_count': float(cold_spot_count),
            'cold_spot_density': float(cold_spot_count / image.size),
            'cold_spot_min_temp': float(np.min(image[cold_spots])) if cold_spot_count > 0 else 0.0
        }

    def prepare_cnn_input(self, thermal_image: np.ndarray) -> np.ndarray:
        """
        Prepare thermal image for CNN input.

        Args:
            thermal_image: Raw thermal image

        Returns:
            Normalized and resized image for CNN
        """
        # Resize to standard size
        resized = cv2.resize(thermal_image, self.image_size)

        # Normalize to [0, 1]
        normalized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized) + 1e-10)

        # Add channel dimension for CNN
        cnn_input = np.expand_dims(normalized, axis=-1)

        return cnn_input
