"""
PPG Logger with Diabetes Detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import serial
import serial.tools.list_ports
import threading
import time
import queue
from collections import deque
import os
import csv
import numpy as np
from scipy.io import savemat
from scipy.signal import firwin, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis
import joblib
import json

# Configuration class for easy parameter management
class Config:
    DEFAULT_BAUDRATE = 115200
    DEFAULT_BUFFER_MAXLEN = 60000
    MAX_RECORDING_SAMPLES = 1000000  # Prevent memory issues
    PLOT_SECONDS_WINDOW = 30
    GUI_UPDATE_INTERVAL_MS = 50
    SERIAL_TIMEOUT = 1.0
    RECONNECT_DELAY = 2.0
    CONFIG_FILE = "ppg_logger_config.json"
    
    @classmethod
    def load_config(cls):
        """Load configuration from file if exists"""
        try:
            if os.path.exists(cls.CONFIG_FILE):
                with open(cls.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    for key, value in config.items():
                        if hasattr(cls, key):
                            setattr(cls, key, value)
                return True
        except Exception as e:
            print(f"Could not load config: {e}")
        return False
    
    @classmethod
    def save_config(cls):
        """Save current configuration to file"""
        try:
            config = {
                'DEFAULT_BAUDRATE': cls.DEFAULT_BAUDRATE,
                'PLOT_SECONDS_WINDOW': cls.PLOT_SECONDS_WINDOW,
                'MAX_RECORDING_SAMPLES': cls.MAX_RECORDING_SAMPLES
            }
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Could not save config: {e}")
        return False


class InputValidator:
    """Centralized input validation"""
    
    @staticmethod
    def validate_float(value, min_val=None, max_val=None, name="Value"):
        """Validate float input with optional bounds"""
        try:
            val = float(value)
            if min_val is not None and val < min_val:
                raise ValueError(f"{name} must be >= {min_val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{name} must be <= {max_val}")
            return val, None
        except ValueError as e:
            return None, str(e)
    
    @staticmethod
    def validate_int(value, min_val=None, max_val=None, name="Value"):
        """Validate integer input with optional bounds"""
        try:
            val = int(value)
            if min_val is not None and val < min_val:
                raise ValueError(f"{name} must be >= {min_val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{name} must be <= {max_val}")
            return val, None
        except ValueError as e:
            return None, str(e)
    
    @staticmethod
    def validate_port(port_name):
        """Validate serial port name"""
        if not port_name or port_name.strip() == "":
            return None, "Port name cannot be empty"
        return port_name.strip(), None


class DiabetesAnalyzer:
    """Embedded diabetes detection analyzer for PPG analysis"""
    
    def __init__(self, model_path=None, Fs_default=200):
        self.model_path = model_path
        self.Fs_default = Fs_default
        self.pipeline = None
        self.is_model_loaded = False
        
        if model_path and os.path.exists(model_path):
            self.load_model()
    
    def load_model(self):
        """Load the trained pipeline model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.pipeline = joblib.load(self.model_path)
                self.is_model_loaded = True
                print(f"✓ ML Model loaded from: {self.model_path}")
                return True
            else:
                print("⚠ ML Model file not found")
                return False
        except Exception as e:
            print(f"✗ Error loading ML model: {e}")
            self.is_model_loaded = False
            return False
    
    def bandpass_fir(self, sig, Fs, low=0.5, high=15.0, numtaps=401):
        """Apply FIR bandpass filter to signal."""
        nyq = Fs / 2.0
        if high >= nyq:
            high = nyq * 0.99
        taps = firwin(numtaps, [low/nyq, high/nyq], pass_zero=False)
        return filtfilt(taps, 1.0, sig)
    
    def simple_sqi(self, sig, Fs):
        """Compute simple Signal Quality Index."""
        f, Pxx = welch(sig, fs=Fs, nperseg=min(len(sig), 2048))
        total_power = np.trapz(Pxx, f) if len(f) > 0 else 0.0
        band_idx = (f >= 0.5) & (f <= 15)
        band_power = np.trapz(Pxx[band_idx], f[band_idx]) if band_idx.any() else 0.0
        return 0.0 if total_power == 0 else (band_power / total_power)
    
    def detect_peaks(self, sig, Fs):
        """Detect systolic peaks in PPG signal."""
        distance = int(0.4 * Fs)
        prominence = (np.std(sig) * 0.1) if len(sig) > 0 else 0.0
        peaks, props = find_peaks(sig, distance=distance, prominence=prominence)
        return peaks, props
    
    def compute_ibis(self, peaks, Fs):
        """Compute inter-beat intervals from peak locations."""
        if len(peaks) < 2:
            return np.array([])
        times = peaks / Fs
        return np.diff(times)
    
    def extract_features_from_signal(self, sig, Fs):
        """Extract comprehensive features from PPG signal."""
        sig = np.asarray(sig, dtype=float).squeeze()
        if sig.ndim != 1:
            sig = sig.flatten()
        
        feats = {}
        
        # Basic statistical features
        feats['len'] = len(sig)
        feats['mean'] = np.mean(sig)
        feats['std'] = np.std(sig)
        feats['skew'] = float(skew(sig)) if len(sig) > 2 else 0.0
        feats['kurtosis'] = float(kurtosis(sig)) if len(sig) > 2 else 0.0
        feats['mad'] = np.mean(np.abs(sig - np.mean(sig)))
        feats['rms'] = np.sqrt(np.mean(sig**2))
        feats['ae'] = np.sum(np.abs(sig - np.mean(sig)))
        
        # Filter signal for morphological analysis
        try:
            sig_f = self.bandpass_fir(sig, Fs)
        except:
            sig_f = sig
        
        # Signal quality
        feats['sqi'] = self.simple_sqi(sig_f, Fs)
        
        # Peak detection and heart rate variability features
        peaks, props = self.detect_peaks(sig_f, Fs)
        feats['n_peaks'] = len(peaks)
        
        ibis = self.compute_ibis(peaks, Fs)
        if len(ibis) > 0:
            feats['ibi_mean'] = np.mean(ibis)
            feats['ibi_sdnn'] = np.std(ibis)
            diffs = np.diff(ibis)
            feats['ibi_rmssd'] = np.sqrt(np.mean(diffs**2)) if len(diffs) > 0 else 0.0
            feats['pnn50'] = np.sum(np.abs(diffs) > 0.05) / len(diffs) if len(diffs) > 0 else 0.0
        else:
            feats['ibi_mean'] = np.nan
            feats['ibi_sdnn'] = np.nan
            feats['ibi_rmssd'] = np.nan
            feats['pnn50'] = np.nan
        
        # Morphological features
        amps, rise_times, decay_times, widths = [], [], [], []
        for i in range(len(peaks)):
            pk = peaks[i]
            left_search_start = max(0, pk - int(0.6*Fs))
            foot_idx = left_search_start + np.argmin(sig_f[left_search_start:pk+1]) if pk >= left_search_start else pk
            amp = sig_f[pk] - sig_f[foot_idx] if pk >= foot_idx else 0
            amps.append(amp)
            rt = (pk - foot_idx)/Fs if pk >= foot_idx else 0
            rise_times.append(rt)
            right_search_end = min(len(sig_f)-1, pk + int(0.6*Fs))
            end_idx = pk + np.argmin(sig_f[pk:right_search_end+1]) if right_search_end > pk else pk
            dt = (end_idx - pk)/Fs if end_idx >= pk else 0
            decay_times.append(dt)
            widths.append((end_idx - foot_idx)/Fs if end_idx >= foot_idx else 0)
        
        feats['amp_mean'] = np.mean(amps) if len(amps) > 0 else np.nan
        feats['amp_std'] = np.std(amps) if len(amps) > 0 else np.nan
        feats['rise_mean'] = np.mean(rise_times) if len(rise_times) > 0 else np.nan
        feats['decay_mean'] = np.mean(decay_times) if len(decay_times) > 0 else np.nan
        feats['width_mean'] = np.mean(widths) if len(widths) > 0 else np.nan
        
        # Spectral features
        try:
            f, Pxx = welch(sig_f, fs=Fs, nperseg=min(1024, len(sig_f)))
            lf_idx = (f >= 0.04) & (f <= 0.15)
            hf_idx = (f > 0.15) & (f <= 0.4)
            lf_power = np.trapz(Pxx[lf_idx], f[lf_idx]) if lf_idx.any() else 0.0
            hf_power = np.trapz(Pxx[hf_idx], f[hf_idx]) if hf_idx.any() else 0.0
            feats['lf_power'] = lf_power
            feats['hf_power'] = hf_power
            feats['lf_hf'] = (lf_power / hf_power) if hf_power > 0 else np.nan
        except:
            feats['lf_power'] = np.nan
            feats['hf_power'] = np.nan
            feats['lf_hf'] = np.nan
        
        # Sample entropy (if pyentrp is available)
        try:
            from pyentrp import entropy as ent
            feats['sampen'] = ent.sample_entropy(sig_f.tolist(), 2, 0.2*np.std(sig_f))[0]
        except:
            feats['sampen'] = np.nan
        
        return feats
    
    def analyze_signal(self, signal_data, Fs, metadata=None):
        """Analyze signal for diabetes risk with flexible duration requirements."""
        if not self.is_model_loaded:
            return {"error": "ML model not loaded"}
        
        try:
            # More flexible minimum duration requirement
            min_duration_seconds = 5.0
            min_length = int(Fs * min_duration_seconds)
            
            actual_duration = len(signal_data) / Fs
            
            if len(signal_data) < min_length:
                if actual_duration >= 3.0:
                    print(f"WARNING: Using {actual_duration:.1f}s of data (less than ideal {min_duration_seconds}s)")
                else:
                    return {"error": f"Insufficient data for analysis (got {len(signal_data)} samples = {actual_duration:.1f}s, need ≥{min_duration_seconds}s for reliable analysis)"}
            
            # Extract features
            feats = self.extract_features_from_signal(signal_data, Fs)
            
            # Add metadata
            if metadata is None:
                metadata = {}
            
            # Encode gender (0=Male, 1=Female)
            gender = metadata.get("Gender", "Male")
            feats["Gender"] = 0 if str(gender).lower() == "male" else 1
            
            # Add other metadata with defaults
            feats["Age"] = metadata.get("Age", np.nan)
            feats["Height"] = metadata.get("Height", np.nan)
            feats["Weight"] = metadata.get("Weight", np.nan)
            
            # Prepare feature vector
            import pandas as pd
            X_row = pd.DataFrame([feats])
            X_row = X_row.fillna(X_row.median(axis=0))
            
            # Make prediction
            prob = self.pipeline.predict_proba(X_row.values)[:, 1][0]
            pred = int(prob >= 0.5)
            
            # Confidence adjustment for shorter signals
            confidence_factor = min(1.0, actual_duration / 10.0)
            confidence_factor = max(0.5, confidence_factor)
            
            # Risk interpretation
            if prob < 0.3:
                risk_level = "Low"
                interpretation = "Low diabetes risk - glucose levels likely normal"
            elif prob < 0.7:
                risk_level = "Moderate"
                interpretation = "Moderate diabetes risk - recommend medical evaluation"
            else:
                risk_level = "High"
                interpretation = "High diabetes risk - strongly recommend immediate medical consultation"
            
            # Add confidence warning for short signals
            if confidence_factor < 0.8:
                interpretation += f" (Analysis confidence: {confidence_factor*100:.0f}% due to {actual_duration:.1f}s signal duration)"
            
            result = {
                "probability": float(prob),
                "prediction": pred,
                "risk_level": risk_level,
                "interpretation": interpretation,
                "signal_quality": feats.get('sqi', 0),
                "peaks_detected": feats.get('n_peaks', 0),
                "heart_rate": 60.0 / feats['ibi_mean'] if feats.get('ibi_mean') and feats['ibi_mean'] > 0 else np.nan,
                "signal_duration": actual_duration,
                "sampling_rate": Fs,
                "confidence_factor": confidence_factor
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}


class PPGLoggerFixed:
    def __init__(self, root):
        self.root = root
        self.root.title("PPG Logger with Diabetes Detection - Fixed Version")
        self.root.geometry("1200x900")

        # Load configuration
        Config.load_config()

        # Thread-safe locks
        self.data_lock = threading.Lock()
        self.recording_lock = threading.Lock()
        
        # Serial & state
        self.ser = None
        self.is_connected = False
        self.connection_lost = False
        self.data_q = queue.Queue()
        self.read_thread = None
        self.stop_read_thread = threading.Event()
        self.reconnect_thread = None
        self.stop_reconnect = threading.Event()

        # Dark mode state
        self.dark_mode = tk.BooleanVar(value=False)

        # Thread-safe buffers with locks
        self.time_buf = deque(maxlen=Config.DEFAULT_BUFFER_MAXLEN)
        self.ppg_buf = deque(maxlen=Config.DEFAULT_BUFFER_MAXLEN)
        self.recorded = []
        
        # Cached plot data to avoid recomputing
        self.cached_plot_data = None
        self.cache_valid = False

        # Recording control
        self.is_recording = False
        self.record_start_timestamp = None
        self.record_target_duration = 0
        self.recording_thread = None
        self.stop_recording_thread = threading.Event()

        # Baseline for normalized mode
        self.running_baseline_window = deque(maxlen=200)
        self.baseline = None

        # ML Analysis
        self.ml_analyzer = DiabetesAnalyzer()

        # UI variables with validation
        self.port_var = tk.StringVar()
        self.expected_input_rate_var = tk.StringVar(value="200")
        self.duration_var = tk.StringVar(value="30")
        self.finger_threshold_var = tk.StringVar(value="5000")
        self.zero_on_no_finger_var = tk.BooleanVar(value=True)
        self.display_mode_var = tk.StringVar(value="Scaled")
        self.scale_factor_var = tk.StringVar(value="1000")
        self.time_display_var = tk.StringVar(value="Rolling")
        
        # Patient metadata variables
        self.age_var = tk.StringVar(value="35")
        self.gender_var = tk.StringVar(value="Male")
        self.height_var = tk.StringVar(value="170")
        self.weight_var = tk.StringVar(value="70")
        self.model_path_var = tk.StringVar()
        self.model_status_var = tk.StringVar(value="No model loaded")
        
        # Start time for absolute time display
        self.start_time = None
        
        # Navigation and display control
        self.is_live_view = True
        self.nav_position = 1.0

        # Build UI
        self._build_ui()
        self._build_plot()
        
        # Auto-load model
        self._auto_load_model()
        
        self.refresh_ports()
        
        # Apply initial theme
        self._apply_theme()
        
        # Start GUI update loop
        self.root.after(Config.GUI_UPDATE_INTERVAL_MS, self._gui_update)

    def _apply_theme(self):
        """Apply light or dark theme based on current mode"""
        if self.dark_mode.get():
            bg_color = "#2b2b2b"
            fg_color = "#ffffff"
            entry_bg = "#404040"
            entry_fg = "#ffffff"
            button_bg = "#404040"
            button_fg = "#ffffff"
            frame_bg = "#333333"
            label_frame_bg = "#404040"
            
            import matplotlib.pyplot as plt
            plt.style.use('dark_background')
            self.fig.patch.set_facecolor('#2b2b2b')
            self.ax.set_facecolor('#1e1e1e')
            self.ax.tick_params(colors='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.title.set_color('white')
            self.ax.grid(True, alpha=0.3, color='gray')
            self.line.set_color('#00ff00')
        else:
            bg_color = "#f0f0f0"
            fg_color = "#000000"
            entry_bg = "#ffffff"
            entry_fg = "#000000"
            button_bg = "#e0e0e0"
            button_fg = "#000000"
            frame_bg = "#f4f4f4"
            label_frame_bg = "#e8f4f8"
            
            import matplotlib.pyplot as plt
            plt.style.use('default')
            self.fig.patch.set_facecolor('white')
            self.ax.set_facecolor('white')
            self.ax.tick_params(colors='black')
            self.ax.xaxis.label.set_color('black')
            self.ax.yaxis.label.set_color('black')
            self.ax.title.set_color('black')
            self.ax.grid(True, alpha=0.3, color='gray')
            self.line.set_color('blue')

        self.root.configure(bg=bg_color)
        self._update_widget_theme(self.root, bg_color, fg_color, entry_bg, entry_fg, 
                                button_bg, button_fg, frame_bg, label_frame_bg)
        self.canvas.draw_idle()

    def _update_widget_theme(self, widget, bg_color, fg_color, entry_bg, entry_fg, 
                           button_bg, button_fg, frame_bg, label_frame_bg):
        """Recursively update theme for all widgets"""
        widget_class = widget.winfo_class()
        
        try:
            if widget_class == 'Frame':
                widget.configure(bg=frame_bg)
            elif widget_class == 'Labelframe':
                widget.configure(bg=label_frame_bg, fg=fg_color)
            elif widget_class == 'Label':
                widget.configure(bg=frame_bg, fg=fg_color)
            elif widget_class == 'Button':
                current_bg = widget.cget('bg')
                if current_bg in ['#9dd3ff', '#8cf58c', '#ff8c8c', '#87CEEB', '#FFB347']:
                    widget.configure(fg='black')
                else:
                    widget.configure(bg=button_bg, fg=button_fg)
            elif widget_class == 'Entry':
                widget.configure(bg=entry_bg, fg=entry_fg, insertbackground=entry_fg)
            elif widget_class == 'Checkbutton':
                widget.configure(bg=frame_bg, fg=fg_color, selectcolor=entry_bg)
        except tk.TclError:
            pass
        
        for child in widget.winfo_children():
            self._update_widget_theme(child, bg_color, fg_color, entry_bg, entry_fg,
                                    button_bg, button_fg, frame_bg, label_frame_bg)

    def toggle_theme(self):
        """Toggle between light and dark theme"""
        self.dark_mode.set(not self.dark_mode.get())
        self._apply_theme()

    def _auto_load_model(self):
        """Automatically search for and load ML model"""
        search_locations = [
            "final_pipeline_xgb.joblib",
            "model.joblib",
            "diabetes_model.joblib",
            "models/final_pipeline_xgb.joblib",
            "models/model.joblib",
            "models/diabetes_model.joblib",
            "./final_pipeline_xgb.joblib",
            "../models/final_pipeline_xgb.joblib"
        ]
        
        for model_path in search_locations:
            if os.path.exists(model_path):
                self.ml_analyzer.model_path = model_path
                if self.ml_analyzer.load_model():
                    self.model_path_var.set(model_path)
                    self.model_status_var.set("Model auto-loaded ✓")
                    return True
                    
        self.model_status_var.set("No model found - please browse manually")
        return False

    def calculate_actual_sampling_rate(self):
        """Calculate actual sampling rate from recorded data."""
        with self.recording_lock:
            if len(self.recorded) < 10:
                return None
            
            timestamps = [row[0] for row in self.recorded]
            duration = timestamps[-1] - timestamps[0]
            
            if duration > 0:
                return (len(self.recorded) - 1) / duration
        
        return None

    def _build_ui(self):
        # Theme toggle
        theme_frame = tk.Frame(self.root)
        theme_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        
        theme_button = tk.Button(theme_frame, text="🌙 Dark Mode", command=self.toggle_theme,
                               font=("Arial", 9), width=12)
        theme_button.pack(side=tk.RIGHT, padx=4)
        
        # STEP 1: Patient Information
        patient_frame = tk.LabelFrame(self.root, text="Step 1: Patient Information", 
                                     padx=10, pady=8, font=("Arial", 10, "bold"))
        patient_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8,4))

        row1 = tk.Frame(patient_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1, text="Age:").pack(side=tk.LEFT, padx=(0,4))
        ttk.Entry(row1, textvariable=self.age_var, width=6).pack(side=tk.LEFT, padx=(0,12))
        
        ttk.Label(row1, text="Gender:").pack(side=tk.LEFT, padx=(0,4))
        gender_combo = ttk.Combobox(row1, textvariable=self.gender_var, width=8, state="readonly")
        gender_combo['values'] = ("Male", "Female")
        gender_combo.pack(side=tk.LEFT, padx=(0,12))
        
        ttk.Label(row1, text="Height (cm):").pack(side=tk.LEFT, padx=(0,4))
        ttk.Entry(row1, textvariable=self.height_var, width=6).pack(side=tk.LEFT, padx=(0,12))
        
        ttk.Label(row1, text="Weight (kg):").pack(side=tk.LEFT, padx=(0,4))
        ttk.Entry(row1, textvariable=self.weight_var, width=6).pack(side=tk.LEFT, padx=(0,12))

        row2 = tk.Frame(patient_frame)
        row2.pack(fill=tk.X, pady=4)
        
        ttk.Label(row2, text="ML Model:").pack(side=tk.LEFT, padx=(0,4))
        ttk.Entry(row2, textvariable=self.model_path_var, width=50).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(row2, text="Browse", command=self.browse_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=4)
        
        ttk.Label(row2, textvariable=self.model_status_var, foreground="red").pack(side=tk.LEFT, padx=12)

        # STEP 2: Connection and Recording
        control_frame = tk.LabelFrame(self.root, text="Step 2: Connect Device and Record Data", 
                                     padx=10, pady=8, font=("Arial", 10, "bold"))
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        conn_row = tk.Frame(control_frame)
        conn_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(conn_row, text="Port:").pack(side=tk.LEFT, padx=(0,4))
        self.port_combo = ttk.Combobox(conn_row, textvariable=self.port_var, width=12)
        self.port_combo.pack(side=tk.LEFT, padx=(0,6))

        ttk.Button(conn_row, text="Refresh Ports", command=self.refresh_ports).pack(side=tk.LEFT, padx=4)
        self.connect_btn = tk.Button(conn_row, text="Connect Device", command=self.toggle_connection, 
                                    bg="#9dd3ff", width=12, font=("Arial", 9, "bold"))
        self.connect_btn.pack(side=tk.LEFT, padx=8)

        ttk.Label(conn_row, text="Expected Rate:").pack(side=tk.LEFT, padx=(12,4))
        ttk.Entry(conn_row, textvariable=self.expected_input_rate_var, width=6).pack(side=tk.LEFT)
        ttk.Label(conn_row, text="Hz").pack(side=tk.LEFT, padx=(2,12))

        rec_row = tk.Frame(control_frame)
        rec_row.pack(fill=tk.X, pady=4)
        
        ttk.Label(rec_row, text="Recording Duration:").pack(side=tk.LEFT, padx=(0,4))
        ttk.Entry(rec_row, textvariable=self.duration_var, width=6).pack(side=tk.LEFT)
        ttk.Label(rec_row, text="seconds").pack(side=tk.LEFT, padx=(2,12))
        
        ttk.Label(rec_row, text="Finger Detection:").pack(side=tk.LEFT, padx=(0,4))
        ttk.Entry(rec_row, textvariable=self.finger_threshold_var, width=8).pack(side=tk.LEFT)
        ttk.Checkbutton(rec_row, text="Zero if below threshold", variable=self.zero_on_no_finger_var).pack(side=tk.LEFT, padx=(4,12))

        self.rec_btn = tk.Button(rec_row, text="Start Recording", command=self.toggle_recording, 
                                state=tk.DISABLED, bg="#8cf58c", width=15, font=("Arial", 9, "bold"))
        self.rec_btn.pack(side=tk.LEFT, padx=12)

        self.status_var = tk.StringVar(value="Disconnected")
        status_label = ttk.Label(rec_row, textvariable=self.status_var, font=("Arial", 9, "bold"))
        status_label.pack(side=tk.RIGHT, padx=12)

        # STEP 3: After Recording Actions
        self.action_frame = tk.LabelFrame(self.root, text="Step 3: After Recording - Choose Action", 
                                         padx=10, pady=8, font=("Arial", 10, "bold"))
        self.action_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        action_row = tk.Frame(self.action_frame)
        action_row.pack(fill=tk.X, pady=4)

        self.save_btn = tk.Button(action_row, text="Save Data", command=self.save_recording, 
                                 bg="#87CEEB", width=15, height=2, font=("Arial", 11, "bold"),
                                 state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=12)

        self.analyze_btn = tk.Button(action_row, text="Check Diabetes", command=self.check_diabetes, 
                                    bg="#FFB347", width=15, height=2, font=("Arial", 11, "bold"),
                                    state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=12)

        self.recording_status_var = tk.StringVar(value="No recording available")
        ttk.Label(action_row, textvariable=self.recording_status_var, font=("Arial", 9)).pack(side=tk.LEFT, padx=24)

        self.results_frame = tk.Frame(self.action_frame)
        self.results_frame.pack(fill=tk.X, pady=8)

        # Display controls
        disp_frame = tk.Frame(self.root, padx=8, pady=4)
        disp_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(disp_frame, text="Display Mode:").pack(side=tk.LEFT, padx=(0,4))
        disp_combo = ttk.Combobox(disp_frame, textvariable=self.display_mode_var, width=10, state="readonly")
        disp_combo['values'] = ("Raw", "Scaled", "Normalized")
        disp_combo.pack(side=tk.LEFT, padx=(0,8))
        disp_combo.bind('<<ComboboxSelected>>', lambda e: self._invalidate_cache())
        
        ttk.Label(disp_frame, text="Scale Factor:").pack(side=tk.LEFT, padx=(0,2))
        scale_entry = ttk.Entry(disp_frame, textvariable=self.scale_factor_var, width=6)
        scale_entry.pack(side=tk.LEFT, padx=(0,12))
        scale_entry.bind('<Return>', lambda e: self._invalidate_cache())
        
        ttk.Label(disp_frame, text="Time Display:").pack(side=tk.LEFT, padx=(0,4))
        time_combo = ttk.Combobox(disp_frame, textvariable=self.time_display_var, width=8, state="readonly")
        time_combo['values'] = ("Rolling", "Absolute")
        time_combo.pack(side=tk.LEFT, padx=(0,12))
        time_combo.bind('<<ComboboxSelected>>', lambda e: self._invalidate_cache())
        
        ttk.Button(disp_frame, text="Clear Graph", command=self.clear_graph).pack(side=tk.LEFT, padx=8)

        # Navigation controls
        nav_frame = tk.Frame(self.root, padx=8, pady=4)
        nav_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(nav_frame, text="Navigate:").pack(side=tk.LEFT, padx=(2,6))
        self.nav_slider_var = tk.DoubleVar(value=1.0)
        self.nav_slider = ttk.Scale(nav_frame, from_=0.0, to=1.0, variable=self.nav_slider_var, 
                                   orient=tk.HORIZONTAL, length=300, command=self._on_slider_change)
        self.nav_slider.pack(side=tk.LEFT, padx=6)
        
        self.nav_info_var = tk.StringVar(value="Live")
        ttk.Label(nav_frame, textvariable=self.nav_info_var).pack(side=tk.LEFT, padx=6)
        
        ttk.Button(nav_frame, text="Go Live", command=self._go_live).pack(side=tk.LEFT, padx=6)

        self.auto_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(nav_frame, text="Auto Y-Scale", variable=self.auto_scale_var).pack(side=tk.LEFT, padx=(12,6))
        
        ttk.Label(nav_frame, text="Y-Range:").pack(side=tk.LEFT, padx=6)
        self.y_min_var = tk.StringVar(value="-1000")
        self.y_max_var = tk.StringVar(value="6000")
        ttk.Entry(nav_frame, textvariable=self.y_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(nav_frame, text="to").pack(side=tk.LEFT, padx=2)
        ttk.Entry(nav_frame, textvariable=self.y_max_var, width=8).pack(side=tk.LEFT, padx=2)

    def _build_plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(12,6))
        self.ax.set_title("PPG Real-time Signal")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Signal Amplitude")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], linewidth=1, color='blue')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _invalidate_cache(self):
        """Invalidate plot data cache when settings change"""
        self.cache_valid = False

    def browse_model(self):
        """Browse for ML model file."""
        filename = filedialog.askopenfilename(
            title="Select Trained ML Model File",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)

    def load_model(self):
        """Load ML model with validation."""
        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showerror("Error", "Please select model file first.")
            return
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file does not exist.")
            return
            
        self.ml_analyzer.model_path = model_path
        if self.ml_analyzer.load_model():
            self.model_status_var.set("Model loaded ✓")
            messagebox.showinfo("Success", "ML model loaded successfully!")
        else:
            self.model_status_var.set("Model loading failed")
            messagebox.showerror("Error", "Failed to load ML model.")

    def get_metadata(self):
        """Get patient metadata from UI with validation."""
        try:
            age_val, age_err = InputValidator.validate_float(self.age_var.get(), 0, 150, "Age")
            height_val, height_err = InputValidator.validate_float(self.height_var.get(), 50, 300, "Height")
            weight_val, weight_err = InputValidator.validate_float(self.weight_var.get(), 20, 300, "Weight")
            
            errors = []
            if age_err:
                errors.append(age_err)
            if height_err:
                errors.append(height_err)
            if weight_err:
                errors.append(weight_err)
            
            if errors:
                messagebox.showerror("Invalid Input", "\n".join(errors))
                return None
            
            metadata = {
                "Age": age_val if age_val is not None else np.nan,
                "Gender": self.gender_var.get(),
                "Height": height_val if height_val is not None else np.nan,
                "Weight": weight_val if weight_val is not None else np.nan
            }
            return metadata
        except Exception as e:
            messagebox.showerror("Validation Error", f"Error validating metadata: {str(e)}")
            return None

    def refresh_ports(self):
        """Refresh available serial ports."""
        try:
            ports = serial.tools.list_ports.comports()
            names = [p.device for p in ports]
            self.port_combo['values'] = names
            if names and not self.port_var.get():
                self.port_combo.current(0)
            print(f"Available ports: {names}")
        except Exception as e:
            print(f"Error refreshing ports: {e}")
            messagebox.showerror("Port Error", f"Failed to refresh ports: {str(e)}")

    def toggle_connection(self):
        """Toggle serial connection with validation."""
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        """Connect to serial port with validation."""
        port, port_err = InputValidator.validate_port(self.port_var.get())
        if port_err:
            messagebox.showerror("Port Error", port_err)
            return
        
        try:
            self.ser = serial.Serial(port, Config.DEFAULT_BAUDRATE, timeout=Config.SERIAL_TIMEOUT)
            self.is_connected = True
            self.connection_lost = False
            self.connect_btn.config(text="Disconnect", bg="#ffb86b")
            self.status_var.set(f"Connected to {port}")
            self.rec_btn.config(state=tk.NORMAL)
            
            # Start serial reader thread
            self.stop_read_thread.clear()
            self.read_thread = threading.Thread(target=self._serial_reader, daemon=True)
            self.read_thread.start()
            
            print(f"✓ Connected to {port}")
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to {port}:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"Unexpected error during connection:\n{str(e)}")

    def disconnect(self):
        """Disconnect from serial port safely."""
        # Stop threads first
        self.stop_read_thread.set()
        self.stop_reconnect.set()
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
        
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            self.reconnect_thread.join(timeout=1.0)
        
        # Close serial port
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception as e:
            print(f"Error closing serial port: {e}")
        
        self.is_connected = False
        self.connection_lost = False
        self.connect_btn.config(text="Connect Device", bg="#9dd3ff")
        self.status_var.set("Disconnected")
        self.rec_btn.config(state=tk.DISABLED)
        print("✓ Disconnected from device")

    def _attempt_reconnect(self):
        """Attempt to reconnect to serial port in background."""
        port = self.port_var.get()
        while not self.stop_reconnect.is_set():
            try:
                time.sleep(Config.RECONNECT_DELAY)
                if self.stop_reconnect.is_set():
                    break
                
                print(f"Attempting to reconnect to {port}...")
                test_ser = serial.Serial(port, Config.DEFAULT_BAUDRATE, timeout=Config.SERIAL_TIMEOUT)
                
                # Successfully reconnected
                with self.data_lock:
                    self.ser = test_ser
                    self.connection_lost = False
                    self.is_connected = True
                
                # Restart reader thread
                self.stop_read_thread.clear()
                self.read_thread = threading.Thread(target=self._serial_reader, daemon=True)
                self.read_thread.start()
                
                self.root.after(0, lambda: self.status_var.set(f"Reconnected to {port}"))
                print(f"✓ Reconnected to {port}")
                break
                
            except Exception as e:
                if not self.stop_reconnect.is_set():
                    print(f"Reconnection attempt failed: {e}")
                continue

    def _serial_reader(self):
        """Serial reader thread with improved error handling."""
        print("✓ Serial reader thread started")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self.stop_read_thread.is_set() and self.ser and self.ser.is_open:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
                try:
                    line = raw.decode('utf-8', errors='ignore').strip()
                except Exception:
                    continue
                
                if not line:
                    continue
                
                # Parse data
                parts = [p.strip() for p in line.split(',') if p.strip()]
                timestamp_ms = None
                ppg = None
                
                if len(parts) >= 2:
                    try:
                        timestamp_ms = int(float(parts[0]))
                        ppg = float(parts[1])
                    except Exception:
                        try:
                            ppg = float(parts[-1])
                            timestamp_ms = int(time.time() * 1000)
                        except Exception:
                            continue
                else:
                    try:
                        ppg = float(parts[0])
                        timestamp_ms = int(time.time() * 1000)
                    except Exception:
                        continue

                # Thread-safe data queueing
                self.data_q.put((timestamp_ms, ppg))
                
                # If recording, capture directly in serial thread for ALL samples
                if self.is_recording:
                    self._record_sample(timestamp_ms, ppg)
                
            except serial.SerialException as e:
                consecutive_errors += 1
                print(f"Serial read error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("⚠ Connection lost - attempting reconnect...")
                    self.connection_lost = True
                    self.root.after(0, lambda: self.status_var.set("Connection lost - reconnecting..."))
                    
                    # Start reconnection thread
                    self.stop_reconnect.clear()
                    self.reconnect_thread = threading.Thread(target=self._attempt_reconnect, daemon=True)
                    self.reconnect_thread.start()
                    break
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Unexpected serial error: {e}")
                break
        
        print("✓ Serial reader thread ended")

    def _record_sample(self, timestamp_ms, ppg):
        """Record a single sample - called from serial thread."""
        try:
            timestamp_s = timestamp_ms / 1000.0
            
            # Apply finger threshold
            thr_val, _ = InputValidator.validate_float(self.finger_threshold_var.get(), 0, 100000)
            thr = thr_val if thr_val is not None else 5000.0
            
            if self.zero_on_no_finger_var.get() and ppg < thr:
                ppg_display = 0.0
            else:
                ppg_display = float(ppg)
            
            raw_ppg = float(ppg)
            
            # Scale factor
            scale_val, _ = InputValidator.validate_float(self.scale_factor_var.get(), 0.001, 1000000)
            scale = scale_val if scale_val is not None and scale_val != 0 else 1000.0
            scaled_ppg = raw_ppg / scale
            
            # Normalized (simplified for recording thread)
            normalized_ppg = scaled_ppg
            
            with self.recording_lock:
                if not self.is_recording:
                    return
                
                # Set start timestamp on first sample
                if self.record_start_timestamp is None:
                    self.record_start_timestamp = timestamp_s
                    print(f"✓ Recording started at timestamp: {timestamp_s:.3f}s")
                
                # Check memory limit
                if len(self.recorded) >= Config.MAX_RECORDING_SAMPLES:
                    print(f"⚠ Maximum recording length reached ({Config.MAX_RECORDING_SAMPLES} samples)")
                    self.root.after(0, self.stop_recording)
                    return
                
                # Record sample
                self.recorded.append((timestamp_s, raw_ppg, scaled_ppg, normalized_ppg))
                
                # Check if duration reached
                elapsed = timestamp_s - self.record_start_timestamp
                if elapsed >= self.record_target_duration:
                    print(f"✓ Recording duration reached: {elapsed:.2f}s")
                    self.root.after(0, self.stop_recording)
                    
        except Exception as e:
            print(f"Error recording sample: {e}")

    def toggle_recording(self):
        """Toggle recording state."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start recording with validation."""
        # Validate metadata first
        if self.get_metadata() is None:
            return
        
        # Validate duration
        dur, dur_err = InputValidator.validate_float(self.duration_var.get(), 0.1, 3600, "Duration")
        if dur_err:
            messagebox.showerror("Invalid Duration", dur_err)
            return
        
        print(f"\n=== STARTING RECORDING ===")
        print(f"Target duration: {dur} seconds")
        
        # Clear previous recording
        with self.recording_lock:
            self.recorded = []
            self.is_recording = True
            self.record_start_timestamp = None
            self.record_target_duration = dur
        
        # Update UI
        self.rec_btn.config(text="Stop Recording", bg="#ff8c8c")
        self.status_var.set("Recording... waiting for data")
        self.save_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)
        self.recording_status_var.set("Recording in progress...")
        self._invalidate_cache()
        
        print("✓ Recording state activated")

    def stop_recording(self):
        """Stop recording with statistics."""
        print(f"\n=== STOPPING RECORDING ===")
        
        with self.recording_lock:
            if not self.is_recording:
                print("⚠ stop_recording called but not recording")
                return
            
            self.is_recording = False
            
            # Calculate statistics
            if self.record_start_timestamp and len(self.recorded) > 0:
                first_timestamp = self.recorded[0][0]
                last_timestamp = self.recorded[-1][0]
                actual_duration = last_timestamp - first_timestamp
                
                print(f"✓ Recording complete:")
                print(f"  Duration: {actual_duration:.2f}s (target: {self.record_target_duration:.2f}s)")
                print(f"  Samples: {len(self.recorded)}")
                
                actual_rate = self.calculate_actual_sampling_rate()
                if actual_rate:
                    expected_rate, _ = InputValidator.validate_float(self.expected_input_rate_var.get(), 1, 10000)
                    expected_rate = expected_rate if expected_rate else 200
                    print(f"  Actual rate: {actual_rate:.1f} Hz")
                    print(f"  Expected rate: {expected_rate} Hz")
                    print(f"  Rate accuracy: {(actual_rate/expected_rate)*100:.1f}%")
            else:
                print(f"⚠ No recording data captured")
        
        # Update UI
        self.rec_btn.config(text="Start Recording", bg="#8cf58c")
        self.status_var.set("Connected" if self.is_connected else "Disconnected")
        
        # Update action buttons
        with self.recording_lock:
            if len(self.recorded) > 0:
                self.save_btn.config(state=tk.NORMAL)
                self.analyze_btn.config(state=tk.NORMAL if self.ml_analyzer.is_model_loaded else tk.DISABLED)
                
                duration = self.recorded[-1][0] - self.recorded[0][0]
                actual_rate = self.calculate_actual_sampling_rate()
                rate_text = f" @ {actual_rate:.0f}Hz" if actual_rate else ""
                self.recording_status_var.set(f"Recording complete: {len(self.recorded)} samples ({duration:.1f}s{rate_text})")
            else:
                self.recording_status_var.set("Recording failed - no data captured")
        
        print("=== RECORDING STOPPED ===\n")

    def save_recording(self):
        """Save recorded data with validation."""
        with self.recording_lock:
            if not self.recorded:
                messagebox.showwarning("No Data", "No recorded data to save.")
                return
            
            recorded_copy = self.recorded.copy()
        
        save_dir = filedialog.askdirectory(title="Select directory to save files")
        if not save_dir:
            return
        
        metadata = self.get_metadata()
        if metadata is None:
            return
        
        base_name = time.strftime("PPG_%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = os.path.join(save_dir, base_name + "_data.csv")
        try:
            with open(csv_path, "w", newline='') as f:
                w = csv.writer(f)
                w.writerow(["timestamp_s", "raw_ppg", "scaled_ppg", "normalized_ppg"])
                for row in recorded_copy:
                    w.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}"])
            print(f"✓ Saved CSV: {csv_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save CSV file:\n{str(e)}")
            return

        # Save .mat file
        mat_path = os.path.join(save_dir, base_name + "_signal.mat")
        try:
            signal_data = np.array([row[1] for row in recorded_copy])
            savemat(mat_path, {'signal': signal_data})
            print(f"✓ Saved MAT: {mat_path}")
        except Exception as e:
            print(f"⚠ Could not save .mat file: {e}")
            mat_path = None

        # Save metadata
        meta_path = os.path.join(save_dir, base_name + "_metadata.txt")
        try:
            with open(meta_path, "w") as f:
                f.write("PPG Recording Metadata\n")
                f.write("=" * 30 + "\n")
                f.write(f"Recording Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                if len(recorded_copy) > 0:
                    duration = recorded_copy[-1][0] - recorded_copy[0][0]
                    actual_rate = self.calculate_actual_sampling_rate()
                    f.write(f"Signal Duration: {duration:.2f} seconds\n")
                    f.write(f"Target Duration: {self.record_target_duration:.2f} seconds\n")
                    f.write(f"Total Samples: {len(recorded_copy)}\n")
                    if actual_rate:
                        f.write(f"Actual Sample Rate: {actual_rate:.1f} Hz\n")
                    expected_rate, _ = InputValidator.validate_float(self.expected_input_rate_var.get())
                    if expected_rate:
                        f.write(f"Expected Sample Rate: {expected_rate} Hz\n")
                
                f.write("\nPatient Information:\n")
                for key, value in metadata.items():
                    f.write(f"  {key}: {value}\n")
            print(f"✓ Saved metadata: {meta_path}")
        except Exception as e:
            print(f"⚠ Could not save metadata file: {e}")

        saved_files = [os.path.basename(csv_path)]
        if mat_path:
            saved_files.append(os.path.basename(mat_path))
        saved_files.append(os.path.basename(meta_path))
        
        messagebox.showinfo("Data Saved", f"Files saved successfully:\n" + "\n".join([f"• {f}" for f in saved_files]))

    def check_diabetes(self):
        """Run diabetes risk analysis with validation."""
        with self.recording_lock:
            if not self.recorded:
                messagebox.showwarning("No Data", "No recorded data to analyze.")
                return
            
            recorded_copy = self.recorded.copy()
        
        if not self.ml_analyzer.is_model_loaded:
            messagebox.showerror("Model Not Loaded", "Please load the ML model first.")
            return
        
        metadata = self.get_metadata()
        if metadata is None:
            return
        
        # Use actual sampling rate
        actual_Fs = self.calculate_actual_sampling_rate()
        expected_Fs_val, _ = InputValidator.validate_float(self.expected_input_rate_var.get(), 1, 10000)
        expected_Fs = expected_Fs_val if expected_Fs_val else 200
        
        if actual_Fs:
            Fs = actual_Fs
            print(f"Using actual sampling rate: {Fs:.1f} Hz")
        else:
            Fs = expected_Fs
            print(f"Using expected sampling rate: {Fs} Hz")
        
        # Extract signal
        signal_data = np.array([row[1] for row in recorded_copy])
        
        # Show progress
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Analyzing...")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        tk.Label(progress_window, text="Analyzing PPG signal for diabetes risk...", 
                font=("Arial", 10)).pack(pady=20)
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()
        
        self.root.update()
        
        # Run analysis
        try:
            result = self.ml_analyzer.analyze_signal(signal_data, Fs, metadata)
            progress_window.destroy()
            
            if result and "error" not in result:
                self.show_analysis_results(result, metadata)
            else:
                error_msg = result.get("error", "Unknown analysis error") if result else "Analysis failed"
                messagebox.showerror("Analysis Error", error_msg)
                
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("Analysis Error", f"Analysis failed:\n{str(e)}")

    def show_analysis_results(self, result, metadata):
        """Display analysis results in new window."""
        results_window = tk.Toplevel(self.root)
        results_window.title("Diabetes Risk Analysis Results")
        results_window.geometry("600x550")
        results_window.transient(self.root)
        
        if self.dark_mode.get():
            results_window.configure(bg="#2b2b2b")
        
        main_frame = tk.Frame(results_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = tk.Frame(main_frame, relief=tk.RAISED, bd=1)
        title_frame.pack(fill=tk.X, pady=(0,10))
        tk.Label(title_frame, text="PPG-Based Diabetes Risk Analysis", 
                font=("Arial", 14, "bold")).pack(pady=8)
        
        # Patient info
        patient_frame = tk.LabelFrame(main_frame, text="Patient Information", 
                                     font=("Arial", 10, "bold"), padx=10, pady=5)
        patient_frame.pack(fill=tk.X, pady=(0,10))
        
        patient_text = f"Age: {metadata.get('Age', 'N/A')} years  |  " \
                      f"Gender: {metadata.get('Gender', 'N/A')}  |  " \
                      f"Height: {metadata.get('Height', 'N/A')} cm  |  " \
                      f"Weight: {metadata.get('Weight', 'N/A')} kg"
        tk.Label(patient_frame, text=patient_text, font=("Arial", 9)).pack(pady=2)
        
        # Risk assessment
        risk_frame = tk.LabelFrame(main_frame, text="Risk Assessment", 
                                  font=("Arial", 10, "bold"), padx=10, pady=5)
        risk_frame.pack(fill=tk.X, pady=(0,10))
        
        risk_level = result.get('risk_level', 'Unknown')
        probability = result.get('probability', 0)
        
        risk_colors = {"Low": "#2e8b57", "Moderate": "#ff8c00", "High": "#dc143c"}
        risk_color = risk_colors.get(risk_level, "black")
        
        risk_text = tk.Label(risk_frame, text=f"RISK LEVEL: {risk_level.upper()}", 
                           font=("Arial", 16, "bold"), fg=risk_color)
        risk_text.pack(pady=5)
        
        prob_text = tk.Label(risk_frame, text=f"Diabetes Probability: {probability:.1%}", 
                           font=("Arial", 12))
        prob_text.pack(pady=2)
        
        interpretation = result.get('interpretation', 'No interpretation available')
        interp_text = tk.Label(risk_frame, text=interpretation, 
                             font=("Arial", 10), wraplength=550)
        interp_text.pack(pady=5)
        
        # Technical details
        tech_frame = tk.LabelFrame(main_frame, text="Technical Details", 
                                  font=("Arial", 10, "bold"), padx=10, pady=5)
        tech_frame.pack(fill=tk.X, pady=(0,10))
        
        signal_quality = result.get('signal_quality', 0)
        peaks_detected = result.get('peaks_detected', 0)
        heart_rate = result.get('heart_rate', np.nan)
        signal_duration = result.get('signal_duration', 0)
        sampling_rate = result.get('sampling_rate', 0)
        confidence = result.get('confidence_factor', 1.0)
        
        tech_info = f"Signal Quality Index: {signal_quality:.3f}\n"
        tech_info += f"Cardiac Peaks Detected: {peaks_detected}\n"
        if not np.isnan(heart_rate):
            tech_info += f"Estimated Heart Rate: {heart_rate:.1f} BPM\n"
        
        tech_info += f"Signal Duration: {signal_duration:.1f} seconds\n"
        tech_info += f"Sampling Rate Used: {sampling_rate:.1f} Hz\n"
        with self.recording_lock:
            tech_info += f"Total Samples: {len(self.recorded)}\n"
        tech_info += f"Analysis Confidence: {confidence*100:.0f}%"
        
        tk.Label(tech_frame, text=tech_info, font=("Arial", 9), justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        
        # Disclaimer
        disclaimer_frame = tk.LabelFrame(main_frame, text="Important Disclaimer", 
                                        font=("Arial", 10, "bold"), padx=10, pady=5)
        disclaimer_frame.pack(fill=tk.X, pady=(0,10))
        
        disclaimer_text = ("This analysis is for research/educational purposes only and should NOT be used "
                          "for medical diagnosis. Always consult with qualified healthcare professionals "
                          "for proper medical evaluation and diagnosis.")
        tk.Label(disclaimer_frame, text=disclaimer_text, font=("Arial", 9), 
                wraplength=550, fg="#8B0000").pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(button_frame, text="Close", command=results_window.destroy, 
                 font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=5)
        
        self._update_widget_theme(results_window, "#2b2b2b" if self.dark_mode.get() else "#f0f0f0", 
                                "#ffffff" if self.dark_mode.get() else "#000000", 
                                "#404040" if self.dark_mode.get() else "#ffffff",
                                "#ffffff" if self.dark_mode.get() else "#000000",
                                "#404040" if self.dark_mode.get() else "#e0e0e0",
                                "#ffffff" if self.dark_mode.get() else "#000000",
                                "#333333" if self.dark_mode.get() else "#f4f4f4",
                                "#404040" if self.dark_mode.get() else "#e8f4f8")

    def clear_graph(self):
        """Clear all data and reset."""
        with self.data_lock:
            self.time_buf.clear()
            self.ppg_buf.clear()
            self.running_baseline_window.clear()
            self.baseline = None
            self.start_time = None
        
        self.is_live_view = True
        self.nav_position = 1.0
        self.nav_slider_var.set(1.0)
        self.nav_info_var.set("Live")
        
        with self.recording_lock:
            self.recorded = []
        
        self.save_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)
        self.recording_status_var.set("No recording available")
        
        self.line.set_data([], [])
        self.ax.set_xlim(0, Config.PLOT_SECONDS_WINDOW)
        self.ax.set_ylim(-1, 1)
        self._invalidate_cache()
        self.canvas.draw_idle()
        
        print("✓ Graph and data cleared")
    
    def _on_slider_change(self, value):
        """Handle navigation slider changes."""
        self.nav_position = float(value)
        self.is_live_view = (self.nav_position >= 0.99)
        self._invalidate_cache()
        
        with self.data_lock:
            if not self.is_live_view and len(self.time_buf) > 0:
                total_duration = (self.time_buf[-1] - self.time_buf[0]) / 1000.0
                current_offset = total_duration * (1.0 - self.nav_position)
                self.nav_info_var.set(f"-{current_offset:.1f}s from live")
            elif self.is_live_view:
                self.nav_info_var.set("Live")
            else:
                self.nav_info_var.set("No data")
    
    def _go_live(self):
        """Return to live view."""
        self.is_live_view = True
        self.nav_position = 1.0
        self.nav_slider_var.set(1.0)
        self.nav_info_var.set("Live")
        self._invalidate_cache()

    def _gui_update(self):
        """Main GUI update loop with thread-safe data access."""
        new_count = 0
        
        # Process queue data and update display buffers
        while not self.data_q.empty():
            try:
                timestamp_ms, ppg = self.data_q.get_nowait()
            except queue.Empty:
                break

            timestamp_s = timestamp_ms / 1000.0
            
            if self.start_time is None:
                self.start_time = timestamp_s

            # Finger detection
            thr_val, _ = InputValidator.validate_float(self.finger_threshold_var.get(), 0, 100000)
            thr = thr_val if thr_val is not None else 5000.0
                
            if self.zero_on_no_finger_var.get() and ppg < thr:
                ppg_display = 0.0
            else:
                ppg_display = float(ppg)

            # Update running baseline
            if ppg_display > 0:
                self.running_baseline_window.append(ppg_display)
                if len(self.running_baseline_window) >= 5:
                    self.baseline = np.median(self.running_baseline_window)
                else:
                    self.baseline = np.mean(self.running_baseline_window) if self.running_baseline_window else None

            # Thread-safe append to display buffers
            with self.data_lock:
                self.time_buf.append(timestamp_ms)
                self.ppg_buf.append(ppg_display)

            new_count += 1

        # Update plot if we have new data
        if new_count > 0:
            self._invalidate_cache()
            self._update_plot()

        # Schedule next update
        self.root.after(Config.GUI_UPDATE_INTERVAL_MS, self._gui_update)

    def _update_plot(self):
        """Update plot with caching for performance."""
        with self.data_lock:
            if len(self.time_buf) == 0:
                return
            
            # Use cached data if valid
            if self.cache_valid and self.cached_plot_data is not None:
                times_plot, ys_plot, x_min, x_max, ylabel = self.cached_plot_data
            else:
                times_ms = np.array(self.time_buf, dtype=np.float64)
                ys_raw = np.array(self.ppg_buf, dtype=np.float64)
                times_s = times_ms / 1000.0
                
                # Update slider range
                if len(self.time_buf) > Config.PLOT_SECONDS_WINDOW * 200:
                    self.nav_slider.config(state='normal')
                else:
                    self.nav_slider.config(state='disabled')
                
                # Determine time window
                if self.is_live_view:
                    if len(times_s) > 0:
                        latest_s = times_s[-1]
                        start_time_threshold = latest_s - Config.PLOT_SECONDS_WINDOW
                        valid_indices = times_s >= start_time_threshold
                        if np.any(valid_indices):
                            times_window = times_s[valid_indices]
                            ys_window = ys_raw[valid_indices]
                        else:
                            times_window = times_s
                            ys_window = ys_raw
                    else:
                        return
                else:
                    total_duration = times_s[-1] - times_s[0] if len(times_s) > 1 else Config.PLOT_SECONDS_WINDOW
                    
                    if total_duration > Config.PLOT_SECONDS_WINDOW:
                        window_end_offset = total_duration * self.nav_position
                        window_end_time = times_s[0] + window_end_offset
                        window_start_time = max(times_s[0], window_end_time - Config.PLOT_SECONDS_WINDOW)
                        
                        valid_indices = (times_s >= window_start_time) & (times_s <= window_end_time)
                        if np.any(valid_indices):
                            times_window = times_s[valid_indices]
                            ys_window = ys_raw[valid_indices]
                        else:
                            times_window = times_s[-100:] if len(times_s) > 100 else times_s
                            ys_window = ys_raw[-100:] if len(ys_raw) > 100 else ys_raw
                    else:
                        times_window = times_s
                        ys_window = ys_raw

                # Time display mode
                time_mode = self.time_display_var.get()
                if time_mode == "Rolling":
                    if len(times_window) > 0:
                        times_plot = times_window - times_window[0]
                        x_min, x_max = 0, Config.PLOT_SECONDS_WINDOW
                        xlabel = "Time (s) - Rolling Window"
                    else:
                        return
                else:
                    if self.start_time is not None and len(times_window) > 0:
                        times_plot = times_window - self.start_time
                        x_min = times_plot[0] if len(times_plot) > 0 else 0
                        x_max = times_plot[-1] if len(times_plot) > 0 else Config.PLOT_SECONDS_WINDOW
                        x_range = x_max - x_min
                        if x_range > 0:
                            x_min -= x_range * 0.02
                            x_max += x_range * 0.02
                        else:
                            x_max = x_min + Config.PLOT_SECONDS_WINDOW
                        xlabel = "Time (s) - Elapsed since start"
                    else:
                        return

                # Apply display transformation
                mode = self.display_mode_var.get()
                scale_val, _ = InputValidator.validate_float(self.scale_factor_var.get(), 0.001, 1000000)
                scale = scale_val if scale_val is not None and scale_val != 0 else 1000.0

                if len(ys_window) > 0:
                    if mode == "Raw":
                        ys_plot = ys_window
                        ylabel = "Raw ADC Value"
                    elif mode == "Scaled":
                        ys_plot = ys_window / scale
                        ylabel = f"Scaled (Raw ÷ {scale})"
                    else:  # Normalized
                        if self.baseline and self.baseline != 0:
                            ys_plot = (ys_window - self.baseline) / self.baseline
                        else:
                            ys_plot = ys_window * 0.0
                        ylabel = "Normalized (Fractional Change)"
                else:
                    return
                
                # Cache the computed data
                self.cached_plot_data = (times_plot, ys_plot, x_min, x_max, ylabel)
                self.cache_valid = True

        # Update plot
        self.line.set_data(times_plot, ys_plot)
        self.ax.set_xlim(x_min, x_max)
        
        # Y-axis scaling
        if self.auto_scale_var.get() and len(ys_plot) > 1:
            ymin, ymax = np.percentile(ys_plot, [1, 99])
            if ymax - ymin < 1e-6:
                self.ax.set_ylim(ymin - 1, ymax + 1)
            else:
                padding = (ymax - ymin) * 0.1
                self.ax.set_ylim(ymin - padding, ymax + padding)
        else:
            y_min_val, _ = InputValidator.validate_float(self.y_min_var.get())
            y_max_val, _ = InputValidator.validate_float(self.y_max_var.get())
            
            if y_min_val is not None and y_max_val is not None:
                mode = self.display_mode_var.get()
                if mode == "Scaled":
                    scale_val, _ = InputValidator.validate_float(self.scale_factor_var.get(), 0.001, 1000000)
                    scale = scale_val if scale_val is not None and scale_val != 0 else 1000.0
                    y_min_val /= scale
                    y_max_val /= scale
                self.ax.set_ylim(y_min_val, y_max_val)
            else:
                self.ax.set_ylim(-1, 1)
        
        self.ax.set_ylabel(ylabel)
        self.canvas.draw_idle()

    def on_closing(self):
        """Handle application closing safely."""
        if self.is_recording:
            if messagebox.askyesno("Recording Active", 
                                  "Recording is in progress. Stop recording and exit?"):
                self.stop_recording()
            else:
                return
        
        if self.is_connected:
            self.disconnect()
        
        # Save configuration
        Config.save_config()
        
        self.root.destroy()


if __name__ == "__main__":
    try:
        import pandas as pd  # Required for ML
        
        root = tk.Tk()
        app = PPGLoggerFixed(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please install required packages: pip install pandas scipy matplotlib joblib pyserial")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback

        traceback.print_exc()
