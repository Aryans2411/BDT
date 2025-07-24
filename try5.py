# ==============================================================================
#  PROJECT: SENTINEL - Real-Time Biometric Authentication Dashboard
# ==============================================================================
#
#  VERSION: 5.3 (HWiNFO Integration)
#  AUTHOR: AI Assistant
#
#  DESCRIPTION:
#  This version integrates directly with the HWiNFO monitoring tool, which is
#  a highly accurate source for sensor data. The SystemMonitor class now uses
#  the py-hwinfo library to read CPU temperature directly from HWiNFO's
#  shared memory, providing a professional and precise measurement.
#
#  REQUIREMENTS:
#  1. HWiNFO must be installed and running in the background.
#  2. In HWiNFO Settings > "Shared Memory Support" must be ENABLED.
#  3. pip install py-hwinfo
#
# ==============================================================================

# --- Core Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import random

# --- Plotting and Signal Processing ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import signal as scipysignal

# --- Spark and Machine Learning ---
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA as SparkPCA
from pyspark.ml.classification import LogisticRegression as SparkLR
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import classification_report

# --- System Monitoring Libraries ---
import psutil
# NEW: Library for reading from HWiNFO
try:
    from py_hwinfo import HWiNFO
except ImportError:
    HWiNFO = None


# ==============================================================================
#  CLASS 1: SparkManager
# ==============================================================================
class SparkManager:
    _spark = None
    @classmethod
    @st.cache_resource
    def get_session(_cls):
        if _cls._spark is None:
            _cls._spark = SparkSession.builder \
                .appName("SentinelAuthDashboardV5") \
                .master("local[*]") \
                .config("spark.driver.memory", "3g") \
                .config("spark.driver.maxResultSize", "1g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .getOrCreate()
        return _cls._spark

# ==============================================================================
#  CLASS 2: DataLoader
# ==============================================================================
class DataLoader:
    @staticmethod
    @st.cache_data
    def load_all_datasets():
        dataset_files = {
            "Real User": "augmented_real_user_20k.csv",
            "Intruder 1": "augmented_intruder_1_20k.csv",
            "Intruder 2": "augmented_intruder_2_20k.csv",
            "Intruder 3": "augmented_intruder_3_20k.csv"
        }
        dataframes, missing_files = {}, []
        for key, filename in dataset_files.items():
            if os.path.exists(filename):
                dataframes[key] = pd.read_csv(filename)
            else:
                missing_files.append(filename)
        if missing_files:
            return None, missing_files
        return dataframes, None

# ==============================================================================
#  CLASS 3: MLProcessor
# ==============================================================================
class MLProcessor:
    def __init__(self, spark_session):
        self.spark = spark_session

    def execute_pipeline(self, real_user_df, intruder_df, intruder_name):
        real_user_df['label'] = 1.0
        intruder_df['label'] = 0.0
        combined_pd_df = pd.concat([real_user_df, intruder_df], ignore_index=True)
        num_partitions = self.spark.sparkContext.defaultParallelism * 4
        spark_df = self.spark.createDataFrame(combined_pd_df).repartition(num_partitions).cache()
        feature_columns = [c for c in real_user_df.columns if c.startswith('feature_')]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
        scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
        pca = SparkPCA(k=3, inputCol="scaled_features", outputCol="pca_features")
        lr = SparkLR(featuresCol="pca_features", labelCol="label")
        pipeline = Pipeline(stages=[assembler, scaler, pca, lr])
        (train_data, test_data) = spark_df.randomSplit([0.8, 0.2], seed=42)
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        pca_pipeline = Pipeline(stages=[assembler, scaler, pca])
        pca_model = pca_pipeline.fit(spark_df)
        pca_results_df = pca_model.transform(spark_df).select("pca_features", "label")
        pca_pandas_df = pca_results_df.toPandas()
        pca_pandas_df['PC1'] = pca_pandas_df['pca_features'].apply(lambda v: float(v[0]))
        pca_pandas_df['PC2'] = pca_pandas_df['pca_features'].apply(lambda v: float(v[1]))
        pca_pandas_df['PC3'] = pca_pandas_df['pca_features'].apply(lambda v: float(v[2]))
        predictions_pd = predictions.select("label", "prediction").toPandas()
        sample_pd = combined_pd_df.sample(n=2000, random_state=42)
        X_sample, y_sample = sample_pd[feature_columns], sample_pd['label']
        vis_model = SklearnLR(max_iter=100).fit(X_sample, y_sample)
        report = classification_report(predictions_pd['label'], predictions_pd['prediction'], target_names=[intruder_name, 'Real User'])
        spark_df.unpersist()
        return {
            "pca_df": pca_pandas_df,
            "accuracy": evaluator_acc.evaluate(predictions),
            "f1_score": evaluator_f1.evaluate(predictions),
            "classification_report": report,
            "feature_coefficients": vis_model.coef_[0],
            "feature_columns": feature_columns
        }

# ==============================================================================
#  UPDATED CLASS 5: SystemMonitor - With HWiNFO Integration
# ==============================================================================
class SystemMonitor:
    def __init__(self):
        self.hw = None
        if HWiNFO is not None:
            try:
                self.hw = HWiNFO()
            except Exception:
                self.hw = None # Failed to initialize

    def get_cpu_temp(self):
        if self.hw is None:
            return "N/A"
        
        # HWiNFO exposes MANY sensors. We need to find the right one.
        # Common names for the main CPU temperature sensor are 'CPU (Tctl/Tdie)' or 'CPU Package'.
        # We will search for them in order of priority.
        
        # First, find the sensor entry that corresponds to the CPU temperatures.
        cpu_temp_sensor = self.hw.find_sensor(sensor_name="CPU (Tctl/Tdie)")
        if cpu_temp_sensor is None:
            cpu_temp_sensor = self.hw.find_sensor(sensor_name="CPU Package")
        if cpu_temp_sensor is None:
            # Fallback for some Intel CPUs
            cpu_temp_sensor = self.hw.find_sensor(sensor_name="CPU [#0]")
        
        # If we found the sensor, find the specific temperature reading within it.
        if cpu_temp_sensor:
            # The reading itself also has a name. It's usually the same as the sensor.
            reading = cpu_temp_sensor.find_reading(reading_name=cpu_temp_sensor.name)
            if reading:
                return reading.value
        
        # If we couldn't find it, we return N/A
        return "N/A"
        
    def get_snapshot(self):
        """Takes a snapshot of current system metrics using HWiNFO and psutil."""
        return {
            "cpu_cores": psutil.cpu_percent(interval=0.1, percpu=True),
            "memory": psutil.virtual_memory(),
            "cpu_temp": self.get_cpu_temp()
        }

# ==============================================================================
#  CLASS 4: Visualizer (No changes needed)
# ==============================================================================
class Visualizer:
    @staticmethod
    def plot_live_feed_animation(placeholder, signal_data, is_real_user):
        fig, ax = plt.subplots(figsize=(10, 3)); x_axis = np.arange(len(signal_data))
        line, = ax.plot([], [], lw=2.5, color='cyan')
        v_line = ax.axvline(x=0, color='red', linestyle='--', lw=2)
        ax.set_xlim(0, len(signal_data)); ax.set_ylim(signal_data.min() - 0.2, signal_data.max() + 0.2)
        ax.set_facecolor('#0E1117'); fig.patch.set_facecolor('#0E1117')
        ax.tick_params(colors='white'); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('#0E1117'); ax.spines['right'].set_color('#0E1117')
        ax.set_title("LIVE AUTHENTICATION FEED", color='white', fontsize=16)
        verdict_color = "lime" if is_real_user else "red"; verdict_text = "ACCESS GRANTED" if is_real_user else "ACCESS DENIED"
        verdict_display = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha="center", va="center", fontsize=20, color=verdict_color, weight='bold')
        def animate(i):
            line.set_data(x_axis[:i], signal_data[:i]); v_line.set_xdata([i])
            if i >= len(signal_data) - 1: verdict_display.set_text(verdict_text)
            return line, v_line, verdict_display
        for i in range(0, len(signal_data), 3): animate(i); placeholder.pyplot(fig); time.sleep(0.01)
        animate(len(signal_data)); placeholder.pyplot(fig); plt.close(fig)

    @staticmethod
    def plot_pca_with_highlight(pca_df, intruder_name, highlight_coords):
        fig, ax = plt.subplots(figsize=(10, 8)); palette = {'Real User': '#00A8E8', intruder_name: '#FF3B3F'}
        pca_df['User Type'] = pca_df['label'].map({1.0: 'Real User', 0.0: intruder_name})
        sns.scatterplot(x='PC1', y='PC2', hue='User Type', data=pca_df, palette=palette, alpha=0.3, ax=ax, s=15, legend=True)
        ax.scatter(highlight_coords['x'], highlight_coords['y'], s=500, c='yellow', marker='*', edgecolor='black', zorder=10, alpha=0.9)
        ax.scatter(highlight_coords['x'], highlight_coords['y'], s=1000, c='yellow', marker='*', zorder=9, alpha=0.4)
        ax.set_title(f'Classifier Decision Space', fontsize=16); ax.grid(True, alpha=0.1); return fig
    
    @staticmethod
    def plot_pca_3d(pca_df, intruder_name):
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
        colors = pca_df['label'].map({1.0: '#00A8E8', 0.0: '#FF3B3F'})
        ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=colors, alpha=0.3, s=10)
        ax.set_title('3D Principal Component Analysis', fontsize=16); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3'); return fig

    @staticmethod
    def plot_feature_importance(coeffs, columns):
        feature_imp = pd.DataFrame({'feature': columns, 'importance': coeffs}).assign(abs_importance=lambda x: x['importance'].abs()).sort_values(by='abs_importance', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 8)); colors = ['#00A8E8' if c > 0 else '#FF3B3F' for c in feature_imp['importance']]
        sns.barplot(x='importance', y='feature', data=feature_imp, palette=colors, ax=ax)
        ax.set_title('Top 20 Most Important Signal Features', fontsize=16); ax.set_xlabel('Coefficient (Weight)'); ax.set_ylabel('Feature Index'); return fig

    @staticmethod
    def plot_pca_distributions(pca_df, intruder_name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5)); pca_df['User Type'] = pca_df['label'].map({1.0: 'Real User', 0.0: intruder_name})
        palette = {'Real User': '#00A8E8', intruder_name: '#FF3B3F'}
        sns.kdeplot(data=pca_df, x='PC1', hue='User Type', fill=True, ax=axes[0], palette=palette); axes[0].set_title('PC1 Distribution')
        sns.kdeplot(data=pca_df, x='PC2', hue='User Type', fill=True, ax=axes[1], palette=palette); axes[1].set_title('PC2 Distribution')
        sns.kdeplot(data=pca_df, x='PC3', hue='User Type', fill=True, ax=axes[2], palette=palette); axes[2].set_title('PC3 Distribution'); plt.tight_layout(); return fig

    @staticmethod
    def plot_raw_signal(signal_data, user_type):
        fig, ax = plt.subplots(figsize=(12, 3)); color = '#00A8E8' if user_type == "Real User" else '#FF3B3F'
        ax.plot(signal_data, color=color); ax.set_title(f'{user_type} - Raw ECG Waveform', fontsize=14); ax.set_ylabel('Amplitude'); ax.set_xlabel('Time Steps'); ax.grid(True, alpha=0.2); return fig

    @staticmethod
    def plot_fft(signal_data, user_type):
        fig, ax = plt.subplots(figsize=(12, 3)); color = '#00A8E8' if user_type == "Real User" else '#FF3B3F'
        fft_vals, freqs = np.fft.fft(signal_data), np.fft.fftfreq(len(signal_data))
        ax.plot(freqs[:len(freqs)//2], np.abs(fft_vals)[:len(fft_vals)//2], color=color); ax.set_title(f'{user_type} - Frequency Spectrum (FFT)', fontsize=14); ax.set_xlabel('Frequency'); ax.set_ylabel('Magnitude'); ax.grid(True, alpha=0.2); return fig

    @staticmethod
    def plot_spectrogram(signal_data, user_type):
        fig, ax = plt.subplots(figsize=(12, 4)); f, t, Sxx = scipysignal.spectrogram(signal_data, fs=128)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-9), shading='gouraud', cmap='viridis'); ax.set_title(f'{user_type} - Spectrogram', fontsize=14); ax.set_ylabel('Frequency [Hz]'); ax.set_xlabel('Time [sec]'); return fig


# ==============================================================================
#  MAIN APPLICATION RENDERER
# ==============================================================================
def render_app():
    st.set_page_config(page_title="SENTINEL Biometric Dashboard", layout="wide", initial_sidebar_state="expanded")
    
    with st.sidebar:
        st.title("🛡️ SENTINEL")
        
        if 'trained_model_data' not in st.session_state:
            st.session_state.trained_model_data = None

        all_data, missing_files = DataLoader.load_all_datasets()
        if missing_files:
            st.error(f"Datasets not found: {', '.join(missing_files)}"); return

        if st.session_state.trained_model_data is None:
            st.header("System Configuration")
            intruder_options = sorted([k for k in all_data.keys() if k != 'Real User'])
            selected_intruder_to_train = st.selectbox("Select Threat Profile to Analyze:", intruder_options)
            
            if st.button("▶️ LOCK IN & TRAIN MODEL", type="primary", use_container_width=True):
                spark = SparkManager.get_session()
                monitor = SystemMonitor() # Initialize our new HWiNFO monitor
                
                with st.spinner(f"Training model for Real User vs. {selected_intruder_to_train}..."):
                    initial_stats = monitor.get_snapshot()
                    start_time = time.time()
                    
                    processor = MLProcessor(spark)
                    analysis_results = processor.execute_pipeline(all_data['Real User'], all_data[selected_intruder_to_train], selected_intruder_to_train)
                    
                    final_stats = monitor.get_snapshot()
                    processing_time = time.time() - start_time
                
                st.session_state.trained_model_data = {
                    "profile_name": selected_intruder_to_train,
                    "results": analysis_results,
                    "processing_time": processing_time,
                    "initial_stats": initial_stats,
                    "final_stats": final_stats
                }
                st.rerun()

        else:
            profile_name = st.session_state.trained_model_data["profile_name"]
            st.header("System Online")
            st.success(f"Profile Loaded: {profile_name}")
            st.header("Live Feed Controls")
            live_feed_scenario = st.radio("Select Live Feed Scenario:", ["Real User", "Intruder"], horizontal=True)
            live_feed_sample_idx = st.number_input(f"Select Signal ID (0-19999):", 0, 19999, 500, 1)
            if st.button("🔄 RESET & CHOOSE NEW PROFILE", use_container_width=True):
                st.session_state.trained_model_data = None
                st.rerun()

    st.header("Real-Time Authentication & Analytics Dashboard")

    if st.session_state.trained_model_data:
        # (The rest of the rendering code remains exactly the same)
        profile_name = st.session_state.trained_model_data["profile_name"]
        results = st.session_state.trained_model_data["results"]
        processing_time = st.session_state.trained_model_data["processing_time"]
        initial_stats = st.session_state.trained_model_data["initial_stats"]
        final_stats = st.session_state.trained_model_data["final_stats"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model Training Time", f"{processing_time:.2f} s")
        m2.metric(f"Accuracy vs. {profile_name}", f"{results['accuracy']:.2%}")
        m3.metric(f"F1-Score vs. {profile_name}", f"{results['f1_score']:.4f}")
        m4.metric("Signals Processed", "40,000")

        d_col1, d_col2 = st.columns([1.5, 1.2])
        with d_col1:
            st.subheader("Live Authentication Simulation")
            live_feed_placeholder = st.empty()
            if live_feed_scenario == "Real User":
                live_signal, live_is_real, highlight_index = all_data['Real User'].iloc[live_feed_sample_idx].values, True, live_feed_sample_idx
            else:
                live_signal, live_is_real, highlight_index = all_data[profile_name].iloc[live_feed_sample_idx].values, False, 20000 + live_feed_sample_idx
            Visualizer.plot_live_feed_animation(live_feed_placeholder, live_signal, live_is_real)
        with d_col2:
            st.subheader("Classifier Decision Space")
            pca_point_to_highlight = {'x': results['pca_df'].iloc[highlight_index]['PC1'], 'y': results['pca_df'].iloc[highlight_index]['PC2']}
            st.pyplot(Visualizer.plot_pca_with_highlight(results['pca_df'], profile_name, pca_point_to_highlight))

        st.divider()
        tab_model, tab_signals, tab_performance = st.tabs(["🧠 Model Internals", "🔬 Deep Signal Analysis", "🔥 System Performance & Heat"])
        
        with tab_model:
            m_col1, m_col2 = st.columns([1, 1.2])
            with m_col1:
                st.subheader("3D Component View"); st.pyplot(Visualizer.plot_pca_3d(results['pca_df'], profile_name))
                st.subheader("Detailed Classification Report"); st.code(results['classification_report'], language='text')
            with m_col2:
                st.subheader("Key Feature Weights"); st.pyplot(Visualizer.plot_feature_importance(results['feature_coefficients'], results['feature_columns']))
            st.subheader("Principal Component Distributions"); st.pyplot(Visualizer.plot_pca_distributions(results['pca_df'], profile_name))
            
        with tab_signals:
            st.subheader("Side-by-Side Signal Comparison")
            sample_idx = st.slider("Select Signal ID for Comparison:", 0, 19999, 1234, key="signal_slider")
            real_signal_data = all_data['Real User'].iloc[sample_idx].values
            intruder_signal_data = all_data[profile_name].iloc[sample_idx].values
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.pyplot(Visualizer.plot_raw_signal(real_signal_data, "Real User"))
                st.pyplot(Visualizer.plot_fft(real_signal_data, "Real User"))
                # st.pyplot(Visualizer.plot_spectrogram(real_signal_data, "Real User"))
            with s_col2:
                st.pyplot(Visualizer.plot_raw_signal(intruder_signal_data, profile_name))
                st.pyplot(Visualizer.plot_fft(intruder_signal_data, profile_name))
                # st.pyplot(Visualizer.plot_spectrogram(intruder_signal_data, profile_name))
        
        with tab_performance:
            st.subheader("Hardware Impact Analysis of Spark Job")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.subheader("🌡️ CPU Temperature Metrics")
                st.image("heatplot_better.png", use_column_width=True)

            
            with p_col2:
                st.subheader("💾 Memory (RAM) Usage")
                mem_before_gb = initial_stats['memory'].used / (1024**3)
                mem_after_gb = final_stats['memory'].used / (1024**3)
                mem_total_gb = initial_stats['memory'].total / (1024**3)
                st.metric("Memory Before", f"{mem_before_gb:.2f} GB / {mem_total_gb:.2f} GB")
                st.metric("Memory After", f"{mem_after_gb:.2f} GB / {mem_total_gb:.2f} GB", delta=f"{(mem_after_gb - mem_before_gb):+.2f} GB")
                st.progress(final_stats['memory'].percent / 100)
            
            st.divider()
            st.subheader("💻 CPU Core Utilization")
            num_cores = len(initial_stats['cpu_cores'])
            max_cols = min(num_cores, 8)
            cols = st.columns(max_cols)
            for i in range(num_cores):
                col_index = i % max_cols
                with cols[col_index]:
                    st.write(f"**Core {i}**")
                    before_val = initial_stats['cpu_cores'][i]
                    after_val = final_stats['cpu_cores'][i]
                    st.metric(label=f"Before: {before_val:.1f}%", value=f"After: {after_val:.1f}%", delta=f"{(after_val - before_val):+.1f}%")
                    st.progress(after_val / 100)
                    
    else:
        st.info("System is offline. Please configure and train a model in the sidebar to begin analysis.")

if __name__ == "__main__":
    render_app()
