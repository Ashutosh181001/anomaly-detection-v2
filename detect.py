"""
Main Detection Module
Real-time anomaly detection pipeline for crypto trades
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
import joblib
import os
import logging
from kafka import KafkaConsumer
from sklearn.ensemble import IsolationForest

# Import local modules
import config
from features import compute_features
from db_writer import DataWriter
from alerts import AlertManager
from benchmark_logger import BenchmarkLogger

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG["level"]),
    format=config.LOGGING_CONFIG["format"],
    datefmt=config.LOGGING_CONFIG["date_format"]
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Main anomaly detection pipeline"""

    def __init__(self):
        # Initialize components
        self.data_writer = DataWriter()
        self.alert_manager = AlertManager()

        # Load configuration
        self.cfg = config.get_config()

        # Initialize model
        self.model = self._load_model()
        self.is_model_fitted = False

        # Initialize benchmarking
        self.benchmark = BenchmarkLogger(
            csv_path='performance_log.csv',
            json_path='benchmark_summary.json',
            report_interval=100
        )

        # Check if loaded model is already fitted
        if self.model is not None:
            try:
                _ = self.model.predict([[0, 0, 0]])
                self.is_model_fitted = True
                logger.info("✅ Model is fitted and ready for predictions")
            except:
                self.is_model_fitted = False
                logger.info("⏳ Model needs training - will train after collecting sufficient data")

        # Initialize buffers and counters
        self.history_df = pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'is_buyer_maker'])
        self.buffer_for_training = []
        self.trade_counter = 0
        self.anomaly_counter = 0
        self.last_status_time = time.time()

        # Load historical data
        self._load_historical_data()

        # Initialize Kafka consumer
        self.consumer = self._init_kafka_consumer()

    def _load_model(self):
        """Load or initialize the anomaly detection model"""
        # Try tuned model
        if os.path.exists(config.MODEL_PATHS["tuned_model"]):
            try:
                model = joblib.load(config.MODEL_PATHS["tuned_model"])
                logger.info("✅ Loaded tuned Isolation Forest model")
                try:
                    _ = model.predict([[0, 0, 0]])
                    return model
                except:
                    logger.warning("Tuned model not fitted, will retrain")
                    return model
            except Exception as e:
                logger.error(f"Failed to load tuned model: {e}")

        # Try base model
        if os.path.exists(config.MODEL_PATHS["base_model"]):
            try:
                model = joblib.load(config.MODEL_PATHS["base_model"])
                logger.info("✅ Loaded base Isolation Forest model")
                try:
                    _ = model.predict([[0, 0, 0]])
                    return model
                except:
                    logger.warning("Base model not fitted, will retrain")
                    return model
            except Exception as e:
                logger.error(f"Failed to load base model: {e}")

        # Initialize new model
        logger.info("🔄 Initializing new Isolation Forest model (will train after collecting data)")
        return IsolationForest(
            n_estimators=config.ANOMALY_CONFIG["n_estimators"],
            contamination=config.ANOMALY_CONFIG["contamination"],
            random_state=config.ANOMALY_CONFIG["random_state"]
        )

    def _load_historical_data(self):
        """Load historical trades for warm start"""
        try:
            recent_trades = self.data_writer.get_recent_trades(minutes=60)
            if not recent_trades.empty:
                window = config.DATA_CONFIG["rolling_window"]
                self.history_df = recent_trades[
                    ['timestamp', 'price', 'quantity', 'is_buyer_maker']
                ].tail(window).copy()
                logger.info(f"📊 Loaded {len(self.history_df)} historical trades for warm start")
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")

    def _init_kafka_consumer(self):
        """Initialize Kafka consumer"""
        try:
            consumer = KafkaConsumer(
                config.KAFKA_CONFIG["topic"],
                bootstrap_servers=config.KAFKA_CONFIG["broker"],
                auto_offset_reset=config.KAFKA_CONFIG["auto_offset_reset"],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                group_id=config.KAFKA_CONFIG["consumer_group"]
            )
            logger.info("✅ Kafka consumer initialized successfully")
            return consumer
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise

    def detect_anomalies(self, features: dict) -> list:
        """
        Detect anomalies using z-score and Isolation Forest
        Returns list of detected anomaly types
        """
        detected = []
        # Z-score detection
        if abs(features.get('z_score', 0)) > config.ANOMALY_CONFIG["z_score_threshold"]:
            detected.append("z_score")
        # Isolation Forest detection
        if self.is_model_fitted and len(self.buffer_for_training) >= config.DATA_CONFIG["min_history"]:
            try:
                model_feats = [
                    features.get('z_score', 0),
                    features.get('price_change_pct', 0),
                    features.get('time_gap_sec', 0)
                ]
                pred = self.model.predict([model_feats])[0]
                if pred == -1:
                    tag = "filtered_isoforest" if "z_score" in detected else "isoforest"
                    detected.append(tag)
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
        return detected

    def update_model(self):
        """Retrain model with recent data"""
        min_samples = 100
        if len(self.buffer_for_training) < min_samples:
            return
        try:
            data = np.array(
                self.buffer_for_training[-config.DATA_CONFIG["rolling_window"]:]
                if len(self.buffer_for_training) > config.DATA_CONFIG["rolling_window"]
                else self.buffer_for_training
            )
            self.model.fit(data)
            self.is_model_fitted = True
            joblib.dump(self.model, config.MODEL_PATHS["base_model"])
            logger.info(f"🔄 Model trained with {len(data)} samples and saved")
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    def print_status(self):
        """Print periodic status update"""
        now = time.time()
        if now - self.last_status_time >= 30:
            rate = (self.anomaly_counter / self.trade_counter * 100) if self.trade_counter else 0
            logger.info(f"""
            📊 STATUS UPDATE
            ├─ Trades processed: {self.trade_counter:,}
            ├─ Anomalies detected: {self.anomaly_counter:,}
            ├─ Detection rate: {rate:.2f}%
            └─ Model fitted: {self.is_model_fitted}
            """)
            self.last_status_time = now

    def run(self):
        """Main detection loop"""
        logger.info("🚀 Starting anomaly detection pipeline...")
        logger.info(f"📍 Monitoring {config.DATA_CONFIG['asset_pair']} trades")

        for message in self.consumer:
            try:
                trade = message.value

                # --- Benchmark: processing timer start ---
                proc_start = time.perf_counter()

                # Compute features
                features, self.history_df = compute_features(
                    trade,
                    self.history_df,
                    config.DATA_CONFIG["rolling_window"]
                )

                # Log trade
                trade_id = self.data_writer.log_trade(features)
                self.trade_counter += 1

                # --- Benchmark: processing timer end & log ---
                proc_end = time.perf_counter()
                self.benchmark.log_event(
                    'trade',
                    trade_id,
                    duration=proc_end - proc_start
                )

                # Print trade info
                if self.trade_counter % config.PIPELINE_CONFIG["status_interval"] == 0:
                    logger.info(
                        f"[{features['timestamp']}] "
                        f"Price: ${features['price']:,.2f} | "
                        f"Z: {features.get('z_score', 0):.2f} | "
                        f"Δ%: {features.get('price_change_pct', 0):.2f}"
                    )

                # Skip if not enough history
                if len(self.history_df) < config.DATA_CONFIG["min_history"]:
                    continue

                # Update training buffer
                self.buffer_for_training.append([
                    features.get('z_score', 0),
                    features.get('price_change_pct', 0),
                    features.get('time_gap_sec', 0)
                ])

                # --- Benchmark: detection timer start ---
                det_start = time.perf_counter()
                anomalies = self.detect_anomalies(features)
                det_end = time.perf_counter()
                self.benchmark.log_event(
                    'detection',
                    trade_id,
                    duration=det_end - det_start
                )

                # Process detected anomalies
                if anomalies:
                    self.anomaly_counter += len(anomalies)
                    # Benchmark: anomaly event
                    self.benchmark.log_event('anomaly', trade_id)
                    for anomaly_type in anomalies:
                        # Log anomaly
                        self.data_writer.log_anomaly(features, anomaly_type, trade_id)

                        # Send alert
                        self.alert_manager.send_alert(features, anomaly_type)

                        # Benchmark: alert sent
                        self.benchmark.log_event('alert', trade_id)

                        logger.warning(f"🚨 {anomaly_type.upper()} anomaly detected!")

                # Retrain model periodically
                if not self.is_model_fitted and len(self.buffer_for_training) >= 100:
                    logger.info("📊 Sufficient data collected for initial model training")
                    self.update_model()
                elif self.trade_counter % config.PIPELINE_CONFIG["retrain_interval"] == 0:
                    self.update_model()

                # Print status
                self.print_status()

                # Optional sleep
                if config.PIPELINE_CONFIG["sleep_between_trades"]:
                    time.sleep(config.PIPELINE_CONFIG["sleep_duration"])

            except Exception as e:
                logger.error(f"Error processing trade: {e}")
                continue

    def shutdown(self):
        """Clean shutdown"""
        try:
            self.consumer.close()
            logger.info("✅ Detector shutdown complete")

            # Print final statistics
            if self.trade_counter > 0:
                anomaly_rate = self.anomaly_counter / self.trade_counter * 100
                logger.info(f"""
                📊 FINAL STATISTICS
                ├─ Total trades: {self.trade_counter:,}
                ├─ Total anomalies: {self.anomaly_counter:,}
                └─ Detection rate: {anomaly_rate:.2f}%
                """)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    detector = AnomalyDetector()
    try:
        detector.run()
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        detector.shutdown()
