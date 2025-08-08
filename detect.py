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

        # Check if loaded model is already fitted
        if self.model is not None:
            try:
                # Test if model can make predictions
                self.model.predict([[0, 0, 0]])
                self.is_model_fitted = True
                logger.info("✅ Model is fitted and ready for predictions")
            except:
                self.is_model_fitted = False
                logger.info("⏳ Model needs training - will train after collecting sufficient data")

        # Initialize buffers
        self.history_df = pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'is_buyer_maker'])
        self.buffer_for_training = []

        # Counters
        self.trade_counter = 0
        self.anomaly_counter = 0
        self.last_status_time = time.time()

        # Load historical data
        self._load_historical_data()

        # Initialize Kafka consumer
        self.consumer = self._init_kafka_consumer()

    def _load_model(self):
        """Load pre-trained model or initialize new one"""
        # Try loading tuned model first
        if os.path.exists(config.MODEL_PATHS["tuned_model"]):
            try:
                model = joblib.load(config.MODEL_PATHS["tuned_model"])
                logger.info("✅ Loaded tuned Isolation Forest model")
                # Check if model is fitted
                try:
                    # Test prediction with dummy data
                    model.predict([[0, 0, 0]])
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
                # Check if model is fitted
                try:
                    # Test prediction with dummy data
                    model.predict([[0, 0, 0]])
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
                # Take last N trades for history
                window_size = config.DATA_CONFIG["rolling_window"]
                self.history_df = recent_trades[
                    ['timestamp', 'price', 'quantity', 'is_buyer_maker']
                ].tail(window_size).copy()

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

        # Isolation Forest detection - only if model is fitted and we have enough buffer
        if self.is_model_fitted and len(self.buffer_for_training) >= config.DATA_CONFIG["min_history"]:
            try:
                # Prepare features for model
                model_features = [
                    features.get('z_score', 0),
                    features.get('price_change_pct', 0),
                    features.get('time_gap_sec', 0)
                ]

                # Predict
                prediction = self.model.predict([model_features])[0]

                if prediction == -1:  # Anomaly
                    if "z_score" in detected:
                        detected.append("filtered_isoforest")
                    else:
                        detected.append("isoforest")

            except Exception as e:
                logger.error(f"Model prediction failed: {e}")

        return detected

    def update_model(self):
        """Retrain model with recent data"""
        min_samples = 100  # Minimum samples needed to train

        if len(self.buffer_for_training) < min_samples:
            logger.debug(f"Not enough data for training: {len(self.buffer_for_training)}/{min_samples}")
            return

        try:
            # Get recent training data
            recent_data = np.array(
                self.buffer_for_training[-config.DATA_CONFIG["rolling_window"]:]
                if len(self.buffer_for_training) > config.DATA_CONFIG["rolling_window"]
                else self.buffer_for_training
            )

            # Fit model
            self.model.fit(recent_data)
            self.is_model_fitted = True

            # Save model
            joblib.dump(self.model, config.MODEL_PATHS["base_model"])
            logger.info(f"🔄 Model trained with {len(recent_data)} samples and saved")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    def print_status(self):
        """Print periodic status update"""
        current_time = time.time()

        if current_time - self.last_status_time >= 30:  # Every 30 seconds
            anomaly_rate = (
                self.anomaly_counter / self.trade_counter * 100
                if self.trade_counter > 0 else 0
            )

            logger.info(f"""
            📊 STATUS UPDATE
            ├─ Trades processed: {self.trade_counter:,}
            ├─ Anomalies detected: {self.anomaly_counter:,}
            ├─ Detection rate: {anomaly_rate:.2f}%
            └─ Model fitted: {self.is_model_fitted}
            """)

            self.last_status_time = current_time

    def run(self):
        """Main detection loop"""
        logger.info("🚀 Starting anomaly detection pipeline...")
        logger.info(f"📍 Monitoring {config.DATA_CONFIG['asset_pair']} trades")

        for message in self.consumer:
            try:
                trade = message.value

                # Compute features
                features, self.history_df = compute_features(
                    trade,
                    self.history_df,
                    config.DATA_CONFIG["rolling_window"]
                )

                # Log trade
                trade_id = self.data_writer.log_trade(features)
                self.trade_counter += 1

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

                # Detect anomalies
                anomalies = self.detect_anomalies(features)

                # Process detected anomalies
                if anomalies:
                    self.anomaly_counter += len(anomalies)

                    for anomaly_type in anomalies:
                        # Log anomaly
                        self.data_writer.log_anomaly(features, anomaly_type, trade_id)

                        # Send alert
                        self.alert_manager.send_alert(features, anomaly_type)

                        logger.warning(f"🚨 {anomaly_type.upper()} anomaly detected!")

                # Retrain model periodically or when we have enough initial data
                if not self.is_model_fitted and len(self.buffer_for_training) >= 100:
                    # Initial training when we have enough data
                    logger.info("📊 Sufficient data collected for initial model training")
                    self.update_model()
                elif self.trade_counter % config.PIPELINE_CONFIG["retrain_interval"] == 0:
                    # Periodic retraining
                    self.update_model()

                # Print status
                self.print_status()

                # Optional sleep for debugging
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