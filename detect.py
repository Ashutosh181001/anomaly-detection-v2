"""
Main Detection Module
Real-time anomaly detection pipeline for multiple crypto pairs
Supports: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, XRP/USDT
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# Supported symbols
SUPPORTED_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]


class MultiSymbolAnomalyDetector:
    """Multi-symbol anomaly detection pipeline with cooldown"""

    def __init__(self):
        # Initialize components
        self.data_writer = DataWriter()
        self.alert_manager = AlertManager()

        # Load configuration
        self.cfg = config.get_config()

        # Initialize models and buffers for each symbol
        self.models = {}
        self.is_model_fitted = {}
        self.history_dict = {}
        self.buffer_for_training = {}
        self.last_anomaly_time = {}  # For cooldown tracking

        # Initialize counters per symbol
        self.trade_counter = {symbol: 0 for symbol in SUPPORTED_SYMBOLS}
        self.anomaly_counter = {symbol: 0 for symbol in SUPPORTED_SYMBOLS}

        # Global counters
        self.total_trades = 0
        self.last_status_time = time.time()

        # Cooldown period in seconds (configurable)
        self.anomaly_cooldown = config.ANOMALY_CONFIG.get("cooldown_seconds", 60)

        # Initialize benchmarking
        self.benchmark = BenchmarkLogger(
            csv_path='performance_log.csv',
            json_path='benchmark_summary.json',
            report_interval=100
        )

        # Initialize models for each symbol
        self._init_models()

        # Load historical data for each symbol
        self._load_historical_data()

        # Initialize Kafka consumer
        self.consumer = self._init_kafka_consumer()

    def _init_models(self):
        """Initialize models for each symbol"""
        base_model = self._load_model()

        for symbol in SUPPORTED_SYMBOLS:
            # For now, use the same model configuration for all symbols
            # In future, could load symbol-specific models
            if base_model is not None:
                # Clone the model for each symbol
                self.models[symbol] = IsolationForest(
                    n_estimators=config.ANOMALY_CONFIG["n_estimators"],
                    contamination=config.ANOMALY_CONFIG["contamination"],
                    random_state=config.ANOMALY_CONFIG["random_state"]
                )

                # Check if we can copy fitted state (if base model was fitted)
                try:
                    _ = base_model.predict([[0, 0, 0]])
                    # If base model is fitted, we'll mark for retraining per symbol
                    self.is_model_fitted[symbol] = False
                    logger.info(f"⏳ {symbol} model needs training")
                except:
                    self.is_model_fitted[symbol] = False
            else:
                # Initialize new model for symbol
                self.models[symbol] = IsolationForest(
                    n_estimators=config.ANOMALY_CONFIG["n_estimators"],
                    contamination=config.ANOMALY_CONFIG["contamination"],
                    random_state=config.ANOMALY_CONFIG["random_state"]
                )
                self.is_model_fitted[symbol] = False

            # Initialize buffers
            self.history_dict[symbol] = pd.DataFrame(
                columns=['timestamp', 'price', 'quantity', 'is_buyer_maker']
            )
            self.buffer_for_training[symbol] = []
            self.last_anomaly_time[symbol] = {}  # Track by anomaly type

        logger.info(f"✅ Initialized models for {len(SUPPORTED_SYMBOLS)} symbols")

    def _load_model(self):
        """Load or initialize the base anomaly detection model"""
        # Try tuned model
        if os.path.exists(config.MODEL_PATHS["tuned_model"]):
            try:
                model = joblib.load(config.MODEL_PATHS["tuned_model"])
                logger.info("✅ Loaded tuned Isolation Forest model as base")
                return model
            except Exception as e:
                logger.error(f"Failed to load tuned model: {e}")

        # Try base model
        if os.path.exists(config.MODEL_PATHS["base_model"]):
            try:
                model = joblib.load(config.MODEL_PATHS["base_model"])
                logger.info("✅ Loaded base Isolation Forest model")
                return model
            except Exception as e:
                logger.error(f"Failed to load base model: {e}")

        # Return None to signal new model needed
        logger.info("🔄 No pre-trained model found, will train per symbol")
        return None

    def _load_historical_data(self):
        """Load historical trades for warm start for each symbol"""
        for symbol in SUPPORTED_SYMBOLS:
            try:
                recent_trades = self.data_writer.get_recent_trades(symbol=symbol, minutes=60)
                if not recent_trades.empty:
                    window = config.DATA_CONFIG["rolling_window"]
                    self.history_dict[symbol] = recent_trades[
                        ['timestamp', 'price', 'quantity', 'is_buyer_maker']
                    ].tail(window).copy()
                    logger.info(f"📊 Loaded {len(self.history_dict[symbol])} historical trades for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load historical data for {symbol}: {e}")

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

    def check_cooldown(self, symbol: str, anomaly_type: str) -> bool:
        """Check if cooldown period has passed for this anomaly type"""
        if symbol not in self.last_anomaly_time:
            return True

        if anomaly_type not in self.last_anomaly_time[symbol]:
            return True

        last_time = self.last_anomaly_time[symbol][anomaly_type]
        current_time = time.time()

        return (current_time - last_time) >= self.anomaly_cooldown

    def update_cooldown(self, symbol: str, anomaly_type: str):
        """Update the last anomaly time for cooldown tracking"""
        if symbol not in self.last_anomaly_time:
            self.last_anomaly_time[symbol] = {}

        self.last_anomaly_time[symbol][anomaly_type] = time.time()

    def detect_anomalies(self, features: dict, symbol: str) -> list:
        """
        Detect anomalies using z-score and Isolation Forest for a specific symbol
        Returns list of detected anomaly types
        """
        detected = []

        # Z-score detection
        if abs(features.get('z_score', 0)) > config.ANOMALY_CONFIG["z_score_threshold"]:
            detected.append("z_score")

        # Isolation Forest detection
        if (self.is_model_fitted[symbol] and
            len(self.buffer_for_training[symbol]) >= config.DATA_CONFIG["min_history"]):
            try:
                model_feats = [
                    features.get('z_score', 0),
                    features.get('price_change_pct', 0),
                    features.get('time_gap_sec', 0)
                ]
                pred = self.models[symbol].predict([model_feats])[0]
                if pred == -1:
                    tag = "filtered_isoforest" if "z_score" in detected else "isoforest"
                    detected.append(tag)
            except Exception as e:
                logger.error(f"Model prediction failed for {symbol}: {e}")

        return detected

    def update_model(self, symbol: str):
        """Retrain model for a specific symbol"""
        min_samples = 100
        if len(self.buffer_for_training[symbol]) < min_samples:
            return

        try:
            data = np.array(
                self.buffer_for_training[symbol][-config.DATA_CONFIG["rolling_window"]:]
                if len(self.buffer_for_training[symbol]) > config.DATA_CONFIG["rolling_window"]
                else self.buffer_for_training[symbol]
            )

            self.models[symbol].fit(data)
            self.is_model_fitted[symbol] = True

            # Save model with symbol suffix
            model_path = config.MODEL_PATHS["base_model"].replace('.pkl', f'_{symbol.replace("/", "_")}.pkl')
            joblib.dump(self.models[symbol], model_path)

            logger.info(f"🔄 {symbol} model trained with {len(data)} samples and saved")
        except Exception as e:
            logger.error(f"Model retraining failed for {symbol}: {e}")

    def print_status(self):
        """Print periodic status update"""
        now = time.time()
        if now - self.last_status_time >= 30:
            logger.info("=" * 60)
            logger.info("📊 MULTI-SYMBOL DETECTOR STATUS")

            for symbol in SUPPORTED_SYMBOLS:
                trades = self.trade_counter[symbol]
                anomalies = self.anomaly_counter[symbol]
                fitted = self.is_model_fitted[symbol]

                if trades > 0:
                    rate = (anomalies / trades * 100)
                    logger.info(
                        f"├─ {symbol}: {trades:,} trades, {anomalies} anomalies ({rate:.2f}%), "
                        f"Model: {'✓' if fitted else '⏳'}"
                    )

            logger.info(f"└─ TOTAL: {self.total_trades:,} trades")
            logger.info("=" * 60)

            self.last_status_time = now

    def run(self):
        """Main detection loop"""
        logger.info("🚀 Starting multi-symbol anomaly detection pipeline...")
        logger.info(f"📍 Monitoring: {', '.join(SUPPORTED_SYMBOLS)}")
        logger.info(f"⏱️ Anomaly cooldown: {self.anomaly_cooldown} seconds")

        for message in self.consumer:
            try:
                trade = message.value

                # Extract symbol (new field from producer)
                symbol = trade.get('symbol', 'BTC/USDT')  # Default to BTC if missing

                # Skip if unsupported symbol
                if symbol not in SUPPORTED_SYMBOLS:
                    logger.warning(f"Unsupported symbol: {symbol}")
                    continue

                # --- Benchmark: processing timer start ---
                proc_start = time.perf_counter()

                # Compute features for this symbol
                features, self.history_dict[symbol] = compute_features(
                    trade,
                    self.history_dict[symbol],
                    config.DATA_CONFIG["rolling_window"]
                )

                # Add symbol to features
                features['symbol'] = symbol

                # Log trade
                trade_id = self.data_writer.log_trade(features)
                self.trade_counter[symbol] += 1
                self.total_trades += 1

                # --- Benchmark: processing timer end & log ---
                proc_end = time.perf_counter()
                self.benchmark.log_event(
                    'trade',
                    trade_id,
                    duration=proc_end - proc_start
                )

                # Print trade info periodically
                if self.trade_counter[symbol] % config.PIPELINE_CONFIG["status_interval"] == 0:
                    logger.info(
                        f"[{symbol}] {features['timestamp']} | "
                        f"Price: ${features['price']:,.2f} | "
                        f"Z: {features.get('z_score', 0):.2f} | "
                        f"Δ%: {features.get('price_change_pct', 0):.2f}"
                    )

                # Skip if not enough history for this symbol
                if len(self.history_dict[symbol]) < config.DATA_CONFIG["min_history"]:
                    continue

                # Update training buffer for this symbol
                self.buffer_for_training[symbol].append([
                    features.get('z_score', 0),
                    features.get('price_change_pct', 0),
                    features.get('time_gap_sec', 0)
                ])

                # --- Benchmark: detection timer start ---
                det_start = time.perf_counter()
                anomalies = self.detect_anomalies(features, symbol)
                det_end = time.perf_counter()
                self.benchmark.log_event(
                    'detection',
                    trade_id,
                    duration=det_end - det_start
                )

                # Process detected anomalies with cooldown
                if anomalies:
                    for anomaly_type in anomalies:
                        # Check cooldown
                        if self.check_cooldown(symbol, anomaly_type):
                            # Log anomaly
                            self.data_writer.log_anomaly(features, anomaly_type, trade_id)
                            self.anomaly_counter[symbol] += 1

                            # Send alert
                            self.alert_manager.send_alert(features, anomaly_type)

                            # Update cooldown
                            self.update_cooldown(symbol, anomaly_type)

                            # Benchmark: anomaly & alert events
                            self.benchmark.log_event('anomaly', trade_id)
                            self.benchmark.log_event('alert', trade_id)

                            logger.warning(
                                f"🚨 [{symbol}] {anomaly_type.upper()} anomaly detected! "
                                f"Price: ${features['price']:,.2f}"
                            )
                        else:
                            # Cooldown active
                            time_left = self.anomaly_cooldown - (
                                time.time() - self.last_anomaly_time[symbol][anomaly_type]
                            )
                            logger.debug(
                                f"[{symbol}] {anomaly_type} anomaly in cooldown "
                                f"({time_left:.0f}s remaining)"
                            )

                # Retrain model periodically for each symbol
                if not self.is_model_fitted[symbol] and len(self.buffer_for_training[symbol]) >= 100:
                    logger.info(f"📊 [{symbol}] Sufficient data for initial model training")
                    self.update_model(symbol)
                elif self.trade_counter[symbol] % config.PIPELINE_CONFIG["retrain_interval"] == 0:
                    self.update_model(symbol)

                # Print status
                if self.total_trades % 500 == 0:
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
            logger.info("=" * 60)
            logger.info("📊 FINAL STATISTICS")

            for symbol in SUPPORTED_SYMBOLS:
                trades = self.trade_counter[symbol]
                anomalies = self.anomaly_counter[symbol]

                if trades > 0:
                    rate = (anomalies / trades * 100)
                    logger.info(f"├─ {symbol}: {trades:,} trades, {anomalies} anomalies ({rate:.2f}%)")

            logger.info(f"└─ TOTAL: {self.total_trades:,} trades")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    detector = MultiSymbolAnomalyDetector()
    try:
        detector.run()
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        detector.shutdown()