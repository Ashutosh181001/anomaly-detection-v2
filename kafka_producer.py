"""
Kafka Producer Module
Streams BTC/USDT trades from Binance to Kafka
"""

import asyncio
import json
import websockets
from kafka import KafkaProducer
from datetime import datetime
import numpy as np
import time
import logging
import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG["level"]),
    format=config.LOGGING_CONFIG["format"],
    datefmt=config.LOGGING_CONFIG["date_format"]
)
logger = logging.getLogger(__name__)

# Binance WebSocket URL
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"


class BinanceKafkaProducer:
    """Streams Binance trades to Kafka with optional synthetic anomaly injection"""

    def __init__(self):
        # Load configuration
        self.kafka_config = config.KAFKA_CONFIG
        self.evaluation_config = config.EVALUATION_CONFIG

        # Initialize Kafka producer
        self.producer = self._init_kafka_producer()

        # Statistics
        self.total_trades = 0
        self.synthetic_anomalies = 0
        self.connection_attempts = 0
        self.last_status_time = time.time()

    def _init_kafka_producer(self):
        """Initialize Kafka producer"""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.kafka_config["broker"],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=5,
                retry_backoff_ms=100,
                request_timeout_ms=30000,
                api_version=(0, 10, 1)
            )
            logger.info("✅ Kafka producer initialized successfully")
            return producer
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def generate_synthetic_anomaly(self, original_price: float) -> dict:
        """Generate synthetic anomaly for academic evaluation"""
        anomaly_types = [
            {"multiplier": 1.05, "type": "price_spike_5pct"},
            {"multiplier": 0.95, "type": "price_drop_5pct"},
            {"multiplier": 1.08, "type": "price_spike_8pct"},
            {"multiplier": 0.92, "type": "price_drop_8pct"},
        ]

        anomaly = np.random.choice(anomaly_types)
        modified_price = original_price * anomaly["multiplier"]

        return {
            "modified_price": modified_price,
            "type": anomaly["type"],
            "multiplier": anomaly["multiplier"]
        }

    async def connect_with_retry(self, max_retries: int = 5):
        """Connect to Binance WebSocket with retry logic"""
        while self.connection_attempts < max_retries:
            try:
                self.connection_attempts += 1
                logger.info(f"Connecting to Binance (attempt {self.connection_attempts}/{max_retries})...")

                websocket = await websockets.connect(
                    BINANCE_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                )

                logger.info("✅ Connected to Binance WebSocket")
                self.connection_attempts = 0
                return websocket

            except Exception as e:
                logger.error(f"Connection attempt {self.connection_attempts} failed: {e}")

                if self.connection_attempts >= max_retries:
                    raise

                wait_time = min(2 ** self.connection_attempts, 30)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)

    def print_status(self):
        """Print periodic status update"""
        current_time = time.time()

        if current_time - self.last_status_time >= 30:  # Every 30 seconds
            if self.evaluation_config["enable_synthetic_injection"]:
                injection_rate = (
                    self.synthetic_anomalies / self.total_trades * 100
                    if self.total_trades > 0 else 0
                )

                logger.info(f"""
                📊 PRODUCER STATUS
                ├─ Total trades: {self.total_trades:,}
                ├─ Synthetic anomalies: {self.synthetic_anomalies:,}
                ├─ Injection rate: {injection_rate:.2f}%
                └─ Target rate: {self.evaluation_config['synthetic_anomaly_rate'] * 100:.1f}%
                """)
            else:
                logger.info(f"📊 Total trades produced: {self.total_trades:,}")

            self.last_status_time = current_time

    async def process_trades(self, websocket):
        """Process incoming trades and send to Kafka"""
        logger.info("Starting to process trades from Binance...")

        while True:
            try:
                # Receive message
                msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                data = json.loads(msg)

                # Log first trade for verification
                if self.total_trades == 0:
                    logger.info(f"First trade received: {data.get('s')} @ ${float(data['p']):,.2f}")

                # Extract trade data
                original_price = float(data['p'])
                current_price = original_price
                injected = False
                anomaly_info = None

                # Inject synthetic anomaly if enabled
                if self.evaluation_config["enable_synthetic_injection"]:
                    if np.random.random() < self.evaluation_config["synthetic_anomaly_rate"]:
                        anomaly_info = self.generate_synthetic_anomaly(original_price)
                        current_price = anomaly_info["modified_price"]
                        injected = True
                        self.synthetic_anomalies += 1

                        logger.warning(
                            f"🚨 Synthetic anomaly #{self.synthetic_anomalies}: "
                            f"{anomaly_info['type']} - "
                            f"${original_price:,.2f} → ${current_price:,.2f}"
                        )

                # Create trade record
                trade_record = {
                    "timestamp": datetime.utcfromtimestamp(data['T'] / 1000).isoformat(),
                    "price": current_price,
                    "quantity": float(data['q']),
                    "trade_id": data['t'],
                    "is_buyer_maker": data['m'],
                    "injected": injected
                }

                # Send to Kafka
                try:
                    future = self.producer.send(self.kafka_config["topic"], value=trade_record)
                    # Don't block on every message for performance
                    self.total_trades += 1

                    # Print status every 10 trades initially, then every 100
                    if self.total_trades <= 10 or self.total_trades % 100 == 0:
                        logger.info(f"✓ Trade #{self.total_trades}: ${current_price:,.2f} - {trade_record['quantity']:.8f} BTC")

                except Exception as kafka_error:
                    logger.error(f"Failed to send to Kafka: {kafka_error}")
                    # Try to recreate producer if it failed
                    try:
                        self.producer = self._init_kafka_producer()
                    except:
                        pass

                self.print_status()

            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout - checking connection...")
                try:
                    pong = await websocket.ping()
                    logger.debug("WebSocket ping successful")
                except:
                    logger.error("WebSocket ping failed - connection lost")
                    break

            except websockets.ConnectionClosed as e:
                logger.error(f"WebSocket connection closed: {e}")
                break

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.debug(f"Raw message: {msg[:200]}")
                continue

            except Exception as e:
                logger.error(f"Unexpected error processing trade: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

    async def run(self):
        """Main producer loop"""
        logger.info("=" * 60)
        logger.info("STARTING KAFKA PRODUCER")
        logger.info(f"Topic: {self.kafka_config['topic']}")
        logger.info(f"Broker: {self.kafka_config['broker']}")
        logger.info(f"Asset: {config.DATA_CONFIG['asset_pair']}")

        if self.evaluation_config["enable_synthetic_injection"]:
            logger.info(f"Synthetic injection: ENABLED ({self.evaluation_config['synthetic_anomaly_rate'] * 100:.1f}%)")
        else:
            logger.info("Synthetic injection: DISABLED")
        logger.info("=" * 60)

        while True:
            try:
                websocket = await self.connect_with_retry()
                await self.process_trades(websocket)

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.info("Restarting in 5 seconds...")
                await asyncio.sleep(5)

            finally:
                try:
                    if 'websocket' in locals():
                        await websocket.close()
                except:
                    pass

    def shutdown(self):
        """Clean shutdown"""
        try:
            self.producer.flush()
            self.producer.close()

            # Print final statistics
            logger.info("=" * 60)
            logger.info("PRODUCER SHUTDOWN COMPLETE")
            logger.info(f"Total trades: {self.total_trades:,}")

            if self.evaluation_config["enable_synthetic_injection"]:
                injection_rate = (
                    self.synthetic_anomalies / self.total_trades * 100
                    if self.total_trades > 0 else 0
                )
                logger.info(f"Synthetic anomalies: {self.synthetic_anomalies:,}")
                logger.info(f"Actual injection rate: {injection_rate:.2f}%")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point"""
    producer = BinanceKafkaProducer()

    try:
        await producer.run()
    except KeyboardInterrupt:
        logger.info("\n🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"Producer crashed: {e}")
    finally:
        producer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())