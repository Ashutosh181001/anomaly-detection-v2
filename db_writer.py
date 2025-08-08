"""
Database Writer Module
Handles data persistence to SQLite or CSV
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Optional
from contextlib import contextmanager
import logging
import config

logger = logging.getLogger(__name__)


class DataWriter:
    """Handles data persistence with SQLite preference and CSV fallback"""

    def __init__(self):
        self.use_db = config.STORAGE_CONFIG["use_database"]
        self.db_path = config.STORAGE_CONFIG["database_path"]
        self.trades_csv = config.STORAGE_CONFIG["trades_csv"]
        self.anomalies_csv = config.STORAGE_CONFIG["anomalies_csv"]

        if self.use_db:
            self._init_database()

    def _init_database(self):
        """Initialize SQLite database with tables"""
        try:
            with self.get_connection() as conn:
                # Trades table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        is_buyer_maker INTEGER,
                        z_score REAL,
                        rolling_mean REAL,
                        rolling_std REAL,
                        price_change_pct REAL,
                        time_gap_sec REAL,
                        volume_ratio REAL,
                        buy_pressure REAL,
                        vwap REAL,
                        vwap_deviation REAL,
                        price_velocity REAL,
                        volume_spike REAL,
                        hour INTEGER,
                        day_of_week INTEGER,
                        trading_session TEXT,
                        is_weekend INTEGER,
                        injected INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Anomalies table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS anomalies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id INTEGER,
                        timestamp DATETIME NOT NULL,
                        anomaly_type TEXT NOT NULL,
                        price REAL,
                        z_score REAL,
                        price_change_pct REAL,
                        volume_spike REAL,
                        vwap_deviation REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (trade_id) REFERENCES trades(id)
                    )
                ''')

                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_type ON anomalies(anomaly_type)')

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.use_db = False

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def log_trade(self, features: Dict) -> Optional[int]:
        """
        Log trade to database or CSV

        Returns trade_id if using database, None otherwise
        """
        if self.use_db:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        INSERT INTO trades (
                            timestamp, price, quantity, is_buyer_maker,
                            z_score, rolling_mean, rolling_std,
                            price_change_pct, time_gap_sec, volume_ratio,
                            buy_pressure, vwap, vwap_deviation,
                            price_velocity, volume_spike,
                            hour, day_of_week, trading_session, is_weekend,
                            injected
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        features['timestamp'],
                        features['price'],
                        features['quantity'],
                        features.get('is_buyer_maker', 0),
                        features.get('z_score', 0),
                        features.get('rolling_mean', 0),
                        features.get('rolling_std', 0),
                        features.get('price_change_pct', 0),
                        features.get('time_gap_sec', 0),
                        features.get('volume_ratio', 1),
                        features.get('buy_pressure', 0.5),
                        features.get('vwap', features['price']),
                        features.get('vwap_deviation', 0),
                        features.get('price_velocity', 0),
                        features.get('volume_spike', 1),
                        features.get('hour', 0),
                        features.get('day_of_week', 0),
                        features.get('trading_session', 'unknown'),
                        features.get('is_weekend', 0),
                        features.get('injected', 0)
                    ))

                    conn.commit()
                    return cursor.lastrowid

            except Exception as e:
                logger.error(f"Database trade insertion failed: {e}")
                self.use_db = False  # Fallback to CSV

        # CSV fallback
        try:
            pd.DataFrame([features]).to_csv(
                self.trades_csv,
                mode='a',
                header=not os.path.exists(self.trades_csv),
                index=False
            )
        except Exception as e:
            logger.error(f"CSV trade logging failed: {e}")

        return None

    def log_anomaly(self, features: Dict, method: str, trade_id: Optional[int] = None):
        """Log anomaly to database or CSV"""

        # Prepare anomaly record
        anomaly_record = {
            'timestamp': features['timestamp'],
            'anomaly_type': method + ("_injected" if features.get('injected') else ""),
            'price': features['price'],
            'z_score': features.get('z_score', 0),
            'price_change_pct': features.get('price_change_pct', 0),
            'volume_spike': features.get('volume_spike', 1),
            'vwap_deviation': features.get('vwap_deviation', 0),
        }

        if self.use_db and trade_id is not None:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute('''
                        INSERT INTO anomalies (
                            trade_id, timestamp, anomaly_type,
                            price, z_score, price_change_pct,
                            volume_spike, vwap_deviation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade_id,
                        anomaly_record['timestamp'],
                        anomaly_record['anomaly_type'],
                        anomaly_record['price'],
                        anomaly_record['z_score'],
                        anomaly_record['price_change_pct'],
                        anomaly_record['volume_spike'],
                        anomaly_record['vwap_deviation']
                    ))

                    conn.commit()
                    return

            except Exception as e:
                logger.error(f"Database anomaly insertion failed: {e}")
                self.use_db = False

        # CSV fallback
        try:
            pd.DataFrame([anomaly_record]).to_csv(
                self.anomalies_csv,
                mode='a',
                header=not os.path.exists(self.anomalies_csv),
                index=False
            )
        except Exception as e:
            logger.error(f"CSV anomaly logging failed: {e}")

    def get_recent_trades(self, minutes: int = 60) -> pd.DataFrame:
        """Get recent trades from storage"""
        if self.use_db:
            try:
                with self.get_connection() as conn:
                    query = '''
                        SELECT * FROM trades
                        WHERE datetime(timestamp) >= datetime('now', '-{} minutes')
                        ORDER BY timestamp DESC
                    '''.format(minutes)

                    return pd.read_sql_query(query, conn)
            except Exception as e:
                logger.error(f"Failed to read from database: {e}")

        # CSV fallback
        if os.path.exists(self.trades_csv):
            try:
                df = pd.read_csv(self.trades_csv)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cutoff = pd.Timestamp.now() - pd.Timedelta(minutes=minutes)
                return df[df['timestamp'] >= cutoff].sort_values('timestamp', ascending=False)
            except Exception as e:
                logger.error(f"Failed to read from CSV: {e}")

        return pd.DataFrame()

    def get_recent_anomalies(self, minutes: int = 60) -> pd.DataFrame:
        """Get recent anomalies from storage"""
        if self.use_db:
            try:
                with self.get_connection() as conn:
                    query = '''
                        SELECT * FROM anomalies
                        WHERE datetime(timestamp) >= datetime('now', '-{} minutes')
                        ORDER BY timestamp DESC
                    '''.format(minutes)

                    return pd.read_sql_query(query, conn)
            except Exception as e:
                logger.error(f"Failed to read anomalies from database: {e}")

        # CSV fallback
        if os.path.exists(self.anomalies_csv):
            try:
                df = pd.read_csv(self.anomalies_csv)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cutoff = pd.Timestamp.now() - pd.Timedelta(minutes=minutes)
                return df[df['timestamp'] >= cutoff].sort_values('timestamp', ascending=False)
            except Exception as e:
                logger.error(f"Failed to read anomalies from CSV: {e}")

        return pd.DataFrame()