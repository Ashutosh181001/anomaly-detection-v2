import time
import csv
import json
import atexit

# Optional system metrics
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _current_timestamp():
    """Return ISO-formatted UTC timestamp."""
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class BenchmarkLogger:
    """
    Tracks performance metrics for the anomaly detection pipeline.

    Metrics tracked:
      - total trades processed
      - total anomalies detected
      - total alerts sent
      - cumulative processing time per trade
      - cumulative detection time per trade
      - periodic writes to CSV every N trades
      - final summary dump to JSON on shutdown
      - optional CPU and memory usage snapshots
    """

    def __init__(self,
                 csv_path='performance_log.csv',
                 json_path='benchmark_summary.json',
                 report_interval=1000):
        """
        Initialize the benchmark logger.

        Args:
            csv_path (str): Path to write periodic metrics CSV.
            json_path (str): Path to write final summary JSON on shutdown.
            report_interval (int): Number of trades between periodic reports.
        """
        self.csv_path = csv_path
        self.json_path = json_path
        self.report_interval = report_interval

        # Counters and accumulators
        self.total_trades = 0
        self.total_anomalies = 0
        self.total_alerts = 0
        self._sum_processing_time = 0.0
        self._sum_detection_time = 0.0

        # Initialize CSV file with header
        self._init_csv()

        # Ensure final summary is dumped on interpreter exit
        atexit.register(self._shutdown)

    def _init_csv(self):
        """Write CSV header. Overwrites existing file."""
        header = [
            'timestamp',
            'total_trades',
            'total_anomalies',
            'detection_rate_pct',
            'avg_processing_time_sec',
            'avg_detection_time_sec',
            'total_alerts',
            'memory_usage_mb',
            'cpu_usage_pct'
        ]
        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        except Exception as e:
            print(f"[BenchmarkLogger] Failed to initialize CSV: {e}")

    def log_event(self, event_type, trade_id=None, duration=None):
        """
        Log a benchmarking event.

        event_type (str): One of 'trade', 'detection', 'anomaly', 'alert'.
        trade_id (any): Identifier for the trade (optional).
        duration (float): Duration in seconds for processing or detection.
        """
        if event_type == 'trade':
            self.total_trades += 1
            if duration is not None:
                self._sum_processing_time += duration

        elif event_type == 'detection':
            if duration is not None:
                self._sum_detection_time += duration

        elif event_type == 'anomaly':
            self.total_anomalies += 1

        elif event_type == 'alert':
            self.total_alerts += 1

        else:
            # Unknown event type
            return

        # Periodic report based on number of trades
        if self.total_trades > 0 and (self.total_trades % self.report_interval == 0):
            self._write_report()

    def _write_report(self):
        """Append a row of aggregated metrics to the CSV."""
        try:
            detection_rate = (self.total_anomalies / self.total_trades * 100)
        except ZeroDivisionError:
            detection_rate = 0.0
        try:
            avg_proc = self._sum_processing_time / self.total_trades
        except ZeroDivisionError:
            avg_proc = 0.0
        try:
            avg_det = self._sum_detection_time / self.total_trades
        except ZeroDivisionError:
            avg_det = 0.0

        # System metrics snapshot
        mem_mb = None
        cpu_pct = None
        if _HAS_PSUTIL:
            try:
                proc = psutil.Process()
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                cpu_pct = psutil.cpu_percent(interval=None)
            except Exception:
                mem_mb = None
                cpu_pct = None

        row = [
            _current_timestamp(),
            self.total_trades,
            self.total_anomalies,
            f"{detection_rate:.2f}",
            f"{avg_proc:.6f}",
            f"{avg_det:.6f}",
            self.total_alerts,
            f"{mem_mb:.2f}" if mem_mb is not None else '',
            f"{cpu_pct:.2f}" if cpu_pct is not None else ''
        ]

        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"[BenchmarkLogger] Failed to write report: {e}")

    def _shutdown(self):
        """Dump final summary to JSON file on shutdown."""
        try:
            summary = {
                'total_trades': self.total_trades,
                'total_anomalies': self.total_anomalies,
                'detection_rate_pct': (
                    self.total_anomalies / self.total_trades * 100
                    if self.total_trades else 0.0
                ),
                'avg_processing_time_sec': (
                    self._sum_processing_time / self.total_trades
                    if self.total_trades else 0.0
                ),
                'avg_detection_time_sec': (
                    self._sum_detection_time / self.total_trades
                    if self.total_trades else 0.0
                ),
                'total_alerts': self.total_alerts
            }
            with open(self.json_path, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"[BenchmarkLogger] Failed to write summary JSON: {e}")
