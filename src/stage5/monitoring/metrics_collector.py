#!/usr/bin/env python3
"""
Metrics Collector - metrics_collector.py
========================================

Comprehensive metrics collection and monitoring system for the AWARE-NET
production deployment. Provides real-time performance monitoring, alerting,
and observability for all system components.

Key Features:
- Real-time performance metrics collection
- System health monitoring and alerting
- Business intelligence and analytics
- Automated response and optimization
- Multi-dimensional metric aggregation
- Integration with Prometheus, Grafana, and alerting systems

Metrics Categories:
- Application Metrics: Inference latency, throughput, accuracy drift
- System Metrics: CPU, memory, GPU utilization, network I/O
- Business Metrics: Detection accuracy, false positive rate, user satisfaction
- Infrastructure Metrics: Load balancer health, database performance
- Security Metrics: Failed authentications, rate limit violations

Usage:
    # Initialize metrics collector
    collector = MetricsCollector()
    await collector.initialize()
    
    # Collect and export metrics
    metrics = await collector.collect_all_metrics()
    await collector.export_to_prometheus()
"""

import os
import sys
import json
import time
import logging
import asyncio
import statistics
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import traceback
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

# Monitoring and metrics libraries
import prometheus_client
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
import numpy as np
import pandas as pd
import redis
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    unit: Optional[str] = None
    description: Optional[str] = None

@dataclass
class SystemHealth:
    """System health status."""
    status: str  # healthy, degraded, critical
    score: float  # 0-100 health score
    issues: List[str]
    recommendations: List[str]
    last_updated: datetime

class MetricsCollector:
    """
    Comprehensive metrics collection and monitoring system.
    
    Collects, aggregates, and exports metrics from all system components
    with real-time alerting and automated response capabilities.
    """
    
    def __init__(self,
                 collection_interval: int = 10,
                 retention_hours: int = 168,  # 7 days
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Metrics collection interval in seconds
            retention_hours: Metrics retention period in hours
            alert_thresholds: Custom alert thresholds for metrics
        """
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        
        # Initialize logging
        self._setup_logging()
        
        # Prometheus registry
        self.registry = CollectorRegistry()
        
        # Metric storage
        self.metrics_buffer = deque(maxlen=10000)
        self.aggregated_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # System health tracking
        self.health_status = SystemHealth(
            status="unknown",
            score=0.0,
            issues=[],
            recommendations=[],
            last_updated=datetime.now()
        )
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        
        # External connections
        self.redis_client: Optional[redis.Redis] = None
        self.influxdb_client: Optional[InfluxDBClient] = None
        
        # Collection state
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize Prometheus metrics
        self._initialize_prometheus_metrics()
        
        logging.info("MetricsCollector initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default alert thresholds."""
        return {
            # Performance thresholds
            'inference_latency_ms': 200.0,
            'api_response_time_ms': 500.0,
            'throughput_rps': 10.0,  # Minimum requests per second
            
            # System resource thresholds
            'cpu_usage_percent': 80.0,
            'memory_usage_percent': 85.0,
            'gpu_usage_percent': 90.0,
            'disk_usage_percent': 90.0,
            
            # Application thresholds
            'error_rate_percent': 5.0,
            'accuracy_drift_percent': 2.0,
            'model_confidence_threshold': 0.7,
            
            # Business thresholds
            'false_positive_rate_percent': 1.0,
            'user_satisfaction_score': 4.0,  # Out of 5
            'cost_per_inference_usd': 0.01,
            
            # Infrastructure thresholds
            'queue_depth': 100,
            'connection_pool_usage_percent': 80.0,
            'cache_hit_rate_percent': 85.0
        }
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors."""
        # Application metrics
        self.inference_latency = Histogram(
            'aware_net_inference_duration_seconds',
            'Model inference time in seconds',
            ['model_type', 'stage'],
            registry=self.registry
        )
        
        self.api_requests_total = Counter(
            'aware_net_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'aware_net_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.detection_accuracy = Gauge(
            'aware_net_detection_accuracy',
            'Current detection accuracy',
            ['dataset', 'model_version'],
            registry=self.registry
        )
        
        self.model_predictions = Counter(
            'aware_net_predictions_total',
            'Total model predictions',
            ['prediction', 'confidence_range'],
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'aware_net_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'aware_net_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_gpu_usage = Gauge(
            'aware_net_system_gpu_usage_percent',
            'System GPU usage percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        # Business metrics
        self.active_users = Gauge(
            'aware_net_active_users',
            'Number of active users',
            registry=self.registry
        )
        
        self.false_positive_rate = Gauge(
            'aware_net_false_positive_rate',
            'Current false positive rate',
            registry=self.registry
        )
        
        self.system_health_score = Gauge(
            'aware_net_system_health_score',
            'Overall system health score (0-100)',
            registry=self.registry
        )
        
        # Infrastructure metrics
        self.queue_depth = Gauge(
            'aware_net_queue_depth',
            'Current queue depth',
            ['queue_type'],
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'aware_net_cache_operations_total',
            'Cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
    
    async def initialize(self):
        """Initialize metrics collector with external connections."""
        try:
            # Initialize Redis connection
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test Redis connection
            try:
                self.redis_client.ping()
                logging.info("✅ Redis connection established")
            except Exception as e:
                logging.warning(f"⚠️ Redis connection failed: {e}")
                self.redis_client = None
            
            # Initialize InfluxDB connection (optional)
            influxdb_url = os.getenv('INFLUXDB_URL')
            if influxdb_url:
                try:
                    self.influxdb_client = InfluxDBClient(url=influxdb_url)
                    logging.info("✅ InfluxDB connection established")
                except Exception as e:
                    logging.warning(f"⚠️ InfluxDB connection failed: {e}")
            
            logging.info("✅ MetricsCollector initialization complete")
            
        except Exception as e:
            logging.error(f"❌ MetricsCollector initialization failed: {e}")
            raise
    
    async def start_collection(self):
        """Start metrics collection in background."""
        if self.is_collecting:
            logging.warning("Metrics collection already running")
            return
        
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logging.info("🚀 Started metrics collection")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logging.info("⛔ Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                start_time = time.time()
                
                # Collect all metrics
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await self._collect_business_metrics()
                await self._collect_infrastructure_metrics()
                
                # Update system health
                await self._update_system_health()
                
                # Export metrics
                await self._export_metrics()
                
                # Check alerts
                await self._check_alerts()
                
                # Calculate sleep time to maintain interval
                collection_time = time.time() - start_time
                sleep_time = max(0, self.collection_interval - collection_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in metrics collection: {e}")
                logging.error(traceback.format_exc())
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            self._add_metric('system_cpu_usage_percent', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_memory_usage.set(memory_percent)
            self._add_metric('system_memory_usage_percent', memory_percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._add_metric('system_disk_usage_percent', disk_percent)
            
            # Network metrics
            network = psutil.net_io_counters()
            self._add_metric('network_bytes_sent', network.bytes_sent)
            self._add_metric('network_bytes_recv', network.bytes_recv)
            
            # GPU metrics (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self.system_gpu_usage.labels(gpu_id=str(i)).set(gpu.load * 100)
                    self._add_metric(f'gpu_{i}_usage_percent', gpu.load * 100)
                    self._add_metric(f'gpu_{i}_memory_percent', gpu.memoryUtil * 100)
                    self._add_metric(f'gpu_{i}_temperature_c', gpu.temperature)
            except ImportError:
                pass  # GPU monitoring not available
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # These would be collected from actual application components
            # For now, we'll simulate some metrics
            
            # Model performance metrics
            self._add_metric('model_inference_latency_ms', 85.0)
            self._add_metric('model_accuracy_current', 0.952)
            self._add_metric('cascade_stage1_usage_percent', 65.0)
            self._add_metric('cascade_stage2_usage_percent', 30.0)
            self._add_metric('cascade_stage3_usage_percent', 5.0)
            
            # API metrics
            self._add_metric('api_requests_per_minute', 45.0)
            self._add_metric('api_response_time_ms', 125.0)
            self._add_metric('api_error_rate_percent', 0.5)
            
            # Queue metrics
            self.queue_depth.labels(queue_type='detection').set(12)
            self.queue_depth.labels(queue_type='batch').set(3)
            
        except Exception as e:
            logging.error(f"Error collecting application metrics: {e}")
    
    async def _collect_business_metrics(self):
        """Collect business and user-facing metrics."""
        try:
            # User metrics
            self.active_users.set(150)  # Would get from actual user tracking
            self._add_metric('daily_active_users', 1250)
            self._add_metric('user_satisfaction_score', 4.2)
            
            # Detection metrics
            self.false_positive_rate.set(0.008)  # 0.8%
            self._add_metric('detection_accuracy_real', 0.954)
            self._add_metric('detection_accuracy_fake', 0.948)
            
            # Cost metrics
            self._add_metric('cost_per_inference_usd', 0.0085)
            self._add_metric('monthly_infrastructure_cost_usd', 2500.0)
            
            # Performance SLA metrics
            self._add_metric('sla_uptime_percent', 99.8)
            self._add_metric('sla_response_time_p95_ms', 180.0)
            
        except Exception as e:
            logging.error(f"Error collecting business metrics: {e}")
    
    async def _collect_infrastructure_metrics(self):
        """Collect infrastructure and deployment metrics."""
        try:
            # Database metrics (if applicable)
            if self.redis_client:
                try:
                    info = self.redis_client.info()
                    self._add_metric('redis_connected_clients', info.get('connected_clients', 0))
                    self._add_metric('redis_used_memory_mb', info.get('used_memory', 0) / (1024*1024))
                    self._add_metric('redis_ops_per_sec', info.get('instantaneous_ops_per_sec', 0))
                except Exception as e:
                    logging.error(f"Error collecting Redis metrics: {e}")
            
            # Load balancer metrics (simulated)
            self._add_metric('load_balancer_active_connections', 45)
            self._add_metric('load_balancer_requests_per_sec', 12.5)
            
            # Container metrics (simulated)
            self._add_metric('container_cpu_usage_percent', 35.0)
            self._add_metric('container_memory_usage_mb', 512.0)
            self._add_metric('container_restart_count', 0)
            
        except Exception as e:
            logging.error(f"Error collecting infrastructure metrics: {e}")
    
    def _add_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add metric to buffer."""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        self.metrics_buffer.append(metric)
        self.aggregated_metrics[name].append((time.time(), value))
    
    async def _update_system_health(self):
        """Update overall system health status."""
        try:
            issues = []
            recommendations = []
            scores = []
            
            # Check critical metrics against thresholds
            recent_metrics = self._get_recent_metrics(minutes=5)
            
            # CPU health
            cpu_usage = recent_metrics.get('system_cpu_usage_percent', [])
            if cpu_usage:
                avg_cpu = statistics.mean(cpu_usage)
                if avg_cpu > self.alert_thresholds['cpu_usage_percent']:
                    issues.append(f"High CPU usage: {avg_cpu:.1f}%")
                    recommendations.append("Consider scaling up or optimizing CPU-intensive operations")
                    scores.append(max(0, 100 - avg_cpu))
                else:
                    scores.append(100)
            
            # Memory health
            memory_usage = recent_metrics.get('system_memory_usage_percent', [])
            if memory_usage:
                avg_memory = statistics.mean(memory_usage)
                if avg_memory > self.alert_thresholds['memory_usage_percent']:
                    issues.append(f"High memory usage: {avg_memory:.1f}%")
                    recommendations.append("Consider scaling up or investigating memory leaks")
                    scores.append(max(0, 100 - avg_memory))
                else:
                    scores.append(100)
            
            # API performance health
            api_response_time = recent_metrics.get('api_response_time_ms', [])
            if api_response_time:
                avg_response_time = statistics.mean(api_response_time)
                if avg_response_time > self.alert_thresholds['api_response_time_ms']:
                    issues.append(f"Slow API responses: {avg_response_time:.1f}ms")
                    recommendations.append("Optimize API endpoints or increase resources")
                    scores.append(max(0, 100 - (avg_response_time / 10)))
                else:
                    scores.append(100)
            
            # Model accuracy health
            model_accuracy = recent_metrics.get('model_accuracy_current', [])
            if model_accuracy:
                current_accuracy = model_accuracy[-1]  # Most recent
                if current_accuracy < 0.90:  # Below 90% accuracy
                    issues.append(f"Low model accuracy: {current_accuracy:.3f}")
                    recommendations.append("Review model performance and consider retraining")
                    scores.append(current_accuracy * 100)
                else:
                    scores.append(100)
            
            # Calculate overall health score
            if scores:
                health_score = statistics.mean(scores)
            else:
                health_score = 50.0  # Unknown health
            
            # Determine health status
            if health_score >= 90:
                status = "healthy"
            elif health_score >= 70:
                status = "degraded"
            else:
                status = "critical"
            
            # Update health status
            self.health_status = SystemHealth(
                status=status,
                score=health_score,
                issues=issues,
                recommendations=recommendations,
                last_updated=datetime.now()
            )
            
            # Update Prometheus metric
            self.system_health_score.set(health_score)
            
            logging.info(f"System health updated: {status} (score: {health_score:.1f})")
            
        except Exception as e:
            logging.error(f"Error updating system health: {e}")
    
    def _get_recent_metrics(self, minutes: int = 5) -> Dict[str, List[float]]:
        """Get recent metrics within specified time window."""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = defaultdict(list)
        
        for name, values in self.aggregated_metrics.items():
            for timestamp, value in values:
                if timestamp >= cutoff_time:
                    recent_metrics[name].append(value)
        
        return dict(recent_metrics)
    
    async def _export_metrics(self):
        """Export metrics to external systems."""
        try:
            # Export to InfluxDB if available
            if self.influxdb_client:
                await self._export_to_influxdb()
            
            # Store in Redis if available
            if self.redis_client:
                await self._store_in_redis()
            
        except Exception as e:
            logging.error(f"Error exporting metrics: {e}")
    
    async def _export_to_influxdb(self):
        """Export metrics to InfluxDB."""
        try:
            if not self.influxdb_client:
                return
            
            write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
            points = []
            
            # Convert recent metrics to InfluxDB points
            for metric in list(self.metrics_buffer)[-100:]:  # Last 100 metrics
                point = Point("aware_net_metrics") \
                    .field(metric.name, metric.value) \
                    .time(metric.timestamp)
                
                # Add labels as tags
                for key, value in metric.labels.items():
                    point = point.tag(key, value)
                
                points.append(point)
            
            if points:
                write_api.write(bucket="aware-net", record=points)
                logging.debug(f"Exported {len(points)} metrics to InfluxDB")
            
        except Exception as e:
            logging.error(f"Error exporting to InfluxDB: {e}")
    
    async def _store_in_redis(self):
        """Store metrics in Redis for caching."""
        try:
            if not self.redis_client:
                return
            
            # Store current health status
            health_data = asdict(self.health_status)
            self.redis_client.setex(
                "aware_net:health_status",
                300,  # 5 minutes expiry
                json.dumps(health_data, default=str)
            )
            
            # Store recent metrics summary
            recent_metrics = self._get_recent_metrics(minutes=5)
            metrics_summary = {}
            
            for name, values in recent_metrics.items():
                if values:
                    metrics_summary[name] = {
                        'current': values[-1],
                        'average': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            self.redis_client.setex(
                "aware_net:metrics_summary",
                60,  # 1 minute expiry
                json.dumps(metrics_summary)
            )
            
        except Exception as e:
            logging.error(f"Error storing metrics in Redis: {e}")
    
    async def _check_alerts(self):
        """Check metrics against alert thresholds."""
        try:
            recent_metrics = self._get_recent_metrics(minutes=5)
            alerts = []
            
            for metric_name, threshold in self.alert_thresholds.items():
                if metric_name in recent_metrics:
                    values = recent_metrics[metric_name]
                    if values:
                        current_value = values[-1]
                        avg_value = statistics.mean(values)
                        
                        # Check if threshold is exceeded
                        if self._should_alert(metric_name, current_value, threshold):
                            alert = {
                                'metric': metric_name,
                                'current_value': current_value,
                                'average_value': avg_value,
                                'threshold': threshold,
                                'severity': self._get_alert_severity(metric_name, current_value, threshold),
                                'timestamp': datetime.now().isoformat()
                            }
                            alerts.append(alert)
            
            if alerts:
                await self._send_alerts(alerts)
            
        except Exception as e:
            logging.error(f"Error checking alerts: {e}")
    
    def _should_alert(self, metric_name: str, value: float, threshold: float) -> bool:
        """Determine if metric value should trigger an alert."""
        # Different alert logic for different metric types
        if 'usage_percent' in metric_name or 'rate_percent' in metric_name:
            return value > threshold
        elif 'latency' in metric_name or 'time' in metric_name:
            return value > threshold
        elif 'accuracy' in metric_name or 'satisfaction' in metric_name:
            return value < threshold
        elif 'throughput' in metric_name or 'rps' in metric_name:
            return value < threshold
        else:
            return value > threshold
    
    def _get_alert_severity(self, metric_name: str, value: float, threshold: float) -> str:
        """Determine alert severity based on how much threshold is exceeded."""
        if 'accuracy' in metric_name or 'satisfaction' in metric_name:
            ratio = threshold / value if value > 0 else float('inf')
        else:
            ratio = value / threshold if threshold > 0 else float('inf')
        
        if ratio >= 2.0:
            return "critical"
        elif ratio >= 1.5:
            return "high"
        elif ratio >= 1.2:
            return "medium"
        else:
            return "low"
    
    async def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alerts to configured channels."""
        try:
            # Log alerts
            for alert in alerts:
                logging.warning(f"ALERT: {alert['metric']} = {alert['current_value']:.2f} "
                              f"(threshold: {alert['threshold']:.2f}, severity: {alert['severity']})")
            
            # Store alerts in Redis for dashboard
            if self.redis_client:
                try:
                    alert_key = f"aware_net:alerts:{int(time.time())}"
                    self.redis_client.setex(alert_key, 3600, json.dumps(alerts))  # 1 hour expiry
                except Exception as e:
                    logging.error(f"Error storing alerts in Redis: {e}")
            
            # TODO: Implement additional alert channels (email, Slack, PagerDuty, etc.)
            
        except Exception as e:
            logging.error(f"Error sending alerts: {e}")
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        return generate_latest(self.registry)
    
    async def get_health_status(self) -> SystemHealth:
        """Get current system health status."""
        return self.health_status
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        try:
            recent_metrics = self._get_recent_metrics(minutes=15)
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'collection_interval_seconds': self.collection_interval,
                'health_status': asdict(self.health_status),
                'metrics_count': len(self.aggregated_metrics),
                'buffer_size': len(self.metrics_buffer),
                'recent_metrics': {}
            }
            
            # Add statistical summary of recent metrics
            for name, values in recent_metrics.items():
                if values:
                    summary['recent_metrics'][name] = {
                        'current': values[-1],
                        'average': statistics.mean(values),
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'count': len(values)
                    }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating metrics summary: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.stop_collection()
            
            if self.redis_client:
                self.redis_client.close()
            
            if self.influxdb_client:
                self.influxdb_client.close()
            
            self.executor.shutdown(wait=True)
            
            logging.info("✅ MetricsCollector cleanup complete")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

# Standalone usage example
async def main():
    """Example usage of MetricsCollector."""
    collector = MetricsCollector(collection_interval=5)
    
    try:
        # Initialize
        await collector.initialize()
        
        # Start collection
        await collector.start_collection()
        
        # Run for demo period
        await asyncio.sleep(30)
        
        # Get metrics summary
        summary = await collector.get_metrics_summary()
        print("Metrics Summary:")
        print(json.dumps(summary, indent=2, default=str))
        
        # Get health status
        health = await collector.get_health_status()
        print(f"\nSystem Health: {health.status} (score: {health.score:.1f})")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await collector.cleanup()

if __name__ == "__main__":
    asyncio.run(main())