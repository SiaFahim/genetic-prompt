"""
Performance monitoring system for genetic algorithm evolution.
"""

import time
import psutil
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
else:
    from .config import config


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    api_calls_total: int
    api_calls_rate: float  # calls per minute
    tokens_used_total: int
    tokens_rate: float  # tokens per minute
    evaluation_time_avg: float
    cache_hit_rate: float
    active_threads: int
    disk_usage_mb: float


class PerformanceMonitor:
    """Monitors system and application performance during evolution."""
    
    def __init__(self, experiment_id: str, monitoring_interval: float = 30.0):
        """
        Initialize performance monitor.
        
        Args:
            experiment_id: Experiment identifier
            monitoring_interval: Interval between performance snapshots (seconds)
        """
        self.experiment_id = experiment_id
        self.monitoring_interval = monitoring_interval
        
        # Performance data storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.recent_metrics = deque(maxlen=100)  # Keep last 100 snapshots
        
        # Tracking variables
        self.start_time = time.time()
        self.last_snapshot_time = self.start_time
        
        # API usage tracking
        self.api_calls_total = 0
        self.tokens_used_total = 0
        self.api_calls_history = deque(maxlen=60)  # Last 60 minutes
        self.tokens_history = deque(maxlen=60)
        
        # Evaluation timing
        self.evaluation_times = deque(maxlen=50)  # Last 50 evaluations
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # System info
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance thresholds
        self.cpu_threshold = 90.0  # %
        self.memory_threshold = 85.0  # %
        self.api_rate_threshold = 100.0  # calls per minute
        
        print(f"📊 Performance monitor initialized for experiment: {experiment_id}")
    
    def record_api_call(self, tokens_used: int = 0):
        """
        Record an API call.
        
        Args:
            tokens_used: Number of tokens used in the call
        """
        current_time = time.time()
        
        self.api_calls_total += 1
        self.tokens_used_total += tokens_used
        
        # Add to history with timestamp
        self.api_calls_history.append(current_time)
        self.tokens_history.append((current_time, tokens_used))
    
    def record_evaluation_time(self, evaluation_time: float):
        """
        Record evaluation timing.
        
        Args:
            evaluation_time: Time taken for evaluation (seconds)
        """
        self.evaluation_times.append(evaluation_time)
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    def take_snapshot(self) -> PerformanceMetrics:
        """
        Take a performance snapshot.
        
        Returns:
            PerformanceMetrics object
        """
        current_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        memory_used_mb = memory_info.rss / 1024 / 1024
        
        # API rate calculations
        api_calls_rate = self._calculate_api_rate()
        tokens_rate = self._calculate_tokens_rate()
        
        # Evaluation timing
        eval_time_avg = (sum(self.evaluation_times) / len(self.evaluation_times) 
                        if self.evaluation_times else 0.0)
        
        # Cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache_requests 
                         if total_cache_requests > 0 else 0.0)
        
        # Thread count
        active_threads = self.process.num_threads()
        
        # Disk usage (experiment directory)
        disk_usage_mb = self._calculate_disk_usage()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            api_calls_total=self.api_calls_total,
            api_calls_rate=api_calls_rate,
            tokens_used_total=self.tokens_used_total,
            tokens_rate=tokens_rate,
            evaluation_time_avg=eval_time_avg,
            cache_hit_rate=cache_hit_rate,
            active_threads=active_threads,
            disk_usage_mb=disk_usage_mb
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.recent_metrics.append(metrics)
        self.last_snapshot_time = current_time
        
        return metrics
    
    def _calculate_api_rate(self) -> float:
        """Calculate API calls per minute."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Count calls in last minute
        recent_calls = sum(1 for call_time in self.api_calls_history 
                          if call_time > minute_ago)
        
        return float(recent_calls)
    
    def _calculate_tokens_rate(self) -> float:
        """Calculate tokens per minute."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Sum tokens in last minute
        recent_tokens = sum(tokens for call_time, tokens in self.tokens_history 
                           if call_time > minute_ago)
        
        return float(recent_tokens)
    
    def _calculate_disk_usage(self) -> float:
        """Calculate disk usage for experiment data."""
        try:
            experiments_dir = config.get_data_dir() / "experiments" / self.experiment_id
            if experiments_dir.exists():
                total_size = sum(f.stat().st_size for f in experiments_dir.rglob('*') if f.is_file())
                return total_size / 1024 / 1024  # MB
        except Exception:
            pass
        return 0.0
    
    def check_performance_alerts(self) -> List[str]:
        """
        Check for performance alerts.
        
        Returns:
            List of alert messages
        """
        alerts = []
        
        if not self.recent_metrics:
            return alerts
        
        latest = self.recent_metrics[-1]
        
        # CPU usage alert
        if latest.cpu_percent > self.cpu_threshold:
            alerts.append(f"High CPU usage: {latest.cpu_percent:.1f}%")
        
        # Memory usage alert
        if latest.memory_percent > self.memory_threshold:
            alerts.append(f"High memory usage: {latest.memory_percent:.1f}%")
        
        # API rate alert
        if latest.api_calls_rate > self.api_rate_threshold:
            alerts.append(f"High API rate: {latest.api_calls_rate:.1f} calls/min")
        
        # Memory growth alert
        if len(self.recent_metrics) >= 10:
            memory_growth = latest.memory_used_mb - self.recent_metrics[-10].memory_used_mb
            if memory_growth > 100:  # 100MB growth
                alerts.append(f"Memory growth: +{memory_growth:.1f}MB in last 10 snapshots")
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {'no_data': True}
        
        # Calculate averages
        cpu_avg = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
        memory_avg = sum(m.memory_used_mb for m in self.metrics_history) / len(self.metrics_history)
        
        # Peak values
        cpu_peak = max(m.cpu_percent for m in self.metrics_history)
        memory_peak = max(m.memory_used_mb for m in self.metrics_history)
        
        # Current values
        latest = self.metrics_history[-1]
        
        # Total runtime
        total_runtime = latest.timestamp - self.start_time
        
        # API efficiency
        api_efficiency = (self.tokens_used_total / self.api_calls_total 
                         if self.api_calls_total > 0 else 0)
        
        return {
            'experiment_id': self.experiment_id,
            'total_runtime_minutes': total_runtime / 60,
            'snapshots_taken': len(self.metrics_history),
            'cpu_usage': {
                'current': latest.cpu_percent,
                'average': cpu_avg,
                'peak': cpu_peak
            },
            'memory_usage': {
                'current_mb': latest.memory_used_mb,
                'average_mb': memory_avg,
                'peak_mb': memory_peak,
                'growth_mb': latest.memory_used_mb - self.initial_memory
            },
            'api_usage': {
                'total_calls': self.api_calls_total,
                'total_tokens': self.tokens_used_total,
                'current_rate_per_min': latest.api_calls_rate,
                'tokens_per_call': api_efficiency
            },
            'cache_performance': {
                'hit_rate': latest.cache_hit_rate,
                'total_hits': self.cache_hits,
                'total_misses': self.cache_misses
            },
            'evaluation_performance': {
                'average_time_seconds': latest.evaluation_time_avg,
                'total_evaluations': len(self.evaluation_times)
            },
            'disk_usage_mb': latest.disk_usage_mb
        }
    
    def save_performance_report(self, filepath: Optional[Path] = None):
        """Save detailed performance report."""
        if filepath is None:
            reports_dir = config.get_data_dir() / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            filepath = reports_dir / f"performance_report_{self.experiment_id}.json"
        
        report = {
            'experiment_id': self.experiment_id,
            'report_generated_at': time.time(),
            'monitoring_duration_minutes': (time.time() - self.start_time) / 60,
            'summary': self.get_performance_summary(),
            'metrics_history': [asdict(m) for m in self.metrics_history],
            'alerts_detected': self.check_performance_alerts()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Performance report saved: {filepath}")
        return str(filepath)
    
    def get_resource_recommendations(self) -> List[str]:
        """Get resource optimization recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        summary = self.get_performance_summary()
        
        # CPU recommendations
        if summary['cpu_usage']['average'] > 80:
            recommendations.append("Consider reducing population size or using parallel processing")
        
        # Memory recommendations
        if summary['memory_usage']['growth_mb'] > 500:
            recommendations.append("Memory usage is growing - check for memory leaks")
        
        # API recommendations
        if summary['api_usage']['tokens_per_call'] < 50:
            recommendations.append("API calls are using few tokens - consider batching")
        
        # Cache recommendations
        if summary['cache_performance']['hit_rate'] < 0.3:
            recommendations.append("Low cache hit rate - consider increasing cache size")
        
        # Evaluation recommendations
        if summary['evaluation_performance']['average_time_seconds'] > 60:
            recommendations.append("Evaluations are slow - consider reducing problem set size")
        
        return recommendations


if __name__ == "__main__":
    # Test performance monitor
    print("Testing performance monitor...")
    
    # Create monitor
    monitor = PerformanceMonitor("test_performance", monitoring_interval=1.0)
    
    # Simulate some activity
    for i in range(5):
        # Record API calls
        monitor.record_api_call(tokens_used=100 + i * 20)
        
        # Record evaluation time
        monitor.record_evaluation_time(10.0 + i * 2)
        
        # Record cache activity
        if i % 2 == 0:
            monitor.record_cache_hit()
        else:
            monitor.record_cache_miss()
        
        # Take snapshot
        metrics = monitor.take_snapshot()
        print(f"✅ Snapshot {i+1}: CPU={metrics.cpu_percent:.1f}%, "
              f"Memory={metrics.memory_used_mb:.1f}MB")
        
        time.sleep(0.1)  # Small delay
    
    # Check alerts
    alerts = monitor.check_performance_alerts()
    print(f"✅ Performance alerts: {len(alerts)}")
    
    # Get summary
    summary = monitor.get_performance_summary()
    print(f"✅ Performance summary: {summary['snapshots_taken']} snapshots")
    
    # Save report
    report_path = monitor.save_performance_report()
    print(f"✅ Report saved: {report_path}")
    
    # Get recommendations
    recommendations = monitor.get_resource_recommendations()
    print(f"✅ Recommendations: {len(recommendations)}")
    
    print("\n🎯 Performance monitor tests completed successfully!")
