import time
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict
import statistics

@dataclass
class TaskMetrics:
    success_count: int = 0
    failure_count: int = 0
    total_processing_time: float = 0.0
    processing_times: List[float] = None
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = []
    
    def add_success(self, processing_time: float):
        self.success_count += 1
        self.total_processing_time += processing_time
        self.processing_times.append(processing_time)
    
    def add_failure(self):
        self.failure_count += 1
    
    def get_avg_processing_time(self) -> float:
        if self.success_count == 0:
            return 0.0
        return self.total_processing_time / self.success_count
    
    def get_success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def get_percentile(self, percentile: float) -> float:
        if not self.processing_times:
            return 0.0
        
        sorted_times = sorted(self.processing_times)
        index = int(len(sorted_times) * percentile / 100)
        return sorted_times[index]

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(TaskMetrics)
        self.agent_metrics = defaultdict(lambda: defaultdict(TaskMetrics))
        self.start_time = datetime.now()
    
    def record_task_metrics(self, task_type: str, processing_time: float, success: bool):
        """Record metrics for a task"""
        if success:
            self.metrics[task_type].add_success(processing_time)
        else:
            self.metrics[task_type].add_failure()
    
    def record_agent_metrics(self, agent_name: str, task_type: str, processing_time: float, success: bool):
        """Record metrics for an agent's task"""
        if success:
            self.agent_metrics[agent_name][task_type].add_success(processing_time)
        else:
            self.agent_metrics[agent_name][task_type].add_failure()
    
    def get_task_metrics(self, task_type: str = None) -> Dict[str, Any]:
        """Get metrics for tasks"""
        if task_type:
            metrics = self.metrics.get(task_type, TaskMetrics())
            return {
                'success_count': metrics.success_count,
                'failure_count': metrics.failure_count,
                'avg_processing_time': metrics.get_avg_processing_time(),
                'success_rate': metrics.get_success_rate(),
                'p95_processing_time': metrics.get_percentile(95)
            }
        
        return {t: self.get_task_metrics(t) for t in self.metrics.keys()}
    
    def get_agent_metrics(self, agent_name: str = None) -> Dict[str, Any]:
        """Get metrics for agents"""
        if agent_name:
            agent_data = self.agent_metrics.get(agent_name, {})
            return {
                task_type: {
                    'success_count': metrics.success_count,
                    'failure_count': metrics.failure_count,
                    'avg_processing_time': metrics.get_avg_processing_time(),
                    'success_rate': metrics.get_success_rate()
                }
                for task_type, metrics in agent_data.items()
            }
        
        return {agent: self.get_agent_metrics(agent) for agent in self.agent_metrics.keys()}
    
    def get_uptime(self) -> float:
        """Get the uptime of the monitor in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.agent_metrics.clear()
        self.start_time = datetime.now()
