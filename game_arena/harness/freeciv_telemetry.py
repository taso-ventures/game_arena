"""Production monitoring and telemetry for FreeCiv LLM Agent.

This module provides comprehensive monitoring, metrics collection, and
telemetry for production deployments of the FreeCiv LLM Agent.

Example usage:
    >>> from game_arena.harness.freeciv_telemetry import TelemetryManager
    >>> telemetry = TelemetryManager(agent_id="freeciv-agent-1")
    >>> telemetry.record_action_start("unit_move")
    >>> # ... perform action ...
    >>> telemetry.record_action_complete("unit_move", success=True, duration=1.2)
    >>> metrics = telemetry.get_metrics_summary()
"""

import json
import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from absl import logging


@dataclass
class ActionMetric:
  """Metrics for a single action execution."""
  action_type: str
  start_time: float
  end_time: Optional[float] = None
  duration: Optional[float] = None
  success: bool = False
  error_type: Optional[str] = None
  error_message: Optional[str] = None
  model_calls: int = 0
  tokens_used: int = 0
  retries: int = 0
  metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceWindow:
  """Performance metrics over a time window."""
  window_start: float
  window_end: float
  total_actions: int = 0
  successful_actions: int = 0
  total_duration: float = 0.0
  total_tokens: int = 0
  total_model_calls: int = 0
  error_count: int = 0
  action_types: Dict[str, int] = field(default_factory=dict)
  errors_by_type: Dict[str, int] = field(default_factory=dict)


class TelemetryCollector:
  """Thread-safe collector for telemetry data."""

  def __init__(self, max_history: int = 10000):
    """Initialize telemetry collector.

    Args:
      max_history: Maximum number of action metrics to keep in memory
    """
    self.max_history = max_history
    self._lock = threading.Lock()

    # Action metrics storage
    self._action_history = deque(maxlen=max_history)
    self._active_actions: Dict[str, ActionMetric] = {}

    # Aggregated metrics
    self._total_actions = 0
    self._total_successful = 0
    self._total_duration = 0.0
    self._total_tokens = 0
    self._total_model_calls = 0
    self._error_counts = defaultdict(int)
    self._action_type_counts = defaultdict(int)

    # Performance windows (for rate calculations)
    self._performance_windows = deque(maxlen=288)  # 24 hours of 5-min windows
    self._current_window_start = time.time()

  def start_action(self, action_id: str, action_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Start tracking an action.

    Args:
      action_id: Unique identifier for this action instance
      action_type: Type of action being performed
      metadata: Optional metadata about the action

    Returns:
      Action ID for tracking
    """
    with self._lock:
      metric = ActionMetric(
          action_type=action_type,
          start_time=time.time(),
          metadata=metadata or {}
      )
      self._active_actions[action_id] = metric
      return action_id

  def complete_action(self, action_id: str, success: bool,
                     error_type: Optional[str] = None,
                     error_message: Optional[str] = None,
                     model_calls: int = 0,
                     tokens_used: int = 0,
                     retries: int = 0) -> None:
    """Complete tracking an action.

    Args:
      action_id: Action identifier
      success: Whether the action succeeded
      error_type: Type of error if failed
      error_message: Error message if failed
      model_calls: Number of model API calls made
      tokens_used: Total tokens consumed
      retries: Number of retries attempted
    """
    with self._lock:
      if action_id not in self._active_actions:
        logging.warning("Completing unknown action: %s", action_id)
        return

      metric = self._active_actions.pop(action_id)
      metric.end_time = time.time()
      metric.duration = metric.end_time - metric.start_time
      metric.success = success
      metric.error_type = error_type
      metric.error_message = error_message
      metric.model_calls = model_calls
      metric.tokens_used = tokens_used
      metric.retries = retries

      # Add to history
      self._action_history.append(metric)

      # Update aggregated metrics
      self._total_actions += 1
      if success:
        self._total_successful += 1
      else:
        self._error_counts[error_type or "unknown"] += 1

      self._total_duration += metric.duration or 0
      self._total_tokens += tokens_used
      self._total_model_calls += model_calls
      self._action_type_counts[metric.action_type] += 1

      # Update performance windows
      self._update_performance_windows(metric)

  def _update_performance_windows(self, metric: ActionMetric) -> None:
    """Update performance tracking windows."""
    window_duration = 300  # 5 minutes
    current_time = time.time()

    # Check if we need a new window
    if current_time - self._current_window_start >= window_duration:
      # Close current window if it exists and has data
      if self._performance_windows and self._performance_windows[-1].total_actions > 0:
        self._performance_windows[-1].window_end = current_time

      # Start new window
      new_window = PerformanceWindow(
          window_start=current_time,
          window_end=current_time + window_duration
      )
      self._performance_windows.append(new_window)
      self._current_window_start = current_time

    # Add metric to current window
    if self._performance_windows:
      window = self._performance_windows[-1]
      window.total_actions += 1
      if metric.success:
        window.successful_actions += 1
      else:
        window.error_count += 1
        if metric.error_type:
          window.errors_by_type[metric.error_type] += 1

      window.total_duration += metric.duration or 0
      window.total_tokens += metric.tokens_used
      window.total_model_calls += metric.model_calls
      window.action_types[metric.action_type] += 1

  def get_current_metrics(self) -> Dict[str, Any]:
    """Get current aggregated metrics.

    Returns:
      Dictionary with current metrics
    """
    with self._lock:
      success_rate = self._total_successful / max(1, self._total_actions)
      avg_duration = self._total_duration / max(1, self._total_actions)
      avg_tokens = self._total_tokens / max(1, self._total_actions)

      return {
          "total_actions": self._total_actions,
          "successful_actions": self._total_successful,
          "success_rate": success_rate,
          "total_duration": self._total_duration,
          "average_duration": avg_duration,
          "total_tokens": self._total_tokens,
          "average_tokens": avg_tokens,
          "total_model_calls": self._total_model_calls,
          "error_counts": dict(self._error_counts),
          "action_type_counts": dict(self._action_type_counts),
          "active_actions": len(self._active_actions),
          "history_size": len(self._action_history)
      }

  def get_recent_performance(self, minutes: int = 30) -> Dict[str, Any]:
    """Get performance metrics for recent time period.

    Args:
      minutes: Number of minutes to look back

    Returns:
      Performance metrics for the specified period
    """
    with self._lock:
      cutoff_time = time.time() - (minutes * 60)
      recent_actions = [
          action for action in self._action_history
          if action.start_time >= cutoff_time
      ]

      if not recent_actions:
        return {"period_minutes": minutes, "total_actions": 0}

      successful = sum(1 for action in recent_actions if action.success)
      total_duration = sum(action.duration or 0 for action in recent_actions)
      total_tokens = sum(action.tokens_used for action in recent_actions)
      total_calls = sum(action.model_calls for action in recent_actions)

      action_types = defaultdict(int)
      errors = defaultdict(int)
      for action in recent_actions:
        action_types[action.action_type] += 1
        if not action.success and action.error_type:
          errors[action.error_type] += 1

      return {
          "period_minutes": minutes,
          "total_actions": len(recent_actions),
          "successful_actions": successful,
          "success_rate": successful / len(recent_actions),
          "total_duration": total_duration,
          "average_duration": total_duration / len(recent_actions),
          "total_tokens": total_tokens,
          "average_tokens": total_tokens / len(recent_actions),
          "total_model_calls": total_calls,
          "actions_per_minute": len(recent_actions) / minutes,
          "action_types": dict(action_types),
          "errors": dict(errors)
      }

  def get_performance_trend(self) -> List[Dict[str, Any]]:
    """Get performance trend over time windows.

    Returns:
      List of performance windows with metrics
    """
    with self._lock:
      return [
          {
              "window_start": window.window_start,
              "window_end": window.window_end,
              "total_actions": window.total_actions,
              "success_rate": window.successful_actions / max(1, window.total_actions),
              "average_duration": window.total_duration / max(1, window.total_actions),
              "actions_per_minute": window.total_actions / 5.0,  # 5-minute windows
              "tokens_per_action": window.total_tokens / max(1, window.total_actions),
              "error_count": window.error_count,
              "action_types": dict(window.action_types),
              "errors_by_type": dict(window.errors_by_type)
          }
          for window in self._performance_windows
      ]


class TelemetryManager:
  """High-level telemetry manager for FreeCiv LLM Agent."""

  def __init__(self, agent_id: str, max_history: int = 10000):
    """Initialize telemetry manager.

    Args:
      agent_id: Unique identifier for this agent instance
      max_history: Maximum history to keep in memory
    """
    self.agent_id = agent_id
    self.collector = TelemetryCollector(max_history)
    self._action_counter = 0
    self._action_counter_lock = threading.Lock()

  def _generate_action_id(self) -> str:
    """Generate unique action ID."""
    with self._action_counter_lock:
      self._action_counter += 1
      return f"{self.agent_id}_action_{self._action_counter}"

  @contextmanager
  def track_action(self, action_type: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for tracking action execution.

    Args:
      action_type: Type of action being performed
      metadata: Optional metadata about the action

    Yields:
      ActionTracker instance for recording additional metrics
    """
    action_id = self._generate_action_id()

    # Start tracking
    self.collector.start_action(action_id, action_type, metadata)

    tracker = ActionTracker(self.collector, action_id)
    try:
      yield tracker
      # If we get here, action succeeded
      tracker.mark_success()
    except Exception as e:
      # Action failed
      tracker.mark_failure(
          error_type=type(e).__name__,
          error_message=str(e)
      )
      raise
    finally:
      # Complete tracking
      tracker.complete()

  def record_action_start(self, action_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Start tracking an action (manual mode).

    Args:
      action_type: Type of action
      metadata: Optional metadata

    Returns:
      Action ID for completion
    """
    action_id = self._generate_action_id()
    self.collector.start_action(action_id, action_type, metadata)
    return action_id

  def record_action_complete(self, action_id: str, success: bool,
                           duration: Optional[float] = None,
                           error_type: Optional[str] = None,
                           error_message: Optional[str] = None,
                           model_calls: int = 0,
                           tokens_used: int = 0,
                           retries: int = 0) -> None:
    """Complete tracking an action (manual mode).

    Args:
      action_id: Action ID from record_action_start
      success: Whether action succeeded
      duration: Optional duration override
      error_type: Error type if failed
      error_message: Error message if failed
      model_calls: Number of model calls
      tokens_used: Tokens consumed
      retries: Number of retries
    """
    self.collector.complete_action(
        action_id=action_id,
        success=success,
        error_type=error_type,
        error_message=error_message,
        model_calls=model_calls,
        tokens_used=tokens_used,
        retries=retries
    )

  def get_metrics_summary(self) -> Dict[str, Any]:
    """Get comprehensive metrics summary.

    Returns:
      Complete metrics summary
    """
    current = self.collector.get_current_metrics()
    recent = self.collector.get_recent_performance(30)

    return {
        "agent_id": self.agent_id,
        "timestamp": time.time(),
        "lifetime_metrics": current,
        "recent_performance": recent,
        "health_status": self._assess_health(current, recent)
    }

  def _assess_health(self, current: Dict[str, Any], recent: Dict[str, Any]) -> str:
    """Assess agent health based on metrics.

    Args:
      current: Current lifetime metrics
      recent: Recent performance metrics

    Returns:
      Health status string
    """
    recent_success_rate = recent.get("success_rate", 0)
    recent_actions = recent.get("total_actions", 0)
    avg_duration = recent.get("average_duration", 0)

    # Health criteria
    if recent_actions == 0:
      return "idle"
    elif recent_success_rate < 0.5:
      return "unhealthy"
    elif recent_success_rate < 0.8 or avg_duration > 10.0:
      return "degraded"
    else:
      return "healthy"

  def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
    """Export metrics in specified format.

    Args:
      format: Export format ("json" or "dict")

    Returns:
      Metrics in requested format
    """
    metrics = self.get_metrics_summary()

    if format == "json":
      return json.dumps(metrics, indent=2)
    else:
      return metrics

  def get_performance_report(self) -> str:
    """Generate human-readable performance report.

    Returns:
      Formatted performance report
    """
    metrics = self.get_metrics_summary()
    lifetime = metrics["lifetime_metrics"]
    recent = metrics["recent_performance"]
    health = metrics["health_status"]

    report = f"""
FreeCiv LLM Agent Performance Report
Agent ID: {self.agent_id}
Health Status: {health.upper()}
Report Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

=== Lifetime Statistics ===
Total Actions: {lifetime['total_actions']:,}
Success Rate: {lifetime['success_rate']:.1%}
Average Duration: {lifetime['average_duration']:.2f}s
Total Tokens: {lifetime['total_tokens']:,}
Average Tokens/Action: {lifetime['average_tokens']:.0f}

=== Recent Performance (30 min) ===
Actions: {recent['total_actions']}
Success Rate: {recent.get('success_rate', 0):.1%}
Actions/Minute: {recent.get('actions_per_minute', 0):.1f}
Average Duration: {recent.get('average_duration', 0):.2f}s

=== Action Types ==="""

    for action_type, count in lifetime.get('action_type_counts', {}).items():
      report += f"\n  {action_type}: {count}"

    if lifetime.get('error_counts'):
      report += "\n\n=== Errors ==="
      for error_type, count in lifetime['error_counts'].items():
        report += f"\n  {error_type}: {count}"

    return report


class ActionTracker:
  """Helper class for tracking individual action metrics."""

  def __init__(self, collector: TelemetryCollector, action_id: str):
    """Initialize action tracker.

    Args:
      collector: Telemetry collector instance
      action_id: Action identifier
    """
    self.collector = collector
    self.action_id = action_id
    self.model_calls = 0
    self.tokens_used = 0
    self.retries = 0
    self.success = False
    self.error_type: Optional[str] = None
    self.error_message: Optional[str] = None

  def add_model_call(self, tokens: int = 0) -> None:
    """Record a model API call.

    Args:
      tokens: Number of tokens used in this call
    """
    self.model_calls += 1
    self.tokens_used += tokens

  def add_retry(self) -> None:
    """Record a retry attempt."""
    self.retries += 1

  def mark_success(self) -> None:
    """Mark action as successful."""
    self.success = True

  def mark_failure(self, error_type: str, error_message: str) -> None:
    """Mark action as failed.

    Args:
      error_type: Type of error
      error_message: Error message
    """
    self.success = False
    self.error_type = error_type
    self.error_message = error_message

  def complete(self) -> None:
    """Complete the action tracking."""
    self.collector.complete_action(
        action_id=self.action_id,
        success=self.success,
        error_type=self.error_type,
        error_message=self.error_message,
        model_calls=self.model_calls,
        tokens_used=self.tokens_used,
        retries=self.retries
    )