"""
Step Time Manager - Persistent storage and analysis of step-level timing data.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import statistics


class StepTimeManager:
    """Manages persistent storage and analysis of step timing data."""
    
    def __init__(self, storage_path: str = "data/step_times.json"):
        """Initialize the step time manager."""
        self.storage_path = storage_path
        self.step_times: Dict[int, List[float]] = {}
        self.step_stats: Dict[int, Dict[str, float]] = {}
        self.max_samples_per_step = 500  # Keep up to 500 samples per step
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Load existing data
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load step times from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                # Convert string keys back to integers
                self.step_times = {int(k): v for k, v in data.get('step_times', {}).items()}
                self.step_stats = {int(k): v for k, v in data.get('step_stats', {}).items()}
                
                print(f"✅ Loaded step timing data: {sum(len(times) for times in self.step_times.values())} samples across {len(self.step_times)} steps")
            except Exception as e:
                print(f"⚠️ Could not load step times: {e}")
                self.step_times = {}
                self.step_stats = {}
        else:
            print("📊 No existing step timing data found. Starting fresh.")
            self.step_times = {}
            self.step_stats = {}
    
    def _save_to_disk(self):
        """Save step times to disk."""
        try:
            data = {
                'step_times': {str(k): v for k, v in self.step_times.items()},
                'step_stats': {str(k): v for k, v in self.step_stats.items()},
                'last_updated': datetime.now().isoformat(),
                'total_samples': sum(len(times) for times in self.step_times.values())
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not save step times: {e}")
    
    def record_step_time(self, step_number: int, duration: float):
        """Record a step duration."""
        if step_number not in self.step_times:
            self.step_times[step_number] = []
        
        self.step_times[step_number].append(duration)
        
        # Keep only the most recent samples
        if len(self.step_times[step_number]) > self.max_samples_per_step:
            self.step_times[step_number] = self.step_times[step_number][-self.max_samples_per_step:]
        
        # Update statistics for this step
        self._update_step_stats(step_number)
        
        # Save to disk periodically (every 10 samples)
        total_samples = sum(len(times) for times in self.step_times.values())
        if total_samples % 10 == 0:
            self._save_to_disk()
    
    def _update_step_stats(self, step_number: int):
        """Update statistics for a specific step."""
        times = self.step_times.get(step_number, [])
        
        if not times:
            return
        
        self.step_stats[step_number] = {
            'avg': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'count': len(times),
            'p25': statistics.quantiles(times, n=4)[0] if len(times) >= 4 else times[0],
            'p75': statistics.quantiles(times, n=4)[2] if len(times) >= 4 else times[-1],
        }
    
    def get_step_estimate(self, step_number: int, use_conservative: bool = False) -> float:
        """
        Get time estimate for a specific step.
        
        Args:
            step_number: The step to estimate
            use_conservative: If True, use 75th percentile instead of average (more conservative)
        
        Returns:
            Estimated duration in seconds
        """
        if step_number in self.step_stats:
            stats = self.step_stats[step_number]
            
            # Use different estimates based on data quality
            if stats['count'] >= 20:
                # Good data, use percentile or average
                return stats['p75'] if use_conservative else stats['avg']
            elif stats['count'] >= 5:
                # Some data, use average with slight padding
                return stats['avg'] * 1.1
            else:
                # Limited data, use max of what we have
                return stats['max']
        
        # No data for this step, use default
        step_defaults = {
            1: 3.5,   # Data gathering - fundamentals
            2: 4.0,   # Data gathering - market data
            3: 5.5,   # Value agent
            4: 5.0,   # Growth/Momentum agent
            5: 4.5,   # Macro regime agent
            6: 5.0,   # Risk agent
            7: 6.0,   # Sentiment agent
            8: 3.0,   # Client layer agent
            9: 2.5,   # Final scoring
            10: 4.0,  # Comprehensive rationale
        }
        
        return step_defaults.get(step_number, 4.0)
    
    def get_total_estimate(self, remaining_steps: List[int], use_conservative: bool = False) -> float:
        """Get total time estimate for remaining steps."""
        total = 0.0
        for step in remaining_steps:
            total += self.get_step_estimate(step, use_conservative)
        return total
    
    def get_all_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for all steps."""
        return self.step_stats.copy()
    
    def get_summary(self) -> str:
        """Get a formatted summary of step timing data."""
        lines = []
        lines.append("=" * 80)
        lines.append("STEP TIMING STATISTICS")
        lines.append("=" * 80)
        
        total_samples = sum(len(times) for times in self.step_times.values())
        lines.append(f"Total samples collected: {total_samples}")
        lines.append(f"Steps with data: {len(self.step_times)}")
        lines.append("")
        
        step_names = {
            1: "Data Gathering - Fundamentals",
            2: "Data Gathering - Market Data",
            3: "Value Agent Analysis",
            4: "Growth/Momentum Agent Analysis",
            5: "Macro Regime Agent Analysis",
            6: "Risk Agent Analysis",
            7: "Sentiment Agent Analysis",
            8: "Score Blending",
            9: "Client Layer Validation",
            10: "Final Analysis"
        }
        
        for step in sorted(self.step_stats.keys()):
            stats = self.step_stats[step]
            name = step_names.get(step, f"Step {step}")
            
            lines.append(f"Step {step}: {name}")
            lines.append(f"  Samples: {stats['count']}")
            lines.append(f"  Average: {stats['avg']:.2f}s")
            lines.append(f"  Median:  {stats['median']:.2f}s")
            lines.append(f"  Range:   {stats['min']:.2f}s - {stats['max']:.2f}s")
            lines.append(f"  Std Dev: {stats['std_dev']:.2f}s")
            lines.append(f"  P25-P75: {stats['p25']:.2f}s - {stats['p75']:.2f}s")
            lines.append("")
        
        if self.step_stats:
            total_avg = sum(s['avg'] for s in self.step_stats.values())
            lines.append(f"Total estimated time per analysis: {total_avg:.1f}s ({total_avg/60:.1f}m)")
        
        lines.append("=" * 80)
        return '\n'.join(lines)
    
    def force_save(self):
        """Force save current data to disk."""
        self._save_to_disk()
