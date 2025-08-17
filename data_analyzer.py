#!/usr/bin/env python3
"""
Data Analyzer - A comprehensive tool for analyzing and visualizing data
This program demonstrates various Python features including:
- Object-oriented programming
- Data processing with built-in libraries
- Mathematical operations
- File I/O operations
- Error handling
"""

import json
import csv
import math
import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class DataPoint:
    """Represents a single data point with value and timestamp."""

    def __init__(self, value: float, timestamp: Optional[datetime] = None):
        self.value = value
        self.timestamp = timestamp or datetime.now()

    def __repr__(self) -> str:
        return f"DataPoint(value={self.value}, timestamp={self.timestamp})"

    def __lt__(self, other) -> bool:
        return self.value < other.value


class DataSet:
    """A collection of data points with analysis capabilities."""

    def __init__(self, name: str = "Unnamed Dataset"):
        self.name = name
        self.data_points: List[DataPoint] = []
        self._cached_stats: Optional[Dict[str, float]] = None

    def add_data_point(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a new data point to the dataset."""
        self.data_points.append(DataPoint(value, timestamp))
        self._cached_stats = None  # Invalidate cache

    def add_multiple_values(self, values: List[float]) -> None:
        """Add multiple values as data points."""
        for value in values:
            self.add_data_point(value)

    def get_values(self) -> List[float]:
        """Extract all values from data points."""
        return [dp.value for dp in self.data_points]

    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate comprehensive statistics for the dataset."""
        if not self.data_points:
            return {}

        if self._cached_stats is not None:
            return self._cached_stats

        values = self.get_values()

        stats = {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'mode': statistics.mode(values) if len(set(values)) < len(values) else values[0],
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'variance': statistics.variance(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
            'sum': sum(values)
        }

        # Calculate percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)
        stats['q1'] = sorted_values[n // 4] if n >= 4 else sorted_values[0]
        stats['q3'] = sorted_values[3 * n // 4] if n >= 4 else sorted_values[-1]
        stats['iqr'] = stats['q3'] - stats['q1']

        self._cached_stats = stats
        return stats

    def detect_outliers(self, method: str = 'iqr') -> List[DataPoint]:
        """Detect outliers using specified method."""
        if method == 'iqr':
            return self._detect_outliers_iqr()
        elif method == 'zscore':
            return self._detect_outliers_zscore()
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def _detect_outliers_iqr(self) -> List[DataPoint]:
        """Detect outliers using IQR method."""
        stats = self.calculate_statistics()
        if not stats:
            return []

        q1, q3 = stats['q1'], stats['q3']
        iqr = stats['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return [dp for dp in self.data_points
                if dp.value < lower_bound or dp.value > upper_bound]

    def _detect_outliers_zscore(self, threshold: float = 2.0) -> List[DataPoint]:
        """Detect outliers using Z-score method."""
        stats = self.calculate_statistics()
        if not stats or stats['std_dev'] == 0:
            return []

        mean, std_dev = stats['mean'], stats['std_dev']
        outliers = []

        for dp in self.data_points:
            z_score = abs((dp.value - mean) / std_dev)
            if z_score > threshold:
                outliers.append(dp)

        return outliers

    def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> 'DataSet':
        """Create a new dataset filtered by date range."""
        filtered_dataset = DataSet(f"{self.name} (filtered)")

        for dp in self.data_points:
            if start_date <= dp.timestamp <= end_date:
                filtered_dataset.data_points.append(dp)

        return filtered_dataset

    def transform_values(self, transformation: str) -> 'DataSet':
        """Apply mathematical transformation to values."""
        transformed_dataset = DataSet(f"{self.name} ({transformation})")

        for dp in self.data_points:
            if transformation == 'log':
                if dp.value > 0:
                    new_value = math.log(dp.value)
                    transformed_dataset.add_data_point(new_value, dp.timestamp)
            elif transformation == 'sqrt':
                if dp.value >= 0:
                    new_value = math.sqrt(dp.value)
                    transformed_dataset.add_data_point(new_value, dp.timestamp)
            elif transformation == 'square':
                new_value = dp.value ** 2
                transformed_dataset.add_data_point(new_value, dp.timestamp)
            elif transformation == 'normalize':
                stats = self.calculate_statistics()
                if stats['std_dev'] != 0:
                    new_value = (dp.value - stats['mean']) / stats['std_dev']
                    transformed_dataset.add_data_point(new_value, dp.timestamp)

        return transformed_dataset


class DataAnalyzer:
    """Main analyzer class for managing multiple datasets."""

    def __init__(self):
        self.datasets: Dict[str, DataSet] = {}
        self.analysis_history: List[Dict[str, Any]] = []

    def add_dataset(self, name: str, dataset: DataSet) -> None:
        """Add a dataset to the analyzer."""
        self.datasets[name] = dataset
        dataset.name = name

    def create_dataset_from_csv(self, name: str, filename: str,
                               value_column: str, timestamp_column: Optional[str] = None) -> None:
        """Create a dataset from CSV file."""
        dataset = DataSet(name)

        try:
            with open(filename, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    value = float(row[value_column])
                    timestamp = None

                    if timestamp_column and timestamp_column in row:
                        try:
                            timestamp = datetime.fromisoformat(row[timestamp_column])
                        except ValueError:
                            timestamp = datetime.now()

                    dataset.add_data_point(value, timestamp)

            self.add_dataset(name, dataset)

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except KeyError as e:
            print(f"Error: Column '{e}' not found in CSV file.")
        except ValueError as e:
            print(f"Error: Invalid data format - {e}")

    def generate_sample_data(self, name: str, size: int = 100,
                           distribution: str = 'normal') -> None:
        """Generate sample data for testing purposes."""
        import random

        dataset = DataSet(name)

        if distribution == 'normal':
            # Generate normal distribution data
            for i in range(size):
                value = random.gauss(50, 15)  # mean=50, std=15
                timestamp = datetime.now() - timedelta(days=size-i)
                dataset.add_data_point(value, timestamp)

        elif distribution == 'uniform':
            # Generate uniform distribution data
            for i in range(size):
                value = random.uniform(0, 100)
                timestamp = datetime.now() - timedelta(days=size-i)
                dataset.add_data_point(value, timestamp)

        elif distribution == 'exponential':
            # Generate exponential distribution data
            for i in range(size):
                value = random.expovariate(0.1)
                timestamp = datetime.now() - timedelta(days=size-i)
                dataset.add_data_point(value, timestamp)

        self.add_dataset(name, dataset)

    def compare_datasets(self, dataset1_name: str, dataset2_name: str) -> Dict[str, Any]:
        """Compare statistics between two datasets."""
        if dataset1_name not in self.datasets or dataset2_name not in self.datasets:
            raise ValueError("One or both datasets not found")

        ds1 = self.datasets[dataset1_name]
        ds2 = self.datasets[dataset2_name]

        stats1 = ds1.calculate_statistics()
        stats2 = ds2.calculate_statistics()

        comparison = {
            'dataset1': dataset1_name,
            'dataset2': dataset2_name,
            'stats_comparison': {},
            'correlation': self._calculate_correlation(ds1, ds2)
        }

        for stat in stats1:
            if stat in stats2:
                comparison['stats_comparison'][stat] = {
                    'dataset1': stats1[stat],
                    'dataset2': stats2[stat],
                    'difference': stats2[stat] - stats1[stat],
                    'percent_change': ((stats2[stat] - stats1[stat]) / stats1[stat] * 100)
                                    if stats1[stat] != 0 else float('inf')
                }

        return comparison

    def _calculate_correlation(self, ds1: DataSet, ds2: DataSet) -> Optional[float]:
        """Calculate correlation coefficient between two datasets."""
        values1 = ds1.get_values()
        values2 = ds2.get_values()

        if len(values1) != len(values2) or len(values1) < 2:
            return None

        try:
            return statistics.correlation(values1, values2)
        except:
            return None

    def export_analysis_report(self, filename: str) -> None:
        """Export comprehensive analysis report to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'analysis_history': self.analysis_history
        }

        for name, dataset in self.datasets.items():
            stats = dataset.calculate_statistics()
            outliers = dataset.detect_outliers()

            report['datasets'][name] = {
                'statistics': stats,
                'outlier_count': len(outliers),
                'data_point_count': len(dataset.data_points),
                'outlier_values': [dp.value for dp in outliers[:10]]  # First 10 outliers
            }

        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Analysis report exported to {filename}")
        except Exception as e:
            print(f"Error exporting report: {e}")

    def print_summary(self) -> None:
        """Print a summary of all datasets and their key statistics."""
        print("=" * 60)
        print("DATA ANALYZER SUMMARY")
        print("=" * 60)

        for name, dataset in self.datasets.items():
            print(f"\nDataset: {name}")
            print("-" * 40)

            stats = dataset.calculate_statistics()
            if stats:
                print(f"Count: {stats['count']:.0f}")
                print(f"Mean: {stats['mean']:.2f}")
                print(f"Median: {stats['median']:.2f}")
                print(f"Std Dev: {stats['std_dev']:.2f}")
                print(f"Range: {stats['min']:.2f} to {stats['max']:.2f}")

                outliers = dataset.detect_outliers()
                print(f"Outliers: {len(outliers)}")
            else:
                print("No data points")


def main():
    """Demonstration of the data analyzer functionality."""
    analyzer = DataAnalyzer()

    # Generate sample datasets
    analyzer.generate_sample_data("Normal Distribution", 100, "normal")
    analyzer.generate_sample_data("Uniform Distribution", 80, "uniform")
    analyzer.generate_sample_data("Exponential Distribution", 60, "exponential")

    # Print summary
    analyzer.print_summary()

    # Compare datasets
    try:
        comparison = analyzer.compare_datasets("Normal Distribution", "Uniform Distribution")
        print(f"\nCorrelation between datasets: {comparison['correlation']:.4f}")
    except Exception as e:
        print(f"Error in comparison: {e}")

    # Export report
    analyzer.export_analysis_report("analysis_report.json")

    print("\nData analysis complete!")


class AdvancedStatistics:
    """Advanced statistical analysis functions."""

    @staticmethod
    def calculate_skewness(values: List[float]) -> float:
        """Calculate the skewness of a dataset."""
        if len(values) < 3:
            return 0.0

        n = len(values)
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)

        if std_dev == 0:
            return 0.0

        skewness = sum(((x - mean) / std_dev) ** 3 for x in values) * n / ((n - 1) * (n - 2))
        return skewness

    @staticmethod
    def calculate_kurtosis(values: List[float]) -> float:
        """Calculate the kurtosis of a dataset."""
        if len(values) < 4:
            return 0.0

        n = len(values)
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)

        if std_dev == 0:
            return 0.0

        kurtosis = sum(((x - mean) / std_dev) ** 4 for x in values) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))
        kurtosis -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return kurtosis

    @staticmethod
    def moving_average(values: List[float], window_size: int) -> List[float]:
        """Calculate moving average with specified window size."""
        if window_size <= 0 or window_size > len(values):
            return values.copy()

        moving_avg = []
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            moving_avg.append(sum(window) / window_size)

        return moving_avg

    @staticmethod
    def exponential_smoothing(values: List[float], alpha: float = 0.3) -> List[float]:
        """Apply exponential smoothing to the data."""
        if not values or alpha <= 0 or alpha > 1:
            return values.copy()

        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed_value = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
            smoothed.append(smoothed_value)

        return smoothed


class DataVisualizer:
    """Simple text-based data visualization tools."""

    @staticmethod
    def create_histogram(values: List[float], bins: int = 10) -> None:
        """Create a simple text-based histogram."""
        if not values:
            print("No data to visualize")
            return

        min_val, max_val = min(values), max(values)
        bin_width = (max_val - min_val) / bins

        bin_counts = [0] * bins
        for value in values:
            bin_index = min(int((value - min_val) / bin_width), bins - 1)
            bin_counts[bin_index] += 1

        max_count = max(bin_counts) if bin_counts else 1
        scale = 50 / max_count  # Scale to 50 characters max

        print("\nHistogram:")
        print("-" * 60)
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            bar_length = int(bin_counts[i] * scale)
            bar = "█" * bar_length

            print(f"{bin_start:6.1f}-{bin_end:6.1f} |{bar:<50} ({bin_counts[i]})")

    @staticmethod
    def create_box_plot_ascii(values: List[float]) -> None:
        """Create a simple ASCII box plot."""
        if not values:
            print("No data to visualize")
            return

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        q1 = sorted_vals[n // 4]
        median = statistics.median(sorted_vals)
        q3 = sorted_vals[3 * n // 4]
        min_val, max_val = min(values), max(values)

        # Simple ASCII box plot
        print("\nBox Plot:")
        print("-" * 60)
        print(f"Min: {min_val:.2f}")
        print(f"Q1:  {q1:.2f}    ┌─────┐")
        print(f"Med: {median:.2f}    │  ●  │")
        print(f"Q3:  {q3:.2f}    └─────┘")
        print(f"Max: {max_val:.2f}")


class TimeSeriesAnalyzer:
    """Specialized analyzer for time series data."""

    def __init__(self, dataset: DataSet):
        self.dataset = dataset

    def detect_trend(self) -> str:
        """Detect overall trend in time series."""
        if len(self.dataset.data_points) < 2:
            return "insufficient_data"

        # Sort by timestamp
        sorted_points = sorted(self.dataset.data_points, key=lambda dp: dp.timestamp)
        values = [dp.value for dp in sorted_points]

        # Simple linear trend detection
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "no_trend"

        slope = numerator / denominator

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def find_seasonal_patterns(self, period: int = 7) -> Dict[int, float]:
        """Find seasonal patterns with given period."""
        if len(self.dataset.data_points) < period:
            return {}

        # Group by position in period
        period_groups = {}
        for i, dp in enumerate(self.dataset.data_points):
            period_pos = i % period
            if period_pos not in period_groups:
                period_groups[period_pos] = []
            period_groups[period_pos].append(dp.value)

        # Calculate average for each position
        period_averages = {}
        for pos, values in period_groups.items():
            period_averages[pos] = statistics.mean(values)

        return period_averages

    def calculate_volatility(self, window_size: int = 30) -> List[float]:
        """Calculate rolling volatility."""
        values = self.dataset.get_values()
        if len(values) < window_size:
            return [statistics.stdev(values)] if len(values) > 1 else [0.0]

        volatilities = []
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            volatility = statistics.stdev(window) if len(window) > 1 else 0.0
            volatilities.append(volatility)

        return volatilities


def run_comprehensive_analysis():
    """Run a comprehensive analysis demonstration."""
    print("=" * 80)
    print("COMPREHENSIVE DATA ANALYSIS DEMONSTRATION")
    print("=" * 80)

    # Create analyzer with multiple datasets
    analyzer = DataAnalyzer()

    # Generate different types of data
    analyzer.generate_sample_data("Stock Prices", 200, "normal")
    analyzer.generate_sample_data("Daily Temperature", 150, "uniform")
    analyzer.generate_sample_data("Website Traffic", 100, "exponential")

    # Perform advanced analysis
    for name, dataset in analyzer.datasets.items():
        print(f"\n{name} Analysis:")
        print("-" * 50)

        values = dataset.get_values()
        stats = dataset.calculate_statistics()

        # Basic statistics
        print(f"Count: {stats['count']}")
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Std Dev: {stats['std_dev']:.2f}")

        # Advanced statistics
        adv_stats = AdvancedStatistics()
        skewness = adv_stats.calculate_skewness(values)
        kurtosis = adv_stats.calculate_kurtosis(values)

        print(f"Skewness: {skewness:.3f}")
        print(f"Kurtosis: {kurtosis:.3f}")

        # Time series analysis
        ts_analyzer = TimeSeriesAnalyzer(dataset)
        trend = ts_analyzer.detect_trend()
        print(f"Trend: {trend}")

        # Visualizations
        visualizer = DataVisualizer()
        visualizer.create_histogram(values[:50], bins=8)  # Limit for display
        visualizer.create_box_plot_ascii(values)

        # Moving averages
        moving_avg = adv_stats.moving_average(values, 10)
        print(f"10-period moving average (last 5): {moving_avg[-5:]}")

        print("\n" + "="*50)


if __name__ == "__main__":
    # Run main demonstration
    main()

    # Run comprehensive analysis
    run_comprehensive_analysis()
