#!/usr/bin/env python3
"""
Comprehensive test suite for the monitoring and logging system.
"""

import sys
import time
import random
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config
from src.utils.evolution_logging import EvolutionLogger
from src.utils.visualization import EvolutionVisualizer
from src.utils.experiment_manager import ExperimentManager
from src.utils.performance_monitor import PerformanceMonitor
from src.genetics.evolution import EvolutionConfig


def test_logging_system():
    """Test comprehensive logging system."""
    print("📝 Testing Logging System")
    print("-" * 40)
    
    # Create logger
    logger = EvolutionLogger("test_logging", "DEBUG")
    
    # Test experiment start
    test_config = {
        'population_size': 30,
        'max_generations': 50,
        'mutation_rate': 0.25
    }
    logger.log_experiment_start(test_config)
    print("✅ Experiment start logged")
    
    # Test generation logging
    for gen in range(1, 6):
        result = {
            'best_fitness': 0.4 + gen * 0.08,
            'mean_fitness': 0.2 + gen * 0.06,
            'diversity': 0.9 - gen * 0.1
        }
        logger.log_generation(gen, result)
    print("✅ Generation logs created")
    
    # Test performance logging
    logger.log_performance("evaluation_time", 42.5, "population_30")
    logger.log_api_usage(150, 7500, 0.375)
    print("✅ Performance logs created")
    
    # Test error logging
    try:
        raise RuntimeError("Test error for logging validation")
    except Exception as e:
        logger.log_error(e, "test_error_context")
    print("✅ Error logs created")
    
    # Test convergence logging
    logger.log_convergence("fitness_plateau", 15, 0.82)
    print("✅ Convergence logged")
    
    # Test experiment end
    final_results = {
        'best_fitness': 0.82,
        'total_generations': 15,
        'convergence_reason': 'fitness_plateau'
    }
    logger.log_experiment_end(final_results)
    print("✅ Experiment end logged")
    
    # Verify log summary
    summary = logger.get_log_summary()
    assert summary['total_generations'] == 5
    assert summary['total_errors'] == 1
    print(f"✅ Log summary verified: {summary['total_generations']} generations")
    
    return True


def test_visualization_system():
    """Test visualization system."""
    print("\n📊 Testing Visualization System")
    print("-" * 40)
    
    # Create visualizer
    visualizer = EvolutionVisualizer("test_visualization", save_plots=True)
    
    # Generate realistic evolution data
    random.seed(42)
    
    for generation in range(1, 16):
        # Simulate realistic fitness evolution
        if generation <= 5:
            # Initial rapid improvement
            best_fitness = 0.3 + generation * 0.08
        elif generation <= 10:
            # Slower improvement
            best_fitness = 0.7 + (generation - 5) * 0.03
        else:
            # Plateau with small variations
            best_fitness = 0.85 + random.uniform(-0.02, 0.02)
        
        mean_fitness = best_fitness - random.uniform(0.1, 0.2)
        diversity = max(0.1, 0.95 - generation * 0.05 + random.uniform(-0.05, 0.05))
        eval_time = 25 + random.uniform(-5, 15)
        
        result = {
            'best_fitness': best_fitness,
            'mean_fitness': mean_fitness,
            'diversity': diversity,
            'evaluation_time': eval_time
        }
        
        # Add convergence event
        if generation == 12:
            result['convergence_status'] = {
                'converged': True,
                'reason': 'fitness_plateau'
            }
        
        visualizer.update_data(generation, result)
    
    print("✅ Evolution data generated")
    
    # Update plots
    visualizer.update_plots()
    print("✅ Main plots updated")
    
    # Create fitness distribution plot
    test_population_fitness = [random.uniform(0.4, 0.9) for _ in range(40)]
    dist_plot = visualizer.create_fitness_distribution_plot(test_population_fitness, 12)
    print(f"✅ Fitness distribution plot created: {Path(dist_plot).name}")
    
    # Create convergence analysis
    conv_plot = visualizer.create_convergence_analysis_plot()
    print(f"✅ Convergence analysis plot created: {Path(conv_plot).name}")
    
    # Save final plots
    visualizer.save_final_plots()
    print("✅ Final plots saved")
    
    # Verify statistics
    stats = visualizer.get_statistics()
    assert stats['total_generations'] == 15
    assert stats['total_convergence_events'] == 1
    print(f"✅ Visualization statistics verified: {stats['total_generations']} generations")
    
    return True


def test_experiment_manager():
    """Test experiment management system."""
    print("\n🧪 Testing Experiment Manager")
    print("-" * 40)
    
    # Create manager
    manager = ExperimentManager()
    
    # Create multiple test experiments
    experiments = []
    
    for i in range(3):
        config = EvolutionConfig(
            population_size=20 + i * 10,
            max_generations=50 + i * 25,
            target_fitness=0.8 + i * 0.05
        )
        
        exp_id = manager.create_experiment(
            experiment_name=f"test_experiment_{i+1}",
            description=f"Test experiment {i+1} for validation",
            evolution_config=config
        )
        experiments.append(exp_id)
    
    print(f"✅ Created {len(experiments)} experiments")
    
    # Start and complete experiments
    for i, exp_id in enumerate(experiments):
        # Start experiment
        logger, visualizer = manager.start_experiment(exp_id)
        
        # Simulate completion
        final_results = {
            'best_fitness': 0.75 + i * 0.05,
            'total_generations': 20 + i * 10,
            'convergence_reason': ['target_reached', 'fitness_plateau', 'max_generations'][i],
            'total_time': 100.0 + i * 50
        }
        
        if i == 2:  # Fail the last experiment
            manager.fail_experiment(exp_id, "Simulated failure for testing")
        else:
            manager.complete_experiment(exp_id, final_results)
    
    print("✅ Experiments started and completed")
    
    # List experiments
    all_experiments = manager.list_experiments()
    completed_experiments = manager.list_experiments(status_filter='completed')
    failed_experiments = manager.list_experiments(status_filter='failed')
    
    print(f"✅ Found {len(all_experiments)} total experiments")
    print(f"   - {len(completed_experiments)} completed")
    print(f"   - {len(failed_experiments)} failed")
    
    # Get summary
    summary = manager.get_experiment_summary()
    assert summary['total_experiments'] >= 3
    assert summary['status_counts']['completed'] >= 2
    assert summary['status_counts']['failed'] >= 1
    print(f"✅ Experiment summary verified: {summary['total_experiments']} experiments")
    
    return True


def test_performance_monitor():
    """Test performance monitoring system."""
    print("\n⚡ Testing Performance Monitor")
    print("-" * 40)
    
    # Create monitor
    monitor = PerformanceMonitor("test_performance_monitoring", monitoring_interval=0.5)
    
    # Simulate realistic workload
    for i in range(8):
        # Simulate API calls with varying token usage
        tokens = random.randint(50, 200)
        monitor.record_api_call(tokens_used=tokens)
        
        # Simulate evaluation times
        eval_time = random.uniform(15.0, 45.0)
        monitor.record_evaluation_time(eval_time)
        
        # Simulate cache activity
        if random.random() < 0.7:  # 70% cache hit rate
            monitor.record_cache_hit()
        else:
            monitor.record_cache_miss()
        
        # Take performance snapshot
        metrics = monitor.take_snapshot()
        
        if i % 2 == 0:  # Print every other snapshot
            print(f"   Snapshot {i+1}: CPU={metrics.cpu_percent:.1f}%, "
                  f"Memory={metrics.memory_used_mb:.1f}MB, "
                  f"API Rate={metrics.api_calls_rate:.1f}/min")
        
        time.sleep(0.1)  # Small delay to simulate work
    
    print("✅ Performance data collected")
    
    # Check for alerts
    alerts = monitor.check_performance_alerts()
    print(f"✅ Performance alerts checked: {len(alerts)} alerts")
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    assert summary['snapshots_taken'] == 8
    assert summary['api_usage']['total_calls'] == 8
    print(f"✅ Performance summary verified: {summary['snapshots_taken']} snapshots")
    
    # Save performance report
    report_path = monitor.save_performance_report()
    print(f"✅ Performance report saved: {Path(report_path).name}")
    
    # Get recommendations
    recommendations = monitor.get_resource_recommendations()
    print(f"✅ Resource recommendations: {len(recommendations)} suggestions")
    
    return True


def test_integrated_monitoring():
    """Test integrated monitoring system."""
    print("\n🔗 Testing Integrated Monitoring")
    print("-" * 40)
    
    # Create integrated monitoring setup
    manager = ExperimentManager()
    
    # Create experiment
    config = EvolutionConfig(
        population_size=25,
        max_generations=30,
        target_fitness=0.85
    )
    
    exp_id = manager.create_experiment(
        experiment_name="integrated_monitoring_test",
        description="Test integrated monitoring system",
        evolution_config=config
    )
    
    # Start experiment with all monitoring components
    logger, visualizer = manager.start_experiment(exp_id)
    monitor = PerformanceMonitor(exp_id, monitoring_interval=1.0)
    
    print("✅ Integrated monitoring setup created")
    
    # Simulate evolution with all monitoring active
    for generation in range(1, 6):
        # Simulate generation data
        best_fitness = 0.5 + generation * 0.07
        mean_fitness = best_fitness - 0.15
        diversity = 0.8 - generation * 0.1
        eval_time = 30.0 + random.uniform(-5, 10)
        
        result = {
            'best_fitness': best_fitness,
            'mean_fitness': mean_fitness,
            'diversity': diversity,
            'evaluation_time': eval_time
        }
        
        # Log generation
        logger.log_generation(generation, result)
        
        # Update visualization
        visualizer.update_data(generation, result)
        
        # Record performance
        monitor.record_api_call(tokens_used=random.randint(80, 150))
        monitor.record_evaluation_time(eval_time)
        monitor.record_cache_hit() if random.random() < 0.6 else monitor.record_cache_miss()
        
        # Take performance snapshot
        perf_metrics = monitor.take_snapshot()
        
        # Log performance
        logger.log_performance("cpu_usage", perf_metrics.cpu_percent)
        logger.log_performance("memory_usage", perf_metrics.memory_used_mb)
        
        print(f"   Generation {generation}: fitness={best_fitness:.3f}, "
              f"CPU={perf_metrics.cpu_percent:.1f}%")
    
    # Finalize monitoring
    visualizer.update_plots()
    visualizer.save_final_plots()
    
    final_results = {
        'best_fitness': 0.85,
        'total_generations': 5,
        'convergence_reason': 'target_reached',
        'total_time': 150.0
    }
    
    logger.log_experiment_end(final_results)
    manager.complete_experiment(exp_id, final_results)
    monitor.save_performance_report()
    
    print("✅ Integrated monitoring test completed")
    
    # Verify all components worked together
    log_summary = logger.get_log_summary()
    viz_stats = visualizer.get_statistics()
    perf_summary = monitor.get_performance_summary()
    
    assert log_summary['total_generations'] == 5
    assert viz_stats['total_generations'] == 5
    assert perf_summary['snapshots_taken'] == 5
    
    print("✅ All monitoring components verified")
    
    return True


def run_comprehensive_test():
    """Run comprehensive monitoring system test."""
    print("🧪 Comprehensive Monitoring System Test Suite")
    print("=" * 60)
    
    # Run tests
    tests = [
        test_logging_system,
        test_visualization_system,
        test_experiment_manager,
        test_performance_monitor,
        test_integrated_monitoring
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All monitoring system tests passed!")
        print("\n🚀 Monitoring system ready for production use!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    if not success:
        sys.exit(1)
