#!/usr/bin/env python3
"""
Analyze results from an existing GSM8K genetic algorithm experiment.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config
from src.utils.experiment_manager import ExperimentManager


def find_latest_experiment() -> Optional[str]:
    """Find the most recent experiment."""
    experiment_manager = ExperimentManager()
    experiments = experiment_manager.list_experiments()
    
    if not experiments:
        return None
    
    # Sort by creation time, most recent first
    experiments.sort(key=lambda x: x.created_at, reverse=True)
    return experiments[0].experiment_id


def load_experiment_data(experiment_id: str) -> Dict[str, Any]:
    """Load all available data for an experiment."""
    data_dir = config.get_data_dir()
    experiment_dir = data_dir / "experiments" / experiment_id
    
    experiment_data = {
        'experiment_id': experiment_id,
        'logs': None,
        'checkpoints': [],
        'plots': [],
        'performance': None,
        'final_results': None
    }
    
    # Load experiment metadata
    metadata_file = experiment_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            experiment_data['metadata'] = json.load(f)
    
    # Load final results
    results_file = experiment_dir / "final_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            experiment_data['final_results'] = json.load(f)
    
    # Load performance data
    performance_file = experiment_dir / "performance_report.json"
    if performance_file.exists():
        with open(performance_file, 'r') as f:
            experiment_data['performance'] = json.load(f)
    
    # Find checkpoints
    checkpoints_dir = experiment_dir / "checkpoints"
    if checkpoints_dir.exists():
        experiment_data['checkpoints'] = list(checkpoints_dir.glob("generation_*.json"))
        experiment_data['checkpoints'].sort()
    
    # Find plots
    plots_dir = data_dir / "plots" / experiment_id
    if plots_dir.exists():
        experiment_data['plots'] = list(plots_dir.glob("*.png"))
    
    # Load evolution log
    log_file = experiment_dir / "evolution.log"
    if log_file.exists():
        with open(log_file, 'r') as f:
            experiment_data['logs'] = f.read()
    
    return experiment_data


def analyze_experiment_progress(experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze experiment progress from checkpoints and logs."""
    analysis = {
        'total_generations': 0,
        'best_fitness_progression': [],
        'mean_fitness_progression': [],
        'diversity_progression': [],
        'evaluation_times': [],
        'convergence_info': None,
        'final_status': 'unknown'
    }
    
    # Analyze checkpoints
    for checkpoint_file in experiment_data['checkpoints']:
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            generation = checkpoint.get('generation', 0)
            analysis['total_generations'] = max(analysis['total_generations'], generation + 1)
            
            # Extract generation results
            if 'generation_results' in checkpoint:
                for result in checkpoint['generation_results']:
                    if result.get('generation') == generation:
                        analysis['best_fitness_progression'].append({
                            'generation': generation,
                            'fitness': result.get('best_fitness', 0)
                        })
                        analysis['mean_fitness_progression'].append({
                            'generation': generation,
                            'fitness': result.get('mean_fitness', 0)
                        })
                        analysis['diversity_progression'].append({
                            'generation': generation,
                            'diversity': result.get('diversity', 0)
                        })
                        analysis['evaluation_times'].append({
                            'generation': generation,
                            'time': result.get('evaluation_time', 0)
                        })
                        
                        # Check convergence status
                        conv_status = result.get('convergence_status')
                        if conv_status and conv_status.get('converged'):
                            analysis['convergence_info'] = {
                                'generation': generation,
                                'reason': conv_status.get('reason'),
                                'confidence': conv_status.get('confidence', 0)
                            }
        except Exception as e:
            print(f"⚠️  Error reading checkpoint {checkpoint_file}: {e}")
    
    # Determine final status
    if experiment_data['final_results']:
        analysis['final_status'] = 'completed'
    elif analysis['convergence_info']:
        analysis['final_status'] = 'converged'
    elif analysis['total_generations'] > 0:
        analysis['final_status'] = 'interrupted'
    
    return analysis


def print_experiment_summary(experiment_data: Dict[str, Any], analysis: Dict[str, Any]):
    """Print comprehensive experiment summary."""
    print("📊 Experiment Results Summary")
    print("=" * 50)
    
    # Basic info
    print(f"Experiment ID: {experiment_data['experiment_id']}")
    
    if experiment_data.get('metadata'):
        metadata = experiment_data['metadata']
        print(f"Name: {metadata.get('experiment_name', 'Unknown')}")
        print(f"Description: {metadata.get('description', 'No description')}")
        print(f"Created: {time.ctime(metadata.get('created_at', 0))}")
        print(f"Status: {metadata.get('status', 'unknown')}")
    
    # Progress info
    print(f"\n🧬 Evolution Progress:")
    print(f"   Total Generations: {analysis['total_generations']}")
    print(f"   Final Status: {analysis['final_status']}")
    
    if analysis['best_fitness_progression']:
        best_fitness = max(analysis['best_fitness_progression'], key=lambda x: x['fitness'])
        print(f"   Best Fitness Achieved: {best_fitness['fitness']:.3f} (Generation {best_fitness['generation']})")
        
        final_fitness = analysis['best_fitness_progression'][-1]
        print(f"   Final Best Fitness: {final_fitness['fitness']:.3f}")
    
    if analysis['mean_fitness_progression']:
        final_mean = analysis['mean_fitness_progression'][-1]
        print(f"   Final Mean Fitness: {final_mean['fitness']:.3f}")
    
    if analysis['diversity_progression']:
        final_diversity = analysis['diversity_progression'][-1]
        print(f"   Final Diversity: {final_diversity['diversity']:.3f}")
    
    # Convergence info
    if analysis['convergence_info']:
        conv = analysis['convergence_info']
        print(f"\n🎯 Convergence Information:")
        print(f"   Converged at Generation: {conv['generation']}")
        print(f"   Convergence Reason: {conv['reason']}")
        print(f"   Confidence: {conv.get('confidence', 0):.3f}")
    
    # Performance info
    if experiment_data.get('performance'):
        perf = experiment_data['performance']
        print(f"\n⚡ Performance Statistics:")
        print(f"   Total Runtime: {perf.get('total_runtime_minutes', 0):.1f} minutes")
        
        if 'api_usage' in perf:
            api = perf['api_usage']
            print(f"   Total API Calls: {api.get('total_calls', 0)}")
            print(f"   Total Tokens: {api.get('total_tokens', 0):,}")
            print(f"   Average Tokens per Call: {api.get('tokens_per_call', 0):.1f}")
        
        if 'cache_performance' in perf:
            cache = perf['cache_performance']
            print(f"   Cache Hit Rate: {cache.get('hit_rate', 0):.1%}")
    
    # Final results
    if experiment_data.get('final_results'):
        results = experiment_data['final_results']
        print(f"\n🏆 Final Results:")
        print(f"   Best Fitness: {results.get('best_fitness', 0):.3f}")
        print(f"   Total Evaluations: {results.get('total_evaluations', 0)}")
        print(f"   Convergence Reason: {results.get('convergence_reason', 'unknown')}")
        
        if results.get('best_genome_text'):
            print(f"\n🎯 Best Evolved Prompt:")
            print(f'   "{results["best_genome_text"]}"')
    
    # Available data
    print(f"\n📁 Available Data:")
    print(f"   Checkpoints: {len(experiment_data['checkpoints'])}")
    print(f"   Plots: {len(experiment_data['plots'])}")
    print(f"   Logs: {'✅' if experiment_data['logs'] else '❌'}")
    print(f"   Performance Data: {'✅' if experiment_data['performance'] else '❌'}")


def show_fitness_progression(analysis: Dict[str, Any]):
    """Show fitness progression over generations."""
    if not analysis['best_fitness_progression']:
        print("⚠️  No fitness progression data available")
        return
    
    print("\n📈 Fitness Progression:")
    print("-" * 40)
    print("Gen | Best Fitness | Mean Fitness | Diversity")
    print("-" * 40)
    
    for i, best_point in enumerate(analysis['best_fitness_progression']):
        gen = best_point['generation']
        best_fit = best_point['fitness']
        
        mean_fit = 0
        diversity = 0
        
        # Find corresponding mean fitness and diversity
        for mean_point in analysis['mean_fitness_progression']:
            if mean_point['generation'] == gen:
                mean_fit = mean_point['fitness']
                break
        
        for div_point in analysis['diversity_progression']:
            if div_point['generation'] == gen:
                diversity = div_point['diversity']
                break
        
        print(f"{gen:3d} | {best_fit:11.3f} | {mean_fit:11.3f} | {diversity:8.3f}")


def main():
    """Main analysis function."""
    print("🔍 GSM8K Experiment Results Analyzer")
    print("=" * 50)
    
    # Find latest experiment
    experiment_id = find_latest_experiment()
    if not experiment_id:
        print("❌ No experiments found!")
        return
    
    print(f"📋 Analyzing experiment: {experiment_id}")
    
    # Load experiment data
    experiment_data = load_experiment_data(experiment_id)
    
    # Analyze progress
    analysis = analyze_experiment_progress(experiment_data)
    
    # Print summary
    print_experiment_summary(experiment_data, analysis)
    
    # Show detailed progression
    show_fitness_progression(analysis)
    
    # Show available plots
    if experiment_data['plots']:
        print(f"\n📊 Available Visualizations:")
        for plot in experiment_data['plots']:
            print(f"   - {plot.name}")
        print(f"\nPlots location: {config.get_data_dir() / 'plots' / experiment_id}")
    
    print(f"\n✅ Analysis complete!")
    
    # Return data for notebook use
    return {
        'experiment_data': experiment_data,
        'analysis': analysis,
        'experiment_id': experiment_id
    }


if __name__ == "__main__":
    results = main()
