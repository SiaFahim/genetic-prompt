#!/usr/bin/env python3
"""
Simple results analyzer for Jupyter notebook use.
This script provides functions to analyze and display experiment results in a notebook.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config


def get_latest_experiment_results():
    """Get results from the most recent experiment."""
    data_dir = config.get_data_dir()
    
    # Look for the most recent experiment directory
    experiments_dir = data_dir / "experiments"
    if not experiments_dir.exists():
        return None
    
    # Find most recent experiment
    experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    if not experiment_dirs:
        return None
    
    # Sort by modification time
    latest_exp = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    
    # Look for plots
    plots_dir = data_dir / "plots" / latest_exp.name
    plots = []
    if plots_dir.exists():
        plots = list(plots_dir.glob("*.png"))
    
    # Basic experiment info
    results = {
        'experiment_id': latest_exp.name,
        'experiment_name': latest_exp.name.split('_')[0] if '_' in latest_exp.name else latest_exp.name,
        'status': 'interrupted',  # Since we know it was interrupted
        'plots': plots,
        'has_data': len(plots) > 0
    }
    
    # Try to load any available metadata
    metadata_file = latest_exp / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                results.update(metadata)
        except:
            pass
    
    return results


def display_experiment_summary():
    """Display a summary of the latest experiment for notebook use."""
    results = get_latest_experiment_results()
    
    if not results:
        print("⚠️  No experiment results found.")
        print("\nTo run an experiment, use one of these methods:")
        print("1. Use the experiment runner cells in this notebook")
        print("2. Run: python scripts/run_experiment.py --preset quick_test")
        return None
    
    print("📊 Latest Experiment Results Summary:")
    print("=" * 50)
    print(f"Experiment: {results['experiment_name']}")
    print(f"ID: {results['experiment_id']}")
    print(f"Status: {results['status']}")
    
    if results.get('created_at'):
        import time
        print(f"Created: {time.ctime(results['created_at'])}")
    
    if results['plots']:
        print(f"\n📈 Available Visualizations: {len(results['plots'])}")
        for plot in results['plots']:
            print(f"   - {plot.name}")
    else:
        print("\n📈 No visualizations available")
    
    # Show what we know about the experiment
    if results.get('description'):
        print(f"\nDescription: {results['description']}")
    
    print(f"\n💡 This experiment appears to have been interrupted early.")
    print(f"   The system was likely stopped during the first generation.")
    print(f"   You can run a new experiment using the cells above.")
    
    return results


def show_available_plots():
    """Show available plots from the latest experiment."""
    results = get_latest_experiment_results()
    
    if not results or not results['plots']:
        print("📈 No plots available from recent experiments.")
        return []
    
    print("📊 Available Experiment Plots:")
    print("=" * 40)
    
    plot_info = []
    for plot_path in results['plots']:
        plot_info.append({
            'name': plot_path.name,
            'path': str(plot_path),
            'description': get_plot_description(plot_path.name)
        })
        print(f"🔹 {plot_path.name}")
        print(f"   {get_plot_description(plot_path.name)}")
        print(f"   Path: {plot_path}")
        print()
    
    return plot_info


def get_plot_description(plot_name: str) -> str:
    """Get description for a plot based on its name."""
    if 'evolution_progress' in plot_name:
        return "Evolution progress showing fitness over generations"
    elif 'convergence' in plot_name:
        return "Convergence analysis and stopping criteria"
    elif 'diversity' in plot_name:
        return "Population diversity over time"
    elif 'performance' in plot_name:
        return "Performance metrics and resource usage"
    elif 'fitness_distribution' in plot_name:
        return "Distribution of fitness values in population"
    else:
        return "Experiment visualization plot"


def display_plots_in_notebook():
    """Display plots in Jupyter notebook."""
    try:
        from IPython.display import Image, display
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  IPython or matplotlib not available. Cannot display plots in notebook.")
        return
    
    results = get_latest_experiment_results()
    
    if not results or not results['plots']:
        print("📈 No plots available to display.")
        return
    
    print("📊 Experiment Visualizations:")
    print("=" * 40)
    
    for plot_path in results['plots']:
        print(f"\n🔹 {plot_path.name}")
        print(f"   {get_plot_description(plot_path.name)}")
        
        try:
            display(Image(str(plot_path)))
        except Exception as e:
            print(f"   ⚠️  Could not display plot: {e}")


def get_experiment_recommendations():
    """Get recommendations for running successful experiments."""
    print("💡 Recommendations for Running Successful Experiments:")
    print("=" * 60)
    
    print("\n🎯 For Quick Testing:")
    print("   • Use 'quick_test' preset (10 population, 15 generations)")
    print("   • Start with 20-30 evaluation problems")
    print("   • Default GPT-4o provides best performance (use gpt-3.5-turbo for cost efficiency)")
    
    print("\n🔬 For Research:")
    print("   • Use 'standard' preset (50 population, 100 generations)")
    print("   • Use 100+ evaluation problems for accuracy")
    print("   • Enable checkpoints for long runs")
    
    print("\n⚙️ Configuration Tips:")
    print("   • Set target_fitness to 0.75-0.85 for reasonable stopping")
    print("   • Use convergence_patience of 10-20 generations")
    print("   • Enable caching to avoid re-evaluating identical prompts")
    
    print("\n🚨 Avoiding Interruptions:")
    print("   • Ensure stable internet connection for API calls")
    print("   • Set reasonable API rate limits")
    print("   • Monitor system resources (memory, CPU)")
    print("   • Use smaller problem sets for initial testing")
    
    print("\n📊 Monitoring Progress:")
    print("   • Watch for fitness improvements over generations")
    print("   • Check diversity scores (should be 0.3-0.8)")
    print("   • Monitor evaluation times for performance issues")


# Convenience function for notebook use
def analyze_results():
    """Main function to call from notebook for result analysis."""
    print("🔍 GSM8K Experiment Results Analysis")
    print("=" * 50)
    
    # Display summary
    results = display_experiment_summary()
    
    if results and results['plots']:
        print("\n" + "="*50)
        display_plots_in_notebook()
    
    print("\n" + "="*50)
    get_experiment_recommendations()
    
    return results


# For direct script execution
if __name__ == "__main__":
    analyze_results()
