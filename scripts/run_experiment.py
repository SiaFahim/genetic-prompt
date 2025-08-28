#!/usr/bin/env python3
"""
Command-line interface for running GSM8K genetic algorithm experiments.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.main_runner import run_gsm8k_experiment
from src.config.experiment_configs import ConfigurationManager, ExperimentConfig
from dataclasses import asdict


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run GSM8K genetic algorithm evolution experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test
  python scripts/run_experiment.py --preset quick_test
  
  # Run standard experiment with custom name
  python scripts/run_experiment.py --preset standard --name "my_experiment"
  
  # Run with custom parameters
  python scripts/run_experiment.py --preset standard --population-size 30 --max-generations 50
  
  # List available presets
  python scripts/run_experiment.py --list-presets
  
  # Run thorough experiment
  python scripts/run_experiment.py --preset thorough --description "Full evolution run"
        """
    )
    
    # Main options
    parser.add_argument('--preset', type=str, default='standard',
                       help='Experiment preset to use (default: standard)')
    parser.add_argument('--name', type=str,
                       help='Custom experiment name')
    parser.add_argument('--description', type=str,
                       help='Experiment description')
    
    # Evolution parameters
    parser.add_argument('--population-size', type=int,
                       help='Population size')
    parser.add_argument('--max-generations', type=int,
                       help='Maximum number of generations')
    parser.add_argument('--crossover-rate', type=float,
                       help='Crossover rate (0.0-1.0)')
    parser.add_argument('--mutation-rate', type=float,
                       help='Mutation rate (0.0-1.0)')
    parser.add_argument('--elite-size', type=int,
                       help='Number of elite individuals to preserve')
    parser.add_argument('--target-fitness', type=float,
                       help='Target fitness for early stopping')
    parser.add_argument('--convergence-patience', type=int,
                       help='Generations to wait before convergence')
    
    # Evaluation parameters
    parser.add_argument('--max-problems', type=int,
                       help='Maximum number of problems to evaluate on')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable evaluation caching')
    parser.add_argument('--batch-size', type=int,
                       help='Evaluation batch size')
    
    # Seed parameters
    parser.add_argument('--seed-strategy', type=str, choices=['balanced', 'diverse', 'random'],
                       help='Seed selection strategy')
    
    # API parameters
    parser.add_argument('--model', type=str,
                       help='LLM model to use (default: gpt-4o, alternatives: gpt-3.5-turbo, claude-3-sonnet-20240229)')
    parser.add_argument('--temperature', type=float,
                       help='LLM temperature')
    parser.add_argument('--max-tokens', type=int,
                       help='Maximum tokens per LLM response')
    
    # Output parameters
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--no-checkpoints', action='store_true',
                       help='Disable checkpoint saving')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    # Utility options
    parser.add_argument('--list-presets', action='store_true',
                       help='List available experiment presets')
    parser.add_argument('--show-preset', type=str,
                       help='Show details of a specific preset')
    parser.add_argument('--validate-config', action='store_true',
                       help='Validate configuration and exit')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without running experiment')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize configuration manager
    config_manager = ConfigurationManager()
    
    # Handle utility options
    if args.list_presets:
        print("📋 Available Experiment Presets:")
        print("=" * 40)
        preset_info = config_manager.get_preset_info()
        for name, info in preset_info.items():
            print(f"\n🔹 {name}")
            print(f"   Name: {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Type: {info['type']}")
            print(f"   Population: {info['population_size']}")
            print(f"   Generations: {info['max_generations']}")
            print(f"   Problems: {info['max_problems']}")
        return
    
    if args.show_preset:
        preset_config = config_manager.get_preset(args.show_preset)
        if preset_config is None:
            print(f"❌ Unknown preset: {args.show_preset}")
            print(f"Available presets: {', '.join(config_manager.list_presets())}")
            return
        
        print(config_manager.get_config_summary(preset_config))
        return
    
    # Get base configuration
    if args.preset not in config_manager.list_presets():
        print(f"❌ Unknown preset: {args.preset}")
        print(f"Available presets: {', '.join(config_manager.list_presets())}")
        return
    
    base_config = config_manager.get_preset(args.preset)
    
    # Apply command-line modifications
    modifications = {}
    
    if args.name:
        modifications['name'] = args.name
    if args.description:
        modifications['description'] = args.description
    if args.population_size:
        modifications['population_size'] = args.population_size
    if args.max_generations:
        modifications['max_generations'] = args.max_generations
    if args.crossover_rate is not None:
        modifications['crossover_rate'] = args.crossover_rate
    if args.mutation_rate is not None:
        modifications['mutation_rate'] = args.mutation_rate
    if args.elite_size:
        modifications['elite_size'] = args.elite_size
    if args.target_fitness is not None:
        modifications['target_fitness'] = args.target_fitness
    if args.convergence_patience:
        modifications['convergence_patience'] = args.convergence_patience
    if args.max_problems:
        modifications['max_problems'] = args.max_problems
    if args.no_cache:
        modifications['use_cache'] = False
    if args.batch_size:
        modifications['batch_size'] = args.batch_size
    if args.seed_strategy:
        modifications['seed_strategy'] = args.seed_strategy
    if args.model:
        modifications['model_name'] = args.model
    if args.temperature is not None:
        modifications['temperature'] = args.temperature
    if args.max_tokens:
        modifications['max_tokens'] = args.max_tokens
    if args.no_plots:
        modifications['save_plots'] = False
    if args.no_checkpoints:
        modifications['save_checkpoints'] = False
    if args.quiet:
        modifications['verbose'] = False
    
    # Create final configuration
    if modifications:
        config = config_manager.create_custom_config(args.preset, modifications)
    else:
        config = base_config
    
    # Validate configuration
    validation_errors = config_manager.validate_config(config)
    if validation_errors:
        print("❌ Configuration validation failed:")
        for error in validation_errors:
            print(f"   - {error}")
        return
    
    if args.validate_config:
        print("✅ Configuration is valid!")
        print(config_manager.get_config_summary(config))
        return
    
    # Show configuration
    print("🔧 Experiment Configuration:")
    print("-" * 40)
    print(config_manager.get_config_summary(config))
    
    if args.dry_run:
        print("\n🏃 Dry run completed - configuration shown above")
        return
    
    # Convert to dictionary format for runner
    config_dict = asdict(config)
    
    # Convert enums to strings
    for key, value in config_dict.items():
        if hasattr(value, 'value'):
            config_dict[key] = value.value
    
    # Confirm experiment start
    if not args.quiet:
        print(f"\n🚀 Starting experiment: {config.name}")
        print(f"   Population: {config.population_size}")
        print(f"   Generations: {config.max_generations}")
        print(f"   Problems: {config.max_problems}")
        print(f"   Model: {config.model_name}")
        
        try:
            response = input("\nProceed with experiment? [Y/n]: ").strip().lower()
            if response and response not in ['y', 'yes']:
                print("❌ Experiment cancelled by user")
                return
        except KeyboardInterrupt:
            print("\n❌ Experiment cancelled by user")
            return
    
    # Run experiment
    try:
        print("\n" + "=" * 60)
        print("🧬 STARTING GSM8K GENETIC ALGORITHM EVOLUTION")
        print("=" * 60)
        
        results = run_gsm8k_experiment(config_dict)
        
        print("\n" + "=" * 60)
        print("📊 EXPERIMENT RESULTS")
        print("=" * 60)
        
        if results.get('status') == 'completed':
            print("✅ Experiment completed successfully!")
            print(f"   Experiment ID: {results.get('experiment_id', 'unknown')}")
            print(f"   Best Fitness: {results.get('results', {}).get('best_fitness', 0):.3f}")
            print(f"   Total Generations: {results.get('results', {}).get('total_generations', 0)}")
            print(f"   Convergence Reason: {results.get('results', {}).get('convergence_reason', 'unknown')}")
            print(f"   Total Time: {results.get('results', {}).get('total_time', 0):.1f}s")
            
            if results.get('best_prompt'):
                print(f"   Best Prompt: {results.get('best_prompt')[:100]}...")
            
            # Performance summary
            if 'performance' in results:
                perf = results['performance']
                print(f"\n📈 Performance Summary:")
                print(f"   API Calls: {perf.get('api_usage', {}).get('total_calls', 0)}")
                print(f"   Cache Hit Rate: {perf.get('cache_performance', {}).get('hit_rate', 0):.1%}")
                print(f"   Avg Evaluation Time: {perf.get('evaluation_performance', {}).get('average_time_seconds', 0):.1f}s")
            
        else:
            print(f"❌ Experiment failed: {results.get('status', 'unknown')}")
            
    except KeyboardInterrupt:
        print("\n❌ Experiment interrupted by user")
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
