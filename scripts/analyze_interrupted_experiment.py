#!/usr/bin/env python3
"""
Analyze results from an interrupted GSM8K genetic algorithm experiment.
This script looks for any available data including partial checkpoints, logs, and cached evaluations.
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config


def find_all_experiment_data() -> List[Dict[str, Any]]:
    """Find all experiment-related data files."""
    data_dir = config.get_data_dir()
    experiments = []
    
    # Look in experiments directory
    experiments_dir = data_dir / "experiments"
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                experiments.append({
                    'id': exp_dir.name,
                    'path': exp_dir,
                    'type': 'experiment_dir'
                })
    
    # Look for checkpoint files anywhere
    for checkpoint_file in data_dir.rglob("generation_*.json"):
        experiments.append({
            'id': checkpoint_file.parent.name,
            'path': checkpoint_file,
            'type': 'checkpoint'
        })
    
    # Look for any evolution logs
    for log_file in data_dir.rglob("evolution*.log"):
        experiments.append({
            'id': log_file.parent.name,
            'path': log_file,
            'type': 'log'
        })
    
    # Look for any JSON files that might contain results
    for json_file in data_dir.rglob("*.json"):
        if any(keyword in json_file.name.lower() for keyword in ['result', 'checkpoint', 'generation', 'evolution']):
            experiments.append({
                'id': json_file.parent.name,
                'path': json_file,
                'type': 'data_file'
            })
    
    return experiments


def analyze_checkpoint_file(checkpoint_path: Path) -> Dict[str, Any]:
    """Analyze a single checkpoint file."""
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        analysis = {
            'generation': data.get('generation', 0),
            'population_size': len(data.get('population', [])),
            'best_fitness': None,
            'config': data.get('config', {}),
            'generation_results': data.get('generation_results', [])
        }
        
        # Extract best fitness from population
        population = data.get('population', [])
        if population:
            fitnesses = [genome.get('fitness', 0) for genome in population if genome.get('fitness') is not None]
            if fitnesses:
                analysis['best_fitness'] = max(fitnesses)
                analysis['mean_fitness'] = sum(fitnesses) / len(fitnesses)
        
        # Extract from generation results
        gen_results = data.get('generation_results', [])
        if gen_results:
            latest_result = gen_results[-1]
            analysis['best_fitness'] = latest_result.get('best_fitness', analysis['best_fitness'])
            analysis['mean_fitness'] = latest_result.get('mean_fitness', analysis.get('mean_fitness'))
            analysis['diversity'] = latest_result.get('diversity')
            analysis['evaluation_time'] = latest_result.get('evaluation_time')
        
        return analysis
        
    except Exception as e:
        print(f"⚠️  Error analyzing checkpoint {checkpoint_path}: {e}")
        return {}


def analyze_log_file(log_path: Path) -> Dict[str, Any]:
    """Extract information from evolution log file."""
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        analysis = {
            'total_lines': len(log_content.split('\n')),
            'generations_found': [],
            'fitness_values': [],
            'errors': [],
            'api_calls': 0,
            'evaluation_times': []
        }
        
        # Extract generation information
        gen_pattern = r'Generation (\d+).*?Best: ([\d.-]+).*?Mean: ([\d.-]+).*?Diversity: ([\d.-]+)'
        for match in re.finditer(gen_pattern, log_content):
            gen_num = int(match.group(1))
            best_fitness = float(match.group(2))
            mean_fitness = float(match.group(3))
            diversity = float(match.group(4))
            
            analysis['generations_found'].append({
                'generation': gen_num,
                'best_fitness': best_fitness,
                'mean_fitness': mean_fitness,
                'diversity': diversity
            })
        
        # Extract evaluation times
        time_pattern = r'Evaluation time: ([\d.]+)s'
        for match in re.finditer(time_pattern, log_content):
            analysis['evaluation_times'].append(float(match.group(1)))
        
        # Count API calls
        analysis['api_calls'] = log_content.count('API call') + log_content.count('Evaluating')
        
        # Extract errors
        error_lines = [line for line in log_content.split('\n') if 'ERROR' in line or 'Failed' in line]
        analysis['errors'] = error_lines[:10]  # First 10 errors
        
        return analysis
        
    except Exception as e:
        print(f"⚠️  Error analyzing log {log_path}: {e}")
        return {}


def find_best_genome_from_checkpoint(checkpoint_path: Path) -> Optional[str]:
    """Extract the best genome text from a checkpoint."""
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        # Look for best genome in the checkpoint
        best_genome_data = data.get('best_genome_ever')
        if best_genome_data and 'token_ids' in best_genome_data:
            # Convert token IDs back to text
            from src.embeddings.vocabulary import vocabulary
            
            # Load vocabulary if not already loaded
            vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
            if vocab_file.exists():
                vocabulary.load_vocabulary(vocab_file)
            
            token_ids = best_genome_data['token_ids']
            if token_ids and len(vocabulary.id_to_token) > 0:
                tokens = [vocabulary.id_to_token.get(token_id, f'<UNK_{token_id}>') for token_id in token_ids]
                return ' '.join(tokens)
        
        # Alternative: look in population for highest fitness genome
        population = data.get('population', [])
        if population:
            best_genome = max(population, key=lambda x: x.get('fitness', -float('inf')))
            if 'token_ids' in best_genome:
                from src.embeddings.vocabulary import vocabulary
                token_ids = best_genome['token_ids']
                if token_ids and len(vocabulary.id_to_token) > 0:
                    tokens = [vocabulary.id_to_token.get(token_id, f'<UNK_{token_id}>') for token_id in token_ids]
                    return ' '.join(tokens)
        
        return None
        
    except Exception as e:
        print(f"⚠️  Error extracting genome from {checkpoint_path}: {e}")
        return None


def main():
    """Main analysis function for interrupted experiments."""
    print("🔍 Interrupted GSM8K Experiment Analyzer")
    print("=" * 50)
    
    # Find all experiment data
    all_data = find_all_experiment_data()
    
    if not all_data:
        print("❌ No experiment data found!")
        return
    
    print(f"📁 Found {len(all_data)} data files/directories")
    
    # Group by experiment ID
    experiments = {}
    for item in all_data:
        exp_id = item['id']
        if exp_id not in experiments:
            experiments[exp_id] = []
        experiments[exp_id].append(item)
    
    # Analyze each experiment
    for exp_id, items in experiments.items():
        print(f"\n📊 Analyzing Experiment: {exp_id}")
        print("-" * 60)
        
        checkpoints = [item for item in items if item['type'] == 'checkpoint']
        logs = [item for item in items if item['type'] == 'log']
        data_files = [item for item in items if item['type'] == 'data_file']
        
        print(f"   Checkpoints: {len(checkpoints)}")
        print(f"   Logs: {len(logs)}")
        print(f"   Data files: {len(data_files)}")
        
        # Analyze checkpoints
        best_checkpoint_analysis = None
        for checkpoint_item in checkpoints:
            analysis = analyze_checkpoint_file(checkpoint_item['path'])
            if analysis and (not best_checkpoint_analysis or 
                           analysis.get('generation', 0) > best_checkpoint_analysis.get('generation', 0)):
                best_checkpoint_analysis = analysis
        
        if best_checkpoint_analysis:
            print(f"\n   🧬 Best Checkpoint Analysis:")
            print(f"      Generation: {best_checkpoint_analysis.get('generation', 0)}")
            print(f"      Population Size: {best_checkpoint_analysis.get('population_size', 0)}")
            if best_checkpoint_analysis.get('best_fitness') is not None:
                print(f"      Best Fitness: {best_checkpoint_analysis['best_fitness']:.3f}")
            if best_checkpoint_analysis.get('mean_fitness') is not None:
                print(f"      Mean Fitness: {best_checkpoint_analysis['mean_fitness']:.3f}")
            if best_checkpoint_analysis.get('diversity') is not None:
                print(f"      Diversity: {best_checkpoint_analysis['diversity']:.3f}")
            if best_checkpoint_analysis.get('evaluation_time') is not None:
                print(f"      Evaluation Time: {best_checkpoint_analysis['evaluation_time']:.1f}s")
            
            # Extract configuration
            config_data = best_checkpoint_analysis.get('config', {})
            if config_data:
                print(f"      Target Fitness: {config_data.get('target_fitness', 'unknown')}")
                print(f"      Max Generations: {config_data.get('max_generations', 'unknown')}")
        
        # Analyze logs
        for log_item in logs:
            log_analysis = analyze_log_file(log_item['path'])
            if log_analysis:
                print(f"\n   📝 Log Analysis ({log_item['path'].name}):")
                print(f"      Total Log Lines: {log_analysis.get('total_lines', 0)}")
                print(f"      Generations Found: {len(log_analysis.get('generations_found', []))}")
                print(f"      API Calls: {log_analysis.get('api_calls', 0)}")
                print(f"      Average Evaluation Time: {sum(log_analysis.get('evaluation_times', [0])) / max(1, len(log_analysis.get('evaluation_times', [0]))):.1f}s")
                
                if log_analysis.get('errors'):
                    print(f"      Errors Found: {len(log_analysis['errors'])}")
                    print(f"      First Error: {log_analysis['errors'][0][:100]}...")
        
        # Try to extract best genome
        if checkpoints:
            best_genome_text = find_best_genome_from_checkpoint(checkpoints[0]['path'])
            if best_genome_text:
                print(f"\n   🎯 Best Evolved Prompt:")
                print(f'      "{best_genome_text}"')
            else:
                print(f"\n   ⚠️  Could not extract best genome text")
    
    print(f"\n✅ Analysis complete!")
    
    # Show data locations
    print(f"\n📁 Data Locations:")
    print(f"   Experiments: {config.get_data_dir() / 'experiments'}")
    print(f"   Plots: {config.get_data_dir() / 'plots'}")
    print(f"   Cache: {config.get_data_dir() / 'cache'}")


if __name__ == "__main__":
    main()
