#!/usr/bin/env python3
"""
Test script to verify the notebook configuration is working correctly.
This simulates the exact configuration setup used in the notebook.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.experiment_configs import ConfigurationManager
from dataclasses import asdict


def test_notebook_configuration():
    """Test the exact configuration setup used in the notebook."""
    print("🧪 Testing Notebook Configuration Setup")
    print("=" * 50)
    
    # Replicate the exact notebook configuration
    config_manager = ConfigurationManager()
    
    # Use the same base preset as the notebook
    BASE_PRESET = "quick_test"
    
    # Use the same custom modifications as the notebook (after our fix)
    custom_modifications = {
        'name': 'My GSM8K Evolution Experiment',
        'description': 'Custom experiment for prompt evolution',
        'population_size': 15,
        'max_generations': 20,
        'max_problems': 30,
        'model_name': 'gpt-4o',  # This should now be gpt-4o
        'temperature': 0.0,
        'target_fitness': 0.95,
    }
    
    print(f"📋 Base Preset: {BASE_PRESET}")
    print(f"🔧 Custom Modifications:")
    for key, value in custom_modifications.items():
        print(f"   {key}: {value}")
    
    # Create the configuration (same as notebook)
    experiment_config = config_manager.create_custom_config(BASE_PRESET, custom_modifications)
    
    print(f"\n✅ Final Configuration Created:")
    print(f"   Model Name: {experiment_config.model_name}")
    print(f"   Population Size: {experiment_config.population_size}")
    print(f"   Max Generations: {experiment_config.max_generations}")
    print(f"   Max Problems: {experiment_config.max_problems}")
    print(f"   Target Fitness: {experiment_config.target_fitness}")
    
    # Show the configuration summary (same as notebook)
    print(f"\n📊 Configuration Summary:")
    summary = config_manager.get_config_summary(experiment_config)
    print(summary)
    
    # Convert to dictionary format (same as notebook would do for runner)
    config_dict = asdict(experiment_config)
    
    # Convert enums to strings (same as notebook)
    for key, value in config_dict.items():
        if hasattr(value, 'value'):
            config_dict[key] = value.value
    
    print(f"\n🔄 Dictionary Format (for runner):")
    print(f"   model_name: {config_dict['model_name']}")
    print(f"   population_size: {config_dict['population_size']}")
    print(f"   max_generations: {config_dict['max_generations']}")
    print(f"   max_problems: {config_dict['max_problems']}")
    
    # Verify the model is correct
    assert experiment_config.model_name == "gpt-4o", f"Expected gpt-4o, got {experiment_config.model_name}"
    assert config_dict['model_name'] == "gpt-4o", f"Expected gpt-4o in dict, got {config_dict['model_name']}"
    
    print(f"\n✅ All assertions passed - configuration is using GPT-4o correctly!")
    
    return config_dict


def test_runner_initialization():
    """Test that the runner receives the correct configuration."""
    print(f"\n🚀 Testing Runner Initialization")
    print("=" * 50)
    
    # Get the config from the previous test
    config_dict = test_notebook_configuration()
    
    # Import and test the runner (without actually running the experiment)
    from src.main_runner import GSM8KExperimentRunner
    
    print(f"\n🔧 Initializing GSM8KExperimentRunner...")
    runner = GSM8KExperimentRunner(config_dict)
    
    print(f"   Runner config model_name: {runner.config.get('model_name', 'NOT_SET')}")
    
    # Verify the runner has the correct configuration
    assert runner.config['model_name'] == "gpt-4o", f"Runner config should have gpt-4o, got {runner.config['model_name']}"
    
    print(f"✅ Runner initialization test passed!")
    
    return runner


def simulate_experiment_start_messages():
    """Simulate the messages that would be shown when starting an experiment."""
    print(f"\n📢 Simulating Experiment Start Messages")
    print("=" * 50)
    
    # Get the runner
    runner = test_runner_initialization()
    
    # Simulate the messages that would be shown (based on CLI script)
    config_dict = runner.config
    
    print(f"\n🧬 Starting genetic algorithm evolution...")
    print("=" * 60)
    print(f"Population Size: {config_dict.get('population_size', 'unknown')}")
    print(f"Max Generations: {config_dict.get('max_generations', 'unknown')}")
    print(f"Evaluation Problems: {config_dict.get('max_problems', 'unknown')}")
    print(f"Model: {config_dict.get('model_name', 'unknown')}")
    print("=" * 60)
    
    # This should show gpt-4o, not gpt-3.5-turbo
    expected_model = config_dict.get('model_name')
    assert expected_model == "gpt-4o", f"Expected gpt-4o in start messages, got {expected_model}"
    
    print(f"\n✅ Experiment start messages would show GPT-4o correctly!")


def main():
    """Run all configuration tests."""
    print("🔍 Notebook Configuration Verification")
    print("=" * 60)
    
    try:
        test_notebook_configuration()
        test_runner_initialization()
        simulate_experiment_start_messages()
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"\n📋 Summary:")
        print(f"   ✅ Notebook configuration creates GPT-4o config correctly")
        print(f"   ✅ Runner receives GPT-4o configuration correctly")
        print(f"   ✅ Start messages would display GPT-4o correctly")
        
        print(f"\n💡 If you're still seeing gpt-3.5-turbo in the notebook:")
        print(f"   1. Clear all notebook output and restart kernel")
        print(f"   2. Re-run the configuration cells")
        print(f"   3. The output should now show GPT-4o")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
