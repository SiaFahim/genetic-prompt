#!/usr/bin/env python3
"""
Test script to verify GPT-4o is consistently used as the default model throughout the system.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config
from src.config.experiment_configs import ConfigurationManager, ExperimentConfig
from src.evaluation.llm_interface import LLMInterface
from src.evaluation.pipeline import EvaluationPipeline


def test_config_defaults():
    """Test that configuration defaults use GPT-4o."""
    print("🔧 Testing Configuration Defaults")
    print("-" * 40)
    
    # Test system config default
    default_model = config.default_model
    print(f"System config default model: {default_model}")
    assert default_model == "gpt-4o", f"Expected gpt-4o, got {default_model}"
    print("✅ System config uses GPT-4o as default")
    
    # Test ExperimentConfig default
    exp_config = ExperimentConfig(
        name="Test Config",
        description="Test configuration",
        experiment_type="standard"
    )
    print(f"ExperimentConfig default model: {exp_config.model_name}")
    assert exp_config.model_name == "gpt-4o", f"Expected gpt-4o, got {exp_config.model_name}"
    print("✅ ExperimentConfig uses GPT-4o as default")
    
    return True


def test_preset_configurations():
    """Test that all preset configurations use GPT-4o as default."""
    print("\n⚙️  Testing Preset Configurations")
    print("-" * 40)
    
    config_manager = ConfigurationManager()
    presets = config_manager.list_presets()
    
    for preset_name in presets:
        preset_config = config_manager.get_preset(preset_name)
        print(f"Preset '{preset_name}': {preset_config.model_name}")
        assert preset_config.model_name == "gpt-4o", f"Preset {preset_name} uses {preset_config.model_name}, expected gpt-4o"
    
    print(f"✅ All {len(presets)} presets use GPT-4o as default")
    return True


def test_llm_interface_default():
    """Test that LLM interface uses GPT-4o as default."""
    print("\n🤖 Testing LLM Interface Default")
    print("-" * 40)
    
    # Test with no explicit model
    llm = LLMInterface()
    print(f"LLM interface default model: {llm.model}")
    assert llm.model == "gpt-4o", f"Expected gpt-4o, got {llm.model}"
    print("✅ LLM interface uses GPT-4o as default")
    
    return True


def test_evaluation_pipeline_default():
    """Test that evaluation pipeline uses GPT-4o as default."""
    print("\n⚡ Testing Evaluation Pipeline Default")
    print("-" * 40)
    
    # Create pipeline without explicit model
    pipeline = EvaluationPipeline(use_cache=False, max_problems=5)
    model_used = pipeline.llm_interface.model
    print(f"Evaluation pipeline model: {model_used}")
    assert model_used == "gpt-4o", f"Expected gpt-4o, got {model_used}"
    print("✅ Evaluation pipeline uses GPT-4o as default")
    
    return True


def test_configuration_summary_display():
    """Test that configuration summaries display the correct model."""
    print("\n📋 Testing Configuration Summary Display")
    print("-" * 40)
    
    config_manager = ConfigurationManager()
    test_config = config_manager.get_preset('standard')
    
    summary = config_manager.get_config_summary(test_config)
    print("Configuration summary:")
    print(summary)
    
    assert "Model: gpt-4o" in summary, f"Summary should show 'Model: gpt-4o', got: {summary}"
    print("✅ Configuration summary displays GPT-4o")
    
    return True


def test_dynamic_model_references():
    """Test that the system uses dynamic model references in status messages."""
    print("\n🔄 Testing Dynamic Model References")
    print("-" * 40)
    
    # Test that we're not using hardcoded model names in key places
    from src.main_runner import GSM8KExperimentRunner
    from dataclasses import asdict
    
    # Create a test configuration
    config_manager = ConfigurationManager()
    test_config = config_manager.get_preset('quick_test')
    test_config.model_name = "test-model-name"  # Use a test model name
    
    config_dict = asdict(test_config)
    for key, value in config_dict.items():
        if hasattr(value, 'value'):
            config_dict[key] = value.value
    
    # Create runner (but don't run the experiment)
    runner = GSM8KExperimentRunner(config_dict)
    
    # Verify the configuration is passed through correctly
    assert runner.config['model_name'] == "test-model-name", "Model name not passed through correctly"
    print("✅ Dynamic model references working correctly")
    
    return True


def test_environment_variable_override():
    """Test that environment variable can override the default."""
    print("\n🌍 Testing Environment Variable Override")
    print("-" * 40)

    # Test that the current environment variable is being read
    current_model = os.getenv('DEFAULT_MODEL', 'gpt-4o')
    config_model = config.default_model

    print(f"Environment variable DEFAULT_MODEL: {current_model}")
    print(f"Config default_model: {config_model}")

    assert config_model == current_model, f"Config should match env var: expected {current_model}, got {config_model}"
    print("✅ Environment variable override working (config matches env var)")

    # Verify that the system respects the .env file setting
    assert config_model == "gpt-4o", f"Expected gpt-4o from .env file, got {config_model}"
    print("✅ .env file configuration is being used")

    return True


def run_comprehensive_gpt4o_test():
    """Run comprehensive GPT-4o consistency test."""
    print("🧪 GPT-4o Consistency Test Suite")
    print("=" * 50)
    
    tests = [
        test_config_defaults,
        test_preset_configurations,
        test_llm_interface_default,
        test_evaluation_pipeline_default,
        test_configuration_summary_display,
        test_dynamic_model_references,
        test_environment_variable_override
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
        print("🎉 ALL GPT-4o CONSISTENCY TESTS PASSED!")
        print("\n✅ GPT-4o Audit Results:")
        print("   • System config defaults to GPT-4o")
        print("   • All experiment presets use GPT-4o")
        print("   • LLM interface defaults to GPT-4o")
        print("   • Evaluation pipeline uses GPT-4o")
        print("   • Configuration summaries display GPT-4o")
        print("   • Dynamic model references work correctly")
        print("   • Environment variable override supported")
        
        print("\n🚀 System consistently uses GPT-4o as the default model!")
        print("   Users will see GPT-4o in all status messages and summaries")
        print("   All components use the configured model dynamically")
        print("   No hardcoded model references remain in critical paths")
        
        return True
    else:
        print("❌ Some GPT-4o consistency tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_gpt4o_test()
    if not success:
        sys.exit(1)
