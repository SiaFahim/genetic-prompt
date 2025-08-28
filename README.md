# GSM8K Genetic Algorithm for Prompt Evolution

A comprehensive genetic algorithm system for evolving mathematical reasoning prompts on the GSM8K dataset. This system uses evolutionary computation to automatically discover high-performing prompts for large language models.

## 🎯 Overview

This project implements a complete genetic algorithm pipeline that:
- Evolves prompts for mathematical reasoning tasks
- Uses real-time evaluation with OpenAI/Anthropic APIs
- Provides comprehensive monitoring and visualization
- Includes 50 high-quality seed prompts across 10 categories
- Supports multiple experiment configurations and research scenarios

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/SiaFahim/genetic-prompt.git
cd genetic-prompt

# Create virtual environment
python -m venv gsm8k_ga_env
source gsm8k_ga_env/bin/activate  # On Windows: gsm8k_ga_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here  # Optional
```

### 3. Run Your First Experiment

```bash
# Quick test (recommended first run)
python scripts/run_experiment.py --preset quick_test

# Standard experiment
python scripts/run_experiment.py --preset standard

# List all available presets
python scripts/run_experiment.py --list-presets
```

### 4. Interactive Tutorial

Open the comprehensive Jupyter notebook tutorial:

```bash
jupyter notebook GSM8K_Genetic_Algorithm_Tutorial.ipynb
```

## 📊 System Architecture

### Core Components

- **Genetic Algorithm Engine**: Population management, selection, crossover, mutation
- **Evaluation Pipeline**: LLM-based fitness evaluation with caching
- **Seed Management**: 50 high-quality prompts across 10 reasoning categories
- **Monitoring System**: Real-time logging, visualization, and performance tracking
- **Configuration System**: 8 predefined experiment presets plus custom configurations

### Key Features

- ✅ **50 High-Quality Seed Prompts** across 10 mathematical reasoning categories
- ✅ **Real-time Monitoring** with fitness evolution plots and convergence analysis
- ✅ **Multiple LLM Support** (GPT-4o default, GPT-3.5-turbo, Claude-3)
- ✅ **Intelligent Caching** to minimize API costs
- ✅ **Comprehensive Logging** with experiment tracking and result archival
- ✅ **8 Experiment Presets** for different research scenarios
- ✅ **Command-line Interface** for easy experiment execution
- ✅ **Performance Monitoring** with resource usage tracking

## 🧬 Experiment Presets

| Preset | Population | Generations | Problems | Use Case |
|--------|------------|-------------|----------|----------|
| `quick_test` | 10 | 15 | 20 | System validation |
| `standard` | 50 | 100 | 100 | General research |
| `thorough` | 100 | 200 | 200 | Comprehensive study |
| `high_mutation` | 50 | 100 | 100 | High exploration |
| `large_population` | 150 | 50 | 100 | Diverse search |
| `ablation_no_crossover` | 50 | 100 | 100 | Mutation-only study |
| `ablation_no_mutation` | 50 | 100 | 100 | Crossover-only study |
| `random_search` | 50 | 100 | 100 | Baseline comparison |

## 📈 Usage Examples

### Command Line Interface

```bash
# Run with custom parameters
python scripts/run_experiment.py \
  --preset standard \
  --population-size 30 \
  --max-generations 50 \
  --model gpt-4o \
  --max-problems 150

# Dry run to see configuration
python scripts/run_experiment.py --preset thorough --dry-run

# Show preset details
python scripts/run_experiment.py --show-preset standard
```

### Python API

```python
from src.main_runner import run_gsm8k_experiment

config = {
    'name': 'My Experiment',
    'evolution': {
        'population_size': 30,
        'max_generations': 50,
        'target_fitness': 0.8
    },
    'model_name': 'gpt-4o',
    'max_problems': 100
}

results = run_gsm8k_experiment(config)
print(f"Best fitness: {results['results']['best_fitness']:.3f}")
```

## 🔬 Research Applications

### Ablation Studies
- Compare crossover vs mutation contributions
- Analyze selection strategy effectiveness
- Study population size impact

### Model Comparisons
- GPT-4o vs GPT-3.5-turbo performance
- Temperature sensitivity analysis
- Cross-model prompt transferability

### Prompt Engineering Research
- Automatic prompt discovery
- Reasoning strategy evolution
- Mathematical problem-solving optimization

## 📊 Results and Monitoring

### Real-time Visualization
- Fitness evolution over generations
- Population diversity tracking
- Convergence analysis plots
- Performance metrics dashboard

### Comprehensive Logging
- Generation-by-generation progress
- API usage and cost tracking
- Error logging and debugging
- Experiment result archival

### Performance Monitoring
- Memory usage tracking
- API rate limiting management
- Cache hit rate optimization
- Resource usage recommendations

## 🛠️ System Requirements

- Python 3.8+
- OpenAI API access (required)
- Anthropic API access (optional)
- 4GB+ RAM recommended
- Internet connection for API calls

## 📁 Project Structure

```
genetic-prompt/
├── src/
│   ├── genetics/          # Genetic algorithm core
│   ├── evaluation/        # LLM evaluation pipeline
│   ├── seeds/            # Seed prompt management
│   ├── utils/            # Utilities and monitoring
│   └── config/           # Configuration system
├── scripts/              # Command-line tools
├── data/                 # Generated data and results
├── tests/               # Test suites
└── GSM8K_Genetic_Algorithm_Tutorial.ipynb
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test individual components
python scripts/test_genetics_system.py
python scripts/test_evaluation_system.py
python scripts/test_seed_system.py
python scripts/test_monitoring_system.py

# Test complete system
python scripts/test_complete_system.py
```

## 📚 Documentation

- **Tutorial**: `GSM8K_Genetic_Algorithm_Tutorial.ipynb` - Complete interactive guide
- **API Reference**: Detailed docstrings in all modules
- **Configuration Guide**: `src/config/experiment_configs.py`
- **Seed Prompts**: `src/seeds/` - 50 categorized prompts with validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GSM8K dataset creators for the mathematical reasoning benchmark
- OpenAI and Anthropic for providing powerful language model APIs
- The genetic algorithm and evolutionary computation research community

## 📞 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and research ideas
- **Documentation**: Comprehensive Jupyter notebook tutorial included

---

**Ready to evolve better prompts?** Start with the tutorial notebook or run your first experiment:

```bash
python scripts/run_experiment.py --preset quick_test
```
