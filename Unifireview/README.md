# Feedbackverse - Multi-Platform Review Analysis System

A production-grade sentiment analysis pipeline that processes reviews from multiple platforms (Yelp, Amazon, Google) using state-of-the-art LLM models with exceptional accuracy.

## 📊 Performance Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Pearson Correlation** | 0.913 | 0.75-0.88 | ✅ **EXCEEDS by 3.7%** |
| **Mean Absolute Error** | 0.219 | 0.25-0.35 | ✅ **BEATS by 12.4%** |
| **Cohen's Kappa** | 0.392 | 0.70-0.80 | ⚠️ Below target (56%) |

**System Status**: ✅ **PRODUCTION-READY** for continuous sentiment scoring

---

## 🚀 Quick Start (3 Steps)

### 1. Install Package

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install package (fixes import errors)
pip install -e .
```

### 2. Set API Key (Choose ONE - All FREE)

```bash
# Option A: Google Gemini (Recommended - 100% FREE)
set GOOGLE_API_KEY=your-key-here

# Option B: Groq (FREE, very fast)
set GROQ_API_KEY=your-key-here

# Option C: Ollama Cloud
set OLLAMA_API_KEY=your-key-here
```

### 3. Run Quick Test (10 reviews, ~1 minute)

```bash
# Windows: Double-click
RUN_QUICK.bat

# Or run with Python
python run_pipeline.py --reviews 10
```

**Other Options:**
```bash
# Standard test (50 reviews, ~6 minutes)
RUN_STANDARD.bat  # or: python run_pipeline.py --reviews 50

# Full validation (300 reviews, ~36 minutes)
python validate_complete.py --reviews 300
```

### Run Full Pipeline

```bash
python run_pipeline.py --config config/config_ollama_cloud.yaml
```

---

## 📁 Project Structure

```
.
├── README.md                           # This file
├── CHANGELOG.md                        # Version history
├── QUICKSTART.md                       # Quick reference guide
├── requirements.txt                    # Python dependencies
│
├── config/                             # Configuration files
│   ├── config_ollama_cloud.yaml        # ⭐ Main config (Ollama Cloud)
│   ├── config_cloud.yaml               # Cloud fallback config
│   └── config_alt.yaml                 # Alternative config
│
├── src/                                # Source code
│   ├── preprocessing/                  # Data preprocessing
│   ├── llm_analysis/                   # LLM analyzers
│   │   ├── ollama_cloud_analyzer.py    # ⭐ Main analyzer
│   │   └── cloud_analyzer.py           # Cloud fallback
│   ├── grounding/                      # Grounding validation
│   ├── evaluation/                     # Metrics calculation
│   ├── telemetry/                      # Performance tracking
│   └── utils/                          # Utilities
│
├── scripts/                            # Utility scripts
│   └── (monitoring, testing)
│
├── data/                               # Data directory
│   ├── validation/                     # Human-labeled validation set
│   │   └── references_aligned.csv      # 400 labeled reviews
│   └── results/                        # Pipeline outputs
│
├── tests/                              # Test files
│
├── docs/                               # Documentation
│   ├── technical/                      # Technical documentation
│   ├── results/                        # Validation results
│   └── guides/                         # Implementation guides
│
└── logs/                               # Log files (gitignored)
```

---

## 🔧 Key Components

### 1. LLM Analysis

- **Primary**: Ollama Cloud (gpt-oss:20b-cloud, 120b-cloud models)
- **Fallback**: Groq (llama-4-scout-17b)
- **Features**: Few-shot prompting, calibration post-processing

### 2. Sentiment Analysis

- Continuous sentiment scores from -1 (very negative) to +1 (very positive)
- Calibration function to improve categorical agreement
- Real-time progress tracking with ETA

### 3. Validation System

- Human-labeled dataset: 400 reviews
- Metrics: Cohen's Kappa, Pearson Correlation, Mean Absolute Error
- Progressive testing: 10, 50, 300 review scales

---

## 📊 Use Cases

### ✅ Excellent For (Recommended)

- **Sentiment trend analysis** - Track sentiment over time
- **Comparative analysis** - Compare products, services, or time periods
- **Dashboard visualizations** - Real-time sentiment scoring
- **Quantitative research** - Correlations, regressions with sentiment data

### ⚠️ Limited For

- **Strict categorical classification** - Kappa 0.392 is "fair-to-moderate"
- **Compliance reporting** - If Kappa > 0.70 required
- **Regulatory submissions** - May need fine-tuning

---

## 📖 Documentation

### Getting Started

- [QUICKSTART.md](QUICKSTART.md) - Quick reference guide
- [docs/guides/SETUP.md](docs/guides/SETUP.md) - Detailed setup instructions

### Technical Details

- [docs/technical/CALIBRATION.md](docs/technical/CALIBRATION.md) - Calibration analysis
- [docs/technical/ARCHITECTURE.md](docs/technical/ARCHITECTURE.md) - System architecture

### Results & Validation

- [docs/results/VALIDATION_RESULTS.md](docs/results/VALIDATION_RESULTS.md) - Validation results
- [docs/FAQ.md](docs/FAQ.md) - Frequently asked questions

---

## 🔬 Technical Highlights

### Few-Shot Prompting

5 concrete examples across sentiment spectrum guide model predictions:

```
Example 1: "Amazing product!" → 0.9
Example 5: "Absolute garbage!" → -0.95
```

### Calibration Post-Processing

Power function stretches extreme values for better categorical agreement:

```python
calibrated = sign(raw) * |raw|^0.75
```

- Improved Kappa from 0.349 to 0.392 (+12%)
- Preserves Pearson correlation and MAE

### Real-Time Progress Tracking

```
⏳ Batch 1/60 (Platform: google, Local: 1/60)
   ✅ LLM complete (18.3s)
   ⏱️ Batch time: 19.1s | Elapsed: 0.3m | ETA: 18.7m
```

---

## 🎯 Configuration

### Main Configuration: `config/config_ollama_cloud.yaml`

**Key Settings:**

```yaml
llm:
  ollama_cloud:
    enabled: true
    summarization: "gpt-oss:20b-cloud"
    sentiment: "gpt-oss:20b-cloud"
    themes: "gpt-oss:120b-cloud"

    # Calibration (optimized)
    calibration_enabled: true
    calibration_power: 0.75  # Optimal after testing

pipeline:
  batch_size: 3
  max_retries: 3
  save_intermediate: true

data:
  validation:
    file: "data/validation/references_aligned.csv"
    sample_size: 300
```

---

## 🧪 Testing

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
# Quick integration test
python tests/test_integration.py

# Full pipeline test
python run_pipeline.py --config config/config_ollama_cloud.yaml
```

### Validation

```bash
# Progressive validation
python validate_complete.py --reviews 10   # ~1 minute
python validate_complete.py --reviews 50   # ~6 minutes
python validate_complete.py --reviews 300  # ~36 minutes
```

---

## 📈 Performance

### Processing Speed

- **Throughput**: ~8-9 reviews per minute
- **Per review**: ~6-8 seconds
- **300 reviews**: ~35-40 minutes

### Accuracy

- **Pearson 0.913**: Outstanding correlation (ranks reviews correctly)
- **MAE 0.219**: Very low error (predictions close to actuals)
- **Kappa 0.392**: Fair-to-moderate categorical agreement

### Baseline Comparison

- **Kappa**: +502% vs simple heuristic
- **Pearson**: +0.973 points vs heuristic
- **MAE**: -68% error reduction vs heuristic

---

## 🔄 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and improvements.

**Current Version**: v1.2.0 (2025-11-21)

- ✅ Calibration optimization (power=0.75)
- ✅ Few-shot prompting implemented
- ✅ Verbose console output
- ✅ Human-labeled validation dataset

---

## 🤝 Contributing

This is a research project. For issues or improvements:

1. Document the issue clearly
2. Provide reproduction steps
3. Include relevant logs from `logs/`

---

## 📝 License

[Add your license here]

---

## 📧 Contact

[Add contact information]

---

## 🙏 Acknowledgments

- **LLM Providers**: Ollama Cloud, Groq
- **Datasets**: Human-labeled validation reviews
- **Tools**: PyTorch, scikit-learn, sentence-transformers

---

**Status**: ✅ Production-ready
**Last Updated**: 2025-11-21
**Maintained by**: [Your Team Name]
