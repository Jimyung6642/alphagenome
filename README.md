# AlphaGenome + Tahoe-100M + State Model Perturbation Analysis Pipeline

A comprehensive genomic perturbation analysis pipeline that integrates multiple state-of-the-art models for transcription factor (TF) binding prediction, baseline expression analysis, and zero-shot perturbation effects.

## 🧬 Pipeline Overview

This pipeline combines three powerful genomic models in a streamlined workflow:

1. **AlphaGenome**: TF binding predictions for target genes using Google DeepMind's unified DNA sequence model
2. **Tahoe-100M**: Baseline TF expression analysis from 100+ million single cells across 50+ cancer cell lines
3. **ST-Tahoe State Model**: Zero-shot TF perturbation effect predictions without requiring model training

**Main Workflow**: `AlphaGenome TF prediction → Tahoe-100M DMSO_TF data → ST-Tahoe perturbation prediction`

## 🚀 Quick Start

### Basic Analysis
```bash
# Analyze single gene with organ focus
python3 main.py --genes TP53 --organ stomach

# Multiple genes analysis
python3 main.py --genes TP53,BRCA1,CLDN18 --organ lung

# Custom perturbation parameters
python3 main.py --genes CLDN18 --perturbation-strength 0.8 --deg-p-threshold 0.01

# Verbose output for debugging
python3 main.py --genes TP53 --organ stomach --verbose
```

### Advanced Options
```bash
# TF organ filtering for AlphaGenome predictions
python3 main.py --genes TP53 --tf-organ stomach --organ lung

# Custom DEG thresholds
python3 main.py --genes CLDN18 --deg-p-threshold 0.01 --deg-fc-threshold 1.0

# Top TF percentage filtering
python3 main.py --genes TP53 --tf-percent 20 --organ stomach
```

## 📋 Requirements

### System Requirements
- **Python**: 3.10-3.13
- **Operating System**: macOS, Linux, Windows
- **Memory**: 8GB+ RAM recommended
- **Storage**: 5GB+ free space for model files and cache

### Required Models & Data Sources

#### 1. AlphaGenome API
- **API Key**: Required (free for non-commercial use)
- **Get API Key**: https://deepmind.google.com/science/alphagenome
- **Setup**: Add `ALPHAGENOME_API_KEY=your_key_here` to `config.env`

#### 2. ST-Tahoe State Model
- **Location**: `models/state/ST-Tahoe/`
- **Source**: Hugging Face (arcinstitute/ST-Tahoe)
- **Files Required**:
  - `config.yaml`
  - `final_from_preprint.ckpt`
  - `pert_onehot_map.pt`
  - `cell_type_onehot_map.pkl`
- **Auto-download**: The pipeline will attempt to download these automatically

#### 3. Tahoe-100M Dataset
- **Source**: Google Cloud Storage (arc-ctc-tahoe100 bucket)
- **Access**: Public dataset, no credentials required
- **Cache**: Files are cached locally in `output/bridge_cache/`
- **Size**: ~100M cells across 50+ cancer cell lines

### Package Dependencies

#### Core Scientific Stack
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `scipy >= 1.10.0`
- `matplotlib >= 3.6.0`
- `seaborn >= 0.12.0`

#### Genomics & Single-Cell
- `anndata >= 0.9.0`
- `scanpy >= 1.9.0`
- `gcsfs >= 2023.0.0` (for Tahoe-100M data access)

#### Machine Learning & State Model
- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- State model dependencies (installed via `uv tool run --from arc-state state`)

#### AlphaGenome Client
- `grpcio >= 1.50.0`
- `protobuf >= 4.0.0`
- `jaxtyping >= 0.2.0`

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd alphagenome
```

### 2. Install Package
```bash
# Development installation (recommended)
pip3 install -e .

# Or standard installation
pip3 install .
```

### 3. Setup Configuration
```bash
# Create config.env file
cat > config.env << EOF
# AlphaGenome API Key (Required)
ALPHAGENOME_API_KEY=your_api_key_here

# Ensembl API timeout (seconds)
TAHOE_ENSEMBL_TIMEOUT=10

# Optional: Google Cloud Service Account for Tahoe-100M data
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service/account/key.json
EOF
```

### 4. Download ST-Tahoe Model (Auto or Manual)
The pipeline will attempt to auto-download the ST-Tahoe model. If this fails:

```bash
# Manual download via git lfs
git lfs clone https://huggingface.co/arcinstitute/ST-Tahoe
mv ST-Tahoe models/state/ST-Tahoe
```

### 5. Verify Installation
```bash
# Test imports
python3 -c "import main; print('✅ Pipeline ready')"

# Check help
python3 main.py --help
```

## 📁 Project Structure

```
alphagenome/
├── main.py                           # Main pipeline entry point
├── standardized_genomic_analyzer.py  # Core analysis engine
├── tahoe_100M_loader.py              # Tahoe-100M data loader
├── state_perturbation_engine.py      # ST-Tahoe model wrapper
├── tahoe_state_bridge.py             # Data format converter
├── deg_analyzer.py                   # Differential expression analysis
├── config.env                        # Configuration file
├── models/state/ST-Tahoe/            # State model files
├── output/                           # Analysis results
├── src/alphagenome/                  # AlphaGenome client library
└── CLAUDE.md                         # Detailed developer documentation
```

## 🔬 Output Files

### Analysis Results
- **CSV files**: Comprehensive perturbation analysis results with timestamps
- **Log files**: Detailed execution logs in `output/perturbation_analysis_YYYYMMDD_HHMMSS.log`
- **Cache directories**: 
  - `output/bridge_cache/`: Tahoe-100M H5AD files
  - `output/state_cache/`: ST-Tahoe model cache
  - `output/deg_analysis/`: DEG analysis results

### Key Output Columns
- `tf_name`: Transcription factor gene symbol
- `mean_binding_score`: AlphaGenome TF binding prediction
- `baseline_expression`: Tahoe-100M DMSO control expression
- `perturbation_effect`: ST-Tahoe predicted expression change
- `deg_pvalue`: Differential expression significance
- `deg_log2fc`: Log2 fold change from perturbation

## 🧪 Testing & Development

### Run Tests
```bash
# Install hatch (build system)
pip install hatch

# Run tests on default Python version
hatch test

# Run tests on all supported Python versions (3.10-3.13)
hatch test --all

# Run specific test file
hatch test src/alphagenome/models/dna_client_test.py
```

### Code Quality
```bash
# Format code (Google Python style)
hatch run check:format

# Lint code
hatch run check:lint

# Run all checks
hatch run check:all
```

### Pipeline Validation
```bash
# Quick validation test
python3 main.py --genes TP53 --organ stomach --verbose

# Check logs for any issues
tail -f output/perturbation_analysis_*.log
```

## 🔧 Model Versions

| Component | Version/Source | Notes |
|-----------|----------------|-------|
| AlphaGenome | API (Latest) | Google DeepMind's unified DNA model |
| ST-Tahoe | arcinstitute/ST-Tahoe | Zero-shot perturbation model |
| Tahoe-100M | arc-ctc-tahoe100 (GCS) | 100M+ single cells, 50+ cell lines |
| GENCODE | v46 | Human genome annotations |

## 🐛 Troubleshooting

### Common Issues

#### 1. AlphaGenome API Key Issues
```bash
# Check if API key is set
grep ALPHAGENOME_API_KEY config.env

# Test API connectivity
python3 -c "from alphagenome.models import dna_client; client = dna_client.create('your_key'); print('✅ API connected')"
```

#### 2. ST-Tahoe Model Download Fails
```bash
# Check model directory
ls -la models/state/ST-Tahoe/

# Manual download
git lfs clone https://huggingface.co/arcinstitute/ST-Tahoe models/state/ST-Tahoe
```

#### 3. Tahoe-100M Data Access Issues
```bash
# Test GCS access
python3 -c "import gcsfs; fs = gcsfs.GCSFileSystem(); print(fs.ls('arc-ctc-tahoe100')[:5])"
```

#### 4. Memory Issues
- Reduce `--tf-percent` to analyze fewer TFs
- Use `--organ` parameter to limit cell lines
- Monitor memory usage: `top -p $(pgrep python)`

### Support
- **Code Issues**: Submit GitHub issues
- **General Questions**: Check `CLAUDE.md` for detailed documentation
- **Model Questions**: Refer to individual model repositories

## 📄 Citation

If you use this pipeline in your research, please cite the underlying models:

```bibtex
# AlphaGenome
@article{alphagenome,
  title={{AlphaGenome}: advancing regulatory variant effect prediction with a unified {DNA} sequence model},
  author={Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and others},
  year={2025},
  journal={bioRxiv},
  doi={https://doi.org/10.1101/2025.06.25.661532}
}

# State Model
@article{state_model,
  title={State: an efficient representation for single-cell transcriptomics},
  journal={In preparation},
  year={2024}
}
```

## 📜 License

This project builds upon several open-source components:
- **AlphaGenome**: Apache 2.0 License (Google LLC)
- **Pipeline Code**: MIT License
- **ST-Tahoe Model**: Check Hugging Face repository for specific terms

See individual model repositories for detailed licensing information.

---

**Pipeline Workflow**: AlphaGenome TF prediction → Tahoe-100M DMSO_TF data → ST-Tahoe perturbation prediction 🧬✨