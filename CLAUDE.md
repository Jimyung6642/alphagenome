# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Building and Testing
- **Install package**: `pip install .` (or `pip install -e .` for development)
- **Run tests**: `hatch test` (runs tests on default Python version)
- **Run all tests**: `hatch test --all` (runs tests on all supported Python versions 3.10-3.13)
- **Format code**: `hatch run check:format` (uses pyink formatter)
- **Lint code**: `hatch run check:lint` (uses pylint)
- **Check all**: `hatch run check:all` (runs both format and lint)

### Development Environment
- Uses hatch as build system and dependency manager
- Tests use absltest framework (Google's testing library)
- Code formatting follows Google Python style guide with pyink
- Proto files are automatically compiled during build via `hatch_build.py`

## Code Architecture

### Core Structure
The codebase is organized into five main modules under `src/alphagenome/`:

1. **`data/`** - Core data structures and utilities for genomic data
   - `genome.py`: Fundamental classes (Interval, Variant, Strand)
   - `track_data.py`: Handles genomic track data (e.g., ChIP-seq, RNA-seq)
   - `ontology.py`: Cell type and tissue ontology mappings
   - `gene_annotation.py`: Gene and transcript annotations
   - `junction_data.py`: Splice junction data handling

2. **`models/`** - API client and prediction interfaces
   - `dna_client.py`: Main client for AlphaGenome API interactions
   - `dna_output.py`: Output data structures for model predictions
   - `variant_scorers.py`: Variant effect prediction scoring
   - `interval_scorers.py`: Interval-based prediction scoring

3. **`visualization/`** - Plotting and visualization components
   - `plot_components.py`: Core plotting functions and component classes
   - `plot.py`: Main plotting interface
   - `plot_transcripts.py`: Transcript visualization utilities

4. **`interpretation/`** - Model interpretation tools
   - `ism.py`: In-silico mutagenesis analysis

5. **`protos/`** - Protocol buffer definitions for gRPC communication
   - Auto-generated Python bindings during build

### Key Design Patterns
- Uses dataclasses extensively for genomic data structures
- gRPC client-server architecture for model predictions
- Modular visualization system with component-based plotting
- Protocol buffers for efficient data serialization
- Type hints throughout with jaxtyping for array types

### Testing
- All modules have corresponding `*_test.py` files
- Uses absltest framework
- Test environment configured with `MPLBACKEND=agg` for headless matplotlib
- Benchmark tests available for performance-critical code

### Dependencies
- Core scientific stack: numpy, pandas, scipy, matplotlib
- gRPC for API communication
- Protocol buffers for data serialization
- anndata for genomics data structures
- Custom tensor utilities for efficient data handling

### API Integration
The client connects to Google DeepMind's AlphaGenome API service. Key supported sequence lengths are defined in `dna_client.py`: 2KB, 16KB, 100KB, 500KB, and 1MB.

**API Key Setup:**
- Obtain API key from https://deepmind.google.com/science/alphagenome
- Create `config.env` file in project root with: `ALPHAGENOME_API_KEY=your_api_key_here`
- API is free for non-commercial use, suitable for 1000s of predictions but not large-scale analyses (>1M predictions)

**Model Capabilities:**
- Multimodal predictions: gene expression, splicing patterns, chromatin features, contact maps
- Single base-pair resolution for most outputs
- Analyzes DNA sequences up to 1 million base pairs
- State-of-the-art performance on genomic prediction benchmarks

### TF Prediction Tool (TF_prediction_test.py)
A comprehensive script for transcription factor binding predictions with the following features:

**Basic Usage:**
- Single gene analysis: `python3 TF_prediction_test.py --gene TP53`
- Interactive mode: `python3 TF_prediction_test.py --interactive`
- Custom regions: `python3 TF_prediction_test.py --gene BRCA1 --region promoter`

**Comprehensive Multi-Tissue Analysis:**
- Full ontology analysis: `python3 TF_prediction_test.py --gene TP53 --comprehensive`
- Limited analysis: `python3 TF_prediction_test.py --gene BRCA1 --comprehensive --max-ontologies 50 --batch-size 10`

**Features:**
- Automated gene coordinate lookup via Ensembl REST API
- Support for 163 different tissue/cell type ontologies (CL, EFO, UBERON, CLO, NTR)
- Batch processing to handle API rate limits
- Comprehensive CSV export with statistical analysis
- Visualization with automatic filtering for display
- Timestamped output files in `output/` folder

**Output Files:**
- `*_comprehensive_tf_results.csv`: Complete results across all ontologies
- `*_comprehensive_tf_results_summary.csv`: Top 5 TFs per ontology
- `*_tf_binding.png`: Visualization plots

## Project Resources

### Documentation and Examples
- **Official Documentation**: https://www.alphagenomedocs.com/
- **Community Forum**: https://www.alphagenomecommunity.com
- **Google Colab Notebooks**: Available in `colabs/` directory for interactive examples
- **Terms of Use**: https://deepmind.google.com/science/alphagenome/terms

### Key Example Notebooks
- `colabs/quick_start.ipynb`: Basic model usage and predictions
- `colabs/visualization_modality_tour.ipynb`: Comprehensive visualization examples
- `colabs/variant_scoring_ui.ipynb`: Variant effect prediction interface
- `colabs/example_analysis_workflow.ipynb`: Complete analysis workflows

### Support and Issues
- **Code Issues**: Submit on GitHub Issues
- **General Questions**: Use community forum (faster response)
- **Direct Contact**: alphagenome@google.com (may have delays due to volume)