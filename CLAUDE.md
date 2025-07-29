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
The codebase consists of a comprehensive genomic analysis pipeline and the core AlphaGenome library:

**Main Pipeline Components:**
- `main.py`: Pipeline orchestrator and entry point
- `standardized_genomic_analyzer.py`: Gene-agnostic analysis engine
- `tahoe_100M_loader.py`: Real transcriptome data from 50 cancer cell lines
- `state_model_client.py`: Clean State model integration components

**Core AlphaGenome Library** (`src/alphagenome/`):

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

### Extended Pipeline Dependencies
- **datasets**: HuggingFace datasets library for Tahoe-100M access (`pip install datasets`)
- **arc-state**: Arc Institute State model CLI (`uv tool install arc-state`)
- **scanpy**: Single-cell analysis tools for transcriptome processing
- **yaml**: Configuration file parsing for State model setup

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

### Comprehensive Genomic Analysis Pipeline (main.py)
The main production pipeline combining AlphaGenome, State model, and Tahoe-100M for comprehensive genomic analysis:

**Basic Usage:**
- Single gene analysis: `python3 main.py --gene TP53`
- Interactive mode: `python3 main.py --interactive`
- Batch analysis: `python3 main.py --genes TP53,BRCA1,CLDN18`

**Key Features:**
- **Gene-agnostic**: Works with ANY human gene symbol
- **Comprehensive coverage**: Uses ALL 50 cancer cell lines from Tahoe-100M
- **Real data only**: No synthetic or placeholder data
- **All ontologies**: Analyzes across ALL available AlphaGenome ontologies
- **Standardized output**: Consistent CSV format for all analyses

**Architecture:**
- `main.py`: Main entry point and pipeline orchestrator
- `standardized_genomic_analyzer.py`: Core analysis engine
- `tahoe_100M_loader.py`: Real transcriptome data loader (50 cancer cell lines)
- `state_model_client.py`: Clean State model integration components

**Output:**
- Timestamped CSV files in `comprehensive_results/` directory
- Combined AlphaGenome + State model predictions
- Analysis statistics and metadata in JSON format
- Comprehensive logging to `main_pipeline.log`

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

## Arc Institute State Model Integration

### State Model Overview
Arc Institute's State model for predicting cellular responses to perturbations, integrated with AlphaGenome genomic predictions through `state_model_client.py`.

**Key Capabilities:**
- Predict cellular responses to drugs, cytokines, and genetic perturbations
- State Embedding (SE-600M): 705M parameter model for transcriptome embeddings
- State Transition model: Bidirectional transformer for cellular response prediction
- Integration with AlphaGenome for genomic variant → cellular response analysis

### Pipeline Components
- **`main.py`**: Main comprehensive analysis pipeline
  - Usage: `python3 main.py --gene TP53` (single gene analysis)
  - Usage: `python3 main.py --interactive` (interactive mode)
  - Usage: `python3 main.py --genes TP53,BRCA1,CLDN18` (batch analysis)
  
- **`state_model_client.py`**: State model integration components
  - Contains `StatePredictionClient` and `AlphaGenomeStateAnalyzer` classes
  - Handles real transcriptome data loading from Tahoe-100M

### Model Setup
- SE-600M model downloaded from HuggingFace (`arcinstitute/SE-600M`)
- Installed via `uv tool install arc-state`
- Model path configurable in `StatePredictionClient` (default: `state/models/SE-600M/`)
- **License**: Non-commercial use only (Arc Research Institute license)

### Tahoe-100M Dataset Integration
- **Dataset**: World's largest single-cell dataset (100M cells, 50 cancer cell lines)
- **Coverage**: Complete transcriptome with 62,710 genes per cell line
- **Integration**: Real transcriptome data loaded via `tahoe_100M_loader.py`
- **Usage**: Automatic caching with efficient streaming for large-scale analysis
- **Cache**: Data cached in `tahoe_cache/` directory for faster subsequent access

### Integration Benefits
- **Multi-scale analysis**: DNA variants → cellular responses
- **Cell type specificity**: Predictions across 70+ cell lines
- **Comprehensive risk scores**: Combined genomic + cellular impact assessment
- **Workflow integration**: Seamless connection with existing TF prediction pipeline

## Important Setup Notes

### Configuration Files
- **`config.env`**: Required for AlphaGenome API key (`ALPHAGENOME_API_KEY=your_key_here`)
- **Git ignored**: Environment files are excluded from version control

### Directory Structure
- **`comprehensive_cache/`**: General analysis cache
- **`tahoe_cache/`**: Tahoe-100M dataset cache
- **`comprehensive_results/`**: Main pipeline output directory
- **`main_pipeline.log`**: Detailed execution logs

### Common Issues
- **AlphaGenome Client**: Use `dna_client.create(api_key)` not `DnaClient()` constructor
- **State Model**: Requires `arc-state` CLI tool in PATH
- **Gene Validation**: Uses Ensembl REST API with fallback if network fails
- **Memory Usage**: Large datasets cached efficiently using streaming

## Core Pipeline Files

The main pipeline consists of these essential components:
- **`main.py`**: Main pipeline entry point - the primary script to use
- **`standardized_genomic_analyzer.py`**: Core analysis engine used by main.py
- **`tahoe_100M_loader.py`**: Real transcriptome data loader used by main.py
- **`state_model_client.py`**: State model integration used by main.py
- **`src/alphagenome/`**: Core AlphaGenome library modules

**Usage**: Use `main.py` for all genomic analysis. It provides comprehensive analysis across all ontologies and cell lines.