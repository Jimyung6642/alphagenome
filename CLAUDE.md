# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Development Principles
- **Real data only**: NEVER use mock data, synthetic data, placeholder data, or hardcoded values in the pipeline. If errors occur, the pipeline should fail with proper error messages rather than fallback to fake data.
- **End-to-end approach**: Use `main.py` as the primary entry point for comprehensive analysis. All parameters, file paths, and configurations should be managed through command-line arguments.
- **Gene-agnostic design**: The pipeline works with ANY human gene symbol and should not be hardcoded for specific genes.

## Quick Start Commands

### Perturbation Analysis Pipeline (main.py)
- **Basic perturbation analysis**: `python3 main.py --genes TP53 --organ stomach`
- **Multiple genes with organ focus**: `python3 main.py --genes TP53,BRCA1,CLDN18 --organ lung`
- **Custom perturbation parameters**: `python3 main.py --genes CLDN18 --perturbation-strength 0.8 --deg-p-threshold 0.01`
- **Verbose output**: `python3 main.py --genes TP53 --verbose` (enables console logging)
- **TF organ filtering**: `python3 main.py --genes TP53 --tf-organ stomach --organ lung`

### Other Commands
- **Web search**: Run `gemini` in bash for internet searches when needed

### Building and Testing
- **Install package**: `pip3 install .` (or `pip3 install -e .` for development)
- **Run tests**: `hatch test` (runs tests on default Python version)
- **Run all tests**: `hatch test --all` (runs tests on all supported Python versions 3.10-3.13)
- **Run single test file**: `hatch test src/alphagenome/models/dna_client_test.py`
- **Format code**: `hatch run check:format` (uses pyink formatter with Google style)
- **Lint code**: `hatch run check:lint` (uses pylint)
- **Check all**: `hatch run check:all` (runs both format and lint)

### Development Environment
- Uses hatch as build system and dependency manager
- Tests use absltest framework (Google's testing library)
- Code formatting follows Google Python style guide with pyink (80 char line length)
- Proto files are automatically compiled during build via `hatch_build.py`
- Python 3.10-3.13 supported, with test matrix for all versions

## Code Architecture

### Core Structure
The codebase consists of a genomic analysis pipeline and the core AlphaGenome library:

**Main Pipeline Components:**
- `main.py`: Comprehensive perturbation analysis pipeline integrating AlphaGenome + Tahoe-100M + ST-Tahoe State Model + DEG Analysis
- `standardized_genomic_analyzer.py`: Gene-agnostic analysis engine with TF expression integration
- `tahoe_100M_loader.py`: Comprehensive Tahoe-100M single-cell data loader (100M+ cells, 50 cancer cell lines)
- `state_perturbation_engine.py`: ST-Tahoe State model wrapper for zero-shot TF perturbation predictions
- `tahoe_state_bridge.py`: Data format converter from Tahoe-100M to State model H5AD format
- `deg_analyzer.py`: Differential expression analysis for target genes and genome-wide DEGs

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

### Comprehensive Perturbation Analysis Pipeline (main.py)
The main analysis pipeline integrating AlphaGenome, Tahoe-100M, ST-Tahoe State Model, and DEG analysis for comprehensive genomic perturbation studies:

**Basic Usage:**
- Single gene analysis: `python3 main.py --genes TP53 --organ stomach`
- Multiple genes with organ focus: `python3 main.py --genes TP53,BRCA1,CLDN18 --organ lung`
- Custom perturbation parameters: `python3 main.py --genes CLDN18 --perturbation-strength 0.8`
- TF organ filtering: `python3 main.py --genes TP53 --tf-organ stomach`
- Verbose output: `python3 main.py --genes TP53 --verbose`

**Key Features:**
- **Comprehensive perturbation analysis**: Integrates AlphaGenome TF predictions, Tahoe-100M expression data, ST-Tahoe zero-shot perturbations, and DEG analysis
- **Gene-agnostic**: Works with ANY human gene symbol
- **Real data only**: No synthetic or placeholder data throughout the pipeline
- **Multi-modal integration**: Combines transcription factor predictions, baseline expression, perturbation effects, and differential expression
- **Organ-specific analysis**: Focused analysis on specific organs and cell lines
- **Zero-shot TF perturbation**: ST-Tahoe State model for cellular perturbation effect predictions without training
- **Standardized output**: Comprehensive CSV and JSON outputs with integrated results

**Command-Line Options:**
- `--genes`: Target gene(s) for analysis (comma-separated list, e.g., "TP53,BRCA1")
- `--organ`: Focus analysis on specific organ (e.g., stomach, lung)
- `--tf-organ`: Filter AlphaGenome TF predictions to specific organ context
- `--perturbation-strength`: Perturbation strength for ST-Tahoe model (default: 0.5)
- `--deg-p-threshold`: P-value threshold for DEG analysis (default: 0.05)
- `--deg-fc-threshold`: Log2 fold-change threshold for DEG analysis (default: 0.5)
- `--tf-percent`: Filter to top N% of TF predictions (default: 10)
- `--verbose`: Enable console logging output
- `--tahoe-cache-dir`: Directory for Tahoe-100M H5AD cache (default: output/bridge_cache)

**Architecture:**
- `main.py`: Main entry point and comprehensive perturbation analysis orchestrator
- `standardized_genomic_analyzer.py`: Core analysis engine with TF expression integration
- `tahoe_100M_loader.py`: Tahoe-100M data loader and processor (100M+ cells)
- `state_perturbation_engine.py`: ST-Tahoe State model wrapper for zero-shot TF perturbation predictions
- `tahoe_state_bridge.py`: Data format converter for Tahoe-100M to State model compatibility
- `deg_analyzer.py`: Differential expression analysis engine

**Output:**
- Timestamped CSV files with comprehensive perturbation analysis results
- AlphaGenome TF predictions integrated with Tahoe-100M expression data
- ST-Tahoe zero-shot perturbation effect predictions
- Differential expression analysis results (target genes and genome-wide DEGs)
- Statistical significance testing and p-value corrections
- Analysis metadata and parameters in JSON format

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


## Important Setup Notes

### Configuration Files
- **`config.env`**: Required for AlphaGenome API key (`ALPHAGENOME_API_KEY=your_key_here`)
- **Git ignored**: Environment files are excluded from version control

### Directory Structure
- **`comprehensive_cache/`**: General analysis cache
- **`output/`**: Main pipeline output directory with timestamped logs and results
- **`tahoe_cache/`**: Tahoe-100M data cache directory (configurable via --tahoe-cache-dir)
- **`config.env`**: Environment configuration file (git ignored)

### Common Issues
- **AlphaGenome Client**: Use `dna_client.create(api_key)` not `DnaClient()` constructor
- **Gene Validation**: Uses Ensembl REST API with fallback if network fails
- **Tahoe-100M Data**: Expression data is distributed across 1,026 pseudobulk files, searches multiple files automatically
- **Proto Compilation**: Protocol buffer files are auto-generated during build - don't edit `*_pb2.py` files directly
- **Missing Dependencies**: If imports fail, ensure AlphaGenome package is installed with `pip3 install -e .`
- **API Key Setup**: Ensure `config.env` exists in project root with valid `ALPHAGENOME_API_KEY`

## Core Pipeline Files

The main pipeline consists of these essential components:
- **`main.py`**: Main pipeline entry point - the primary script to use
- **`standardized_genomic_analyzer.py`**: Core analysis engine used by main.py
- **`src/alphagenome/`**: Core AlphaGenome library modules

**Usage**: Use `main.py` for integrated genomic analysis combining AlphaGenome predictions with Tahoe-100M TF expression data.

## Development Workflow

### Making Changes
1. **Install in development mode**: `pip3 install -e .` 
2. **Run tests after changes**: `hatch test` or `hatch test --all`
3. **Format and lint code**: `hatch run check:all` before committing
4. **Test pipeline changes**: Use `python3 main.py --gene TP53 --fast` for quick validation

### Testing Integration
- **Pipeline validation**: `python3 test_tahoe_integration.py` (if available)
- **Single gene test**: `python3 main.py --gene TP53 --fast --verbose`
- **Check logs**: Review timestamped log files in `output/` directory

## Current Pipeline Status

### Completed Improvements
- **Multi-file pseudobulk search**: Searches across 1,026 Tahoe-100M files instead of just one
- **Intelligent file discovery**: Caches file locations (`_cell_line_file_mapping`) for faster subsequent searches
- **Real metadata integration**: Uses 102 real cell lines from Tahoe-100M across 15 organ types
- **Performance optimizations**: Reduced search scope and prioritized common files for efficiency
- **Expression data conversion**: Converts pseudobulk summary data to simulated single-cell matrices

### Pipeline Architecture Status
- ✅ Real AlphaGenome API integration (no synthetic TF predictions)
- ✅ Real Tahoe-100M metadata (102 cell lines, 15 organs)
- ✅ Multi-file expression data loading with caching
- ✅ State model integration for perturbation predictions
- ✅ Differential expression analysis (DEG) for target genes and genome-wide effects
- ✅ Data format bridging between Tahoe-100M and State model
- ✅ Comprehensive error handling and logging throughout pipeline