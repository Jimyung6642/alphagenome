# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Development Principles
- **Real data only**: NEVER use mock data, synthetic data, placeholder data, or hardcoded values in the pipeline. If errors occur, the pipeline should fail with proper error messages rather than fallback to fake data.
- **End-to-end approach**: Use `main.py` as the primary entry point for comprehensive analysis. All parameters, file paths, and configurations should be managed through command-line arguments.
- **Gene-agnostic design**: The pipeline works with ANY human gene symbol and should not be hardcoded for specific genes.

## Quick Start Commands
- **Run basic gene analysis**: `python3 main.py --gene TP53`
- **Fast analysis mode**: `python3 main.py --gene TP53 --fast` (limits to 10 ontologies, 5 TFs per ontology)
- **Interactive mode**: `python3 main.py --interactive`
- **Batch analysis**: `python3 main.py --genes TP53,BRCA1,CLDN18`
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
- `main.py`: Pipeline orchestrator and entry point with Tahoe-100M integration
- `standardized_genomic_analyzer.py`: Gene-agnostic analysis engine with TF expression integration
- `tahoe_100M_loader.py`: Comprehensive Tahoe-100M single-cell data loader (100M+ cells, 50 cancer cell lines)

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

### AlphaGenome + Tahoe-100M Integration Pipeline (main.py)
The main analysis pipeline integrating AlphaGenome TF predictions with Tahoe-100M single-cell expression data:

**Basic Usage:**
- Single gene analysis with Tahoe-100M integration: `python3 main.py --gene TP53`
- Fast mode (10 ontologies, 5 TFs per ontology): `python3 main.py --gene TP53 --fast`
- Analysis with specific cell lines: `python3 main.py --gene TP53 --tahoe-cell-lines "HeLa,A549,MCF7"`
- Analysis with expression threshold: `python3 main.py --gene TP53 --expression-threshold 0.5`
- Interactive mode: `python3 main.py --interactive`
- Batch analysis: `python3 main.py --genes TP53,BRCA1,CLDN18`

**Key Features:**
- **Gene-agnostic**: Works with ANY human gene symbol
- **Tahoe-100M integration**: Real single-cell expression data from 100M+ cells across 50 cancer cell lines
- **TF expression integration**: Combines AlphaGenome predictions with actual TF expression levels
- **Real data only**: No synthetic or placeholder data
- **All ontologies**: Analyzes across available AlphaGenome ontologies
- **Standardized output**: Consistent CSV format with integrated expression data

**New Command-Line Options:**
- `--fast`: Enable fast mode with limits (10 ontologies, 5 TFs per ontology)
- `--tahoe-cell-lines`: Specify target cell lines (e.g., "HeLa,A549,MCF7")
- `--expression-threshold`: Set minimum TF expression threshold (default: 0.1)
- `--tahoe-cache-dir`: Directory for Tahoe-100M data cache (default: tahoe_cache)

**Architecture:**
- `main.py`: Main entry point and pipeline orchestrator
- `standardized_genomic_analyzer.py`: Core analysis engine with Tahoe-100M integration
- `tahoe_100M_loader.py`: Comprehensive Tahoe-100M data loader and processor

**Output:**
- Timestamped CSV files with integrated AlphaGenome + Tahoe-100M results
- Enhanced analysis statistics including expression metrics
- TF expression data from target cell lines
- Expression frequency and cell count data
- Analysis metadata in JSON format

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
- **`output/`**: Main pipeline output directory
- **`main_pipeline.log`**: Detailed execution logs

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
- ✅ TF identification without State model dependency
- ✅ Comprehensive error handling and logging