#!/usr/bin/env python3
"""
AlphaGenome Genomic Analysis Pipeline

A streamlined genomic analysis pipeline using:
- AlphaGenome: TF binding predictions across ontologies

Key Features:
✅ Gene-Agnostic: Works with ANY human gene
✅ Real AlphaGenome API: No synthetic, placeholder, or mock data - uses actual TF discovery
✅ User-Configurable: Optional limits via command line arguments
✅ Standardized: Consistent output format and methodology

Usage Examples:
    # Basic analysis
    python main.py --gene CLDN18
    
    # Interactive mode
    python main.py --interactive
    
    # Batch analysis
    python main.py --genes TP53,BRCA1,CLDN18
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import json
from download_config import DownloadConfig

# Set up logging with timestamped log file in output directory
def setup_logging(output_dir: str = "output", quiet: bool = False):
    """Set up logging with timestamped log file in output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"{timestamp}.log"
    
    handlers = [logging.FileHandler(log_file)]
    
    # Only add console handler if not in quiet mode
    if not quiet:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    return log_file

# Initialize logging (will be updated in main() with proper output directory)
log_file_path = setup_logging()
logger = logging.getLogger(__name__)

# Import our core components
try:
    from standardized_genomic_analyzer import StandardizedGenomicAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    logger.error("StandardizedGenomicAnalyzer not available")


class GenomicPipeline:
    """
    Main genomic analysis pipeline orchestrator.
    """
    
    def __init__(self, output_dir: str = "output", config_overrides: Optional[dict] = None):
        """Initialize the pipeline with optional configuration overrides."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config_overrides = config_overrides or {}
        
        self.analyzer = None
        
        # Pipeline statistics
        self.pipeline_stats = {
            'genes_analyzed': 0,
            'total_results': 0,
            'pipeline_start_time': datetime.now().isoformat(),
            'successful_analyses': 0,
            'failed_analyses': 0
        }
        
        logger.info("🚀 Genomic Pipeline Initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        self._validate_dependencies()
        self._initialize_components()
    
    def _validate_dependencies(self):
        """Validate all required dependencies are available."""
        logger.info("🔍 Validating pipeline dependencies...")
        
        missing_deps = []
        
        if not ANALYZER_AVAILABLE:
            missing_deps.append("StandardizedGenomicAnalyzer")
        
        
        if missing_deps:
            error_msg = f"Missing dependencies: {', '.join(missing_deps)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("✅ All dependencies validated successfully")
        
        # Validate AlphaGenome API availability
        self._validate_alphagenome_api()
    
    def _validate_alphagenome_api(self):
        """Validate that AlphaGenome API is accessible."""
        logger.info("🔍 Validating AlphaGenome API availability...")
        
        # Check AlphaGenome API key availability
        try:
            import os
            from pathlib import Path
            
            # Check for API key in config.env
            config_file = Path("config.env")
            api_key_found = False
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    content = f.read()
                    if 'ALPHAGENOME_API_KEY' in content and '=' in content:
                        api_key_found = True
            
            # Also check environment variable
            if os.environ.get('ALPHAGENOME_API_KEY'):
                api_key_found = True
                
            if not api_key_found:
                raise RuntimeError(
                    "AlphaGenome API key not found. Create config.env with: ALPHAGENOME_API_KEY=your_key_here"
                )
                
            logger.info("✅ AlphaGenome API key configuration found")
        except Exception as e:
            raise RuntimeError(f"AlphaGenome API key validation failed: {e}")
        
        logger.info("🎉 AlphaGenome API is accessible and ready")
        logger.info("   ✅ AlphaGenome API: Key configured")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("🔧 Initializing pipeline components...")
        
        try:
            # Build configuration with user overrides
            base_config = {
                'max_ontologies': None,  # Use all ontologies by default
                'max_tfs_per_ontology': None,  # Use all TFs per ontology (no limits)
                'cache_dir': 'comprehensive_cache',
                'output_dir': str(self.output_dir),
                'real_data_only': True,  # STRICT: Only real data sources allowed
                'comprehensive_tf_discovery': True,  # Enable real AlphaGenome TF discovery
                'strict_data_validation': True  # Enforce strict validation of all data sources
            }
            
            # Apply user configuration overrides
            final_config = {**base_config, **self.config_overrides}
            
            logger.info("🔧 Pipeline Configuration:")
            logger.info(f"   Max ontologies: {final_config['max_ontologies'] or 'ALL'}")
            logger.info(f"   Max TFs per ontology: {final_config['max_tfs_per_ontology'] or 'ALL discovered'}")
            logger.info(f"   Comprehensive TF discovery: {final_config['comprehensive_tf_discovery']}")
            logger.info(f"   Real data only: {final_config['real_data_only']}")
            logger.info(f"   Strict data validation: {final_config['strict_data_validation']}")
            
            # Initialize analyzer
            self.analyzer = StandardizedGenomicAnalyzer(config=final_config)
            logger.info("✅ Standardized Genomic Analyzer initialized")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    def analyze_single_gene(self, gene_symbol: str, focus_organ: Optional[str] = None, top_n_percent: Optional[int] = None, tahoe_organ: Optional[str] = None) -> pd.DataFrame:
        """
        Perform analysis of a single gene.
        
        Args:
            gene_symbol: Human gene symbol to analyze
            focus_organ: Optional organ to prioritize
            
        Returns:
            DataFrame with analysis results
        """
        organ_msg = f" with {focus_organ} focus" if focus_organ else ""
        logger.info(f"🧬 Starting analysis for gene: {gene_symbol}{organ_msg}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Run analysis
        results = self.analyzer.analyze_any_human_gene(gene_symbol, focus_organ=focus_organ, top_n_percent=top_n_percent, tahoe_organ=tahoe_organ)
        
        if results.empty:
            logger.warning(f"No results generated for {gene_symbol}")
            self.pipeline_stats['failed_analyses'] += 1
            return pd.DataFrame()
        
        # Save results
        output_path = self.analyzer.save_comprehensive_results(results, gene_symbol)
        
        # Export gene metadata and sequence information
        gene_export_paths = self.analyzer.export_gene_metadata_and_sequence(gene_symbol)
        if gene_export_paths:
            logger.info(f"📄 Gene metadata exported: {len(gene_export_paths)} files")
            for file_type, path in gene_export_paths.items():
                logger.info(f"   {file_type}: {path}")
        
        # Update statistics
        self.pipeline_stats['genes_analyzed'] += 1
        self.pipeline_stats['successful_analyses'] += 1
        self.pipeline_stats['total_results'] += len(results)
        
        duration = datetime.now() - start_time
        
        logger.info("=" * 80)
        logger.info(f"✅ ANALYSIS COMPLETED for {gene_symbol}")
        logger.info(f"📊 Results: {len(results)} predictions")
        logger.info(f"⏱️  Duration: {duration}")
        logger.info(f"💾 Saved to: {output_path}")
        logger.info("=" * 80)
        
        return results
    
    def analyze_multiple_genes(self, gene_list: List[str]) -> dict:
        """
        Perform comprehensive analysis of multiple genes.
        
        Args:
            gene_list: List of human gene symbols
            
        Returns:
            Dictionary mapping gene symbols to result DataFrames
        """
        logger.info(f"🧬 Starting batch analysis for {len(gene_list)} genes")
        logger.info(f"📋 Genes: {', '.join(gene_list)}")
        
        results_dict = {}
        batch_start_time = datetime.now()
        
        for i, gene in enumerate(gene_list):
            logger.info(f"\n🧪 Analyzing gene {i+1}/{len(gene_list)}: {gene}")
            
            try:
                results = self.analyze_single_gene(gene)
                results_dict[gene] = results
                
            except Exception as e:
                logger.error(f"Failed to analyze {gene}: {e}")
                logger.error(f"Real data requirement failed for {gene} - pipeline must stop")
                self.pipeline_stats['failed_analyses'] += 1
                raise RuntimeError(f"Analysis failed for {gene}: {e}. Pipeline stopped to prevent synthetic data usage.")
        
        # Create batch summary
        batch_duration = datetime.now() - batch_start_time
        successful_genes = [gene for gene, df in results_dict.items() if not df.empty]
        
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 BATCH ANALYSIS SUMMARY")
        logger.info(f"   Total genes: {len(gene_list)}")
        logger.info(f"   Successful: {len(successful_genes)}")
        logger.info(f"   Failed: {len(gene_list) - len(successful_genes)}")
        logger.info(f"   Total duration: {batch_duration}")
        logger.info(f"   Successful genes: {', '.join(successful_genes)}")
        logger.info("=" * 80)
        
        # Save batch summary
        self._save_batch_summary(gene_list, results_dict, batch_duration)
        
        return results_dict
    
    def _save_batch_summary(self, gene_list: List[str], results_dict: dict, duration):
        """Save batch analysis summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"{timestamp}_batch_summary.json"
        
        summary = {
            'batch_timestamp': timestamp,
            'genes_requested': gene_list,
            'total_genes': len(gene_list),
            'successful_analyses': len([g for g, df in results_dict.items() if not df.empty]),
            'failed_analyses': len([g for g, df in results_dict.items() if df.empty]),
            'duration_seconds': duration.total_seconds(),
            'results_summary': {
                gene: {
                    'success': not df.empty,
                    'result_count': len(df) if not df.empty else 0,
                    'unique_cell_lines': df['cell_line'].nunique() if not df.empty else 0,
                    'unique_tfs': df['tf_name'].nunique() if not df.empty else 0
                }
                for gene, df in results_dict.items()
            },
            'pipeline_statistics': self.get_pipeline_statistics()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Batch summary saved to: {summary_path}")
    
    def get_pipeline_statistics(self) -> dict:
        """Get comprehensive pipeline statistics."""
        stats = self.pipeline_stats.copy()
        
        # Add analyzer statistics if available
        if self.analyzer:
            analyzer_stats = self.analyzer.get_analysis_statistics()
            stats.update(analyzer_stats)
        
        # Add system information
        stats['pipeline_end_time'] = datetime.now().isoformat()
        
        return stats
    
    def interactive_mode(self):
        """Run the pipeline in interactive mode."""
        print("\n🧬 Comprehensive Genomic Analysis Pipeline")
        print("=" * 60)
        print("Enter human gene symbols for TF identification analysis")
        print("Features: TF identification, Expression analysis, REAL data only")
        print("Type 'quit' to exit")
        
        while True:
            try:
                gene_input = input("\n🧬 Enter gene symbol (or 'quit'): ").strip()
                
                if gene_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not gene_input:
                    print("Please enter a valid gene symbol")
                    continue
                
                # Support multiple genes separated by comma
                if ',' in gene_input:
                    genes = [g.strip().upper() for g in gene_input.split(',') if g.strip()]
                    print(f"📋 Analyzing {len(genes)} genes: {', '.join(genes)}")
                    results_dict = self.analyze_multiple_genes(genes)
                    
                    # Show summary
                    successful = [g for g, df in results_dict.items() if not df.empty]
                    print(f"\n📊 Batch Results: {len(successful)}/{len(genes)} successful")
                    
                else:
                    gene = gene_input.upper()
                    print(f"🚀 Starting comprehensive analysis for: {gene}")
                    
                    results = self.analyze_single_gene(gene)
                    
                    if not results.empty:
                        print(f"\n✅ Analysis completed successfully!")
                        print(f"📊 Results: {len(results)} predictions")
                        if 'tf_name' in results.columns:
                            print(f"🧬 TFs identified: {results['tf_name'].nunique()}")
                            print(f"🧫 Cell lines: {results['cell_line'].nunique()}")
                            print(f"🏥 Organs: {results['organ'].nunique()}")
                        
                            # Show top 5 results
                            print(f"\n🏆 Top 5 TF-Cell Line Results:")
                            score_col = 'mean_binding_score' if 'mean_binding_score' in results.columns else 'binding_score'
                            for i, (_, row) in enumerate(results.head().iterrows()):
                                print(f"   {i+1}. {row['tf_name']} in {row['cell_line']} ({row['organ']})")
                                print(f"       Score: {row[score_col]:.4f}, Expression: {row['tf_expression_level']:.4f}")
                    else:
                        print("❌ No results generated")
                
            except KeyboardInterrupt:
                print("\n👋 Analysis interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Critical analysis error: {e}")
                print("🛑 Pipeline stopped to prevent synthetic data usage")
                break
        
        # Show final statistics
        final_stats = self.get_pipeline_statistics()
        print(f"\n📈 Session Statistics:")
        print(f"   Genes analyzed: {final_stats['genes_analyzed']}")
        print(f"   Total predictions: {final_stats['total_results']}")
        print(f"   Success rate: {final_stats['successful_analyses']}/{final_stats['genes_analyzed']}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TF Identification Pipeline: AlphaGenome TF Predictions + Tahoe-100M Expression Analysis (REAL DATA ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Identify TFs for a gene with expression analysis
  python main.py --gene CLDN18
  
  # Focus on specific organ for TF identification
  python main.py --gene CLDN18 --organ stomach
  
  # Fast mode for quick TF identification
  python main.py --gene CLDN18 --fast
  
  # Custom limits for TF discovery
  python main.py --gene CLDN18 --max-ontologies 20 --max-tfs-per-ontology 10
  
  # Multiple genes for TF identification
  python main.py --genes TP53,BRCA1,CLDN18
  
  # Interactive TF identification mode
  python main.py --interactive
        """
    )
    
    parser.add_argument(
        '--gene',
        type=str,
        help='Single human gene symbol to analyze'
    )
    
    parser.add_argument(
        '--genes',
        type=str,
        help='Comma-separated list of gene symbols to analyze'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--tf-organ',
        type=str,
        help='Focus TF identification on a specific organ (e.g., "stomach", "lung"). This filters the initial TF predictions to be organ-specific.'
    )
    
    # Optional limitation arguments (comprehensive by default)
    parser.add_argument(
        '--max-ontologies',
        type=int,
        help='Limit number of ontologies to analyze (default: ALL 163 ontologies)'
    )
    
    parser.add_argument(
        '--max-tfs-per-ontology', 
        type=int,
        help='Limit number of TFs per ontology (default: ALL discovered TFs)'
    )
    
    parser.add_argument(
        '--max-tfs',
        type=int,
        help='Limit total number of TFs to analyze across all ontologies (default: ALL discovered TFs)'
    )
    
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Enable fast mode with limits: 10 ontologies, 5 TFs per ontology'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress terminal output, only log to file'
    )
    
    # Timeout and reliability options
    parser.add_argument(
        '--download-timeout',
        type=int,
        default=3600,
        help='Timeout for dataset downloads in seconds (default: 3600s for overnight runs)'
    )
    
    parser.add_argument(
        '--api-timeout',
        type=int,
        default=60,
        help='Timeout for API calls in seconds (default: 60s)'
    )
    
    parser.add_argument(
        '--retry-attempts',
        type=int,
        default=3,
        help='Number of retry attempts for failed operations (default: 3)'
    )
    
    parser.add_argument(
        '--overnight-mode',
        action='store_true',
        help='Enable overnight mode with extended timeouts and aggressive caching'
    )
    
    # TF expression analysis options
    parser.add_argument(
        '--focus-cell-lines',
        type=str,
        help='Comma-separated list of specific cell lines to focus on for TF expression analysis (e.g., "HeLa,A549,MCF7")'
    )
    
    parser.add_argument(
        '--expression-threshold',
        type=float,
        default=0.0,
        help='Minimum TF expression threshold for inclusion in results (default: 0.0 - show all)'
    )
    
    parser.add_argument(
        '--tahoe-cache-dir',
        type=str,
        default='tahoe_cache',
        help='Directory for caching Tahoe-100M data (default: tahoe_cache)'
    )
    
    parser.add_argument(
        '--tahoe-organ',
        type=str,
        help='Focus expression analysis on cell lines from a specific organ in Tahoe-100M (e.g., "stomach", "lung").'
    )
    
    # AlphaGenome TF prediction options
    parser.add_argument(
        '--tf-top-n-percent',
        type=int,
        default=10,
        metavar='N',
        help='Filter to the top N%% of TF predictions based on binding score. Only applied when --tf-organ is NOT specified (organ-specific searches use all relevant TFs). Default is 10.'
    )
    
    return parser.parse_args()


def main():
    """Main pipeline entry point."""
    start_time = datetime.now()
    args = parse_arguments()
    
    # Set up logging with proper output directory and timestamp
    global log_file_path
    log_file_path = setup_logging(args.output, quiet=args.quiet)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🧬 AlphaGenome + Tahoe-100M Genomic Analysis Pipeline")
    print("🔬 AlphaGenome TF Predictions + Tahoe-100M Expression Integration")
    print("🧫 100M+ Single-Cell Transcriptomic Profiles across 50 Cancer Cell Lines")
    print("=" * 80)
    print(f"📝 Logs will be saved to: {log_file_path}")
    
    try:
        # Build configuration from config.env and command line arguments
        config_overrides = {}
        
        # Apply command line overrides to config.env defaults
        if args.download_timeout != 3600:  # Only override if different from default
            config_overrides['download_timeout'] = args.download_timeout
        if args.api_timeout != 60:  # Only override if different from default
            config_overrides['api_timeout'] = args.api_timeout
        if args.retry_attempts != 3:  # Only override if different from default
            config_overrides['retry_attempts'] = args.retry_attempts
        
        # Handle overnight mode
        if args.overnight_mode:
            print("🌙 Overnight mode enabled - using extended timeouts and aggressive caching")
            config_overrides.update({
                'overnight_mode': True,
                'aggressive_caching': True
            })
        
        # Handle fast mode
        if args.fast:
            print("⚡ Fast mode enabled - using limited analysis for quick results")
            config_overrides.update({
                'fast_mode': True,
                'max_ontologies': 10,
                'max_tfs_per_ontology': 5
            })
        
        # Apply individual limits if specified
        if args.max_ontologies:
            config_overrides['max_ontologies'] = args.max_ontologies
        if args.max_tfs_per_ontology:
            config_overrides['max_tfs_per_ontology'] = args.max_tfs_per_ontology
        if args.max_tfs:
            config_overrides['max_tfs'] = args.max_tfs
        if args.output != 'output':  # Only override if different from default
            config_overrides['output_dir'] = args.output
        if args.verbose:
            config_overrides['verbose'] = True
        if args.quiet:
            config_overrides['quiet'] = True
        
        # Tahoe-100M specific configuration
        if args.focus_cell_lines:
            focus_cell_lines = [c.strip() for c in args.focus_cell_lines.split(',') if c.strip()]
            config_overrides['focus_cell_lines'] = focus_cell_lines
            print(f"🧫 Focus cell lines: {', '.join(focus_cell_lines)}")
        
        if args.expression_threshold != 0.0:
            config_overrides['expression_threshold'] = args.expression_threshold
            print(f"📊 Expression threshold: {args.expression_threshold}")
        
        
        if args.tahoe_cache_dir != 'tahoe_cache':
            config_overrides['tahoe_cache_dir'] = args.tahoe_cache_dir
            print(f"💾 Tahoe cache directory: {args.tahoe_cache_dir}")
        
        # Initialize comprehensive configuration
        download_config = DownloadConfig(config_overrides)
        
        # Display current configuration
        print(f"⏱️  Download timeout: {download_config.get('download_timeout')/3600:.1f} hours")
        print(f"🔄 Retry attempts: {download_config.get('retry_attempts')}")
        print(f"📊 Max ontologies: {download_config.get('max_ontologies') or 'ALL (163)'}")
        print(f"🧬 Max TFs per ontology: {download_config.get('max_tfs_per_ontology') or 'ALL discovered'}")
        
        if download_config.get('overnight_mode'):
            print("💤 Optimized for overnight execution with enhanced reliability")
        if download_config.get('fast_mode'):
            print("⚡ Fast mode active - limited analysis for quick results")
        else:
            print("🔬 Full mode - using available data (may take longer)")
        
        # Initialize pipeline
        pipeline = GenomicPipeline(
            output_dir=download_config.get('output_dir'),
            config_overrides=download_config.config
        )
        
        if args.interactive:
            # Interactive mode
            pipeline.interactive_mode()
            
        elif args.gene:
            # Single gene analysis
            gene = args.gene.upper()
            organ_msg = f" with {args.tf_organ} focus" if args.tf_organ else ""
            print(f"🧬 Analyzing gene: {gene}{organ_msg}")
            
            # Only apply top-N filtering when no organ focus is specified
            top_n_to_use = None if args.tf_organ else args.tf_top_n_percent
            
            results = pipeline.analyze_single_gene(gene, focus_organ=args.tf_organ, top_n_percent=top_n_to_use, tahoe_organ=args.tahoe_organ)
            
            if not results.empty:
                print(f"\n✅ Analysis completed successfully!")
                print(f"📊 Generated {len(results)} TF-cell line combinations")
                print(f"🧬 Identified {results['tf_name'].nunique()} unique TFs")
                print(f"🧫 Across {results['cell_line'].nunique()} cell lines and {results['organ'].nunique()} organs")
            else:
                print(f"\n❌ No results generated for {gene}")
                
        elif args.genes:
            # Multiple gene analysis
            gene_list = [g.strip().upper() for g in args.genes.split(',') if g.strip()]
            print(f"📋 TF identification for {len(gene_list)} genes: {', '.join(gene_list)}")
            
            results_dict = {}
            for gene in gene_list:
                print(f"🧬 Analyzing gene {gene}...")
                # Only apply top-N filtering when no organ focus is specified
                top_n_to_use = None if args.tf_organ else args.tf_top_n_percent
                results_dict[gene] = pipeline.analyze_single_gene(gene, focus_organ=args.tf_organ, top_n_percent=top_n_to_use, tahoe_organ=args.tahoe_organ)
            
            successful = len([g for g, df in results_dict.items() if not df.empty])
            print(f"\n📊 Batch analysis completed: {successful}/{len(gene_list)} successful")
            
        else:
            # No specific arguments - show help and enter interactive mode
            print("No gene specified. Entering interactive TF identification mode...")
            pipeline.interactive_mode()
        
        # Show final pipeline statistics
        final_stats = pipeline.get_pipeline_statistics()
        print(f"\n📈 Pipeline Statistics:")
        print(f"   Genes analyzed: {final_stats['genes_analyzed']}")
        print(f"   Total predictions: {final_stats['total_results']}")
        print(f"   Successful analyses: {final_stats['successful_analyses']}")
        print(f"   Failed analyses: {final_stats['failed_analyses']}")
        
        # Log total execution time
        total_duration = datetime.now() - start_time
        logger.info(f"Total pipeline execution time: {total_duration}")
        print(f"⏱️  Total analysis time: {total_duration}")
        
        print(f"\n🎉 TF Identification Pipeline completed!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()