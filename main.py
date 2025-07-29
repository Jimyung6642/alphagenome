#!/usr/bin/env python3
"""
Comprehensive Genomic Analysis Pipeline

The ultimate comprehensive pipeline combining:
- AlphaGenome: TF binding predictions across ALL 163 ontologies BY DEFAULT
- State Model: Cellular perturbation analysis using SE-600M
- Tahoe-100M: Real transcriptome data from ALL 50 cancer cell lines

Key Features:
✅ Gene-Agnostic: Works with ANY human gene
✅ COMPREHENSIVE BY DEFAULT: Uses ALL ontologies, ALL discovered TFs, ALL cell lines
✅ Real AlphaGenome API: No synthetic, placeholder, or mock data - uses actual TF discovery
✅ User-Configurable: Optional limits via command line arguments
✅ Standardized: Consistent output format and methodology
✅ Scalable: Efficient data streaming and caching

Default Behavior (Comprehensive):
    - ALL 163 ontologies analyzed
    - ALL discovered TFs per ontology (no artificial limits)
    - ALL 50 cancer cell lines from Tahoe-100M
    - Real AlphaGenome API calls for TF discovery

Usage Examples:
    # Comprehensive analysis (DEFAULT - may take 30+ minutes)
    python main.py --gene CLDN18
    
    # Fast mode for testing (limited analysis)
    python main.py --gene CLDN18 --fast
    
    # Custom limits
    python main.py --gene CLDN18 --max-ontologies 20 --max-tfs-per-ontology 10
    
    # Interactive mode
    python main.py --interactive
    
    # Batch analysis (comprehensive by default)
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

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('main_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Import our comprehensive components
try:
    from standardized_genomic_analyzer import StandardizedGenomicAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    logger.error("StandardizedGenomicAnalyzer not available")

try:
    from tahoe_100M_loader import ComprehensiveTahoeDataLoader
    TAHOE_AVAILABLE = True
except ImportError:
    TAHOE_AVAILABLE = False
    logger.error("ComprehensiveTahoeDataLoader not available")


class ComprehensiveGenomicPipeline:
    """
    Main comprehensive genomic analysis pipeline orchestrator.
    """
    
    def __init__(self, output_dir: str = "output", config_overrides: Optional[dict] = None):
        """Initialize the comprehensive pipeline with optional configuration overrides."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config_overrides = config_overrides or {}
        
        self.analyzer = None
        self.tahoe_loader = None
        
        # Pipeline statistics
        self.pipeline_stats = {
            'genes_analyzed': 0,
            'total_results': 0,
            'pipeline_start_time': datetime.now().isoformat(),
            'successful_analyses': 0,
            'failed_analyses': 0
        }
        
        logger.info("🚀 Comprehensive Genomic Pipeline Initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        self._validate_dependencies()
        self._initialize_components()
    
    def _validate_dependencies(self):
        """Validate all required dependencies are available."""
        logger.info("🔍 Validating pipeline dependencies...")
        
        missing_deps = []
        
        if not ANALYZER_AVAILABLE:
            missing_deps.append("StandardizedGenomicAnalyzer")
        
        if not TAHOE_AVAILABLE:
            missing_deps.append("ComprehensiveTahoeDataLoader")
        
        # Check for required external tools
        try:
            import datasets
            logger.info("✅ Hugging Face datasets available")
        except ImportError:
            missing_deps.append("datasets (pip install datasets)")
        
        try:
            import anndata
            logger.info("✅ AnnData available")
        except ImportError:
            missing_deps.append("anndata (pip install anndata)")
        
        # Check for State model CLI
        import subprocess
        try:
            result = subprocess.run(['uv', 'tool', 'run', '--from', 'arc-state', 'state', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ State model CLI available via uv tool run")
            else:
                missing_deps.append("arc-state CLI (uv tool install arc-state)")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing_deps.append("arc-state CLI (uv tool install arc-state)")
        
        if missing_deps:
            error_msg = f"Missing dependencies: {', '.join(missing_deps)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("✅ All dependencies validated successfully")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("🔧 Initializing pipeline components...")
        
        try:
            # Build comprehensive configuration with user overrides
            base_config = {
                'max_ontologies': None,  # Use ALL 163 ontologies by default
                'max_cell_lines': None,  # Use ALL 50 cell lines by default  
                'max_tfs_per_ontology': None,  # Use ALL TFs per ontology (no limits)
                'max_cells_per_line': 1000,  # More cells for better State predictions
                'cache_dir': 'comprehensive_cache',
                'output_dir': str(self.output_dir),
                'streaming_mode': True,
                'real_data_only': True,
                'comprehensive_tf_discovery': True  # Enable real AlphaGenome TF discovery
            }
            
            # Apply user configuration overrides
            final_config = {**base_config, **self.config_overrides}
            
            logger.info("🔧 Pipeline Configuration:")
            logger.info(f"   Max ontologies: {final_config['max_ontologies'] or 'ALL (163)'}")
            logger.info(f"   Max TFs per ontology: {final_config['max_tfs_per_ontology'] or 'ALL discovered'}")
            logger.info(f"   Max cell lines: {final_config['max_cell_lines'] or 'ALL (50)'}")
            logger.info(f"   Comprehensive TF discovery: {final_config['comprehensive_tf_discovery']}")
            
            # Initialize comprehensive analyzer
            self.analyzer = StandardizedGenomicAnalyzer(config=final_config)
            logger.info("✅ Standardized Genomic Analyzer initialized")
            
            # Initialize Tahoe loader for direct access
            self.tahoe_loader = ComprehensiveTahoeDataLoader(
                cache_dir='tahoe_cache',
                streaming=True
            )
            logger.info("✅ Tahoe-100M Data Loader initialized")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    def analyze_single_gene(self, gene_symbol: str, focus_organ: Optional[str] = None) -> pd.DataFrame:
        """
        Perform comprehensive analysis of a single gene.
        
        Args:
            gene_symbol: Human gene symbol to analyze
            focus_organ: Optional organ to prioritize for cell line selection
            
        Returns:
            DataFrame with comprehensive analysis results
        """
        organ_msg = f" with {focus_organ} focus" if focus_organ else ""
        logger.info(f"🧬 Starting comprehensive analysis for gene: {gene_symbol}{organ_msg}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Run comprehensive analysis
            results = self.analyzer.analyze_any_human_gene(gene_symbol, focus_organ=focus_organ)
            
            if results.empty:
                logger.warning(f"No results generated for {gene_symbol}")
                self.pipeline_stats['failed_analyses'] += 1
                return pd.DataFrame()
            
            # Save results
            output_path = self.analyzer.save_comprehensive_results(results, gene_symbol)
            
            # Update statistics
            self.pipeline_stats['genes_analyzed'] += 1
            self.pipeline_stats['successful_analyses'] += 1
            self.pipeline_stats['total_results'] += len(results)
            
            duration = datetime.now() - start_time
            
            logger.info("=" * 80)
            logger.info(f"✅ COMPREHENSIVE ANALYSIS COMPLETED for {gene_symbol}")
            logger.info(f"📊 Results: {len(results)} predictions across {results['cell_line'].nunique()} cell lines")
            logger.info(f"⏱️  Duration: {duration}")
            logger.info(f"💾 Saved to: {output_path}")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed for {gene_symbol}: {e}")
            self.pipeline_stats['failed_analyses'] += 1
            raise RuntimeError(f"Gene analysis failed for {gene_symbol}: {e}")
    
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
                results_dict[gene] = pd.DataFrame()
                continue
        
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
        print("Enter human gene symbols for comprehensive analysis")
        print("Features: ALL cell lines, ALL ontologies, REAL data only")
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
                        print(f"🧫 Cell lines: {results['cell_line'].nunique()}")
                        print(f"🧬 TFs: {results['tf_name'].nunique()}")
                        
                        # Show top 5 results
                        print(f"\n🏆 Top 5 TF-Cell Line Combinations:")
                        for i, (_, row) in enumerate(results.head().iterrows()):
                            print(f"   {i+1}. {row['tf_name']} in {row['cell_line']} "
                                  f"(score: {row['combined_impact_score']:.4f})")
                    else:
                        print("❌ No results generated")
                
            except KeyboardInterrupt:
                print("\n👋 Analysis interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Analysis error: {e}")
                continue
        
        # Show final statistics
        final_stats = self.get_pipeline_statistics()
        print(f"\n📈 Session Statistics:")
        print(f"   Genes analyzed: {final_stats['genes_analyzed']}")
        print(f"   Total predictions: {final_stats['total_results']}")
        print(f"   Success rate: {final_stats['successful_analyses']}/{final_stats['genes_analyzed']}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Genomic Analysis Pipeline: AlphaGenome + State + Tahoe-100M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Comprehensive analysis (DEFAULT - ALL ontologies, ALL TFs)
  python main.py --gene CLDN18
  
  # Comprehensive with organ focus
  python main.py --gene CLDN18 --organ stomach
  
  # Fast mode for quick testing (limited analysis)
  python main.py --gene CLDN18 --fast
  
  # Custom limits
  python main.py --gene CLDN18 --max-ontologies 20 --max-tfs-per-ontology 10
  
  # Multiple genes (comprehensive by default)
  python main.py --genes TP53,BRCA1,CLDN18
  
  # Interactive mode
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
        '--organ',
        type=str,
        help='Focus analysis on specific organ (e.g., "stomach", "lung", "breast"). Prioritizes cell lines from this organ.'
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
        '--max-cell-lines',
        type=int, 
        help='Limit number of cell lines to analyze (default: ALL 50 cell lines)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Enable fast mode with limits: 10 ontologies, 5 TFs per ontology, 10 cell lines'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main pipeline entry point."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🧬 Comprehensive Genomic Analysis Pipeline")
    print("🔬 AlphaGenome + State Model + Tahoe-100M Integration")
    print("=" * 70)
    
    try:
        # Build configuration from command line arguments
        config_overrides = {}
        
        # Handle fast mode
        if args.fast:
            print("⚡ Fast mode enabled - using limited analysis for quick results")
            config_overrides.update({
                'max_ontologies': 10,
                'max_tfs_per_ontology': 5,
                'max_cell_lines': 10
            })
        else:
            print("🔬 Comprehensive mode - using ALL ontologies and TFs (may take longer)")
        
        # Apply individual limits if specified
        if args.max_ontologies:
            config_overrides['max_ontologies'] = args.max_ontologies
            print(f"📊 Limited to {args.max_ontologies} ontologies")
            
        if args.max_tfs_per_ontology:
            config_overrides['max_tfs_per_ontology'] = args.max_tfs_per_ontology
            print(f"🧬 Limited to {args.max_tfs_per_ontology} TFs per ontology")
            
        if args.max_cell_lines:
            config_overrides['max_cell_lines'] = args.max_cell_lines
            print(f"🧫 Limited to {args.max_cell_lines} cell lines")
        
        # Initialize pipeline with configuration
        pipeline = ComprehensiveGenomicPipeline(
            output_dir=args.output,
            config_overrides=config_overrides
        )
        
        if args.interactive:
            # Interactive mode
            pipeline.interactive_mode()
            
        elif args.gene:
            # Single gene analysis
            gene = args.gene.upper()
            organ_msg = f" with {args.organ} focus" if args.organ else ""
            print(f"🧬 Analyzing gene: {gene}{organ_msg}")
            
            results = pipeline.analyze_single_gene(gene, focus_organ=args.organ)
            
            if not results.empty:
                print(f"\n✅ Analysis completed successfully!")
                print(f"📊 Generated {len(results)} comprehensive predictions")
            else:
                print(f"\n❌ No results generated for {gene}")
                
        elif args.genes:
            # Multiple gene analysis
            gene_list = [g.strip().upper() for g in args.genes.split(',') if g.strip()]
            print(f"📋 Analyzing {len(gene_list)} genes: {', '.join(gene_list)}")
            
            results_dict = {}
            for gene in gene_list:
                print(f"🧬 Analyzing gene {gene}...")
                results_dict[gene] = pipeline.analyze_single_gene(gene, focus_organ=args.organ)
            
            successful = len([g for g, df in results_dict.items() if not df.empty])
            print(f"\n📊 Batch analysis completed: {successful}/{len(gene_list)} successful")
            
        else:
            # No specific arguments - show help and enter interactive mode
            print("No gene specified. Entering interactive mode...")
            pipeline.interactive_mode()
        
        # Show final pipeline statistics
        final_stats = pipeline.get_pipeline_statistics()
        print(f"\n📈 Pipeline Statistics:")
        print(f"   Genes analyzed: {final_stats['genes_analyzed']}")
        print(f"   Total predictions: {final_stats['total_results']}")
        print(f"   Successful analyses: {final_stats['successful_analyses']}")
        print(f"   Failed analyses: {final_stats['failed_analyses']}")
        
        print(f"\n🎉 Comprehensive Genomic Analysis Pipeline completed!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()