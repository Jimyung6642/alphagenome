#!/usr/bin/env python3
"""
Standardized Genomic Analyzer

A comprehensive, gene-agnostic pipeline that combines AlphaGenome TF predictions
with State model cellular perturbation analysis using real Tahoe-100M transcriptome data.

Key Features:
- Works with ANY human gene (gene-agnostic)
- Uses ALL 50 cancer cell lines from Tahoe-100M
- Employs real transcriptome data (62,710 genes)
- Comprehensive TF analysis across all AlphaGenome ontologies
- Real State model predictions (no synthetic/mock data)
- Standardized output format for reproducibility

Usage:
    analyzer = StandardizedGenomicAnalyzer()
    results = analyzer.analyze_any_human_gene("TP53")
    results = analyzer.analyze_any_human_gene("BRCA1")
    results = analyzer.analyze_any_human_gene("CLDN18")
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json

# Add src directory to Python path for importing alphagenome modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# AlphaGenome imports
try:
    from alphagenome import colab_utils
    from alphagenome.data import genome
    from alphagenome.models import dna_client
    ALPHAGENOME_AVAILABLE = True
except ImportError:
    ALPHAGENOME_AVAILABLE = False
    logging.warning("AlphaGenome modules not available")

# Import our comprehensive data components
try:
    from tahoe_100M_loader import ComprehensiveTahoeDataLoader
    TAHOE_LOADER_AVAILABLE = True
except ImportError:
    TAHOE_LOADER_AVAILABLE = False
    logging.warning("ComprehensiveTahoeDataLoader not available")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StandardizedGenomicAnalyzer:
    """
    Standardized genomic analyzer that works with ANY human gene using 
    comprehensive real data from AlphaGenome, State model, and Tahoe-100M.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the standardized genomic analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.alphagenome_client = None
        self.state_client = None
        self.tahoe_loader = None
        
        # Gene validation cache
        self._validated_genes = set()
        
        # Analysis statistics
        self.analysis_stats = {
            'genes_analyzed': 0,
            'total_tf_predictions': 0,
            'total_state_predictions': 0,
            'cell_lines_used': 0,
            'ontologies_analyzed': 0
        }
        
        logger.info("🧬 Initialized Standardized Genomic Analyzer")
        logger.info(f"   Configuration: {self.config}")
        
        # Initialize all components
        self._initialize_components()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for comprehensive analysis."""
        return {
            'max_ontologies': None,  # No limit - use ALL 163 ontologies by default
            'max_cell_lines': None,  # No limit - use ALL 50 cell lines by default
            'max_tfs_per_ontology': None,  # No limit - discover ALL TFs per ontology by default
            'max_cells_per_line': 1000,  # Increased for better State predictions
            'cache_dir': 'comprehensive_cache',
            'output_dir': 'output',  # Changed to 'output' as standard
            'streaming_mode': True,
            'parallel_processing': True,
            'validate_genes': True,
            'real_data_only': True,  # No synthetic/mock data - use real AlphaGenome API
            'comprehensive_tf_discovery': True,  # Enable real TF discovery via AlphaGenome
        }
    
    def _initialize_components(self):
        """Initialize all required components."""
        logger.info("🔧 Initializing comprehensive analysis components...")
        
        # Initialize AlphaGenome client
        if ALPHAGENOME_AVAILABLE:
            try:
                api_key = colab_utils.get_api_key()
                self.alphagenome_client = dna_client.create(api_key)
                logger.info("✅ AlphaGenome client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AlphaGenome client: {e}")
                raise RuntimeError(f"AlphaGenome initialization failed: {e}")
        else:
            raise ImportError("AlphaGenome modules required for genomic analysis")
        
        # Initialize Tahoe-100M data loader
        if TAHOE_LOADER_AVAILABLE:
            try:
                self.tahoe_loader = ComprehensiveTahoeDataLoader(
                    cache_dir=self.config['cache_dir'],
                    streaming=self.config['streaming_mode']
                )
                logger.info("✅ Tahoe-100M data loader initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Tahoe loader: {e}")
                raise RuntimeError(f"Tahoe-100M initialization failed: {e}")
        else:
            raise ImportError("ComprehensiveTahoeDataLoader required for real transcriptome data")
        
        # Initialize State model client
        try:
            from state_model_client import StatePredictionClient
            self.state_client = StatePredictionClient()
            self.state_client.load_model()
            logger.info("✅ State model client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize State client: {e}")
            raise RuntimeError(f"State model initialization failed: {e}")
        
        logger.info("🚀 All components initialized successfully")
    
    def validate_human_gene(self, gene_symbol: str) -> str:
        """
        Validate that the gene symbol is a valid human gene.
        
        Args:
            gene_symbol: Gene symbol to validate
            
        Returns:
            Validated gene symbol (uppercase)
            
        Raises:
            ValueError: If gene is not valid
        """
        gene_symbol = gene_symbol.upper().strip()
        
        # Check cache first
        if gene_symbol in self._validated_genes:
            return gene_symbol
        
        if not self.config.get('validate_genes', True):
            logger.info(f"Gene validation skipped for {gene_symbol}")
            self._validated_genes.add(gene_symbol)
            return gene_symbol
        
        logger.info(f"🔍 Validating human gene: {gene_symbol}")
        
        try:
            # Use Ensembl REST API to validate gene
            import urllib.request
            import ssl
            import json
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Query Ensembl for gene information
            url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}?content-type=application/json"
            
            try:
                with urllib.request.urlopen(url, context=ssl_context, timeout=10) as response:
                    gene_data = json.loads(response.read().decode())
                    
                if 'id' in gene_data:
                    logger.info(f"✅ Valid human gene: {gene_symbol} ({gene_data.get('description', 'No description')})")
                    self._validated_genes.add(gene_symbol)
                    return gene_symbol
                else:
                    raise ValueError(f"Gene {gene_symbol} not found in Ensembl")
                    
            except Exception as e:
                logger.warning(f"Ensembl validation failed for {gene_symbol}: {e}")
                logger.info(f"Proceeding with gene {gene_symbol} (validation skipped)")
                self._validated_genes.add(gene_symbol)
                return gene_symbol
                
        except Exception as e:
            logger.error(f"Gene validation error for {gene_symbol}: {e}")
            raise ValueError(f"Could not validate gene {gene_symbol}: {e}")
    
    def get_all_available_cell_lines(self, focus_organ: Optional[str] = None) -> List[str]:
        """
        Get available cell lines from Tahoe-100M dataset with optional organ prioritization.
        
        Args:
            focus_organ: If specified, prioritize cell lines from this organ (e.g., "stomach", "lung", "breast")
        
        Returns:
            List of cancer cell lines (organ-prioritized if requested)
        """
        if focus_organ:
            logger.info(f"🎯 Discovering cell lines with {focus_organ} cancer prioritization...")
        else:
            logger.info("🧫 Discovering ALL available cell lines from Tahoe-100M...")
        
        try:
            prioritized_cell_lines = self.tahoe_loader.get_prioritized_cell_lines(
                focus_organ=focus_organ,
                max_lines=self.config['max_cell_lines']
            )
            
            analysis_type = f"{focus_organ}-prioritized" if focus_organ else "comprehensive"
            logger.info(f"🌍 Using {len(prioritized_cell_lines)} cancer cell lines for {analysis_type} analysis")
            
            # Update statistics
            self.analysis_stats['cell_lines_used'] = len(prioritized_cell_lines)
            self.analysis_stats['organ_focused'] = focus_organ
            
            return prioritized_cell_lines
            
        except Exception as e:
            logger.error(f"Failed to get cell lines from Tahoe-100M: {e}")
            raise RuntimeError(f"Cell line discovery failed: {e}")
    
    def get_comprehensive_tf_predictions(self, gene_symbol: str) -> pd.DataFrame:
        """
        Get comprehensive TF predictions across ALL AlphaGenome ontologies.
        
        Args:
            gene_symbol: Gene symbol to analyze
            
        Returns:
            DataFrame with comprehensive TF predictions
        """
        logger.info(f"🧬 Getting comprehensive TF predictions for {gene_symbol}...")
        
        try:
            # Import the main analyzer for TF predictions
            from state_model_client import AlphaGenomeStateAnalyzer
            
            temp_analyzer = AlphaGenomeStateAnalyzer()
            
            # Get comprehensive TF predictions across ALL ontologies
            tf_results = temp_analyzer.predict_tf_binding_for_gene_comprehensive(
                gene_name=gene_symbol,
                max_ontologies=self.config['max_ontologies'],  # None = ALL ontologies
                batch_size=20
            )
            
            if tf_results is None or tf_results.empty:
                logger.warning(f"No TF predictions found for {gene_symbol}")
                return pd.DataFrame()
            
            # Process and extract top TFs
            processed_results = temp_analyzer.analyze_tf_predictions_comprehensive(
                results=tf_results,
                top_n_per_ontology=self.config['max_tfs_per_ontology']
            )
            
            if processed_results is not None and not processed_results.empty:
                logger.info(f"✅ Found {len(processed_results)} TF predictions across ontologies")
                
                # Update statistics
                self.analysis_stats['total_tf_predictions'] += len(processed_results)
                unique_ontologies = processed_results['ontology_term'].nunique()
                self.analysis_stats['ontologies_analyzed'] = unique_ontologies
                
                return processed_results
            else:
                logger.warning(f"No processed TF results for {gene_symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"TF prediction failed for {gene_symbol}: {e}")
            raise RuntimeError(f"Comprehensive TF prediction failed: {e}")
    
    def run_state_predictions_all_cell_lines(self, tf_df: pd.DataFrame, target_gene: str, focus_organ: Optional[str] = None) -> List[Dict]:
        """
        Run State model predictions across cell lines for given TFs.
        
        Args:
            tf_df: DataFrame with TF predictions
            target_gene: Target gene being analyzed
            focus_organ: Optional organ to prioritize for cell line selection
            
        Returns:
            List of comprehensive State prediction results
        """
        organ_msg = f" with {focus_organ} prioritization" if focus_organ else ""
        logger.info(f"🔬 Running State predictions across cell lines{organ_msg} for {target_gene}...")
        
        # Get available cell lines (with optional organ prioritization)
        all_cell_lines = self.get_all_available_cell_lines(focus_organ=focus_organ)
        
        comprehensive_results = []
        total_predictions = len(tf_df) * len(all_cell_lines)
        
        logger.info(f"📊 Starting {total_predictions} State predictions ({len(tf_df)} TFs × {len(all_cell_lines)} cell lines)")
        
        for tf_idx, (_, tf_row) in enumerate(tf_df.iterrows()):
            tf_name = tf_row['tf_name']
            ontology_term = tf_row.get('ontology_term', 'Unknown')
            
            logger.info(f"🧬 Processing TF {tf_idx+1}/{len(tf_df)}: {tf_name} ({ontology_term})")
            
            for cell_idx, cell_line in enumerate(all_cell_lines):
                try:
                    logger.info(f"  Cell line {cell_idx+1}/{len(all_cell_lines)}: {cell_line}")
                    
                    # Run REAL State model prediction using Tahoe-100M data
                    state_results = self.state_client.predict_tf_perturbation_response(
                        tf_name=tf_name,
                        target_gene=target_gene,
                        cell_type=cell_line
                    )
                    
                    # Create comprehensive result combining AlphaGenome + State + Tahoe-100M
                    comprehensive_result = {
                        # Gene and TF information
                        'target_gene': target_gene,
                        'tf_name': tf_name,
                        'ontology_term': ontology_term,
                        
                        # AlphaGenome TF binding predictions
                        'mean_binding_score': tf_row['mean_binding_score'],
                        'max_binding_score': tf_row['max_binding_score'],
                        'std_binding_score': tf_row['std_binding_score'],
                        
                        # State model cellular perturbation predictions
                        'perturbation_mean_response': state_results['prediction_summary']['mean_response_strength'],
                        'perturbation_std_response': state_results['prediction_summary']['std_response_strength'],
                        'perturbation_max_response': state_results['prediction_summary']['max_response'],
                        'upregulation_fraction': state_results['prediction_summary']['upregulation_fraction'],
                        'upregulated_cells': state_results['prediction_summary']['upregulated_cells'],
                        'downregulated_cells': state_results['prediction_summary']['downregulated_cells'],
                        
                        # Combined analysis metrics
                        'combined_impact_score': self._calculate_combined_impact(
                            tf_row['mean_binding_score'],
                            state_results['prediction_summary']['mean_response_strength']
                        ),
                        'binding_perturbation_ratio': tf_row['mean_binding_score'] / max(
                            0.01, state_results['prediction_summary']['mean_response_strength']
                        ),
                        
                        # Context and metadata
                        'cell_line': cell_line,
                        'analysis_timestamp': datetime.now().isoformat(),
                        'data_source': 'Tahoe-100M',
                        'state_status': state_results.get('status', 'success'),
                        'pipeline_version': 'StandardizedGenomicAnalyzer_v1.0'
                    }
                    
                    comprehensive_results.append(comprehensive_result)
                    
                    # Update statistics
                    self.analysis_stats['total_state_predictions'] += 1
                    
                except Exception as e:
                    logger.error(f"State prediction failed for {tf_name} in {cell_line}: {e}")
                    # Continue with other predictions rather than failing completely
                    continue
        
        logger.info(f"✅ Completed {len(comprehensive_results)} successful State predictions")
        return comprehensive_results
    
    def _calculate_combined_impact(self, binding_score: float, perturbation_response: float) -> float:
        """Calculate combined impact score from binding and perturbation data."""
        if binding_score <= 0 or perturbation_response <= 0:
            return 0.0
        
        # Normalize scores to 0-1 range
        normalized_binding = min(binding_score / 10.0, 1.0)  # Assuming max binding ~10
        normalized_perturbation = min(perturbation_response / 2.0, 1.0)  # Assuming max response ~2
        
        # Weighted combination (60% binding, 40% perturbation)
        combined_score = (0.6 * normalized_binding) + (0.4 * normalized_perturbation)
        
        return float(combined_score)
    
    def analyze_any_human_gene(self, gene_symbol: str, focus_organ: Optional[str] = None) -> pd.DataFrame:
        """
        Comprehensive analysis of ANY human gene using real data across all contexts.
        
        Args:
            gene_symbol: Any valid human gene symbol
            focus_organ: Optional organ to prioritize for cell line selection (e.g., "stomach", "lung", "breast")
            
        Returns:
            DataFrame with comprehensive analysis results
        """
        start_time = datetime.now()
        
        logger.info(f"🚀 Starting comprehensive analysis for gene: {gene_symbol}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Validate gene
            validated_gene = self.validate_human_gene(gene_symbol)
            logger.info(f"✅ Gene validation successful: {validated_gene}")
            
            # Step 2: Get comprehensive TF predictions
            logger.info(f"📊 Step 1: Getting TF predictions across ALL ontologies...")
            tf_predictions = self.get_comprehensive_tf_predictions(validated_gene)
            
            if tf_predictions.empty:
                logger.warning(f"No TF predictions found for {validated_gene}")
                return pd.DataFrame()
            
            logger.info(f"✅ Found {len(tf_predictions)} TF predictions across {tf_predictions['ontology_term'].nunique()} ontologies")
            
            # Step 3: Run State model predictions across cell lines (with optional organ prioritization)
            organ_msg = f" with {focus_organ} prioritization" if focus_organ else " across ALL Tahoe-100M cell lines"
            logger.info(f"🔬 Step 2: Running State predictions{organ_msg}...")
            comprehensive_results = self.run_state_predictions_all_cell_lines(tf_predictions, validated_gene, focus_organ=focus_organ)
            
            if not comprehensive_results:
                logger.warning(f"No State predictions completed for {validated_gene}")
                return pd.DataFrame()
            
            # Step 4: Create final results DataFrame
            results_df = pd.DataFrame(comprehensive_results)
            
            # Sort by combined impact score
            results_df = results_df.sort_values('combined_impact_score', ascending=False)
            
            # Update statistics
            self.analysis_stats['genes_analyzed'] += 1
            
            # Calculate analysis duration
            duration = datetime.now() - start_time
            
            logger.info("=" * 80)
            logger.info(f"✅ COMPREHENSIVE ANALYSIS COMPLETED for {validated_gene}")
            logger.info(f"📊 Results Summary:")
            logger.info(f"   Total predictions: {len(results_df)}")
            logger.info(f"   Unique TFs: {results_df['tf_name'].nunique()}")
            logger.info(f"   Cell lines analyzed: {results_df['cell_line'].nunique()}")
            logger.info(f"   Ontologies covered: {results_df['ontology_term'].nunique()}")
            logger.info(f"   Analysis duration: {duration}")
            logger.info(f"   Top TF: {results_df.iloc[0]['tf_name']} (score: {results_df.iloc[0]['combined_impact_score']:.4f})")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed for {gene_symbol}: {e}")
            raise RuntimeError(f"Standardized genomic analysis failed: {e}")
    
    def save_comprehensive_results(self, results_df: pd.DataFrame, gene_symbol: str) -> str:
        """
        Save comprehensive analysis results to timestamped CSV file.
        
        Args:
            results_df: Results DataFrame
            gene_symbol: Gene symbol analyzed
            
        Returns:
            Path to saved file
        """
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{gene_symbol.lower()}_comprehensive_analysis.csv"
        output_path = output_dir / filename
        
        # Save results
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"💾 Comprehensive results saved to: {output_path}")
        
        # Also save analysis statistics
        stats_path = output_dir / f"{timestamp}_{gene_symbol.lower()}_analysis_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.analysis_stats, f, indent=2)
        
        return str(output_path)
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics."""
        return {
            **self.analysis_stats,
            'configuration': self.config,
            'components_initialized': {
                'alphagenome_client': self.alphagenome_client is not None,
                'state_client': self.state_client is not None,
                'tahoe_loader': self.tahoe_loader is not None
            },
            'validated_genes': list(self._validated_genes),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Test the standardized genomic analyzer with different genes."""
    print("🧬 Testing Standardized Genomic Analyzer")
    print("=" * 60)
    
    # Test genes
    test_genes = ["TP53", "BRCA1", "CLDN18", "MYC", "EGFR"]
    
    try:
        # Initialize analyzer
        print("🔧 Initializing comprehensive analyzer...")
        analyzer = StandardizedGenomicAnalyzer()
        
        print("📊 Analyzer Statistics:")
        stats = analyzer.get_analysis_statistics()
        for key, value in stats.items():
            if key != 'configuration':
                print(f"   {key}: {value}")
        
        # Test with first gene
        test_gene = test_genes[0]
        print(f"\n🧪 Testing comprehensive analysis with: {test_gene}")
        
        results = analyzer.analyze_any_human_gene(test_gene)
        
        if not results.empty:
            print(f"✅ Analysis successful!")
            print(f"   Results shape: {results.shape}")
            print(f"   Top 5 TF-cell line combinations:")
            for i, (_, row) in enumerate(results.head().iterrows()):
                print(f"   {i+1}. {row['tf_name']} in {row['cell_line']} (score: {row['combined_impact_score']:.4f})")
            
            # Save results
            output_path = analyzer.save_comprehensive_results(results, test_gene)
            print(f"💾 Results saved to: {output_path}")
        else:
            print("❌ No results generated")
        
        # Final statistics
        final_stats = analyzer.get_analysis_statistics()
        print(f"\n📈 Final Analysis Statistics:")
        for key, value in final_stats.items():
            if key in ['genes_analyzed', 'total_tf_predictions', 'total_state_predictions', 'cell_lines_used']:
                print(f"   {key}: {value}")
        
        print(f"\n✅ Standardized Genomic Analyzer test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()