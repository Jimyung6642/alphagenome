#!/usr/bin/env python3
"""
TF Prediction Test Script for AlphaGenome

This script demonstrates how to use AlphaGenome to predict transcription factor (TF) 
binding patterns for a target gene. It uses the CHIP_TF output type to analyze 
TF binding sites in genomic regions.

Example usage:
    python TF_prediction_test.py --gene TP53
    python TF_prediction_test.py --gene BRCA1 --region promoter
    python TF_prediction_test.py  # Interactive mode
"""

import sys
import os
import argparse
import urllib.request
import json
import ssl
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# Add src directory to Python path for importing alphagenome modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from alphagenome import colab_utils
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_output_path(gene_name: str = None, region_type: str = None, chromosome: str = None, 
                      start: int = None, end: int = None, file_type: str = "png") -> str:
    """
    Create an output file path with timestamp and ensure output directory exists.
    
    Args:
        gene_name: Gene symbol (if applicable)
        region_type: Region type (if applicable)
        chromosome: Chromosome (for coordinate-based analysis)
        start: Start position (for coordinate-based analysis)
        end: End position (for coordinate-based analysis)
        file_type: File extension (png, csv, etc.)
        
    Returns:
        Full path to output file
    """
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate descriptive filename based on analysis type
    if gene_name and region_type:
        if file_type == "csv":
            filename = f"{timestamp}_{gene_name.lower()}_{region_type}_tf_results.csv"
        else:
            filename = f"{timestamp}_{gene_name.lower()}_{region_type}_tf_binding.{file_type}"
    elif chromosome and start and end:
        if file_type == "csv":
            filename = f"{timestamp}_{chromosome}_{start}_{end}_tf_results.csv"
        else:
            filename = f"{timestamp}_{chromosome}_{start}_{end}_tf_binding.{file_type}"
    else:
        if file_type == "csv":
            filename = f"{timestamp}_tf_results.csv"
        else:
            filename = f"{timestamp}_tf_binding.{file_type}"
    
    return str(output_dir / filename)


class GeneLookup:
    """Class for looking up gene coordinates from public databases."""
    
    def __init__(self):
        """Initialize the gene lookup client."""
        self.ensembl_base_url = "https://rest.ensembl.org"
        # Create SSL context that doesn't verify certificates (for development only)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
    def get_gene_info(self, gene_name: str, species: str = "human") -> Optional[Dict]:
        """
        Get gene information from Ensembl REST API.
        
        Args:
            gene_name: Gene symbol (e.g., 'TP53', 'BRCA1')
            species: Species name (default: 'human')
            
        Returns:
            Dictionary with gene information or None if not found
        """
        try:
            # First, search for the gene to get Ensembl ID
            search_url = f"{self.ensembl_base_url}/xrefs/symbol/{species}/{gene_name}"
            headers = {"Content-Type": "application/json"}
            
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                search_results = json.loads(response.read().decode())
            
            if not search_results:
                print(f"Gene '{gene_name}' not found")
                return None
            
            # Get the first gene result
            gene_id = None
            for result in search_results:
                if result.get('type') == 'gene':
                    gene_id = result.get('id')
                    break
            
            if not gene_id:
                print(f"No gene ID found for '{gene_name}'")
                return None
            
            # Get detailed gene information
            lookup_url = f"{self.ensembl_base_url}/lookup/id/{gene_id}"
            req = urllib.request.Request(lookup_url, headers=headers)
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                gene_info = json.loads(response.read().decode())
            
            return gene_info
            
        except Exception as e:
            print(f"Error looking up gene '{gene_name}': {e}")
            return None
    
    def get_gene_coordinates(
        self, 
        gene_name: str, 
        region_type: str = "gene",
        upstream: int = 0,
        downstream: int = 0
    ) -> Optional[Tuple[str, int, int]]:
        """
        Get genomic coordinates for a gene.
        
        Args:
            gene_name: Gene symbol
            region_type: Type of region ('gene', 'promoter', 'tss')
            upstream: Base pairs upstream to include
            downstream: Base pairs downstream to include
            
        Returns:
            Tuple of (chromosome, start, end) or None if not found
        """
        gene_info = self.get_gene_info(gene_name)
        if not gene_info:
            return None
        
        chromosome = f"chr{gene_info['seq_region_name']}"
        gene_start = gene_info['start']
        gene_end = gene_info['end']
        strand = gene_info['strand']
        
        print(f"Found {gene_name}: {chromosome}:{gene_start}-{gene_end} (strand: {strand})")
        
        if region_type == "gene":
            # Return full gene + flanking regions, adjust to supported length
            gene_length = gene_end - gene_start
            total_length = gene_length + upstream + downstream
            
            # Find the closest supported length
            supported_lengths = [2048, 16384, 131072, 524288, 1048576]
            best_length = min(supported_lengths, key=lambda x: abs(x - total_length) if x >= total_length else float('inf'))
            
            # If no supported length is large enough, use the largest
            if best_length < total_length:
                best_length = max(supported_lengths)
            
            # Center the region around the gene
            gene_center = (gene_start + gene_end) // 2
            start = max(1, gene_center - best_length // 2)
            end = start + best_length
            
        elif region_type == "promoter":
            # Return promoter region - use 2KB length (supported by AlphaGenome)
            region_length = 2048  # 2KB supported length
            
            if strand == 1:  # Forward strand
                # Center around TSS with more upstream coverage
                tss = gene_start
                start = max(1, tss - 1536)  # 1.5KB upstream
                end = tss + 512  # 0.5KB downstream
            else:  # Reverse strand
                tss = gene_end
                start = max(1, tss - 512)  # 0.5KB upstream 
                end = tss + 1536  # 1.5KB downstream
            
            # Ensure exact length
            if end - start != region_length:
                end = start + region_length
                
        elif region_type == "tss":
            # Return region around transcription start site - use 2KB length
            region_length = 2048  # 2KB supported length
            
            if strand == 1:  # Forward strand
                tss = gene_start
            else:  # Reverse strand
                tss = gene_end
                
            # Center around TSS
            start = max(1, tss - region_length // 2)
            end = start + region_length
            
        else:
            raise ValueError(f"Unknown region_type: {region_type}")
        
        return chromosome, start, end


class TFPredictor:
    """Class for predicting transcription factor binding using AlphaGenome."""
    
    def __init__(self):
        """Initialize the TF predictor with AlphaGenome client."""
        try:
            # Get API key using the updated colab_utils (loads from config.env)
            api_key = colab_utils.get_api_key()
            self.client = dna_client.create(api_key)
            self.gene_lookup = GeneLookup()
            self._available_ontologies = None
            print("✓ Successfully initialized AlphaGenome client")
        except Exception as e:
            print(f"✗ Failed to initialize client: {e}")
            sys.exit(1)
    
    def get_all_available_ontologies(self) -> List[str]:
        """
        Get all available ontology terms from AlphaGenome metadata.
        
        Returns:
            List of all available ontology CURIEs
        """
        if self._available_ontologies is not None:
            return self._available_ontologies
        
        try:
            print("🔍 Querying AlphaGenome for all available ontology terms...")
            
            # Get metadata for CHIP_TF to see all available ontology terms
            metadata = self.client.output_metadata()
            
            if metadata.chip_tf is not None:
                # Extract unique ontology terms from the metadata
                ontology_terms = set()
                
                if 'ontology_curie' in metadata.chip_tf.columns:
                    for curie in metadata.chip_tf['ontology_curie'].dropna().unique():
                        if curie and isinstance(curie, str):
                            ontology_terms.add(curie)
                
                self._available_ontologies = sorted(list(ontology_terms))
                print(f"✓ Found {len(self._available_ontologies)} unique ontology terms")
                
                # Display some examples
                print("📋 Sample ontology terms:")
                for i, term in enumerate(self._available_ontologies[:10]):
                    print(f"   {i+1}. {term}")
                if len(self._available_ontologies) > 10:
                    print(f"   ... and {len(self._available_ontologies) - 10} more")
                
                return self._available_ontologies
            else:
                print("⚠️  No CHIP_TF metadata available, using default ontology terms")
                # Fallback to some known ontology terms
                self._available_ontologies = ['EFO:0001187']  # HepG2
                return self._available_ontologies
                
        except Exception as e:
            print(f"⚠️  Error getting ontology terms: {e}")
            print("   Using default ontology terms")
            self._available_ontologies = ['EFO:0001187']  # HepG2 fallback
            return self._available_ontologies
    
    def predict_tf_binding_comprehensive(
        self,
        chromosome: str,
        start: int,
        end: int,
        max_ontologies: Optional[int] = None,
        batch_size: int = 50
    ) -> Dict[str, dna_client.Output]:
        """
        Run comprehensive TF binding prediction across all available ontologies.
        
        Args:
            chromosome: Chromosome name (e.g., 'chr1', 'chr22')
            start: Start position (0-based)
            end: End position (0-based)
            max_ontologies: Maximum number of ontologies to analyze (None = all)
            batch_size: Number of ontologies to process in each batch
            
        Returns:
            Dictionary mapping ontology terms to prediction outputs
        """
        # Get all available ontologies
        all_ontologies = self.get_all_available_ontologies()
        
        if max_ontologies is not None:
            ontologies_to_use = all_ontologies[:max_ontologies]
            print(f"🎯 Using first {len(ontologies_to_use)} ontologies (limited by max_ontologies)")
        else:
            ontologies_to_use = all_ontologies
            print(f"🌍 Running comprehensive analysis across all {len(ontologies_to_use)} ontologies")
        
        interval = genome.Interval(
            chromosome=chromosome,
            start=start,
            end=end
        )
        
        print(f"📍 Analyzing region: {chromosome}:{start}-{end}")
        print(f"🔬 Processing in batches of {batch_size} ontologies...")
        
        results = {}
        
        # Process ontologies in batches to avoid overwhelming the API
        for i in range(0, len(ontologies_to_use), batch_size):
            batch_ontologies = ontologies_to_use[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(ontologies_to_use) + batch_size - 1) // batch_size
            
            print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch_ontologies)} ontologies)")
            
            try:
                # Run prediction for this batch
                output = self.client.predict_interval(
                    interval=interval,
                    requested_outputs=[dna_client.OutputType.CHIP_TF],
                    ontology_terms=batch_ontologies,
                )
                
                # Store results for each ontology in the batch
                for ontology in batch_ontologies:
                    results[ontology] = output
                
                print(f"✅ Batch {batch_num} completed successfully")
                
            except Exception as e:
                print(f"❌ Error in batch {batch_num}: {e}")
                # Continue with remaining batches
                continue
        
        print(f"\n🎉 Comprehensive analysis complete!")
        print(f"📊 Successfully analyzed {len(results)} ontology contexts")
        
        return results
    
    def predict_tf_binding_for_gene_comprehensive(
        self,
        gene_name: str,
        region_type: str = "promoter",
        upstream: int = 0,
        downstream: int = 0,
        max_ontologies: Optional[int] = None,
        batch_size: int = 50
    ) -> Optional[Dict[str, dna_client.Output]]:
        """
        Predict comprehensive TF binding for a gene across all available ontologies.
        
        Args:
            gene_name: Gene symbol (e.g., 'TP53', 'BRCA1')
            region_type: Type of region ('gene', 'promoter', 'tss')
            upstream: Base pairs upstream to include
            downstream: Base pairs downstream to include
            max_ontologies: Maximum number of ontologies to analyze (None = all)
            batch_size: Number of ontologies to process in each batch
            
        Returns:
            Dictionary mapping ontology terms to prediction outputs or None if failed
        """
        print(f"\n🧬 Looking up coordinates for gene: {gene_name}")
        
        # Get gene coordinates
        coordinates = self.gene_lookup.get_gene_coordinates(
            gene_name, region_type, upstream, downstream
        )
        
        if not coordinates:
            print(f"✗ Could not find coordinates for gene: {gene_name}")
            return None
        
        chromosome, start, end = coordinates
        
        print(f"🚀 Starting comprehensive TF binding analysis for {gene_name}")
        
        # Run comprehensive prediction
        return self.predict_tf_binding_comprehensive(
            chromosome=chromosome,
            start=start,
            end=end,
            max_ontologies=max_ontologies,
            batch_size=batch_size
        )
    
    def predict_tf_binding_for_gene(
        self,
        gene_name: str,
        region_type: str = "promoter",
        upstream: int = 0,
        downstream: int = 0,
        ontology_terms: Optional[List[str]] = None,
    ) -> Optional[dna_client.Output]:
        """
        Predict TF binding for a gene by name.
        
        Args:
            gene_name: Gene symbol (e.g., 'TP53', 'BRCA1')
            region_type: Type of region ('gene', 'promoter', 'tss')
            upstream: Base pairs upstream to include
            downstream: Base pairs downstream to include
            ontology_terms: List of cell type ontology terms
            
        Returns:
            AlphaGenome output containing TF binding predictions or None if failed
        """
        print(f"\n🧬 Looking up coordinates for gene: {gene_name}")
        
        # Get gene coordinates
        coordinates = self.gene_lookup.get_gene_coordinates(
            gene_name, region_type, upstream, downstream
        )
        
        if not coordinates:
            print(f"✗ Could not find coordinates for gene: {gene_name}")
            return None
        
        chromosome, start, end = coordinates
        
        print(f"📍 Analyzing region: {chromosome}:{start}-{end} ({region_type})")
        
        # Predict TF binding
        return self.predict_tf_binding(
            chromosome=chromosome,
            start=start,
            end=end,
            ontology_terms=ontology_terms
        )
    
    def predict_tf_binding(
        self, 
        chromosome: str, 
        start: int, 
        end: int,
        ontology_terms: Optional[List[str]] = None,
        target_tfs: Optional[List[str]] = None
    ) -> dna_client.Output:
        """
        Predict transcription factor binding for a genomic interval.
        
        Args:
            chromosome: Chromosome name (e.g., 'chr1', 'chr22')
            start: Start position (0-based)
            end: End position (0-based)
            ontology_terms: List of cell type ontology terms (e.g., ['EFO:0001187'])
            target_tfs: Optional list of specific TFs to analyze
            
        Returns:
            AlphaGenome output containing TF binding predictions
        """
        if ontology_terms is None:
            # Default to HepG2 cells which have good TF coverage
            ontology_terms = ['EFO:0001187']  # HepG2 cell line
        
        interval = genome.Interval(
            chromosome=chromosome,
            start=start,
            end=end
        )
        
        print(f"Predicting TF binding for {chromosome}:{start}-{end}")
        print(f"Using cell types: {ontology_terms}")
        
        try:
            output = self.client.predict_interval(
                interval=interval,
                requested_outputs=[dna_client.OutputType.CHIP_TF],
                ontology_terms=ontology_terms,
            )
            
            print("✓ Successfully obtained TF binding predictions")
            return output
            
        except Exception as e:
            print(f"✗ Failed to get predictions: {e}")
            raise
    
    def analyze_tf_predictions_comprehensive(
        self, 
        results: Dict[str, dna_client.Output], 
        top_n_per_ontology: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        Analyze comprehensive TF predictions across multiple ontologies.
        
        Args:
            results: Dictionary mapping ontology terms to prediction outputs
            top_n_per_ontology: Number of top TFs to include per ontology
            
        Returns:
            Comprehensive DataFrame with all TF binding results across ontologies
        """
        if not results:
            print("No comprehensive TF binding predictions available")
            return None
        
        all_results = []
        
        print(f"\n🔬 Comprehensive TF Binding Analysis Across {len(results)} Ontologies")
        print("=" * 80)
        
        for ontology, output in results.items():
            if output.chip_tf is None:
                print(f"⚠️  No TF data for ontology: {ontology}")
                continue
                
            tf_data = output.chip_tf
            
            if not hasattr(tf_data, 'names') or tf_data.names is None:
                print(f"⚠️  No TF names available for ontology: {ontology}")
                continue
            
            # Calculate binding statistics for each TF in this ontology
            mean_scores = np.mean(tf_data.values, axis=0)
            max_scores = np.max(tf_data.values, axis=0)
            min_scores = np.min(tf_data.values, axis=0)
            std_scores = np.std(tf_data.values, axis=0)
            
            # Create DataFrame for this ontology
            ontology_df = pd.DataFrame({
                'ontology_term': ontology,
                'tf_name': tf_data.names,
                'mean_binding_score': mean_scores,
                'max_binding_score': max_scores,
                'min_binding_score': min_scores,
                'std_binding_score': std_scores,
                'interval': str(tf_data.interval),
                'resolution': tf_data.resolution,
                'num_positions': tf_data.values.shape[0],
                'num_tfs_in_ontology': len(tf_data.names)
            })
            
            # Add metadata if available
            if hasattr(tf_data, 'metadata') and tf_data.metadata is not None:
                for col in tf_data.metadata.columns:
                    if col not in ['name']:
                        ontology_df[f'metadata_{col}'] = tf_data.metadata[col].values
            
            all_results.append(ontology_df)
            
            # Display summary for this ontology
            top_tfs = ontology_df.nlargest(min(5, len(ontology_df)), 'mean_binding_score')
            print(f"\n📊 {ontology}: {len(tf_data.names)} TFs")
            print(f"   Top TF: {top_tfs.iloc[0]['tf_name']} (score: {top_tfs.iloc[0]['mean_binding_score']:.4f})")
        
        if not all_results:
            print("❌ No valid TF data found in any ontology")
            return None
        
        # Combine all results
        comprehensive_df = pd.concat(all_results, ignore_index=True)
        
        # Sort by mean binding score globally
        comprehensive_df = comprehensive_df.sort_values('mean_binding_score', ascending=False).reset_index(drop=True)
        
        print(f"\n🌟 Global Top {min(10, len(comprehensive_df))} TFs Across All Ontologies:")
        for i in range(min(10, len(comprehensive_df))):
            row = comprehensive_df.iloc[i]
            print(f"{i+1:2d}. {row['tf_name']} ({row['ontology_term']}): {row['mean_binding_score']:.4f}")
        
        print(f"\n📈 Summary Statistics:")
        print(f"   Total TF-ontology combinations: {len(comprehensive_df)}")
        print(f"   Unique TFs: {comprehensive_df['tf_name'].nunique()}")
        print(f"   Unique ontologies: {comprehensive_df['ontology_term'].nunique()}")
        print(f"   Mean binding score (all): {comprehensive_df['mean_binding_score'].mean():.4f}")
        print(f"   Max binding score (all): {comprehensive_df['mean_binding_score'].max():.4f}")
        
        return comprehensive_df
    
    def analyze_tf_predictions(self, output: dna_client.Output, top_n: int = 10) -> Optional[pd.DataFrame]:
        """
        Analyze and display information about TF binding predictions.
        
        Args:
            output: AlphaGenome output containing TF predictions
            top_n: Number of top TFs to display
            
        Returns:
            DataFrame with TF binding analysis results or None if no data
        """
        if output.chip_tf is None:
            print("No TF binding predictions available")
            return None
        
        tf_data = output.chip_tf
        print(f"\nTF Binding Analysis:")
        print(f"Interval: {tf_data.interval}")
        print(f"Data shape: {tf_data.values.shape}")
        
        # Get track names (TF names)
        if hasattr(tf_data, 'names') and tf_data.names is not None:
            track_names = tf_data.names
            print(f"Number of TFs: {len(track_names)}")
            
            # Calculate binding statistics for each TF
            mean_scores = np.mean(tf_data.values, axis=0)
            max_scores = np.max(tf_data.values, axis=0)
            min_scores = np.min(tf_data.values, axis=0)
            std_scores = np.std(tf_data.values, axis=0)
            
            # Create comprehensive results DataFrame
            results_df = pd.DataFrame({
                'tf_name': track_names,
                'mean_binding_score': mean_scores,
                'max_binding_score': max_scores,
                'min_binding_score': min_scores,
                'std_binding_score': std_scores,
                'interval': str(tf_data.interval),
                'resolution': tf_data.resolution,
                'num_positions': tf_data.values.shape[0]
            })
            
            # Add metadata if available
            if hasattr(tf_data, 'metadata') and tf_data.metadata is not None:
                # Add any additional metadata columns
                for col in tf_data.metadata.columns:
                    if col not in ['name']:  # Skip 'name' as we already have 'tf_name'
                        results_df[f'metadata_{col}'] = tf_data.metadata[col].values
            
            # Sort by mean binding score (descending)
            results_df = results_df.sort_values('mean_binding_score', ascending=False).reset_index(drop=True)
            
            # Display top TFs
            print(f"\nTop {top_n} TFs by mean binding score:")
            for i in range(min(top_n, len(results_df))):
                tf_name = results_df.iloc[i]['tf_name']
                score = results_df.iloc[i]['mean_binding_score']
                print(f"{i+1:2d}. {tf_name}: {score:.4f}")
            
            return results_df
        
        else:
            print("Track names not available")
            return None
    
    def save_comprehensive_results_to_csv(
        self,
        comprehensive_df: pd.DataFrame,
        gene_name: Optional[str] = None,
        region_type: Optional[str] = None,
        chromosome: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> str:
        """
        Save comprehensive TF binding results to CSV file.
        
        Args:
            comprehensive_df: DataFrame containing comprehensive TF binding results
            gene_name: Gene name (for filename generation)
            region_type: Region type (for filename generation)
            chromosome: Chromosome (for coordinate-based filename)
            start: Start position (for coordinate-based filename)
            end: End position (for coordinate-based filename)
            
        Returns:
            Path to saved CSV file
        """
        try:
            # Generate output path for comprehensive CSV
            base_filename = "comprehensive_tf_results"
            if gene_name and region_type:
                base_filename = f"{gene_name.lower()}_{region_type}_comprehensive_tf_results"
            elif chromosome and start and end:
                base_filename = f"{chromosome}_{start}_{end}_comprehensive_tf_results"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            csv_path = output_dir / f"{timestamp}_{base_filename}.csv"
            
            # Save to CSV with comprehensive information
            comprehensive_df.to_csv(csv_path, index=False)
            print(f"📊 Comprehensive results saved to: {csv_path}")
            
            # Also create a summary CSV with top TFs per ontology
            summary_df = (
                comprehensive_df.groupby('ontology_term')
                .apply(lambda x: x.nlargest(5, 'mean_binding_score'))
                .reset_index(drop=True)
            )
            
            summary_path = output_dir / f"{timestamp}_{base_filename}_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"📋 Summary (top 5 per ontology) saved to: {summary_path}")
            
            return str(csv_path)
            
        except Exception as e:
            print(f"❌ Failed to save comprehensive CSV: {e}")
            return ""
    
    def save_results_to_csv(
        self,
        results_df: pd.DataFrame,
        gene_name: Optional[str] = None,
        region_type: Optional[str] = None,
        chromosome: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> str:
        """
        Save TF binding results to CSV file.
        
        Args:
            results_df: DataFrame containing TF binding results
            gene_name: Gene name (for filename generation)
            region_type: Region type (for filename generation)
            chromosome: Chromosome (for coordinate-based filename)
            start: Start position (for coordinate-based filename)
            end: End position (for coordinate-based filename)
            
        Returns:
            Path to saved CSV file
        """
        try:
            # Generate output path for CSV
            csv_path = create_output_path(
                gene_name=gene_name,
                region_type=region_type,
                chromosome=chromosome,
                start=start,
                end=end,
                file_type="csv"
            )
            
            # Save to CSV
            results_df.to_csv(csv_path, index=False)
            print(f"📊 Results saved to: {csv_path}")
            
            return csv_path
            
        except Exception as e:
            print(f"❌ Failed to save CSV: {e}")
            return ""
    
    def plot_tf_binding(
        self, 
        output: dna_client.Output, 
        selected_tfs: Optional[List[str]] = None,
        gene_name: Optional[str] = None,
        region_type: Optional[str] = None,
        chromosome: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> str:
        """
        Plot TF binding predictions and save to output folder with timestamp.
        
        Args:
            output: AlphaGenome output containing TF predictions
            selected_tfs: List of specific TFs to plot (if None, plots top 5)
            gene_name: Gene name (for filename generation)
            region_type: Region type (for filename generation)
            chromosome: Chromosome (for coordinate-based filename)
            start: Start position (for coordinate-based filename)
            end: End position (for coordinate-based filename)
            
        Returns:
            Path to saved plot file
        """
        if output.chip_tf is None:
            print("No TF binding predictions to plot")
            return ""
        
        tf_data = output.chip_tf
        
        if selected_tfs is None and hasattr(tf_data, 'names'):
            # Select top 5 TFs by mean binding score
            mean_scores = np.mean(tf_data.values, axis=0)
            top_indices = np.argsort(mean_scores)[-5:][::-1]
            selected_tfs = [tf_data.names[i] for i in top_indices]
        
        print(f"\n📊 Plotting TF binding patterns...")
        
        try:
            # Generate output path with timestamp
            save_path = create_output_path(
                gene_name=gene_name,
                region_type=region_type,
                chromosome=chromosome,
                start=start,
                end=end
            )
            
            # Create plot components - limit to top 10 TFs to avoid plotting too many tracks
            title = "Transcription Factor Binding Predictions (Top 10 TFs)"
            if gene_name and region_type:
                title += f" - {gene_name} ({region_type})"
            elif chromosome and start and end:
                title += f" - {chromosome}:{start}-{end}"
            
            # Filter to top 10 TFs if there are too many
            if tf_data.num_tracks > 10:
                mean_scores = np.mean(tf_data.values, axis=0)
                top_indices = np.argsort(mean_scores)[-10:][::-1]
                filtered_tf_data = tf_data.select_tracks_by_index(top_indices)
                print(f"Filtered to top 10 TFs for visualization (out of {tf_data.num_tracks} total)")
            else:
                filtered_tf_data = tf_data
            
            components = [
                plot_components.Tracks(tdata=filtered_tf_data)
            ]
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            plot_components.plot(
                components=components,
                interval=filtered_tf_data.interval
            )
            plt.suptitle(title, fontsize=14)
            
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
            
            plt.show()
            
            return save_path
            
        except Exception as e:
            print(f"❌ Failed to create plot: {e}")
            return ""


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict transcription factor binding for genes using AlphaGenome",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python TF_prediction_test.py --gene TP53
  python TF_prediction_test.py --gene BRCA1 --region promoter
  python TF_prediction_test.py --gene MYC --region tss --upstream 2000 --downstream 1000
  python TF_prediction_test.py --gene TP53 --comprehensive  # Comprehensive analysis
  python TF_prediction_test.py --gene BRCA1 --comprehensive --max-ontologies 50 --batch-size 5
  python TF_prediction_test.py  # Interactive mode
        """
    )
    
    parser.add_argument(
        '--gene', '-g',
        type=str,
        help='Gene symbol to analyze (e.g., TP53, BRCA1, MYC)'
    )
    
    parser.add_argument(
        '--region', '-r',
        type=str,
        choices=['gene', 'promoter', 'tss'],
        default='promoter',
        help='Region type to analyze (default: promoter)'
    )
    
    parser.add_argument(
        '--upstream', '-u',
        type=int,
        default=0,
        help='Base pairs upstream to include (0 = use default for region type)'
    )
    
    parser.add_argument(
        '--downstream', '-d',
        type=int,
        default=0,
        help='Base pairs downstream to include (0 = use default for region type)'
    )
    
    parser.add_argument(
        '--cell-type', '-c',
        type=str,
        default='EFO:0001187',
        help='Cell type ontology term (default: EFO:0001187 for HepG2)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--comprehensive', '-comp',
        action='store_true',
        help='Run comprehensive analysis across all available ontologies (tissues/cell types)'
    )
    
    parser.add_argument(
        '--max-ontologies', '-mo',
        type=int,
        default=None,
        help='Maximum number of ontologies to analyze in comprehensive mode (default: all)'
    )
    
    parser.add_argument(
        '--batch-size', '-bs',
        type=int,
        default=10,
        help='Batch size for comprehensive ontology processing (default: 10)'
    )
    
    return parser.parse_args()


def interactive_mode(predictor: TFPredictor):
    """Run the script in interactive mode."""
    print("\n🔬 Interactive TF Prediction Mode")
    print("=" * 50)
    
    while True:
        try:
            print("\nOptions:")
            print("1. Analyze a gene by name (single ontology)")
            print("2. Analyze specific coordinates (single ontology)")
            print("3. Comprehensive analysis - gene across ALL ontologies")
            print("4. Comprehensive analysis - coordinates across ALL ontologies")
            print("5. Exit")
            
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == '5':
                print("Goodbye! 👋")
                break
            elif choice == '3':
                # Comprehensive gene analysis
                gene_name = input("Enter gene symbol (e.g., TP53, BRCA1): ").strip().upper()
                if not gene_name:
                    print("Please enter a gene name.")
                    continue
                
                print("\nRegion types:")
                print("1. Gene (full gene + flanking)")
                print("2. Promoter (2kb upstream, 500bp downstream of TSS)")
                print("3. TSS (1kb upstream and downstream of TSS)")
                
                region_choice = input("Select region type (1-3, default=2): ").strip()
                region_map = {'1': 'gene', '2': 'promoter', '3': 'tss'}
                region_type = region_map.get(region_choice, 'promoter')
                
                # Optional limits
                max_ontologies = None
                if input("Limit number of ontologies? (y/N): ").strip().lower() == 'y':
                    try:
                        max_ontologies = int(input("Maximum ontologies to analyze: "))
                    except ValueError:
                        print("Invalid input, analyzing all ontologies")
                
                batch_size = 10
                if input("Use custom batch size? (y/N): ").strip().lower() == 'y':
                    try:
                        batch_size = int(input("Batch size (default=10): ") or "10")
                    except ValueError:
                        print("Invalid input, using default batch size")
                
                print(f"\n🚀 Starting comprehensive analysis for {gene_name} ({region_type} region)...")
                
                results = predictor.predict_tf_binding_for_gene_comprehensive(
                    gene_name=gene_name,
                    region_type=region_type,
                    max_ontologies=max_ontologies,
                    batch_size=batch_size
                )
                
                if results:
                    # Analyze comprehensive results
                    comprehensive_df = predictor.analyze_tf_predictions_comprehensive(results)
                    
                    if comprehensive_df is not None:
                        # Save comprehensive CSV results
                        csv_path = predictor.save_comprehensive_results_to_csv(
                            comprehensive_df,
                            gene_name=gene_name,
                            region_type=region_type
                        )
                        if csv_path:
                            print(f"📁 Comprehensive results saved: {Path(csv_path).name}")
                
            elif choice == '4':
                # Comprehensive coordinate analysis
                print("\nEnter genomic coordinates:")
                try:
                    chromosome = input("Chromosome (e.g., chr17): ").strip()
                    start = int(input("Start position: ").strip())
                    end = int(input("End position: ").strip())
                    
                    # Optional limits
                    max_ontologies = None
                    if input("Limit number of ontologies? (y/N): ").strip().lower() == 'y':
                        try:
                            max_ontologies = int(input("Maximum ontologies to analyze: "))
                        except ValueError:
                            print("Invalid input, analyzing all ontologies")
                    
                    batch_size = 10
                    if input("Use custom batch size? (y/N): ").strip().lower() == 'y':
                        try:
                            batch_size = int(input("Batch size (default=10): ") or "10")
                        except ValueError:
                            print("Invalid input, using default batch size")
                    
                    print(f"\n🚀 Starting comprehensive analysis for {chromosome}:{start}-{end}...")
                    
                    results = predictor.predict_tf_binding_comprehensive(
                        chromosome=chromosome,
                        start=start,
                        end=end,
                        max_ontologies=max_ontologies,
                        batch_size=batch_size
                    )
                    
                    if results:
                        # Analyze comprehensive results
                        comprehensive_df = predictor.analyze_tf_predictions_comprehensive(results)
                        
                        if comprehensive_df is not None:
                            # Save comprehensive CSV results
                            csv_path = predictor.save_comprehensive_results_to_csv(
                                comprehensive_df,
                                chromosome=chromosome,
                                start=start,
                                end=end
                            )
                            if csv_path:
                                print(f"📁 Comprehensive results saved: {Path(csv_path).name}")
                        
                except ValueError:
                    print("Please enter valid coordinates.")
                    continue
            elif choice == '1':
                # Gene-based analysis
                gene_name = input("Enter gene symbol (e.g., TP53, BRCA1): ").strip().upper()
                if not gene_name:
                    print("Please enter a gene name.")
                    continue
                
                print("\nRegion types:")
                print("1. Gene (full gene + flanking)")
                print("2. Promoter (2kb upstream, 500bp downstream of TSS)")
                print("3. TSS (1kb upstream and downstream of TSS)")
                
                region_choice = input("Select region type (1-3, default=2): ").strip()
                region_map = {'1': 'gene', '2': 'promoter', '3': 'tss'}
                region_type = region_map.get(region_choice, 'promoter')
                
                # Optional custom flanking regions
                upstream = 0
                downstream = 0
                if input("Use custom flanking regions? (y/N): ").strip().lower() == 'y':
                    try:
                        upstream = int(input("Upstream bp (default=0): ") or "0")
                        downstream = int(input("Downstream bp (default=0): ") or "0")
                    except ValueError:
                        print("Invalid input, using defaults")
                
                print(f"\n🚀 Analyzing {gene_name} ({region_type} region)...")
                
                output = predictor.predict_tf_binding_for_gene(
                    gene_name=gene_name,
                    region_type=region_type,
                    upstream=upstream,
                    downstream=downstream,
                    ontology_terms=['EFO:0001187']
                )
                
                if output:
                    # Analyze predictions and get results DataFrame
                    results_df = predictor.analyze_tf_predictions(output, top_n=15)
                    
                    if results_df is not None:
                        # Save CSV results
                        csv_path = predictor.save_results_to_csv(
                            results_df,
                            gene_name=gene_name,
                            region_type=region_type
                        )
                        if csv_path:
                            print(f"📁 CSV saved to output folder: {Path(csv_path).name}")
                    
                    if input("\nGenerate plot? (Y/n): ").strip().lower() != 'n':
                        saved_path = predictor.plot_tf_binding(
                            output, 
                            gene_name=gene_name,
                            region_type=region_type
                        )
                        if saved_path:
                            print(f"📁 Plot saved to output folder: {Path(saved_path).name}")
                
            elif choice == '2':
                # Coordinate-based analysis
                print("\nEnter genomic coordinates:")
                try:
                    chromosome = input("Chromosome (e.g., chr17): ").strip()
                    start = int(input("Start position: ").strip())
                    end = int(input("End position: ").strip())
                    
                    print(f"\n🚀 Analyzing {chromosome}:{start}-{end}...")
                    
                    output = predictor.predict_tf_binding(
                        chromosome=chromosome,
                        start=start,
                        end=end,
                        ontology_terms=['EFO:0001187']
                    )
                    
                    # Analyze predictions and get results DataFrame
                    results_df = predictor.analyze_tf_predictions(output, top_n=15)
                    
                    if results_df is not None:
                        # Save CSV results
                        csv_path = predictor.save_results_to_csv(
                            results_df,
                            chromosome=chromosome,
                            start=start,
                            end=end
                        )
                        if csv_path:
                            print(f"📁 CSV saved to output folder: {Path(csv_path).name}")
                    
                    if input("\nGenerate plot? (Y/n): ").strip().lower() != 'n':
                        saved_path = predictor.plot_tf_binding(
                            output,
                            chromosome=chromosome,
                            start=start,
                            end=end
                        )
                        if saved_path:
                            print(f"📁 Plot saved to output folder: {Path(saved_path).name}")
                        
                except ValueError:
                    print("Please enter valid coordinates.")
                    continue
            else:
                print("Invalid choice. Please select 1, 2, 3, 4, or 5.")
                
        except KeyboardInterrupt:
            print("\n\nExiting... 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run TF prediction analysis."""
    args = parse_arguments()
    
    print("🧬 AlphaGenome TF Prediction Tool")
    print("=" * 50)
    
    # Initialize predictor
    predictor = TFPredictor()
    
    if args.interactive or (not args.gene and not args.comprehensive):
        # Run in interactive mode
        interactive_mode(predictor)
    elif args.comprehensive:
        # Run comprehensive analysis
        if args.gene:
            print(f"\n🌍 Comprehensive analysis for gene: {args.gene}")
            print(f"Region type: {args.region}")
            if args.max_ontologies:
                print(f"Limited to {args.max_ontologies} ontologies")
            print(f"Batch size: {args.batch_size}")
            
            try:
                results = predictor.predict_tf_binding_for_gene_comprehensive(
                    gene_name=args.gene.upper(),
                    region_type=args.region,
                    upstream=args.upstream,
                    downstream=args.downstream,
                    max_ontologies=args.max_ontologies,
                    batch_size=args.batch_size
                )
                
                if results:
                    # Analyze comprehensive results
                    comprehensive_df = predictor.analyze_tf_predictions_comprehensive(results)
                    
                    if comprehensive_df is not None:
                        # Save comprehensive CSV results
                        csv_path = predictor.save_comprehensive_results_to_csv(
                            comprehensive_df,
                            gene_name=args.gene.upper(),
                            region_type=args.region
                        )
                        
                        print(f"\n✅ Comprehensive analysis complete!")
                        if csv_path:
                            print(f"📊 Results saved to output folder: {Path(csv_path).name}1")
                else:
                    print(f"❌ Could not complete comprehensive analysis for gene: {args.gene}")
                    
            except Exception as e:
                print(f"❌ Error in comprehensive analysis: {e}")
        else:
            print("❌ Gene name required for comprehensive analysis. Use --gene argument.")
    else:
        # Run with command line arguments (single ontology)
        print(f"\n🚀 Analyzing gene: {args.gene}")
        print(f"Region type: {args.region}")
        if args.upstream > 0 or args.downstream > 0:
            print(f"Flanking regions: {args.upstream}bp upstream, {args.downstream}bp downstream")
        
        try:
            output = predictor.predict_tf_binding_for_gene(
                gene_name=args.gene.upper(),
                region_type=args.region,
                upstream=args.upstream,
                downstream=args.downstream,
                ontology_terms=[args.cell_type]
            )
            
            if output:
                # Analyze the predictions and get results DataFrame
                results_df = predictor.analyze_tf_predictions(output, top_n=15)
                
                if results_df is not None:
                    # Save CSV results
                    csv_path = predictor.save_results_to_csv(
                        results_df,
                        gene_name=args.gene.upper(),
                        region_type=args.region
                    )
                
                # Generate plot
                saved_path = predictor.plot_tf_binding(
                    output,
                    gene_name=args.gene.upper(),
                    region_type=args.region
                )
                
                print(f"\n✅ Analysis complete!")
                if results_df is not None and csv_path:
                    print(f"📊 CSV saved to output folder: {Path(csv_path).name}")
                if saved_path:
                    print(f"📁 Plot saved to output folder: {Path(saved_path).name}")
            else:
                print(f"❌ Could not analyze gene: {args.gene}")
                
        except Exception as e:
            print(f"❌ Error in analysis: {e}")


if __name__ == "__main__":
    main()