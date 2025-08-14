#!/usr/bin/env python3
"""
Differential Expression Gene (DEG) Analyzer

Analyzes differential gene expression between baseline (DMSO) and perturbation
conditions, with focus on target gene changes and genome-wide effects.

Key Features:
- Target gene-specific expression change analysis
- Genome-wide differential expression identification
- Statistical significance testing
- Cell line and organ-specific analysis
- Integration with State model predictions

Usage:
    analyzer = DEGAnalyzer()
    results = analyzer.analyze_expression_changes(baseline_data, perturbation_data, "TP53")
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import json
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

class DEGAnalyzer:
    """
    Analyzer for differential gene expression between baseline and perturbation conditions.
    """
    
    def __init__(self, output_dir: str = "deg_output"):
        """Initialize the DEG analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analyzer statistics
        self.analyzer_stats = {
            'analyses_performed': 0,
            'genes_tested': 0,
            'significant_degs': 0,
            'analyzer_start_time': datetime.now().isoformat()
        }
        
        logger.info("📊 DEG Analyzer Initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        self._setup_analysis_directories()
    
    def _setup_analysis_directories(self):
        """Setup directories for analysis output."""
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        logger.info("📁 Analysis directories setup complete")
    
    def analyze_expression_changes(self, baseline_data: pd.DataFrame, 
                                 perturbation_data: pd.DataFrame,
                                 target_gene: str,
                                 p_value_threshold: float = 0.05,
                                 log2fc_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze differential expression between baseline and perturbation conditions.
        
        Args:
            baseline_data: DataFrame with baseline expression data
            perturbation_data: DataFrame with perturbation predictions
            target_gene: Target gene symbol
            p_value_threshold: P-value threshold for significance
            log2fc_threshold: Log2 fold-change threshold for significance
        
        Returns:
            Dictionary containing comprehensive DEG analysis results
        """
        logger.info(f"📊 Analyzing expression changes for {target_gene}")
        logger.info(f"🎯 Thresholds: p < {p_value_threshold}, |log2FC| > {log2fc_threshold}")
        
        start_time = datetime.now()
        
        try:
            # Analyze target gene specific changes
            target_gene_results = self._analyze_target_gene_changes(
                baseline_data, perturbation_data, target_gene
            )
            
            # Perform genome-wide differential expression analysis
            genome_wide_results = self._analyze_genome_wide_changes(
                baseline_data, perturbation_data, p_value_threshold, log2fc_threshold
            )
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(
                target_gene_results, genome_wide_results
            )
            
            # Create comprehensive results
            analysis_results = {
                'target_gene': target_gene,
                'target_gene_results': target_gene_results,
                'genome_wide_results': genome_wide_results,
                'summary_statistics': summary_stats,
                'analysis_parameters': {
                    'p_value_threshold': p_value_threshold,
                    'log2fc_threshold': log2fc_threshold,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            # Update statistics
            self.analyzer_stats['analyses_performed'] += 1
            self.analyzer_stats['genes_tested'] += len(genome_wide_results)
            self.analyzer_stats['significant_degs'] += len([g for g in genome_wide_results 
                                                          if g.get('significant', False)])
            
            duration = datetime.now() - start_time
            logger.info(f"✅ DEG analysis completed in {duration}")
            logger.info(f"📊 Results: {len(genome_wide_results)} genes tested")
            total_degs = summary_stats.get('genome_wide_summary', {}).get('total_significant_degs', 0)
            logger.info(f"📈 Significant DEGs: {total_degs}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"DEG analysis failed: {e}")
            raise RuntimeError(f"Differential expression analysis failed: {e}")
    
    def _analyze_target_gene_changes(self, baseline_data: pd.DataFrame,
                                   perturbation_data: pd.DataFrame,
                                   target_gene: str) -> List[Dict[str, Any]]:
        """
        Analyze expression changes for the target gene across all conditions.
        
        Args:
            baseline_data: Baseline expression data
            perturbation_data: Perturbation prediction data
            target_gene: Target gene symbol
        
        Returns:
            List of target gene change records
        """
        logger.info(f"🎯 Analyzing target gene changes: {target_gene}")
        
        target_results = []
        
        try:
            # Extract target gene effects from perturbation data
            if 'target_effects' in perturbation_data.columns:
                target_effects_df = perturbation_data
            else:
                # If perturbation_data is from State engine results
                target_effects_df = perturbation_data.get('target_effects', pd.DataFrame())
            
            if isinstance(target_effects_df, pd.DataFrame) and not target_effects_df.empty:
                for _, row in target_effects_df.iterrows():
                    # Get baseline expression for this cell line
                    baseline_expr = self._get_baseline_expression(
                        baseline_data, row.get('cell_line', ''), target_gene
                    )
                    
                    # Calculate changes
                    fold_change = row.get('predicted_fold_change', 0)
                    predicted_expr = row.get('predicted_expression', baseline_expr)
                    
                    target_results.append({
                        'target_gene': target_gene,
                        'cell_line': row.get('cell_line', 'unknown'),
                        'organ': row.get('organ', 'unknown'),
                        'tf_perturbed': row.get('tf_perturbed', 'unknown'),
                        'baseline_expression': baseline_expr,
                        'predicted_expression': predicted_expr,
                        'fold_change': fold_change,
                        'log2_fold_change': np.log2(abs(fold_change) + 1e-8) * np.sign(fold_change),
                        'p_value': row.get('p_value', 0.05),
                        'effect_direction': row.get('effect_direction', 'unknown'),
                        'significant': row.get('p_value', 0.05) < 0.05 and abs(fold_change) > 0.5
                    })
            
            logger.info(f"✅ Target gene analysis: {len(target_results)} records")
            
            return target_results
            
        except Exception as e:
            logger.warning(f"Target gene analysis failed: {e}")
            return []
    
    def _analyze_genome_wide_changes(self, baseline_data: pd.DataFrame,
                                   perturbation_data: pd.DataFrame,
                                   p_threshold: float,
                                   log2fc_threshold: float) -> List[Dict[str, Any]]:
        """
        Perform genome-wide differential expression analysis.
        
        Args:
            baseline_data: Baseline expression data
            perturbation_data: Perturbation prediction data
            p_threshold: P-value threshold
            log2fc_threshold: Log2 fold-change threshold
        
        Returns:
            List of DEG analysis records
        """
        logger.info("🧬 Performing genome-wide differential expression analysis...")
        
        deg_results = []
        
        try:
            # Extract DEG results from perturbation data
            if isinstance(perturbation_data, dict) and 'differential_genes' in perturbation_data:
                deg_df = perturbation_data['differential_genes']
            else:
                # No DEG data available and simulation is not allowed
                logger.error("No differential gene data found in perturbation results")
                raise RuntimeError("DEG analysis requires real differential expression data from State model. Simulation is not allowed.")
            
            if isinstance(deg_df, pd.DataFrame) and not deg_df.empty:
                for _, row in deg_df.iterrows():
                    # Extract values with defaults
                    log2fc = row.get('log2_fold_change', row.get('log2fc', 0))
                    p_value = row.get('p_value', 0.05)
                    adj_p_value = row.get('adjusted_p_value', row.get('p_adj', p_value))
                    
                    # Determine significance
                    significant = (adj_p_value < p_threshold and abs(log2fc) > log2fc_threshold)
                    
                    deg_results.append({
                        'gene_id': row.get('gene_id', f"GENE_{len(deg_results):04d}"),
                        'tf_perturbed': row.get('tf_perturbed', 'unknown'),
                        'log2_fold_change': log2fc,
                        'fold_change': 2 ** log2fc,
                        'p_value': p_value,
                        'adjusted_p_value': adj_p_value,
                        'expression_direction': row.get('expression_direction', 
                                                      'up' if log2fc > 0 else 'down'),
                        'significant': significant,
                        'deg_category': self._classify_deg(log2fc, adj_p_value, p_threshold, log2fc_threshold)
                    })
            
            # Apply multiple testing correction if needed
            if deg_results:
                p_values = [r['p_value'] for r in deg_results]
                _, adj_p_values, _, _ = multipletests(p_values, method='fdr_bh')
                
                for i, result in enumerate(deg_results):
                    result['adjusted_p_value'] = adj_p_values[i]
                    result['significant'] = (adj_p_values[i] < p_threshold and 
                                           abs(result['log2_fold_change']) > log2fc_threshold)
            
            significant_count = len([r for r in deg_results if r['significant']])
            logger.info(f"✅ Genome-wide analysis: {len(deg_results)} genes tested")
            logger.info(f"📈 Significant DEGs: {significant_count}")
            
            return deg_results
            
        except Exception as e:
            logger.warning(f"Genome-wide analysis failed: {e}")
            return []
    
    
    def _get_baseline_expression(self, baseline_data: pd.DataFrame, 
                               cell_line: str, gene: str) -> float:
        """
        Get baseline expression level for a gene in a cell line.
        
        Args:
            baseline_data: Baseline expression data
            cell_line: Cell line name
            gene: Gene symbol
        
        Returns:
            Baseline expression level
        """
        try:
            if 'expression_level' in baseline_data.columns:
                cell_line_data = baseline_data[baseline_data['cell_line'] == cell_line]
                if not cell_line_data.empty:
                    return cell_line_data['expression_level'].iloc[0]
            
            # Default baseline expression
            return 1.0
            
        except Exception:
            return 1.0
    
    def _classify_deg(self, log2fc: float, adj_p: float, 
                     p_threshold: float, fc_threshold: float) -> str:
        """
        Classify DEG based on fold change and significance.
        
        Args:
            log2fc: Log2 fold change
            adj_p: Adjusted p-value
            p_threshold: P-value threshold
            fc_threshold: Fold change threshold
        
        Returns:
            DEG classification string
        """
        if adj_p >= p_threshold:
            return 'not_significant'
        elif abs(log2fc) < fc_threshold:
            return 'significant_small_change'
        elif log2fc > fc_threshold:
            return 'significantly_upregulated'
        elif log2fc < -fc_threshold:
            return 'significantly_downregulated'
        else:
            return 'other'
    
    def _calculate_summary_statistics(self, target_results: List[Dict[str, Any]],
                                    genome_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics for the DEG analysis.
        
        Args:
            target_results: Target gene analysis results
            genome_results: Genome-wide analysis results
        
        Returns:
            Dictionary with summary statistics
        """
        logger.info("📊 Calculating summary statistics...")
        
        try:
            # Target gene statistics
            significant_target_effects = len([r for r in target_results if r.get('significant', False)])
            
            # Genome-wide statistics
            total_genes_tested = len(genome_results)
            total_significant_degs = len([r for r in genome_results if r.get('significant', False)])
            upregulated_degs = len([r for r in genome_results 
                                  if r.get('significant', False) and r.get('log2_fold_change', 0) > 0])
            downregulated_degs = total_significant_degs - upregulated_degs
            
            # TF-specific statistics
            tf_stats = {}
            for result in genome_results:
                tf = result.get('tf_perturbed', 'unknown')
                if tf not in tf_stats:
                    tf_stats[tf] = {'total': 0, 'significant': 0, 'upregulated': 0, 'downregulated': 0}
                
                tf_stats[tf]['total'] += 1
                if result.get('significant', False):
                    tf_stats[tf]['significant'] += 1
                    if result.get('log2_fold_change', 0) > 0:
                        tf_stats[tf]['upregulated'] += 1
                    else:
                        tf_stats[tf]['downregulated'] += 1
            
            summary = {
                'target_gene_summary': {
                    'total_conditions': len(target_results),
                    'significant_effects': significant_target_effects,
                    'effect_rate': significant_target_effects / max(len(target_results), 1)
                },
                'genome_wide_summary': {
                    'total_genes_tested': total_genes_tested,
                    'total_significant_degs': total_significant_degs,
                    'upregulated_degs': upregulated_degs,
                    'downregulated_degs': downregulated_degs,
                    'deg_rate': total_significant_degs / max(total_genes_tested, 1)
                },
                'tf_specific_summary': tf_stats,
                'overall_statistics': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_comparisons': len(target_results) + len(genome_results)
                }
            }
            
            logger.info(f"✅ Summary statistics calculated")
            logger.info(f"   Target effects: {significant_target_effects}/{len(target_results)}")
            logger.info(f"   Genome-wide DEGs: {total_significant_degs}/{total_genes_tested}")
            
            return summary
            
        except Exception as e:
            logger.warning(f"Summary statistics calculation failed: {e}")
            return {'error': str(e)}
    
    def export_deg_results(self, analysis_results: Dict[str, Any], 
                          target_gene: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export DEG analysis results to files.
        
        Args:
            analysis_results: Complete analysis results
            target_gene: Target gene symbol
            output_dir: Optional output directory override
        
        Returns:
            Dictionary mapping result types to file paths
        """
        logger.info("💾 Exporting DEG analysis results...")
        
        if output_dir:
            export_dir = Path(output_dir)
        else:
            export_dir = self.results_dir
        
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_paths = {}
        
        try:
            # Export target gene changes
            if analysis_results.get('target_gene_results'):
                target_df = pd.DataFrame(analysis_results['target_gene_results'])
                target_path = export_dir / f"{timestamp}_{target_gene}_target_changes.csv"
                target_df.to_csv(target_path, index=False)
                export_paths['target_changes'] = str(target_path)
            
            # Export genome-wide DEGs
            if analysis_results.get('genome_wide_results'):
                deg_df = pd.DataFrame(analysis_results['genome_wide_results'])
                deg_path = export_dir / f"{timestamp}_{target_gene}_degs.csv"
                deg_df.to_csv(deg_path, index=False)
                export_paths['degs'] = str(deg_path)
            
            # Export summary statistics
            summary_path = export_dir / f"{timestamp}_{target_gene}_deg_summary.json"
            summary_data = {
                'analysis_results': {
                    'target_gene': analysis_results['target_gene'],
                    'summary_statistics': analysis_results['summary_statistics'],
                    'analysis_parameters': analysis_results['analysis_parameters']
                },
                'analyzer_statistics': self.get_analyzer_statistics()
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            export_paths['summary'] = str(summary_path)
            
            logger.info(f"✅ Results exported to {len(export_paths)} files")
            for result_type, path in export_paths.items():
                logger.info(f"   {result_type}: {path}")
            
            return export_paths
            
        except Exception as e:
            logger.error(f"Results export failed: {e}")
            raise RuntimeError(f"DEG results export failed: {e}")
    
    def get_analyzer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analyzer statistics."""
        stats = self.analyzer_stats.copy()
        stats['analyzer_end_time'] = datetime.now().isoformat()
        return stats


def main():
    """Test the DEG analyzer."""
    # Test data
    baseline_data = pd.DataFrame({
        'cell_line': ['HeLa', 'A549', 'MCF7'],
        'organ': ['cervix', 'lung', 'breast'],
        'expression_level': [1.2, 0.8, 1.5]
    })
    
    perturbation_data = {
        'target_effects': pd.DataFrame({
            'cell_line': ['HeLa', 'A549'],
            'organ': ['cervix', 'lung'],
            'tf_perturbed': ['STAT1', 'NF-KB'],
            'predicted_fold_change': [2.5, -1.8],
            'predicted_expression': [3.0, 0.4],
            'p_value': [0.01, 0.02]
        }),
        'tfs_analyzed': ['STAT1', 'NF-KB']
    }
    
    analyzer = DEGAnalyzer()
    
    try:
        results = analyzer.analyze_expression_changes(
            baseline_data, perturbation_data, "TP53"
        )
        
        print("✅ DEG analysis successful!")
        print(f"Target gene effects: {len(results['target_gene_results'])}")
        print(f"Genome-wide DEGs: {len(results['genome_wide_results'])}")
        print(f"Significant DEGs: {results['summary_statistics']['genome_wide_summary']['total_significant_degs']}")
        
        # Export results
        export_paths = analyzer.export_deg_results(results, "TP53")
        print(f"Results exported to: {list(export_paths.keys())}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()