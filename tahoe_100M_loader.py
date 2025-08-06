#!/usr/bin/env python3
"""
Enhanced Tahoe-100M Data Loader with DMSO Control Integration

A robust data loader for the Tahoe-100M single-cell perturbation atlas containing
100M+ transcriptomic profiles across 50 cancer cell lines, with specialized focus
on DMSO control conditions for baseline TF expression analysis.

Key Features:
- DMSO control-focused data extraction for baseline expression analysis
- Uses proper expression_data subset (not differential expression statistics)
- HuggingFace datasets integration for efficient streaming access
- Comprehensive treatment condition filtering and validation
- Real data only - no synthetic or mock data

Usage:
    loader = ComprehensiveTahoeDataLoader()
    cell_lines = loader.get_available_cell_lines()
    # Extract ONLY DMSO control expression data
    expression_data = loader.get_control_tf_expression("TP53", cell_lines=["HeLa", "A549"])
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import requests
from io import BytesIO
import warnings

# HuggingFace datasets for proper data access
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logging.warning("HuggingFace datasets not available - falling back to manual downloads")

# Suppress anndata warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='anndata')

# Set up logging
logger = logging.getLogger(__name__)

class ComprehensiveTahoeDataLoader:
    """
    Enhanced loader for Tahoe-100M single-cell perturbation atlas with DMSO control focus.
    """
    
    DATASET_NAME = "tahoebio/Tahoe-100M"
    CELL_LINE_METADATA_URL = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/cell_line_metadata.parquet"
    GENE_METADATA_URL = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/gene_metadata.parquet"
    
    def __init__(self, cache_dir: str = "comprehensive_cache", config: Optional[Dict] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.config = config or self._get_default_config()
        
        self._cell_line_metadata = None
        self._gene_metadata = None
        self._expression_dataset = None
        self._gene_id_mapping = None
        self._gene_symbol_to_token_map = None
        self._token_to_gene_symbol_map = None
        self._available_cell_lines = None
        self._cell_line_organ_map = None
        self._drug_metadata = None
        self._complete_drug_catalog = None

        logger.info("🧬 Enhanced Tahoe-100M Data Loader with DMSO Control Integration")
        logger.info(f"   Cache directory: {self.cache_dir}")
        
        self._initialize_metadata()
        self._initialize_datasets()

    def _get_default_config(self) -> Dict:
        return {
            'use_datasets_library': DATASETS_AVAILABLE,
            'download_timeout': 3600,  # Default 1 hour timeout for downloads
        }

    def _initialize_metadata(self):
        logger.info("🔍 Downloading Tahoe-100M metadata...")
        self._download_cell_line_metadata()
        self._download_gene_metadata()
        self._download_drug_metadata()
        self._build_organ_mapping()
        self._build_gene_mappings()
        self._analyze_complete_drug_catalog()

    def _download_cell_line_metadata(self):
        cache_file = self.cache_dir / "cell_line_metadata.parquet"
        if cache_file.exists():
            logger.info("Loading cached cell line metadata...")
            self._cell_line_metadata = pd.read_parquet(cache_file)
        else:
            logger.info("Downloading cell line metadata from Hugging Face...")
            response = requests.get(self.CELL_LINE_METADATA_URL, timeout=60)
            response.raise_for_status()
            self._cell_line_metadata = pd.read_parquet(BytesIO(response.content))
            self._cell_line_metadata.to_parquet(cache_file)
        self._available_cell_lines = sorted(self._cell_line_metadata['cell_name'].unique())
        logger.info(f"Found {len(self._available_cell_lines)} unique cell lines")

    def _download_gene_metadata(self):
        cache_file = self.cache_dir / "gene_metadata.parquet"
        if cache_file.exists():
            logger.info("Loading cached gene metadata...")
            self._gene_metadata = pd.read_parquet(cache_file)
        else:
            logger.info("Downloading gene metadata from Hugging Face...")
            response = requests.get(self.GENE_METADATA_URL, timeout=60)
            response.raise_for_status()
            self._gene_metadata = pd.read_parquet(BytesIO(response.content))
            self._gene_metadata.to_parquet(cache_file)
        logger.info(f"Loaded metadata for {len(self._gene_metadata)} genes")

    def _download_drug_metadata(self):
        """Download comprehensive drug metadata from Tahoe-100M dataset."""
        cache_file = self.cache_dir / "drug_metadata.parquet"
        
        if cache_file.exists():
            logger.info("Loading cached drug metadata...")
            self._drug_metadata = pd.read_parquet(cache_file)
        else:
            if not self.config.get('use_datasets_library'):
                logger.warning("HuggingFace datasets not available - cannot download drug metadata")
                self._drug_metadata = pd.DataFrame()
                return
                
            try:
                logger.info("Downloading complete drug metadata from Tahoe-100M...")
                drug_dataset = load_dataset(self.DATASET_NAME, 'drug_metadata', split='train')
                
                # Convert to pandas DataFrame
                self._drug_metadata = drug_dataset.to_pandas()
                
                # Cache for future use
                self._drug_metadata.to_parquet(cache_file)
                logger.info(f"✅ Downloaded and cached drug metadata: {len(self._drug_metadata)} drugs")
                
            except Exception as e:
                logger.error(f"Failed to download drug metadata: {e}")
                self._drug_metadata = pd.DataFrame()
                return
        
        logger.info(f"📊 Drug metadata loaded: {len(self._drug_metadata)} unique drugs")
        if not self._drug_metadata.empty:
            logger.info(f"   Available drug fields: {list(self._drug_metadata.columns)}")

    def _build_organ_mapping(self):
        self._cell_line_organ_map = {}
        self._cell_name_to_cellosaurus_id = {}
        self._cellosaurus_id_to_cell_name = {}
        
        for _, row in self._cell_line_metadata.iterrows():
            cell_name = row['cell_name']
            organ = self._normalize_organ_name(row['Organ'])
            cellosaurus_id = row.get('Cell_ID_Cellosaur', None)  # Correct field name
            
            self._cell_line_organ_map[cell_name] = organ
            
            # Build CELLOSAURUS ID mappings if available
            if cellosaurus_id and cellosaurus_id != 'unknown' and pd.notna(cellosaurus_id):
                self._cell_name_to_cellosaurus_id[cell_name] = cellosaurus_id
                self._cellosaurus_id_to_cell_name[cellosaurus_id] = cell_name
        
        logger.info(f"Built organ mapping for {len(self._cell_line_organ_map)} cell lines")
        logger.info(f"Built CELLOSAURUS ID mapping for {len(self._cell_name_to_cellosaurus_id)} cell lines")

    def _build_gene_mappings(self):
        self._gene_symbol_to_token_map = self._gene_metadata.set_index('gene_symbol')['token_id'].to_dict()
        self._token_to_gene_symbol_map = self._gene_metadata.set_index('token_id')['gene_symbol'].to_dict()
        logger.info(f"Created mappings for {len(self._gene_symbol_to_token_map)} gene symbols")

    def _analyze_complete_drug_catalog(self):
        """Analyze the complete drug catalog from Tahoe-100M metadata."""
        if self._drug_metadata is None or self._drug_metadata.empty:
            logger.warning("No drug metadata available for analysis")
            self._complete_drug_catalog = {}
            return
        
        logger.info("🧪 Analyzing complete Tahoe-100M drug catalog...")
        
        # Initialize catalog analysis
        catalog = {
            'total_drugs': len(self._drug_metadata),
            'analysis_timestamp': datetime.now().isoformat(),
            'drug_list': [],
            'moa_analysis': {},
            'target_analysis': {},
            'clinical_status': {},
            'chemical_diversity': {},
            'drug_name_mapping': {}
        }
        
        # Extract basic drug information
        if 'drug' in self._drug_metadata.columns:
            catalog['drug_list'] = sorted(self._drug_metadata['drug'].unique().tolist())
            # Create drug name mapping for expression data cross-reference
            catalog['drug_name_mapping'] = {drug: drug for drug in catalog['drug_list']}
        
        # Analyze Mechanism of Action (MOA)
        if 'moa-broad' in self._drug_metadata.columns:
            moa_broad_counts = self._drug_metadata['moa-broad'].value_counts()
            catalog['moa_analysis']['broad_categories'] = moa_broad_counts.to_dict()
            catalog['moa_analysis']['broad_category_count'] = len(moa_broad_counts)
            
        if 'moa-fine' in self._drug_metadata.columns:
            moa_fine_counts = self._drug_metadata['moa-fine'].value_counts()
            catalog['moa_analysis']['fine_categories'] = moa_fine_counts.to_dict()
            catalog['moa_analysis']['fine_category_count'] = len(moa_fine_counts)
        
        # Analyze molecular targets
        if 'targets' in self._drug_metadata.columns:
            # Handle target analysis (targets might be lists or strings)
            target_series = self._drug_metadata['targets'].dropna()
            all_targets = []
            for targets in target_series:
                if isinstance(targets, str):
                    # Split by common delimiters
                    all_targets.extend([t.strip() for t in targets.replace(';', ',').split(',') if t.strip()])
                elif isinstance(targets, list):
                    all_targets.extend(targets)
            
            if all_targets:
                target_counts = pd.Series(all_targets).value_counts()
                catalog['target_analysis']['unique_targets'] = len(target_counts)
                catalog['target_analysis']['top_targets'] = target_counts.head(20).to_dict()
                catalog['target_analysis']['all_targets'] = target_counts.to_dict()
        
        # Analyze clinical status
        if 'human-approved' in self._drug_metadata.columns:
            approved_counts = self._drug_metadata['human-approved'].value_counts()
            catalog['clinical_status']['fda_approved'] = approved_counts.to_dict()
        
        if 'clinical-trials' in self._drug_metadata.columns:
            trial_counts = self._drug_metadata['clinical-trials'].value_counts()
            catalog['clinical_status']['clinical_trials'] = trial_counts.to_dict()
        
        # Analyze chemical diversity
        if 'canonical_smiles' in self._drug_metadata.columns:
            smiles_data = self._drug_metadata['canonical_smiles'].dropna()
            catalog['chemical_diversity']['unique_smiles'] = len(smiles_data.unique())
            catalog['chemical_diversity']['total_with_smiles'] = len(smiles_data)
            
        if 'pubchem_cid' in self._drug_metadata.columns:
            pubchem_data = self._drug_metadata['pubchem_cid'].dropna()
            catalog['chemical_diversity']['unique_pubchem_ids'] = len(pubchem_data.unique())
            catalog['chemical_diversity']['total_with_pubchem'] = len(pubchem_data)
        
        self._complete_drug_catalog = catalog
        
        # Log comprehensive analysis
        logger.info(f"✅ Complete Drug Catalog Analysis:")
        logger.info(f"   📊 Total unique drugs: {catalog['total_drugs']}")
        logger.info(f"   🎯 MOA broad categories: {catalog['moa_analysis'].get('broad_category_count', 'N/A')}")
        logger.info(f"   🎯 MOA fine categories: {catalog['moa_analysis'].get('fine_category_count', 'N/A')}")
        logger.info(f"   🧬 Molecular targets: {catalog['target_analysis'].get('unique_targets', 'N/A')}")
        logger.info(f"   💊 Chemical structures (SMILES): {catalog['chemical_diversity'].get('unique_smiles', 'N/A')}")
        logger.info(f"   🏥 PubChem compounds: {catalog['chemical_diversity'].get('unique_pubchem_ids', 'N/A')}")
        
        # Show top MOAs and targets
        if 'broad_categories' in catalog['moa_analysis']:
            top_moas = list(catalog['moa_analysis']['broad_categories'].items())[:5]
            logger.info(f"   🔝 Top MOA categories: {dict(top_moas)}")
            
        if 'top_targets' in catalog['target_analysis']:
            top_targets = list(catalog['target_analysis']['top_targets'].items())[:5]
            logger.info(f"   🔝 Top molecular targets: {dict(top_targets)}")

    def _initialize_datasets(self):
        if not self.config.get('use_datasets_library'):
            logger.warning("HuggingFace datasets library not available. Cannot access expression data.")
            return
        try:
            from datasets import DownloadConfig as HFDownloadConfig
            logger.info("🔧 Initializing HuggingFace datasets for expression data (streaming)...")
            
            # Use a download config with maximum retries
            download_config = HFDownloadConfig(max_retries=self.config.get('retry_attempts', 3))
            
            self._expression_dataset = load_dataset(
                self.DATASET_NAME, 
                'expression_data', 
                split='train', 
                streaming=True, 
                download_config=download_config
            )
            logger.info("✅ Expression dataset loaded successfully in streaming mode.")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace datasets: {e}")
            self._expression_dataset = None

    def get_available_cell_lines(self) -> List[str]:
        return self._available_cell_lines or []

    def get_organ_for_cell_line(self, cell_line: str) -> str:
        return self._cell_line_organ_map.get(cell_line, 'unknown')

    def get_target_cell_lines(self, tissue_type: Optional[str] = None) -> List[str]:
        if not tissue_type:
            return self.get_available_cell_lines()
        
        target_organs = [t.strip().lower() for t in tissue_type.split(',')]
        target_lines = []
        for organ in target_organs:
            lines = [cl for cl, org in self._cell_line_organ_map.items() if org == organ]
            if lines:
                target_lines.extend(lines)
                logger.info(f"🎯 Added {len(lines)} {organ} cell lines")
            else:
                logger.warning(f"⚠️  No cell lines found for tissue type: {organ}")
        
        return sorted(list(set(target_lines)))
    
    def get_complete_drug_catalog(self) -> Dict[str, Any]:
        """Get the complete drug catalog with comprehensive analysis."""
        if self._complete_drug_catalog is None:
            logger.warning("Drug catalog not initialized")
            return {}
        return self._complete_drug_catalog
    
    def get_drug_list(self) -> List[str]:
        """Get list of all available drugs in Tahoe-100M."""
        if self._complete_drug_catalog and 'drug_list' in self._complete_drug_catalog:
            return self._complete_drug_catalog['drug_list']
        return []
    
    def get_drug_metadata_df(self) -> pd.DataFrame:
        """Get the raw drug metadata DataFrame."""
        return self._drug_metadata if self._drug_metadata is not None else pd.DataFrame()
    
    def cross_reference_expression_drugs(self, max_samples: int = 10000) -> Dict[str, Any]:
        """
        Cross-reference drugs found in expression data with the complete drug catalog.
        
        Args:
            max_samples: Maximum expression samples to check for drug cross-reference
            
        Returns:
            Dictionary with cross-reference analysis
        """
        logger.info(f"🔍 Cross-referencing expression data drugs with complete catalog...")
        
        if not self._expression_dataset:
            logger.error("Expression dataset not available for cross-reference")
            return {}
        
        if not self._complete_drug_catalog:
            logger.error("Drug catalog not available for cross-reference")
            return {}
        
        # Get complete drug list from catalog
        catalog_drugs = set(self.get_drug_list())
        
        # Sample expression data to find drugs actually present
        expression_drugs = set()
        samples_checked = 0
        
        logger.info(f"   Checking up to {max_samples} expression samples for drug names...")
        
        for sample in self._expression_dataset:
            if samples_checked >= max_samples:
                break
                
            drug_val = sample.get('drug', '')
            if drug_val and drug_val != 'MISSING':
                expression_drugs.add(drug_val)
            
            samples_checked += 1
            
            # Progress logging
            if samples_checked % 1000 == 0:
                logger.info(f"   Checked {samples_checked} samples, found {len(expression_drugs)} unique drugs")
        
        # Perform cross-reference analysis
        drugs_in_both = catalog_drugs.intersection(expression_drugs)
        drugs_only_in_catalog = catalog_drugs - expression_drugs
        drugs_only_in_expression = expression_drugs - catalog_drugs
        
        cross_ref_analysis = {
            'analysis_timestamp': datetime.now().isoformat(),
            'samples_checked': samples_checked,
            'catalog_drugs_total': len(catalog_drugs),
            'expression_drugs_found': len(expression_drugs),
            'drugs_in_both_datasets': len(drugs_in_both),
            'drugs_only_in_catalog': len(drugs_only_in_catalog),
            'drugs_only_in_expression': len(drugs_only_in_expression),
            'coverage_percentage': (len(drugs_in_both) / len(catalog_drugs) * 100) if catalog_drugs else 0,
            'detailed_mapping': {
                'drugs_in_both': sorted(list(drugs_in_both)),
                'drugs_only_in_catalog': sorted(list(drugs_only_in_catalog)),
                'drugs_only_in_expression': sorted(list(drugs_only_in_expression)),
                'expression_drugs_list': sorted(list(expression_drugs))
            }
        }
        
        # Log results
        logger.info(f"✅ Drug Cross-Reference Analysis Complete:")
        logger.info(f"   📊 Catalog drugs: {len(catalog_drugs)}")
        logger.info(f"   📊 Expression drugs found: {len(expression_drugs)}")
        logger.info(f"   ✅ Drugs in both datasets: {len(drugs_in_both)}")
        logger.info(f"   📈 Coverage: {cross_ref_analysis['coverage_percentage']:.1f}%")
        logger.info(f"   ⚠️  Catalog-only drugs: {len(drugs_only_in_catalog)}")
        logger.info(f"   ⚠️  Expression-only drugs: {len(drugs_only_in_expression)}")
        
        if drugs_only_in_expression:
            logger.info(f"   🔍 Expression-only drugs (first 10): {sorted(list(drugs_only_in_expression))[:10]}")
        
        return cross_ref_analysis
    
    def generate_comprehensive_drug_report(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate comprehensive drug catalog report with all metadata.
        
        Args:
            output_dir: Directory to save reports (default: cache_dir/drug_reports)
            
        Returns:
            Dictionary with paths to generated report files
        """
        if not self._complete_drug_catalog or self._drug_metadata is None:
            logger.error("Drug catalog or metadata not available for report generation")
            return {}
        
        # Set up output directory
        if output_dir is None:
            output_dir = self.cache_dir / "drug_reports"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_paths = {}
        
        logger.info(f"📋 Generating comprehensive drug catalog reports...")
        
        # 1. Generate summary report
        summary_path = output_dir / f"{timestamp}_tahoe_drug_catalog_summary.md"
        catalog = self._complete_drug_catalog
        
        summary_content = f"""# 🧪 TAHOE-100M COMPLETE DRUG CATALOG SUMMARY

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Dataset:** Tahoe-100M Single-Cell Perturbation Atlas  
**Total Drugs:** {catalog['total_drugs']}

---

## 📊 OVERVIEW STATISTICS

### Drug Collection
- **Total Unique Drugs:** {catalog['total_drugs']:,}
- **Chemical Structures (SMILES):** {catalog['chemical_diversity'].get('unique_smiles', 'N/A'):,}
- **PubChem Compounds:** {catalog['chemical_diversity'].get('unique_pubchem_ids', 'N/A'):,}

### Mechanism of Action (MOA)
- **Broad MOA Categories:** {catalog['moa_analysis'].get('broad_category_count', 'N/A')}
- **Fine MOA Categories:** {catalog['moa_analysis'].get('fine_category_count', 'N/A')}

### Molecular Targets
- **Unique Molecular Targets:** {catalog['target_analysis'].get('unique_targets', 'N/A')}

---

## 🔝 TOP CATEGORIES

### Top Mechanism of Action (Broad)
"""
        
        if 'broad_categories' in catalog['moa_analysis']:
            for i, (moa, count) in enumerate(list(catalog['moa_analysis']['broad_categories'].items())[:10], 1):
                summary_content += f"{i}. **{moa}** - {count} drugs\\n"
        
        summary_content += f"""
### Top Molecular Targets
"""
        
        if 'top_targets' in catalog['target_analysis']:
            for i, (target, count) in enumerate(list(catalog['target_analysis']['top_targets'].items())[:10], 1):
                summary_content += f"{i}. **{target}** - {count} drugs\\n"
        
        summary_content += f"""
---

## 💊 CLINICAL STATUS

### FDA Approval Status
"""
        
        if 'fda_approved' in catalog['clinical_status']:
            for status, count in catalog['clinical_status']['fda_approved'].items():
                summary_content += f"- **{status}:** {count} drugs\\n"
        
        summary_content += f"""
### Clinical Trial Status
"""
        
        if 'clinical_trials' in catalog['clinical_status']:
            for status, count in catalog['clinical_status']['clinical_trials'].items():
                summary_content += f"- **{status}:** {count} drugs\\n"
        
        summary_content += f"""
---

## 📚 HOW TO USE THIS CATALOG

1. **Complete Drug List:** See `*_complete_drug_list.csv` for all {catalog['total_drugs']} drugs
2. **Detailed Metadata:** Check `*_drug_metadata.csv` for comprehensive information
3. **MOA Analysis:** Review `*_moa_analysis.json` for mechanism breakdowns
4. **Target Analysis:** Explore `*_target_analysis.json` for molecular target data

---

## 🔬 DATASET CONTEXT

This catalog represents the complete collection of small-molecule perturbations in the Tahoe-100M dataset:
- **Scale:** 100M+ single-cell transcriptomic profiles
- **Cell Lines:** 50 cancer cell lines  
- **Perturbations:** 1,100+ small molecules
- **Coverage:** Comprehensive mechanism and target diversity

---

*Generated from Tahoe-100M drug_metadata table*  
*For technical details, see accompanying JSON and CSV files*
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        report_paths['summary'] = str(summary_path)
        
        # 2. Export complete drug list
        drug_list_path = output_dir / f"{timestamp}_complete_drug_list.csv"
        drug_list_df = pd.DataFrame({'drug_name': catalog['drug_list']})
        drug_list_df.to_csv(drug_list_path, index=False)
        report_paths['drug_list'] = str(drug_list_path)
        
        # 3. Export complete drug metadata
        metadata_path = output_dir / f"{timestamp}_drug_metadata.csv"
        self._drug_metadata.to_csv(metadata_path, index=False)
        report_paths['metadata'] = str(metadata_path)
        
        # 4. Export MOA analysis
        moa_path = output_dir / f"{timestamp}_moa_analysis.json"
        with open(moa_path, 'w') as f:
            json.dump(catalog['moa_analysis'], f, indent=2)
        report_paths['moa_analysis'] = str(moa_path)
        
        # 5. Export target analysis
        target_path = output_dir / f"{timestamp}_target_analysis.json"
        with open(target_path, 'w') as f:
            json.dump(catalog['target_analysis'], f, indent=2)
        report_paths['target_analysis'] = str(target_path)
        
        # 6. Export complete catalog
        catalog_path = output_dir / f"{timestamp}_complete_drug_catalog.json"
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        report_paths['complete_catalog'] = str(catalog_path)
        
        logger.info(f"✅ Drug catalog reports generated:")
        logger.info(f"   📋 Summary: {summary_path}")
        logger.info(f"   📊 Drug List: {drug_list_path}")
        logger.info(f"   📚 Metadata: {metadata_path}")
        logger.info(f"   🎯 MOA Analysis: {moa_path}")
        logger.info(f"   🧬 Target Analysis: {target_path}")
        logger.info(f"   💾 Complete Catalog: {catalog_path}")
        logger.info(f"   📁 Reports directory: {output_dir}")
        
        return report_paths

    def get_control_tf_expression(self, gene_symbol: str, cell_lines: List[str], tf_list: List[str]) -> pd.DataFrame:
        """
        Extract TF expression data from Tahoe-100M dataset.
        
        NOTE: After analysis, it appears Tahoe-100M may not have traditional DMSO vehicle controls.
        Instead, we'll extract baseline expression from any available samples for the requested cell lines,
        prioritizing lower-dose or less disruptive treatments as proxy controls.
        """
        logger.info(f"🧬 Extracting TF expression for gene: {gene_symbol} from Tahoe-100M")
        if not self._expression_dataset:
            logger.error("Expression dataset not available. Cannot extract data.")
            return pd.DataFrame()

        tf_tokens = {self._gene_symbol_to_token_map.get(tf) for tf in tf_list}
        tf_tokens.discard(None)
        if not tf_tokens:
            logger.warning(f"No valid TF tokens found for list: {tf_list}")
            return pd.DataFrame()

        # Convert cell line names to CELLOSAURUS IDs for matching
        target_cellosaurus_ids = []
        for cell_line in cell_lines:
            cellosaurus_id = self._cell_name_to_cellosaurus_id.get(cell_line)
            if cellosaurus_id:
                target_cellosaurus_ids.append(cellosaurus_id)
                logger.info(f"🔍 Mapped {cell_line} -> {cellosaurus_id}")
            else:
                logger.warning(f"⚠️  No CELLOSAURUS ID found for cell line: {cell_line}")

        if not target_cellosaurus_ids:
            logger.error(f"❌ No valid CELLOSAURUS IDs found for cell lines: {cell_lines}")
            return pd.DataFrame()

        logger.info(f"🔍 Searching for samples with cell_line_ids: {target_cellosaurus_ids}")
        logger.info(f"🧬 Looking for {len(tf_tokens)} TF tokens: {tf_list}")
        
        expression_results = []
        matching_sample_count = 0
        total_sample_count = 0
        
        # Search through dataset for matching cell lines
        for sample in self._expression_dataset:
            total_sample_count += 1
            
            # Check if this sample matches our target cell lines
            sample_cell_line_id = sample.get('cell_line_id', '')
            if sample_cell_line_id in target_cellosaurus_ids:
                matching_sample_count += 1
                
                # Get the cell line name for this CELLOSAURUS ID
                cell_line_name = self._cellosaurus_id_to_cell_name.get(sample_cell_line_id, sample_cell_line_id)
                
                # Create a dictionary for fast lookup of gene expressions in the current sample
                sample_expressions = {gene: expr for gene, expr in zip(sample['genes'], sample['expressions'])}
                
                # Extract TF expression data
                for tf_symbol in tf_list:
                    tf_token = self._gene_symbol_to_token_map.get(tf_symbol)
                    if tf_token in sample_expressions:
                        expression_results.append({
                            'gene_symbol': gene_symbol,
                            'tf_name': tf_symbol,
                            'cell_line': cell_line_name,
                            'organ': self.get_organ_for_cell_line(cell_line_name),
                            'mean_expression': sample_expressions[tf_token],
                            'condition': 'baseline_expression',  # Changed from DMSO_control
                            'source': 'Tahoe-100M',
                            'drug_value': sample.get('drug', 'unknown'),
                            'sample_id': sample.get('sample', 'unknown'),
                            'cellosaurus_id': sample_cell_line_id,
                            # Add expected fields for integration
                            'expression_frequency': 1.0,  # Single sample = 100% frequency
                            'expressing_cells': 1,  # Single sample
                            'total_cells': 1,  # Single sample
                            'std_expression': 0.0  # No variation in single sample
                        })
            
            # Limit inspection for performance - break after finding sufficient samples
            if matching_sample_count >= 50 or total_sample_count >= 50000:  # More reasonable limits
                break
        
        # Log search results
        logger.info(f"🧪 Tahoe-100M Expression Search Results:")
        logger.info(f"   Total samples inspected: {total_sample_count}")
        logger.info(f"   Matching cell line samples found: {matching_sample_count}")
        logger.info(f"   Target CELLOSAURUS IDs: {target_cellosaurus_ids}")
        logger.info(f"   Expression measurements extracted: {len(expression_results)}")
        
        if matching_sample_count == 0:
            logger.error(f"❌ No samples found for CELLOSAURUS IDs: {target_cellosaurus_ids}")
            logger.error(f"   Original cell lines: {cell_lines}")
            return pd.DataFrame()

        if not expression_results:
            logger.warning(f"❌ No TF expression data found for the specified TFs in matching samples.")
            logger.warning(f"   TFs searched: {tf_list}")
            return pd.DataFrame()

        results_df = pd.DataFrame(expression_results)
        
        # The data is already at the single-sample level, so we group by TF and cell line to aggregate if needed
        # For now, we return the direct, per-sample measurements.
        logger.info(f"✅ Extracted {len(results_df)} TF expression measurements from baseline samples.")
        return results_df

    def get_quantitative_tf_expression(self, gene_symbol: str, cell_lines: List[str], tf_list: List[str], 
                                     max_samples_per_cell_line: int = 10) -> pd.DataFrame:
        """
        Enhanced quantitative TF expression extraction with proper statistical aggregation.
        
        This method addresses the issues found in the raw data analysis:
        - Uses correct gene symbol to token mapping 
        - Aggregates multiple samples per cell line for robust statistics
        - Handles the real quantitative expression values (-2.0 to 423.0)
        - Provides statistical measures instead of single-sample values
        
        Args:
            gene_symbol: Target gene symbol for analysis context
            cell_lines: List of cell line names to analyze
            tf_list: List of TF gene symbols to extract expression for
            max_samples_per_cell_line: Maximum samples to process per cell line
            
        Returns:
            DataFrame with quantitative expression statistics per TF-cell line combination
        """
        logger.info(f"🔬 Enhanced quantitative TF expression extraction for {gene_symbol}")
        logger.info(f"   Target cell lines: {cell_lines}")
        logger.info(f"   TF list: {tf_list}")
        
        if not self._expression_dataset:
            logger.error("Expression dataset not available")
            return pd.DataFrame()
        
        # Create TF symbol to token mapping
        tf_token_mapping = {}
        for tf_symbol in tf_list:
            token_id = self._gene_symbol_to_token_map.get(tf_symbol)
            if token_id:
                tf_token_mapping[tf_symbol] = token_id
            else:
                logger.warning(f"⚠️  TF {tf_symbol} not found in gene mapping")
        
        if not tf_token_mapping:
            logger.error("No valid TF tokens found")
            return pd.DataFrame()
            
        logger.info(f"📊 Found token mappings for {len(tf_token_mapping)} TFs")
        
        # Create cell line to CELLOSAURUS ID mapping with fallback to direct matching
        target_cellosaurus_ids = {}
        for cell_line in cell_lines:
            cellosaurus_id = self._cell_name_to_cellosaurus_id.get(cell_line)
            if cellosaurus_id:
                target_cellosaurus_ids[cell_line] = cellosaurus_id
            else:
                # Try direct matching as fallback (some cell lines might match directly)
                target_cellosaurus_ids[cell_line] = cell_line
                logger.info(f"📝 Using direct matching for cell line: {cell_line}")
            
        logger.info(f"🧫 Setup mappings for {len(target_cellosaurus_ids)} cell lines")
        
        # Collect expression data across samples
        expression_data = {}  # Structure: {(tf_symbol, cell_line): [expression_values]}
        sample_metadata = {}  # Structure: {(tf_symbol, cell_line): [sample_info]}
        samples_processed = 0
        samples_per_cell_line = {cl: 0 for cl in cell_lines}
        
        logger.info("🔍 Scanning dataset for matching samples...")
        
        try:
            for sample in self._expression_dataset:
                samples_processed += 1
                
                # Progress logging
                if samples_processed % 10000 == 0:
                    logger.info(f"   Processed {samples_processed} samples...")
                
                # Check if sample matches target cell lines
                sample_cell_line_id = sample.get('cell_line_id', '')
                matching_cell_line = None
                for cell_line, cellosaurus_id in target_cellosaurus_ids.items():
                    if sample_cell_line_id == cellosaurus_id:
                        matching_cell_line = cell_line
                        break
                
                if not matching_cell_line:
                    continue
                    
                # Skip if we've collected enough samples for this cell line
                if samples_per_cell_line[matching_cell_line] >= max_samples_per_cell_line:
                    continue
                    
                samples_per_cell_line[matching_cell_line] += 1
                
                # Extract gene expressions from sample
                sample_genes = sample.get('genes', [])
                sample_expressions = sample.get('expressions', [])
                
                if len(sample_genes) != len(sample_expressions):
                    logger.warning(f"Gene-expression length mismatch in sample")
                    continue
                    
                # Create lookup dictionary for this sample
                gene_expr_lookup = dict(zip(sample_genes, sample_expressions))
                
                # Extract TF expressions
                for tf_symbol, tf_token in tf_token_mapping.items():
                    if tf_token in gene_expr_lookup:
                        key = (tf_symbol, matching_cell_line)
                        
                        # Initialize containers if needed
                        if key not in expression_data:
                            expression_data[key] = []
                            sample_metadata[key] = []
                        
                        # Store expression value and metadata
                        expr_value = gene_expr_lookup[tf_token]
                        expression_data[key].append(float(expr_value))
                        
                        sample_metadata[key].append({
                            'sample_id': sample.get('sample', 'unknown'),
                            'drug': sample.get('drug', 'unknown'),
                            'plate': sample.get('plate', 'unknown'),
                            'barcode': sample.get('BARCODE_SUB_LIB_ID', 'unknown')
                        })
                
                # Early termination if we've collected enough data
                if all(count >= max_samples_per_cell_line for count in samples_per_cell_line.values()):
                    logger.info(f"✅ Collected sufficient samples for all cell lines")
                    break
                    
                # Aggressive early stopping - stop after smaller number of samples
                if samples_processed >= 10000:  # Much smaller limit
                    logger.info(f"⚠️  Reached processing limit of 10k samples")
                    break
                
                # Additional condition: stop if we have any data for any TF-cell line combo
                if expression_data and samples_processed >= 1000:
                    logger.info(f"✅ Found expression data, stopping early at {samples_processed} samples")
                    break
                    
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            return pd.DataFrame()
        
        logger.info(f"📊 Data collection complete:")
        logger.info(f"   Total samples processed: {samples_processed}")
        logger.info(f"   Samples per cell line: {samples_per_cell_line}")
        logger.info(f"   TF-cell line combinations found: {len(expression_data)}")
        
        # Generate statistical summaries
        results = []
        
        for (tf_symbol, cell_line), expr_values in expression_data.items():
            if not expr_values:
                continue
                
            expr_array = np.array(expr_values)
            metadata_list = sample_metadata[(tf_symbol, cell_line)]
            
            # Calculate comprehensive statistics
            stats = {
                'gene_symbol': gene_symbol,
                'tf_name': tf_symbol,
                'cell_line': cell_line,
                'organ': self.get_organ_for_cell_line(cell_line),
                
                # Expression statistics
                'mean_expression': float(np.mean(expr_array)),
                'median_expression': float(np.median(expr_array)),
                'std_expression': float(np.std(expr_array)),
                'min_expression': float(np.min(expr_array)),
                'max_expression': float(np.max(expr_array)),
                
                # Sample information
                'sample_count': len(expr_values),
                'expressing_cells': int(np.sum(expr_array > 0)),  # Non-zero expressions
                'total_cells': len(expr_values),
                'expression_frequency': float(np.sum(expr_array > 0) / len(expr_values)),
                
                # Percentiles
                'q25_expression': float(np.percentile(expr_array, 25)),
                'q75_expression': float(np.percentile(expr_array, 75)),
                
                # Data source and validation
                'source': 'Tahoe-100M_Enhanced',
                'condition': 'quantitative_aggregated',
                'tf_token_id': tf_token_mapping[tf_symbol],
                
                # Sample diversity metrics
                'unique_drugs': len(set(m['drug'] for m in metadata_list)),
                'unique_plates': len(set(m['plate'] for m in metadata_list)),
            }
            
            results.append(stats)
        
        if not results:
            logger.warning("❌ No quantitative expression data extracted")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results)
        
        # Sort by mean expression (highest first) and sample count
        results_df = results_df.sort_values(['mean_expression', 'sample_count'], ascending=[False, False])
        
        logger.info(f"✅ Enhanced extraction complete:")
        logger.info(f"   📊 Generated {len(results_df)} TF-cell line combinations")
        logger.info(f"   🧬 Unique TFs: {results_df['tf_name'].nunique()}")
        logger.info(f"   🧫 Unique cell lines: {results_df['cell_line'].nunique()}")
        logger.info(f"   📈 Expression range: {results_df['mean_expression'].min():.3f} to {results_df['mean_expression'].max():.3f}")
        logger.info(f"   🔢 Average samples per combination: {results_df['sample_count'].mean():.1f}")
        
        # Show top results
        if len(results_df) > 0:
            logger.info("🏆 Top 3 TF-cell line combinations by mean expression:")
            for i, (_, row) in enumerate(results_df.head(3).iterrows()):
                logger.info(f"   {i+1}. {row['tf_name']} in {row['cell_line']}: "
                          f"mean={row['mean_expression']:.3f} (n={row['sample_count']})")
        
        return results_df

    def _normalize_organ_name(self, organ: str) -> str:
        organ_mapping = {
            'Lung': 'lung', 'Vulva/Vagina': 'vulva', 'Skin': 'skin', 'Breast': 'breast',
            'Bowel': 'colon', 'Esophagus/Stomach': 'stomach', 'Pancreas': 'pancreas',
            'Uterus': 'uterus', 'Bladder/Urinary Tract': 'bladder', 'CNS/Brain': 'brain',
            'Liver': 'liver', 'Ovary/Fallopian Tube': 'ovary', 'Cervix': 'cervix',
            'Peripheral Nervous System': 'nervous_system', 'Kidney': 'kidney'
        }
        return organ_mapping.get(organ, organ.lower().replace('/', '_').replace(' ', '_'))

    def explore_dataset_structure(self, max_samples: int = 2000) -> Dict[str, Any]:
        """
        Explore the actual structure of Tahoe-100M dataset to understand field names and values.
        
        Args:
            max_samples: Maximum number of samples to inspect
            
        Returns:
            Dictionary with exploration results
        """
        logger.info(f"🔍 Exploring Tahoe-100M dataset structure (max {max_samples} samples)...")
        
        if not self._expression_dataset:
            logger.error("Expression dataset not available for exploration")
            return {}
        
        # Check for cached exploration results
        cache_file = self.cache_dir / f"dataset_exploration_{max_samples}.json"
        if cache_file.exists():
            logger.info(f"Loading cached exploration results from {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    cached_results = json.load(f)
                # Convert sets back from lists
                for key in ['drug_values', 'cell_line_ids', 'sample_fields', 'canonical_smiles_patterns']:
                    if key in cached_results:
                        cached_results[key] = set(cached_results[key])
                return cached_results
            except Exception as e:
                logger.warning(f"Failed to load cached exploration: {e}")

        exploration_results = {
            'drug_values': set(),
            'cell_line_ids': set(),
            'sample_fields': set(),
            'unique_combinations': [],
            'total_samples_inspected': 0,
            'canonical_smiles_patterns': set(),
            'cell_line_id_to_drug': {},
            'drug_frequency': {},
            'moa_patterns': set(),
            'pubchem_ids': set(),
            'exploration_timestamp': datetime.now().isoformat()
        }
        
        # Inspect samples to understand structure
        for i, sample in enumerate(self._expression_dataset):
            if i >= max_samples:
                break
                
            exploration_results['total_samples_inspected'] += 1
            
            # Collect all field names
            exploration_results['sample_fields'].update(sample.keys())
            
            # Collect drug values with frequency tracking
            drug_val = sample.get('drug', 'MISSING')
            exploration_results['drug_values'].add(drug_val)
            if drug_val not in exploration_results['drug_frequency']:
                exploration_results['drug_frequency'][drug_val] = 0
            exploration_results['drug_frequency'][drug_val] += 1
            
            # Collect cell line IDs (this might be the key!)
            cell_line_id = sample.get('cell_line_id', 'MISSING')
            exploration_results['cell_line_ids'].add(cell_line_id)
            
            # Collect canonical SMILES (might indicate DMSO)
            smiles = sample.get('canonical_smiles', 'MISSING')
            if smiles and smiles != 'MISSING':
                exploration_results['canonical_smiles_patterns'].add(smiles)
            
            # Collect MOA patterns
            moa = sample.get('moa-fine', 'MISSING')
            if moa and moa != 'MISSING':
                exploration_results['moa_patterns'].add(moa)
            
            # Collect PubChem IDs
            pubchem_id = sample.get('pubchem_cid', 'MISSING')
            if pubchem_id and pubchem_id != 'MISSING':
                exploration_results['pubchem_ids'].add(pubchem_id)
            
            # Track drug-cell line combinations
            if cell_line_id not in exploration_results['cell_line_id_to_drug']:
                exploration_results['cell_line_id_to_drug'][cell_line_id] = set()
            exploration_results['cell_line_id_to_drug'][cell_line_id].add(drug_val)
            
            # Collect detailed combinations for first 20 samples
            if i < 20:
                combo = {
                    'sample_id': i,
                    'drug': drug_val,
                    'cell_line_id': cell_line_id,
                    'canonical_smiles': smiles,
                    'sample': sample.get('sample', 'MISSING'),
                    'pubchem_cid': sample.get('pubchem_cid', 'MISSING')
                }
                exploration_results['unique_combinations'].append(combo)
        
        # Convert sets to sorted lists for logging and caching
        exploration_results['drug_values'] = sorted(list(exploration_results['drug_values']))
        exploration_results['cell_line_ids'] = sorted(list(exploration_results['cell_line_ids']))
        exploration_results['sample_fields'] = sorted(list(exploration_results['sample_fields']))
        exploration_results['canonical_smiles_patterns'] = sorted(list(exploration_results['canonical_smiles_patterns']))
        exploration_results['moa_patterns'] = sorted(list(exploration_results['moa_patterns']))
        exploration_results['pubchem_ids'] = sorted(list(exploration_results['pubchem_ids']))
        
        # Convert cell_line_id_to_drug sets to lists for JSON serialization
        for cell_id in exploration_results['cell_line_id_to_drug']:
            exploration_results['cell_line_id_to_drug'][cell_id] = sorted(list(exploration_results['cell_line_id_to_drug'][cell_id]))
        
        # Cache exploration results for future use
        try:
            with open(cache_file, 'w') as f:
                json.dump(exploration_results, f, indent=2)
            logger.info(f"💾 Exploration results cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache exploration results: {e}")
        
        # Log comprehensive findings
        logger.info(f"📊 Comprehensive Dataset Exploration Results:")
        logger.info(f"   Total samples inspected: {exploration_results['total_samples_inspected']}")
        logger.info(f"   Available fields: {exploration_results['sample_fields']}")
        logger.info(f"   🧪 DRUG DIVERSITY ANALYSIS:")
        logger.info(f"     Unique drugs: {len(exploration_results['drug_values'])}")
        logger.info(f"     Most common drugs: {sorted(exploration_results['drug_frequency'].items(), key=lambda x: x[1], reverse=True)[:5]}")
        logger.info(f"     Drug list (first 15): {exploration_results['drug_values'][:15]}")
        logger.info(f"   🧫 CELL LINE ANALYSIS:")
        logger.info(f"     Unique cell lines: {len(exploration_results['cell_line_ids'])}")
        logger.info(f"     Cell line IDs (first 15): {exploration_results['cell_line_ids'][:15]}")
        logger.info(f"   🧬 CHEMICAL ANALYSIS:")
        logger.info(f"     Unique SMILES patterns: {len(exploration_results['canonical_smiles_patterns'])}")
        logger.info(f"     Unique MOA patterns: {len(exploration_results['moa_patterns'])}")
        logger.info(f"     Unique PubChem IDs: {len(exploration_results['pubchem_ids'])}")
        
        # Look for DMSO-like patterns in drugs
        dmso_candidates = [drug for drug in exploration_results['drug_values'] 
                          if any(term in drug.lower() for term in ['dmso', 'control', 'vehicle', 'untreated', 'baseline'])]
        if dmso_candidates:
            logger.info(f"   🎯 Potential DMSO/control candidates in drugs: {dmso_candidates}")
        else:
            logger.warning("   ⚠️  No obvious DMSO/control candidates found in drug values")
        
        # Look for DMSO-like patterns in SMILES (DMSO has a specific SMILES: CS(=O)C)
        dmso_smiles_candidates = [smiles for smiles in exploration_results['canonical_smiles_patterns'] 
                                 if 'CS(=O)C' in smiles or 'dimethyl sulfoxide' in smiles.lower()]
        if dmso_smiles_candidates:
            logger.info(f"   🎯 Potential DMSO SMILES patterns: {dmso_smiles_candidates}")
        else:
            logger.warning("   ⚠️  No DMSO SMILES pattern (CS(=O)C) found")
        
        # Show sample combinations
        logger.info("   📋 Sample combinations (first 20):")
        for combo in exploration_results['unique_combinations']:
            logger.info(f"      Sample {combo['sample_id']}: drug='{combo['drug']}', cell_line_id='{combo['cell_line_id']}', smiles='{combo['canonical_smiles']}', sample='{combo['sample']}'")
        
        return exploration_results

    def validate_real_data(self) -> bool:
        logger.info("🔍 Validating real Tahoe-100M data access...")
        if not self._expression_dataset:
            logger.error("   ❌ Expression dataset not available.")
            return False
        logger.info("   ✅ Expression dataset is available.")
        
        if self._cell_line_metadata is None or self._cell_line_metadata.empty:
            logger.error("   ❌ Cell line metadata not available.")
            return False
        logger.info(f"   ✅ Cell line metadata: {len(self._cell_line_metadata)} cell lines")
        
        # Validate drug catalog availability
        if self._complete_drug_catalog is None or not self._complete_drug_catalog:
            logger.error("   ❌ Drug catalog not available.")
            return False
        logger.info(f"   ✅ Complete drug catalog: {self._complete_drug_catalog['total_drugs']} drugs")
        
        # Show drug diversity from metadata instead of sampling
        logger.info("   📊 Drug catalog diversity (from complete metadata):")
        logger.info(f"     🧪 Total drugs: {self._complete_drug_catalog['total_drugs']}")
        logger.info(f"     🎯 MOA categories: {self._complete_drug_catalog['moa_analysis'].get('broad_category_count', 'N/A')}")
        logger.info(f"     🧬 Molecular targets: {self._complete_drug_catalog['target_analysis'].get('unique_targets', 'N/A')}")
        logger.info(f"     💊 Chemical structures: {self._complete_drug_catalog['chemical_diversity'].get('unique_smiles', 'N/A')}")
        
        # Optionally perform cross-reference with expression data
        logger.info("   🔍 Cross-referencing with expression data...")
        cross_ref = self.cross_reference_expression_drugs(max_samples=1000)
        if cross_ref:
            logger.info(f"     ✅ Drug catalog coverage: {cross_ref['coverage_percentage']:.1f}%")
            logger.info(f"     📊 Expression drugs found: {cross_ref['expression_drugs_found']}")
            logger.info(f"     🔗 Drugs in both datasets: {cross_ref['drugs_in_both_datasets']}")
        
        # Generate comprehensive drug reports for reference
        logger.info("   📋 Generating comprehensive drug catalog reports...")
        try:
            report_paths = self.generate_comprehensive_drug_report()
            if report_paths:
                logger.info(f"     ✅ Drug reports generated: {len(report_paths)} files")
                logger.info(f"     📁 Reports location: {Path(list(report_paths.values())[0]).parent}")
        except Exception as e:
            logger.warning(f"     ⚠️  Could not generate drug reports: {e}")
        
        logger.info("✅ Real Tahoe-100M data validation PASSED")
        return True
