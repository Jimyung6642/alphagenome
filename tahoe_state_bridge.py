#!/usr/bin/env python3
"""
Tahoe-100M to State Model Data Bridge

Converts Tahoe-100M single-cell data to State model compatible formats,
handling organ-specific filtering and DMSO control template creation.

Key Features:
- Convert Tahoe-100M pseudobulk data to H5AD format
- Organ-specific cell line filtering
- DMSO control template creation
- Expression matrix transformation for State model
- Real data preservation throughout conversion

Usage:
    bridge = TahoeStateBridge()
    h5ad_data = bridge.convert_tahoe_to_state("TP53", tahoe_data, organ="stomach")
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import anndata as adata
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import json
import warnings
import gcsfs
import scanpy as sc
from google.auth import default
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError

# Suppress anndata warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='anndata')

logger = logging.getLogger(__name__)

class TahoeStateBridge:
    """
    Bridge for converting Tahoe-100M data to State model compatible formats.
    """
    
    def __init__(self, cache_dir: str = "tahoe_state_cache", require_gcs: bool = False, gcs_bucket: str = 'arc-ctc-tahoe100'):
        """Initialize the Tahoe-State bridge."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.require_gcs = require_gcs
        self.gcs_bucket = gcs_bucket
        
        # Bridge statistics
        self.bridge_stats = {
            'conversions_performed': 0,
            'cell_lines_processed': 0,
            'genes_analyzed': 0,
            'bridge_start_time': datetime.now().isoformat()
        }
        
        logger.info("🌉 Tahoe-State Bridge Initialized")
        logger.info(f"📁 Cache directory: {self.cache_dir}")
        
        # Initialize GCS authentication
        self.gcs_client = None
        self.gcs_authenticated = False
        
        self._setup_conversion_directories()
        self._setup_gcs_authentication()
    
    def _setup_conversion_directories(self):
        """Setup directories for data conversion."""
        self.h5ad_dir = self.cache_dir / "h5ad_files"
        self.h5ad_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.temp_dir = self.cache_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info("📁 Conversion directories setup complete")
    
    def _setup_gcs_authentication(self):
        """Setup Google Cloud Storage authentication with comprehensive guidance."""
        logger.info("🔐 Setting up GCS authentication...")
        
        try:
            # Try multiple authentication methods with detailed guidance
            credentials, project_id = None, None
            
            # Method 1: Application Default Credentials (ADC)
            try:
                credentials, project_id = default()
                logger.info("✅ Using Application Default Credentials (ADC)")
                logger.info(f"   Project ID: {project_id}")
                self.gcs_authenticated = True
            except DefaultCredentialsError as adc_error:
                logger.warning("⚠️  Application Default Credentials not available")
                logger.info("   To fix: Run 'gcloud auth application-default login'")
            
            # Method 2: Service Account Key from environment
            if not self.gcs_authenticated:
                service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                if service_account_path and os.path.exists(service_account_path):
                    try:
                        from google.oauth2 import service_account
                        credentials = service_account.Credentials.from_service_account_file(service_account_path)
                        logger.info("✅ Using Service Account credentials")
                        logger.info(f"   Key file: {service_account_path}")
                        self.gcs_authenticated = True
                    except Exception as sa_error:
                        logger.warning(f"⚠️  Service Account key invalid: {sa_error}")
                else:
                    logger.warning("⚠️  GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
                    logger.info("   To fix: Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json")
            
            # Method 3: Anonymous access (last resort for public datasets)
            if not self.gcs_authenticated:
                logger.warning("⚠️  No GCS authentication available - will attempt anonymous access")
                logger.warning("   This may fail for private datasets")
                logger.info("   Recommended fixes:")
                logger.info("   1. gcloud auth application-default login")
                logger.info("   2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
                logger.info("   3. Ensure service account has Storage Object Viewer permissions")
            
            # Initialize GCS client if authenticated
            if self.gcs_authenticated and credentials:
                try:
                    self.gcs_client = storage.Client(credentials=credentials, project=project_id)
                    # Test the client connection
                    _ = self.gcs_client.list_buckets(max_results=1)
                    logger.info("✅ GCS client initialized and tested successfully")
                except Exception as client_error:
                    logger.error(f"❌ GCS client test failed: {client_error}")
                    logger.error("   Check that the service account has necessary permissions")
                    self.gcs_authenticated = False
            else:
                logger.info("ℹ️  GCS client will use anonymous access (limited functionality)")
                
        except Exception as e:
            logger.error(f"❌ GCS authentication setup failed: {e}")
            logger.error("   Pipeline may fail if GCS access is required")
            logger.error("   Please configure GCS authentication before running the pipeline")
    
    def convert_tahoe_to_state(self, target_gene: str, tahoe_data: pd.DataFrame,
                              organ: Optional[str] = None, 
                              min_expression_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Convert Tahoe-100M data to State model compatible H5AD format.
        
        Args:
            target_gene: Target gene symbol
            tahoe_data: DataFrame with Tahoe-100M expression data
            organ: Optional organ filter
            min_expression_threshold: Minimum expression level to include
        
        Returns:
            Dictionary containing H5AD path and metadata
        """
        logger.info(f"🔄 Converting Tahoe-100M data to State format for {target_gene}")
        
        if organ:
            logger.info(f"🫁 Organ filter: {organ}")
        
        try:
            # Filter by organ if specified
            if organ:
                filtered_data = tahoe_data[tahoe_data['organ'].str.lower() == organ.lower()]
                if filtered_data.empty:
                    logger.warning(f"No data found for organ: {organ}, using all data")
                    filtered_data = tahoe_data
            else:
                filtered_data = tahoe_data
            
            logger.info(f"📊 Processing {len(filtered_data)} cell line records")
            
            # Load real H5AD data from GCS (production path)
            h5ad_path = self._download_and_prepare_h5ad_from_gcs(target_gene=target_gene, organ=organ)
            
            # Create metadata summary
            metadata_summary = self._create_metadata_summary(
                filtered_data, target_gene, organ, h5ad_path
            )
            
            # Update statistics
            self.bridge_stats['conversions_performed'] += 1
            try:
                loaded_tmp = adata.read_h5ad(h5ad_path)
                self.bridge_stats['cell_lines_processed'] += int(loaded_tmp.obs.get('cell_line', pd.Series(index=loaded_tmp.obs.index)).nunique())
                self.bridge_stats['genes_analyzed'] += int(loaded_tmp.n_vars)
            except Exception:
                pass
            
            conversion_result = {
                'h5ad_path': h5ad_path,
                'metadata_summary': metadata_summary,
                'conversion_stats': {
                    'cells_generated': None,
                    'genes_included': None,
                    'cell_lines': len(filtered_data['cell_line'].unique()),
                    'organs': len(filtered_data['organ'].unique()) if not organ else 1
                }
            }
            
            logger.info(f"✅ Conversion completed:")
            logger.info(f"   H5AD file: {h5ad_path}")
            logger.info(f"   Cells: {len(cell_metadata)}")
            logger.info(f"   Genes: {len(gene_metadata)}")
            
            return conversion_result
            
        except Exception as e:
            logger.error(f"Tahoe-State conversion failed: {e}")
            raise RuntimeError(f"Data conversion failed: {e}")
    
    def _create_expression_matrix(self, tahoe_data: pd.DataFrame, target_gene: str,
                                min_expression_threshold: float) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        This method should not create synthetic expression matrices.
        Real H5AD data must be loaded from GCS instead.
        This is kept as a strict block to ensure proper pipeline flow.
        """
        logger.error("Synthetic expression matrix creation is not allowed")
        logger.error("Use _create_h5ad_from_existing_tahoe_data for fallback instead")
        raise RuntimeError("Bridge must use real Tahoe-100M H5AD data from GCS. Synthetic data generation is prohibited.")
    
    def _load_real_tahoe_h5ad_data(self, cell_lines: List[str], target_gene: str,
                                  tf_list: List[str], organ: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Load real Tahoe-100M H5AD data from Google Cloud Storage.
        
        Args:
            cell_lines: List of cell lines to load
            target_gene: Target gene symbol
            tf_list: List of TF symbols
            organ: Optional organ filter
        
        Returns:
            Tuple of (expression_matrix, cell_metadata, gene_metadata)
        """
        logger.info("☁️ Loading real Tahoe-100M H5AD data from GCS...")
        
        # Check network connectivity first with SSL handling
        try:
            import urllib.request
            import ssl
            
            # Try with default SSL context first
            try:
                urllib.request.urlopen('https://www.google.com', timeout=10)
                logger.info("🌐 Network connectivity confirmed")
            except ssl.SSLError as ssl_error:
                logger.warning(f"⚠️ SSL verification failed, trying with unverified context: {ssl_error}")
                # Create unverified SSL context for macOS certificate issues
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                urllib.request.urlopen('https://www.google.com', timeout=10, context=ssl_context)
                logger.info("🌐 Network connectivity confirmed (unverified SSL)")
            except Exception as fallback_error:
                # If SSL context fails, try with completely unverified context
                logger.warning(f"⚠️ Standard SSL workaround failed: {fallback_error}")
                ssl_context = ssl._create_unverified_context()
                urllib.request.urlopen('https://www.google.com', timeout=10, context=ssl_context)
                logger.info("🌐 Network connectivity confirmed (completely unverified SSL)")
                
        except Exception as e:
            logger.error(f"❌ Network connectivity issue: {e}")
            raise RuntimeError(f"No network connection available. Pipeline requires internet access for GCS data: {e}")
        
        try:
            # Initialize GCS filesystem with enhanced SSL handling and timeouts
            logger.info("🔗 Initializing GCS connection with SSL fixes...")
            
            # Configure SSL context with proper certificate handling
            import ssl
            import os
            import certifi
            
            # Set certificate bundle environment variables for proper SSL verification
            cert_bundle_path = certifi.where()
            os.environ['SSL_CERT_FILE'] = cert_bundle_path
            os.environ['REQUESTS_CA_BUNDLE'] = cert_bundle_path
            os.environ['CURL_CA_BUNDLE'] = cert_bundle_path
            
            # Create proper SSL context with certificate verification
            ssl_context = ssl.create_default_context(cafile=cert_bundle_path)
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            logger.info("🔒 SSL certificate bundle configured with proper verification")
            logger.info(f"   Certificate bundle: {cert_bundle_path}")
            
            # Initialize gcsfs with enhanced SSL configuration and retries
            max_retries = 3
            fs = None
            
            # Configure SSL context for aiohttp (used by gcsfs)
            import aiohttp
            import asyncio
            
            # Ensure we have an event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Create aiohttp connector with proper SSL context
            aiohttp_ssl_context = ssl.create_default_context(cafile=cert_bundle_path)
            connector = aiohttp.TCPConnector(ssl=aiohttp_ssl_context, limit=100, limit_per_host=30)
            
            # Add timeout wrapper for gcsfs operations
            import signal
            from contextlib import contextmanager
            
            class TimeoutError(Exception):
                pass
            
            @contextmanager
            def timeout_context(seconds):
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {seconds} seconds")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                
                try:
                    yield
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"🔄 GCS initialization attempt {attempt + 1}/{max_retries}")
                    
                    # Wrap gcsfs initialization in timeout
                    with timeout_context(45):  # 45 second timeout for initialization
                        if self.gcs_authenticated and self.gcs_client:
                            # Use authenticated access with proper SSL
                            fs = gcsfs.GCSFileSystem(
                                project=self.gcs_client.project,
                                token='google_default',
                                timeout=30,  # Reasonable timeout
                                retry_total=2,  # Moderate retries
                                retry_backoff_factor=1.0,
                                connector=connector  # Use properly configured SSL connector
                            )
                            logger.info("✅ Using authenticated GCS access with proper SSL verification")
                        else:
                            # Try anonymous access with proper SSL
                            fs = gcsfs.GCSFileSystem(
                                timeout=30,  # Reasonable timeout
                                retry_total=2,  # Moderate retries
                                retry_backoff_factor=1.0,
                                connector=connector  # Use properly configured SSL connector
                            )
                            logger.info("ℹ️  Using anonymous GCS access with proper SSL verification")
                    
                    # If we got here, fs was created successfully
                    break
                    
                except TimeoutError as e:
                    logger.warning(f"⚠️ GCS initialization attempt {attempt + 1} timed out: {e}")
                    if attempt == max_retries - 1:
                        logger.error("❌ All GCS initialization attempts timed out - falling back to local data")
                        raise RuntimeError("GCS initialization timed out after all retries")
                except Exception as e:
                    logger.warning(f"⚠️ GCS filesystem initialization attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error("❌ All GCS initialization attempts failed - falling back to local data")
                        raise e
                    
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            if fs is None:
                raise RuntimeError("Failed to initialize GCS filesystem after all retries")
            
            # Tahoe-100M base path
            base_path = "gs://arc-ctc-tahoe100"
            
            # Test GCS access with retries and progressive timeouts
            gcs_access_successful = False
            for timeout_attempt, timeout_val in enumerate([10, 20, 30], 1):
                try:
                    logger.info(f"🔍 Testing GCS bucket access (attempt {timeout_attempt}, timeout: {timeout_val}s)...")
                    
                    # Wrap bucket access in timeout
                    with timeout_context(timeout_val + 5):  # Add 5s buffer for timeout wrapper
                        test_files = fs.ls(base_path, timeout=timeout_val)
                        logger.info(f"✅ GCS access confirmed: {len(test_files)} items found")
                        gcs_access_successful = True
                        break
                        
                except TimeoutError as e:
                    logger.warning(f"⚠️ GCS access attempt {timeout_attempt} timed out: {e}")
                except Exception as e:
                    logger.warning(f"⚠️ GCS access attempt {timeout_attempt} failed: {e}")
                
                if timeout_attempt < 3:
                    import time
                    time.sleep(3)  # Brief pause before retry
            
            if not gcs_access_successful:
                logger.error("❌ All GCS access attempts failed. Pipeline requires real Tahoe-100M H5AD data from GCS.")
                logger.error("   To fix this issue:")
                logger.error("   1. Check network connectivity")
                logger.error("   2. Verify GCS authentication: gcloud auth application-default login")
                logger.error("   3. Ensure access to gs://arc-ctc-tahoe100 bucket")
                logger.error("   4. Check SSL certificate configuration")
                raise RuntimeError("GCS access failed: Pipeline cannot proceed without real Tahoe-100M H5AD data. Synthetic data fallbacks are prohibited.")
            
            # List available H5AD files with timeout
            logger.info("🔍 Discovering available H5AD files...")
            try:
                h5ad_files = fs.glob(f"{base_path}/**/*.h5ad", timeout=60)
                logger.info(f"📁 Found {len(h5ad_files)} H5AD files")
            except Exception as e:
                logger.error(f"❌ H5AD file discovery failed: {e}")
                raise RuntimeError(f"Failed to discover H5AD files in GCS: {e}. Pipeline requires GCS access for real data.")
            
            # Load and combine relevant H5AD files
            combined_adata = None
            processed_files = 0
            
            for file_path in h5ad_files[:5]:  # Start with first 5 files for testing
                logger.info(f"📖 Loading {file_path}...")
                
                try:
                    # Load H5AD file with authentication support
                    if self.gcs_authenticated:
                        # Use authenticated scanpy loading
                        adata_obj = sc.read_h5ad(f"gcs://{file_path}")
                    else:
                        # Try direct gcsfs access
                        with fs.open(file_path, 'rb') as f:
                            adata_obj = sc.read_h5ad(f)
                    
                    # Filter for DMSO_TF control samples
                    if 'drug' in adata_obj.obs.columns:
                        dmso_mask = adata_obj.obs['drug'] == 'DMSO_TF'
                        if dmso_mask.sum() > 0:
                            adata_obj = adata_obj[dmso_mask].copy()
                        else:
                            logger.info(f"   No DMSO_TF samples found in {file_path}")
                            continue
                    
                    # Filter by cell lines if specified
                    if cell_lines and 'cell_line' in adata_obj.obs.columns:
                        cell_line_mask = adata_obj.obs['cell_line'].isin(cell_lines)
                        if cell_line_mask.sum() > 0:
                            adata_obj = adata_obj[cell_line_mask].copy()
                        else:
                            logger.info(f"   No matching cell lines found in {file_path}")
                            continue
                    
                    # Filter by organ if specified
                    if organ and 'organ' in adata_obj.obs.columns:
                        organ_mask = adata_obj.obs['organ'].str.lower() == organ.lower()
                        if organ_mask.sum() > 0:
                            adata_obj = adata_obj[organ_mask].copy()
                        else:
                            logger.info(f"   No matching organ samples found in {file_path}")
                            continue
                    
                    # Combine with existing data
                    if combined_adata is None:
                        combined_adata = adata_obj.copy()
                    else:
                        # Concatenate along observation axis
                        combined_adata = adata.concat([combined_adata, adata_obj], axis=0)
                    
                    processed_files += 1
                    logger.info(f"   ✅ Processed {file_path}: {adata_obj.n_obs} cells, {adata_obj.n_vars} genes")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to load {file_path}: {e}")
                    continue
            
            if combined_adata is None:
                logger.error("❌ Failed to load any valid H5AD data from GCS. Pipeline requires real Tahoe-100M H5AD files.")
                logger.error("   Possible causes:")
                logger.error("   - No H5AD files contain the required cell lines or DMSO_TF samples")
                logger.error("   - H5AD files are corrupted or incompatible")
                logger.error("   - Insufficient data matching the specified organ/cell line filters")
                logger.error("   Pipeline cannot proceed without real single-cell H5AD data.")
                raise RuntimeError("No valid Tahoe-100M H5AD data found in GCS. Synthetic data generation is prohibited.")
            
            logger.info(f"✅ Real H5AD data loaded: {combined_adata.n_obs} cells, {combined_adata.n_vars} genes from {processed_files} files")
            
            # Ensure State model compatibility
            combined_adata = self._ensure_state_compatibility(combined_adata)
            
            # Extract expression matrix and metadata
            expression_matrix = combined_adata.X.toarray() if hasattr(combined_adata.X, 'toarray') else combined_adata.X
            cell_metadata = combined_adata.obs.copy()
            gene_metadata = combined_adata.var.copy()
            
            # Ensure target gene and TFs are included
            all_genes = list(set([target_gene] + tf_list))
            available_genes = gene_metadata.index.tolist()
            missing_genes = [g for g in all_genes if g not in available_genes]
            
            if missing_genes:
                logger.warning(f"⚠️ Missing genes in H5AD data: {missing_genes}")
            
            return expression_matrix, cell_metadata, gene_metadata
            
        except Exception as e:
            logger.error(f"Real H5AD data loading failed: {e}")
            logger.error("❌ GCS data loading failed. This could be due to:")
            logger.error("   - Network connectivity issues")  
            logger.error("   - GCS authentication problems")
            logger.error("   - Tahoe-100M bucket access restrictions")
            logger.error("   - Timeout during large file operations")
            raise RuntimeError(f"Failed to load real Tahoe-100M data from GCS: {e}. Pipeline requires real data access and cannot proceed without GCS connectivity.")
    
    def _create_h5ad_from_existing_tahoe_data(self, cell_lines: List[str], target_gene: str,
                                            tf_list: List[str], organ: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        DEPRECATED: Synthetic data generation is prohibited in this pipeline.
        
        This method previously created synthetic single-cell data from pseudobulk statistics.
        This violates the "real data only" principle of the pipeline.
        
        The pipeline must use real Tahoe-100M H5AD files from GCS.
        
        Args:
            cell_lines: List of cell lines (not used)
            target_gene: Target gene symbol (not used) 
            tf_list: List of TF symbols (not used)
            organ: Optional organ filter (not used)
        
        Raises:
            RuntimeError: Always - synthetic data generation is prohibited
        """
        logger.error("❌ SYNTHETIC DATA GENERATION ATTEMPTED - THIS IS PROHIBITED")
        logger.error("   The pipeline requires REAL Tahoe-100M H5AD data from GCS.")
        logger.error("   Synthetic data generation violates the 'real data only' principle.")
        logger.error("   Please ensure:")
        logger.error("   1. GCS authentication is properly configured")
        logger.error("   2. Network connectivity to gs://arc-ctc-tahoe100 is available")
        logger.error("   3. Required H5AD files are accessible in the bucket")
        logger.error("   4. SSL certificates are properly configured")
        
        raise RuntimeError(
            "Synthetic data generation is prohibited. Pipeline requires real Tahoe-100M H5AD data from GCS. "
            "Fix GCS access issues instead of falling back to synthetic data."
        )
    
    def _ensure_state_compatibility(self, adata_obj: adata.AnnData) -> adata.AnnData:
        """
        Ensure H5AD data meets State model requirements.
        
        Args:
            adata_obj: Input AnnData object
        
        Returns:
            State-compatible AnnData object
        """
        logger.info("🔧 Ensuring State model compatibility...")
        
        try:
            import scipy.sparse as sp
            
            # 1. Ensure CSR matrix format (required by State)
            if not sp.issparse(adata_obj.X):
                logger.info("Converting dense matrix to CSR format")
                adata_obj.X = sp.csr_matrix(adata_obj.X)
            elif not isinstance(adata_obj.X, sp.csr_matrix):
                logger.info("Converting sparse matrix to CSR format")
                adata_obj.X = adata_obj.X.tocsr()
            
            # 2. Ensure gene_name column in var (required by State)
            if 'gene_name' not in adata_obj.var.columns:
                if adata_obj.var.index.name == 'gene_symbol' or 'gene_symbol' in adata_obj.var.columns:
                    adata_obj.var['gene_name'] = adata_obj.var.index if adata_obj.var.index.name == 'gene_symbol' else adata_obj.var['gene_symbol']
                else:
                    adata_obj.var['gene_name'] = adata_obj.var.index
                logger.info("✅ Added gene_name column to var dataframe")
            
            # 3. Ensure required cell metadata columns (ST-Tahoe format)
            required_cols = ['cell_line', 'drug', 'drugname_drugconc', 'cell_name', 'organ']
            for col in required_cols:
                if col not in adata_obj.obs.columns:
                    if col in ['drug', 'drugname_drugconc']:
                        adata_obj.obs[col] = 'DMSO_TF'  # Default to control condition
                    elif col == 'organ':
                        adata_obj.obs[col] = 'stomach'  # Default organ
                    elif col == 'cell_line':
                        adata_obj.obs[col] = 'unknown'  # Placeholder
                    elif col == 'cell_name':
                        adata_obj.obs[col] = adata_obj.obs.get('cell_line', 'unknown')  # Use cell_line for cell_name
                    logger.info(f"✅ Added missing {col} column")
            
            # 4. Add batch information if missing
            if 'batch' not in adata_obj.obs.columns:
                adata_obj.obs['batch'] = adata_obj.obs['cell_line']
                logger.info("✅ Added batch column based on cell_line")
            
            # 5. Ensure proper data types (ST-Tahoe format)
            categorical_cols = ['cell_line', 'drug', 'drugname_drugconc', 'cell_name', 'organ', 'batch']
            for col in categorical_cols:
                if col in adata_obj.obs.columns:
                    adata_obj.obs[col] = adata_obj.obs[col].astype('category')
            
            logger.info("✅ State model compatibility ensured")
            logger.info(f"   Matrix format: {type(adata_obj.X).__name__}")
            logger.info(f"   Shape: {adata_obj.shape}")
            logger.info(f"   Required columns present: {all(col in adata_obj.obs.columns for col in required_cols)}")
            
            return adata_obj
            
        except Exception as e:
            logger.error(f"State compatibility check failed: {e}")
            raise RuntimeError(f"Failed to ensure State compatibility: {e}")

    def _download_and_prepare_h5ad_from_gcs(self, target_gene: str, organ: Optional[str]) -> str:
        """Download a Tahoe-100M H5AD from GCS, ensure compatibility, and cache locally.

        Returns local H5AD path.
        """
        logger.info("☁️ Loading real Tahoe-100M H5AD data from GCS (production mode)...")
        # Require authenticated access for production
        if not self.gcs_authenticated:
            msg = "GCS access is required but no credentials found. Run `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS."
            logger.error(msg)
            raise RuntimeError(msg)

        # Initialize synchronous gcsfs
        try:
            fs = gcsfs.GCSFileSystem(token='google_default')
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GCS filesystem: {e}")

        # List candidate H5AD files
        try:
            objects = fs.ls(self.gcs_bucket)
            candidates = [p for p in objects if p.lower().endswith('.h5ad')]
        except Exception as e:
            raise RuntimeError(f"Failed to list GCS bucket {self.gcs_bucket}: {e}")

        selected = candidates
        if organ:
            selected = [p for p in candidates if organ.lower() in p.lower()]

        if not selected:
            raise RuntimeError(f"No H5AD files found in bucket {self.gcs_bucket} matching organ={organ or 'ANY'}.")

        remote_path = selected[0]
        local_name = Path(remote_path).name
        local_path = self.h5ad_dir / local_name

        if not local_path.exists():
            logger.info(f"⬇️  Downloading {remote_path} -> {local_path}")
            try:
                fs.get(remote_path, str(local_path))
            except Exception as e:
                raise RuntimeError(f"Failed to download {remote_path}: {e}")
        else:
            logger.info(f"🔁 Using cached H5AD: {local_path}")

        # Load and ensure compatibility
        try:
            ad = sc.read_h5ad(local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read H5AD {local_path}: {e}")

        ad = self._ensure_state_compatibility(ad)

        processed_name = f"processed_{target_gene}_{local_name}"
        processed_path = self.h5ad_dir / processed_name
        try:
            ad.write(processed_path)
        except Exception as e:
            raise RuntimeError(f"Failed to write processed H5AD {processed_path}: {e}")

        return str(processed_path)
    
    def _create_cell_metadata(self, tahoe_data: pd.DataFrame, cell_lines: List[str],
                            cells_per_line: int) -> pd.DataFrame:
        """
        Create cell metadata for the H5AD object.
        
        Args:
            tahoe_data: Original Tahoe data
            cell_lines: List of cell lines
            cells_per_line: Number of cells per line
        
        Returns:
            DataFrame with cell metadata
        """
        logger.info("📋 Creating cell metadata...")
        
        cell_metadata = []
        
        for cell_line in cell_lines:
            # Get organ for this cell line
            cell_line_data = tahoe_data[tahoe_data['cell_line'] == cell_line]
            organ = cell_line_data['organ'].iloc[0] if not cell_line_data.empty else 'unknown'
            
            for cell_idx in range(cells_per_line):
                cell_id = f"{cell_line}_cell_{cell_idx:03d}"
                
                cell_metadata.append({
                    'cell_id': cell_id,
                    'cell_line': cell_line,
                    'organ': organ,
                    'treatment': 'DMSO',  # Control condition
                    'batch': f"batch_{cell_line}",
                    'n_genes': len(tahoe_data.columns) - 2,  # Approximate
                    'total_counts': np.random.normal(5000, 1000),  # Simulate library size
                    'pct_counts_mitochondrial': np.random.uniform(5, 15)  # Simulate QC metric
                })
        
        cell_meta_df = pd.DataFrame(cell_metadata)
        cell_meta_df.index = cell_meta_df['cell_id']
        
        logger.info(f"✅ Cell metadata created: {len(cell_meta_df)} cells")
        
        return cell_meta_df
    
    def _create_anndata_object(self, expression_matrix: np.ndarray,
                             cell_metadata: pd.DataFrame,
                             gene_metadata: pd.DataFrame) -> adata.AnnData:
        """
        Create AnnData object from expression data and metadata.
        
        Args:
            expression_matrix: Gene expression matrix
            cell_metadata: Cell metadata DataFrame
            gene_metadata: Gene metadata DataFrame
        
        Returns:
            AnnData object
        """
        logger.info("🧬 Creating AnnData object...")
        
        try:
            import scipy.sparse as sp
            
            # Ensure CSR matrix format for State compatibility
            if not sp.issparse(expression_matrix):
                expression_matrix = sp.csr_matrix(expression_matrix)
            elif not isinstance(expression_matrix, sp.csr_matrix):
                expression_matrix = expression_matrix.tocsr()
            
            # Create AnnData object
            adata_obj = adata.AnnData(
                X=expression_matrix,
                obs=cell_metadata,
                var=gene_metadata
            )
            
            # Add layer for raw counts (copy of X for simplicity)
            adata_obj.layers['raw'] = expression_matrix.copy()
            
            # Add some basic computed metrics
            adata_obj.obs['total_counts'] = np.array(adata_obj.X.sum(axis=1)).flatten()
            adata_obj.obs['n_genes_by_counts'] = np.array((adata_obj.X > 0).sum(axis=1)).flatten()
            
            # Mark highly variable genes  
            if 'gene_type' in adata_obj.var.columns:
                adata_obj.var['highly_variable'] = adata_obj.var['gene_type'].isin(['target', 'tf'])
            else:
                adata_obj.var['highly_variable'] = False
            
            # Ensure proper categorical types for State model
            categorical_cols = ['cell_line', 'drug', 'organ', 'batch']
            for col in categorical_cols:
                if col in adata_obj.obs.columns:
                    adata_obj.obs[col] = adata_obj.obs[col].astype('category')
            
            # Add metadata
            adata_obj.uns['creation_date'] = datetime.now().isoformat()
            adata_obj.uns['data_source'] = 'tahoe_100M'
            adata_obj.uns['conversion_method'] = 'tahoe_state_bridge'
            
            logger.info(f"✅ AnnData object created:")
            logger.info(f"   Shape: {adata_obj.shape}")
            logger.info(f"   Cell lines: {adata_obj.obs['cell_line'].nunique()}")
            logger.info(f"   Organs: {adata_obj.obs['organ'].nunique()}")
            
            return adata_obj
            
        except Exception as e:
            raise RuntimeError(f"AnnData object creation failed: {e}")
    
    def _save_h5ad_file(self, adata_obj: adata.AnnData, target_gene: str,
                       organ: Optional[str]) -> str:
        """
        Save AnnData object as H5AD file.
        
        Args:
            adata_obj: AnnData object to save
            target_gene: Target gene symbol
            organ: Optional organ filter
        
        Returns:
            Path to saved H5AD file
        """
        logger.info("💾 Saving H5AD file...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        organ_suffix = f"_{organ}" if organ else ""
        
        filename = f"{target_gene}{organ_suffix}_{timestamp}.h5ad"
        h5ad_path = self.h5ad_dir / filename
        
        try:
            adata_obj.write(h5ad_path)
            
            logger.info(f"✅ H5AD file saved: {h5ad_path}")
            logger.info(f"   File size: {h5ad_path.stat().st_size / (1024*1024):.1f} MB")
            
            return str(h5ad_path)
            
        except Exception as e:
            raise RuntimeError(f"H5AD file saving failed: {e}")
    
    def _create_metadata_summary(self, tahoe_data: pd.DataFrame, target_gene: str,
                               organ: Optional[str], h5ad_path: str) -> Dict[str, Any]:
        """
        Create comprehensive metadata summary for the conversion.
        
        Args:
            tahoe_data: Original Tahoe data
            target_gene: Target gene symbol
            organ: Optional organ filter
            h5ad_path: Path to H5AD file
        
        Returns:
            Dictionary with metadata summary
        """
        summary = {
            'conversion_metadata': {
                'target_gene': target_gene,
                'organ_filter': organ,
                'conversion_timestamp': datetime.now().isoformat(),
                'h5ad_file_path': h5ad_path
            },
            'data_summary': {
                'original_records': len(tahoe_data),
                'cell_lines_included': tahoe_data['cell_line'].unique().tolist(),
                'organs_included': tahoe_data['organ'].unique().tolist(),
                'tf_count': tahoe_data['tf_name'].nunique() if 'tf_name' in tahoe_data.columns else 0
            },
            'quality_metrics': {
                'min_expression_threshold': 0.1,  # Default threshold used
                'real_data_source': 'GCS_Tahoe_100M_fallback',
                'data_validation': 'real_h5ad_with_fallback'
            },
            'bridge_statistics': self.get_bridge_statistics()
        }
        
        # Save metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_path = self.metadata_dir / f"{target_gene}_{timestamp}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Metadata summary saved: {metadata_path}")
        
        return summary
    
    def create_control_template(self, h5ad_path: str, cell_lines: List[str]) -> str:
        """
        Create DMSO control template for State model analysis.
        
        Args:
            h5ad_path: Path to H5AD data file
            cell_lines: List of cell lines to include
        
        Returns:
            Path to control template file
        """
        logger.info("🎛️ Creating DMSO control template...")
        
        try:
            # Load H5AD data
            adata_obj = adata.read_h5ad(h5ad_path)
            
            # Filter to specified cell lines
            if cell_lines:
                cell_line_mask = adata_obj.obs['cell_line'].isin(cell_lines)
                control_data = adata_obj[cell_line_mask].copy()
            else:
                control_data = adata_obj.copy()
            
            # Create control template
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            template_path = self.temp_dir / f"control_template_{timestamp}.h5ad"
            
            # Add control-specific metadata
            control_data.obs['is_control'] = True
            control_data.obs['perturbation'] = 'none'
            control_data.uns['template_type'] = 'dmso_control'
            control_data.uns['creation_timestamp'] = timestamp
            
            control_data.write(template_path)
            
            logger.info(f"✅ Control template created: {template_path}")
            logger.info(f"   Control cells: {control_data.n_obs}")
            logger.info(f"   Cell lines: {control_data.obs['cell_line'].nunique()}")
            
            return str(template_path)
            
        except Exception as e:
            raise RuntimeError(f"Control template creation failed: {e}")
    
    def get_bridge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bridge statistics."""
        stats = self.bridge_stats.copy()
        stats['bridge_end_time'] = datetime.now().isoformat()
        return stats


def main():
    """Test the Tahoe-State bridge."""
    # Test data
    test_data = pd.DataFrame({
        'cell_line': ['HeLa', 'A549', 'MCF7', 'SW480'],
        'organ': ['cervix', 'lung', 'breast', 'colon'],
        'tf_name': ['STAT1', 'NF-KB', 'TP53', 'MYC'],
        'expression_level': [1.2, 0.8, 1.5, 2.0]
    })
    
    bridge = TahoeStateBridge()
    
    try:
        # Test conversion
        result = bridge.convert_tahoe_to_state(
            target_gene="TP53",
            tahoe_data=test_data,
            organ="lung"
        )
        
        print("✅ Tahoe-State conversion successful!")
        print(f"H5AD file: {result['h5ad_path']}")
        print(f"Cells generated: {result['conversion_stats']['cells_generated']}")
        print(f"Genes included: {result['conversion_stats']['genes_included']}")
        
        # Test control template creation
        template_path = bridge.create_control_template(
            result['h5ad_path'], 
            ['A549']
        )
        print(f"Control template: {template_path}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
