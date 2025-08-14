#!/usr/bin/env python3
"""
AlphaGenome + State Model Perturbation Analysis Pipeline

A comprehensive genomic perturbation analysis pipeline that integrates:
1. AlphaGenome: TF binding predictions for target genes
2. Tahoe-100M: Baseline TF expression analysis (DMSO controls)
3. State Model: Perturbation effect predictions
4. DEG Analysis: Target gene changes and genome-wide differential expression

Workflow:
1. Predict TFs regulating target gene using AlphaGenome
2. Analyze baseline TF expression in Tahoe-100M (organ-focused)
3. Predict perturbation effects using State model
4. Analyze target gene expression changes
5. Identify genome-wide differentially expressed genes (DEGs)

Usage Examples:
    # Basic perturbation analysis
    python main.py --gene TP53 --organ stomach
    
    # Multiple genes with organ focus
    python main.py --genes TP53,BRCA1 --organ lung
    
    # Custom perturbation parameters
    python main.py --genes CLDN18 --organ stomach --perturbation-strength 0.8
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import warnings

# Suppress protobuf version warnings that are not critical
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')
warnings.filterwarnings('ignore', message='.*protobuf gencode version.*')

# Set up logging with timestamped log file in output directory
def setup_logging(output_dir: str = "output", verbose: bool = False):
    """Set up logging with timestamped log file in output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"perturbation_analysis_{timestamp}.log"
    
    handlers = [logging.FileHandler(log_file)]
    
    # Only add console handler if verbose mode is enabled
    if verbose:
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

# Import our pipeline components
try:
    from standardized_genomic_analyzer import StandardizedGenomicAnalyzer
    BASELINE_ANALYZER_AVAILABLE = True
except ImportError:
    BASELINE_ANALYZER_AVAILABLE = False
    logger.error("StandardizedGenomicAnalyzer not available")

try:
    from state_perturbation_engine import StatePerturbationEngine
    STATE_ENGINE_AVAILABLE = True
except ImportError:
    STATE_ENGINE_AVAILABLE = False
    logger.error("StatePerturbationEngine not available")

try:
    from tahoe_state_bridge import TahoeStateBridge
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    logger.error("TahoeStateBridge not available")

try:
    from deg_analyzer import DEGAnalyzer
    DEG_ANALYZER_AVAILABLE = True
except ImportError:
    DEG_ANALYZER_AVAILABLE = False
    logger.error("DEGAnalyzer not available")

try:
    from download_config import DownloadConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logger.warning("DownloadConfig not available - using default configuration")


class PerturbationAnalysisPipeline:
    """
    Main pipeline orchestrator for comprehensive perturbation analysis.
    
    Integrates AlphaGenome TF predictions, Tahoe-100M baseline expression,
    State model perturbation predictions, and differential expression analysis.
    """
    
    def __init__(self, output_dir: str = "output", config_overrides: Optional[dict] = None):
        """Initialize the perturbation analysis pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config_overrides = config_overrides or {}
        
        # Pipeline components
        self.baseline_analyzer = None
        self.state_engine = None
        self.bridge = None
        self.deg_analyzer = None
        
        # Pipeline statistics
        self.pipeline_stats = {
            'genes_analyzed': 0,
            'perturbations_completed': 0,
            'pipeline_start_time': datetime.now().isoformat(),
            'successful_analyses': 0,
            'failed_analyses': 0
        }
        
        logger.info("🚀 Perturbation Analysis Pipeline Initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        self._validate_dependencies()
        self._initialize_components()
    
    def _validate_dependencies(self):
        """Validate all required dependencies are available."""
        logger.info("🔍 Validating pipeline dependencies...")
        
        missing_deps = []
        
        if not BASELINE_ANALYZER_AVAILABLE:
            missing_deps.append("StandardizedGenomicAnalyzer (baseline analysis)")
        if not STATE_ENGINE_AVAILABLE:
            missing_deps.append("StatePerturbationEngine (State model)")
        if not BRIDGE_AVAILABLE:
            missing_deps.append("TahoeStateBridge (data conversion)")
        if not DEG_ANALYZER_AVAILABLE:
            missing_deps.append("DEGAnalyzer (differential expression)")
        
        if missing_deps:
            error_msg = f"Missing dependencies: {', '.join(missing_deps)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("✅ All pipeline dependencies validated")
        
        # Validate external tools and connectivity
        self._validate_external_tools()
        
        # Comprehensive pre-execution validation
        self._validate_pipeline_prerequisites()
    
    def _validate_external_tools(self):
        """Validate external tools and APIs."""
        logger.info("🔍 Validating external tools...")
        
        # Check AlphaGenome API key
        try:
            config_file = Path("config.env")
            api_key_found = False
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    content = f.read()
                    if 'ALPHAGENOME_API_KEY' in content and '=' in content:
                        api_key_found = True
            
            if os.environ.get('ALPHAGENOME_API_KEY'):
                api_key_found = True
                
            if not api_key_found:
                raise RuntimeError(
                    "AlphaGenome API key not found. Create config.env with: ALPHAGENOME_API_KEY=your_key_here"
                )
                
            logger.info("✅ AlphaGenome API key found")
        except Exception as e:
            raise RuntimeError(f"AlphaGenome API validation failed: {e}")
        
        # Validate State model installation and checkpoints
        try:
            import subprocess
            
            # SE-600M model not used in zero-shot TF perturbation workflow
            
            # Validate State CLI
            result = subprocess.run(['uv', 'tool', 'run', '--from', 'arc-state', 'state', '--help'], 
                                 capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error("❌ State CLI not working - pipeline cannot proceed")
                raise RuntimeError("State model CLI validation failed. Pipeline requires working State installation.")
            else:
                logger.info("✅ State model CLI validated")
                
        except FileNotFoundError:
            logger.error("❌ State model not found")
            raise RuntimeError("State model not installed. Please install State model before running pipeline.")
        except Exception as e:
            logger.error(f"❌ State model validation failed: {e}")
            raise RuntimeError(f"State model validation failed: {e}. Pipeline requires working State installation.")
    
    def _validate_pipeline_prerequisites(self):
        """Comprehensive validation of all pipeline prerequisites before execution."""
        logger.info("🔍 Performing comprehensive pipeline validation...")
        
        validation_errors = []
        validation_warnings = []
        
        # 1. Network connectivity validation (non-blocking)
        try:
            logger.info("🌐 Testing network connectivity...")
            import urllib.request
            import ssl
            
            # Test general internet connectivity with multiple fallbacks
            connectivity_ok = False
            test_urls = ['https://www.google.com', 'https://8.8.8.8', 'https://1.1.1.1']
            
            for test_url in test_urls:
                try:
                    urllib.request.urlopen(test_url, timeout=5)
                    logger.info(f"✅ Internet connectivity confirmed via {test_url}")
                    connectivity_ok = True
                    break
                except Exception:
                    continue
            
            if not connectivity_ok:
                validation_warnings.append("Internet connectivity may be limited - pipeline will try to proceed")
                logger.warning("⚠️ Could not confirm internet connectivity, but pipeline will attempt to proceed")
        except Exception as e:
            validation_warnings.append(f"Network validation failed: {e} - pipeline will try to proceed")
        
        # 2. GCS connectivity validation
        try:
            logger.info("☁️  Testing GCS connectivity...")
            import gcsfs
            
            # Test GCS bucket access
            try:
                fs = gcsfs.GCSFileSystem(timeout=10)
                test_files = fs.ls('gs://arc-ctc-tahoe100', timeout=10)
                if test_files:
                    logger.info("✅ GCS bucket access confirmed")
                else:
                    validation_warnings.append("GCS bucket appears empty or inaccessible")
            except Exception as gcs_error:
                validation_warnings.append(f"GCS access issue: {gcs_error}. Pipeline may use fallback authentication.")
        except Exception as e:
            validation_warnings.append(f"GCS validation failed: {e}")
        
        # 3. AlphaGenome API validation
        try:
            logger.info("🧬 Testing AlphaGenome API connectivity...")
            from alphagenome import colab_utils
            from alphagenome.models import dna_client
            
            # Test API key and connection
            api_key = colab_utils.get_api_key('ALPHAGENOME_API_KEY')
            test_client = dna_client.create(api_key)
            metadata = test_client.output_metadata()
            
            if metadata and hasattr(metadata, 'chip_tf'):
                logger.info("✅ AlphaGenome API connection confirmed")
                logger.info(f"   Available ontologies: {len(metadata.chip_tf)}")
            else:
                validation_errors.append("AlphaGenome API returned invalid metadata")
        except Exception as e:
            validation_errors.append(f"AlphaGenome API validation failed: {e}")
        
        # 4. State model files validation with auto-download
        try:
            logger.info("🤖 Validating State model files...")
            from pathlib import Path
            
            # SE-600M model not required for zero-shot TF perturbation workflow
            
            # Check ST model with auto-download
            st_model_path = Path("models/state/ST-Tahoe")
            st_ready = False
            
            if st_model_path.exists():
                required_st_files = ['config.yaml', 'final_from_preprint.ckpt', 'pert_onehot_map.pt']
                missing_files = [f for f in required_st_files if not (st_model_path / f).exists()]
                if not missing_files:
                    logger.info("✅ ST model files validated")
                    st_ready = True
                else:
                    logger.info(f"⬇️ ST model missing files {missing_files}, attempting download...")
            else:
                logger.info("⬇️ ST model directory not found, creating and downloading...")
            
            if not st_ready:
                try:
                    st_download_success = self._download_st_tahoe_model(st_model_path)
                    if st_download_success:
                        logger.info("✅ ST model downloaded successfully")
                        st_ready = True
                    else:
                        validation_errors.append("ST-Tahoe model auto-download failed - zero-shot perturbation analysis will not work")
                except Exception as download_error:
                    validation_errors.append(f"ST-Tahoe model auto-download failed: {download_error} - State analysis disabled")
            
            # ST-Tahoe model summary for zero-shot workflow
            if st_ready:
                logger.info("✅ ST-Tahoe model validated and ready for zero-shot TF perturbation")
            else:
                validation_errors.append("ST-Tahoe model not available - zero-shot perturbation analysis disabled")
                
        except Exception as e:
            validation_warnings.append(f"State model validation failed: {e}")
        
        # 5. Cache directory permissions
        try:
            logger.info("📁 Validating cache directories...")
            from pathlib import Path
            
            cache_dirs = ['comprehensive_cache', 'tahoe_cache', 'output']
            for cache_dir in cache_dirs:
                cache_path = Path(cache_dir)
                cache_path.mkdir(exist_ok=True)
                
                # Test write permissions
                test_file = cache_path / "test_write.tmp"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    logger.info(f"✅ Cache directory {cache_dir} is writable")
                except Exception as write_error:
                    validation_errors.append(f"Cache directory {cache_dir} not writable: {write_error}")
        except Exception as e:
            validation_warnings.append(f"Cache validation failed: {e}")
        
        # 6. Memory and disk space validation
        try:
            logger.info("💾 Checking system resources...")
            import psutil
            
            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            if available_gb < 4:
                validation_warnings.append(f"Low available memory: {available_gb:.1f}GB (recommended: 8GB+)")
            else:
                logger.info(f"✅ Available memory: {available_gb:.1f}GB")
            
            # Check available disk space
            disk = psutil.disk_usage('.')
            available_disk_gb = disk.free / (1024**3)
            if available_disk_gb < 10:
                validation_warnings.append(f"Low disk space: {available_disk_gb:.1f}GB (recommended: 50GB+)")
            else:
                logger.info(f"✅ Available disk space: {available_disk_gb:.1f}GB")
        except ImportError:
            validation_warnings.append("psutil not available - could not check system resources")
        except Exception as e:
            validation_warnings.append(f"Resource validation failed: {e}")
        
        # Report validation results
        logger.info("=" * 60)
        logger.info("📋 PIPELINE VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        if validation_errors:
            logger.error(f"❌ {len(validation_errors)} CRITICAL ISSUES FOUND:")
            for i, error in enumerate(validation_errors, 1):
                logger.error(f"   {i}. {error}")
            logger.error("=" * 60)
            logger.error("❌ PIPELINE CANNOT START - Fix critical issues above")
            raise RuntimeError(f"Pipeline validation failed: {len(validation_errors)} critical issues found. See logs for details.")
        
        if validation_warnings:
            logger.warning(f"⚠️  {len(validation_warnings)} warnings found:")
            for i, warning in enumerate(validation_warnings, 1):
                logger.warning(f"   {i}. {warning}")
            logger.warning("   Pipeline may experience issues - consider addressing warnings")
        
        if not validation_errors and not validation_warnings:
            logger.info("✅ ALL VALIDATIONS PASSED - Pipeline ready to start")
        elif not validation_errors:
            logger.info("✅ CRITICAL VALIDATIONS PASSED - Pipeline can start with warnings")
        
        logger.info("=" * 60)
    
    # SE-600M model download function removed - not needed for zero-shot TF perturbation workflow
    
    def _download_st_tahoe_model(self, st_model_path: Path) -> bool:
        """Download ST-Tahoe model files from Hugging Face."""
        logger.info("⬇️ Downloading ST-Tahoe model files from Hugging Face...")
        
        try:
            import urllib.request
            import shutil
            import ssl
            
            # Create directory
            st_model_path.mkdir(parents=True, exist_ok=True)
            
            # Hugging Face URLs for ST-Tahoe model files
            base_url = "https://huggingface.co/arcinstitute/ST-Tahoe/resolve/main"
            files_to_download = {
                'config.yaml': f"{base_url}/config.yaml",
                'final_from_preprint.ckpt': f"{base_url}/final_from_preprint.ckpt", 
                'pert_onehot_map.pt': f"{base_url}/pert_onehot_map.pt",
                'cell_type_onehot_map.pkl': f"{base_url}/cell_type_onehot_map.pkl"  # Additional file that may be needed
            }
            
            downloaded_files = []
            failed_downloads = []
            
            for filename, url in files_to_download.items():
                file_path = st_model_path / filename
                
                # Skip if file already exists and is not empty
                if file_path.exists() and file_path.stat().st_size > 0:
                    logger.info(f"✅ File already exists: {filename}")
                    downloaded_files.append(filename)
                    continue
                
                logger.info(f"⬇️ Downloading {filename}...")
                
                try:
                    # Create SSL context that handles certificates properly
                    ssl_context = ssl.create_default_context()
                    
                    # Download with retries and SSL handling
                    for attempt in range(3):
                        try:
                            # Try with proper SSL context first
                            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
                            urllib.request.install_opener(opener)
                            urllib.request.urlretrieve(url, file_path)
                            
                            # Verify download
                            if file_path.exists() and file_path.stat().st_size > 0:
                                size_mb = file_path.stat().st_size / (1024**2)
                                logger.info(f"✅ Downloaded {filename}: {size_mb:.1f}MB")
                                downloaded_files.append(filename)
                                break
                            else:
                                raise Exception("Downloaded file is empty or corrupt")
                                
                        except Exception as download_error:
                            if "CERTIFICATE_VERIFY_FAILED" in str(download_error) and attempt < 2:
                                # Try with disabled SSL verification as fallback
                                logger.warning(f"⚠️ SSL certificate verification failed, trying with unverified SSL...")
                                try:
                                    import ssl
                                    ssl_context = ssl._create_unverified_context()
                                    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
                                    urllib.request.install_opener(opener)
                                    urllib.request.urlretrieve(url, file_path)
                                    
                                    # Verify download
                                    if file_path.exists() and file_path.stat().st_size > 0:
                                        size_mb = file_path.stat().st_size / (1024**2)
                                        logger.info(f"✅ Downloaded {filename}: {size_mb:.1f}MB (unverified SSL)")
                                        downloaded_files.append(filename)
                                        break
                                    else:
                                        raise Exception("Downloaded file is empty or corrupt")
                                except Exception as ssl_fallback_error:
                                    logger.warning(f"⚠️ SSL fallback also failed: {ssl_fallback_error}")
                                    if attempt == 2:  # Last attempt
                                        logger.error(f"❌ Failed to download {filename} after all attempts: {download_error}")
                                        failed_downloads.append(filename)
                            elif attempt == 2:  # Last attempt
                                logger.error(f"❌ Failed to download {filename} after 3 attempts: {download_error}")
                                failed_downloads.append(filename)
                            else:
                                logger.warning(f"⚠️ Download attempt {attempt + 1} failed for {filename}, retrying...")
                                
                except Exception as file_error:
                    logger.error(f"❌ Download failed for {filename}: {file_error}")
                    failed_downloads.append(filename)
            
            # Check if critical files were downloaded
            required_files = ['config.yaml', 'final_from_preprint.ckpt', 'pert_onehot_map.pt']
            missing_required = [f for f in required_files if f not in downloaded_files]
            
            if not missing_required:
                logger.info(f"✅ ST-Tahoe model download completed: {len(downloaded_files)} files")
                
                # Create success readme
                readme_path = st_model_path / "DOWNLOAD_SUCCESS.md"
                readme_content = f"""# ST-Tahoe Model Download Success

Downloaded on: {datetime.now().isoformat()}
Source: Hugging Face (arcinstitute/ST-Tahoe)

## Files Downloaded:
{chr(10).join(f'- {f}' for f in downloaded_files)}

Model is ready for zero-shot TF perturbation prediction.
"""
                
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                
                return True
            else:
                logger.error(f"❌ Critical ST-Tahoe files missing: {missing_required}")
                
                # Create manual download instructions
                readme_path = st_model_path / "MANUAL_DOWNLOAD_NEEDED.md"
                readme_content = f"""# ST-Tahoe Manual Download Required

Auto-download failed for critical files: {missing_required}

## Manual Download Steps:
1. Visit: https://huggingface.co/arcinstitute/ST-Tahoe
2. Download the missing files:
{chr(10).join(f'   - {f}' for f in missing_required)}
3. Place files in: {st_model_path}
4. Re-run the pipeline

## Alternative using git-lfs:
```bash
cd {st_model_path.parent}
git lfs clone https://huggingface.co/arcinstitute/ST-Tahoe
```

Partially downloaded files: {downloaded_files}
Failed downloads: {failed_downloads}
"""
                
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                
                return False
                
        except Exception as e:
            logger.error(f"ST-Tahoe download setup failed: {e}")
            return False
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("🔧 Initializing pipeline components...")
        
        try:
            # Build configuration
            base_config = {
                'max_ontologies': None,
                'max_tfs_per_ontology': None,
                'cache_dir': 'comprehensive_cache',
                'output_dir': str(self.output_dir),
                'real_data_only': True,
                'comprehensive_tf_discovery': True,
                'strict_data_validation': True,
                'perturbation_strength': 0.5,
                'deg_p_threshold': 0.05,
                'deg_log2fc_threshold': 0.5
            }
            
            # Apply user configuration overrides
            final_config = {**base_config, **self.config_overrides}
            
            # Ensure critical parameters are properly set
            if 'max_cell_lines' not in final_config:
                final_config['max_cell_lines'] = None
            
            logger.info("🔧 Pipeline Configuration:")
            logger.info(f"   Perturbation strength: {final_config['perturbation_strength']}")
            logger.info(f"   DEG p-value threshold: {final_config['deg_p_threshold']}")
            logger.info(f"   DEG log2FC threshold: {final_config['deg_log2fc_threshold']}")
            
            # Initialize components with enhanced real data support
            self.baseline_analyzer = StandardizedGenomicAnalyzer(config=final_config)
            logger.info("✅ Baseline analyzer initialized")
            
            # Pass model paths to State engine
            state_config = final_config.copy()
            # SE-600M model not needed for zero-shot TF perturbation workflow
            
            self.state_engine = StatePerturbationEngine(
                cache_dir=str(self.output_dir / "state_cache"),
                config_overrides=state_config
            )
            logger.info("✅ State perturbation engine initialized with real model support")
            
            self.bridge = TahoeStateBridge(
                cache_dir=str(Path(self.config_overrides.get('tahoe_cache_dir', self.output_dir / "bridge_cache"))),
                require_gcs=bool(self.config_overrides.get('require_gcs', False)),
                gcs_bucket=str(self.config_overrides.get('gcs_bucket', 'arc-ctc-tahoe100'))
            )
            logger.info("✅ Tahoe-State bridge initialized with GCS access")
            
            self.deg_analyzer = DEGAnalyzer(
                output_dir=str(self.output_dir / "deg_analysis")
            )
            logger.info("✅ DEG analyzer initialized")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    def analyze_gene_perturbation(self, gene_symbol: str, 
                                organ: Optional[str] = None,
                                perturbation_strength: float = 0.5,
                                top_n_percent: Optional[int] = None,
                                tf_organ: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive perturbation analysis for a single gene.
        
        Args:
            gene_symbol: Target gene symbol
            organ: Optional organ focus for analysis
            perturbation_strength: Strength of perturbation (0.1-1.0)
            top_n_percent: Optional filter for top N% of TF predictions
            tf_organ: Optional organ filter for AlphaGenome TF predictions
        
        Returns:
            Dictionary with complete analysis results
        """
        logger.info(f"🧬 Starting perturbation analysis for gene: {gene_symbol}")
        if organ:
            logger.info(f"🫁 Organ focus: {organ}")
        if tf_organ:
            logger.info(f"🧬 TF organ filter: {tf_organ}")
        if tf_organ and organ and tf_organ != organ:
            logger.info(f"ℹ️  Using different organs for TF filtering ({tf_organ}) and perturbation analysis ({organ})")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: AlphaGenome TF Prediction + Tahoe-100M Baseline Analysis
            logger.info("📊 Step 1: AlphaGenome TF prediction + Real Tahoe-100M baseline analysis...")
            baseline_results = self.baseline_analyzer.analyze_any_human_gene(
                gene_symbol, 
                focus_organ=tf_organ or organ,  # Use tf_organ for AlphaGenome TF filtering, fallback to organ
                top_n_percent=top_n_percent,
                tahoe_organ=organ  # Use organ for Tahoe-100M cell line selection
            )
            
            if baseline_results.empty:
                raise RuntimeError(f"No baseline results found for {gene_symbol}")
            
            logger.info(f"✅ Baseline analysis complete: {len(baseline_results)} TF-cell line combinations")
            
            # Extract TF list for perturbation
            tf_list = baseline_results['tf_name'].unique().tolist()
            logger.info(f"🎯 TFs for perturbation: {', '.join(tf_list[:5])}{'...' if len(tf_list) > 5 else ''} ({len(tf_list)} total)")
            
            # Step 2: Real H5AD Data Conversion
            logger.info("📊 Step 2: Loading real Tahoe-100M H5AD data from GCS...")
            conversion_result = self.bridge.convert_tahoe_to_state(
                gene_symbol, baseline_results, organ
            )
            logger.info(f"✅ Real H5AD data loaded: {conversion_result['conversion_stats']['cells_generated']} DMSO_TF control cells")
            
            # Step 3: Real State Model Perturbation Prediction
            logger.info("📊 Step 3: Running real State model zero-shot inference...")
            perturbation_results = self.state_engine.predict_tf_perturbation(
                gene_symbol, tf_list, baseline_results, organ, perturbation_strength
            )
            logger.info(f"✅ Real State model prediction complete")
            
            # Step 4: Target Gene Analysis
            logger.info("📊 Step 4: Analyzing target gene expression changes...")
            
            # Step 5: DEG Analysis
            logger.info("📊 Step 5: Genome-wide differential expression analysis...")
            deg_results = self.deg_analyzer.analyze_expression_changes(
                baseline_results, perturbation_results, gene_symbol,
                p_value_threshold=self.config_overrides.get('deg_p_threshold', 0.05),
                log2fc_threshold=self.config_overrides.get('deg_log2fc_threshold', 0.5)
            )
            # Check if DEG analysis succeeded
            significant_degs = 0
            try:
                if 'summary_statistics' in deg_results and 'genome_wide_summary' in deg_results['summary_statistics']:
                    significant_degs = deg_results['summary_statistics']['genome_wide_summary'].get('total_significant_degs', 0)
            except Exception as e:
                logger.warning(f"Could not extract significant DEGs count: {e}")
                significant_degs = 0
            
            logger.info(f"✅ DEG analysis complete: {significant_degs} significant DEGs")
            
            # Combine results
            comprehensive_results = {
                'gene_symbol': gene_symbol,
                'organ_focus': organ,
                'baseline_analysis': {
                    'tf_count': len(tf_list),
                    'cell_lines': baseline_results['cell_line'].nunique(),
                    'results_df': baseline_results
                },
                'perturbation_analysis': perturbation_results,
                'deg_analysis': deg_results,
                'analysis_metadata': {
                    'perturbation_strength': perturbation_strength,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'pipeline_version': '1.0',
                    'data_sources': ['alphagenome', 'tahoe_100M', 'state_model']
                }
            }
            
            # Export results
            export_paths = self._export_comprehensive_results(comprehensive_results, significant_degs)
            comprehensive_results['export_paths'] = export_paths
            
            # Update statistics
            self.pipeline_stats['genes_analyzed'] += 1
            self.pipeline_stats['perturbations_completed'] += len(tf_list)
            self.pipeline_stats['successful_analyses'] += 1
            
            duration = datetime.now() - start_time
            
            logger.info("=" * 80)
            logger.info(f"✅ PERTURBATION ANALYSIS COMPLETED for {gene_symbol}")
            logger.info(f"📊 TFs analyzed: {len(tf_list)}")
            logger.info(f"📊 Significant DEGs: {significant_degs}")
            logger.info(f"⏱️  Duration: {duration}")
            logger.info(f"💾 Results exported to: {len(export_paths)} files")
            logger.info("=" * 80)
            
            return comprehensive_results
            
        except Exception as e:
            self.pipeline_stats['failed_analyses'] += 1
            logger.error(f"Perturbation analysis failed for {gene_symbol}: {e}")
            raise RuntimeError(f"Analysis failed for {gene_symbol}: {e}")
    
    def analyze_multiple_genes(self, gene_list: List[str], 
                             organ: Optional[str] = None,
                             perturbation_strength: float = 0.5,
                             tf_organ: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform perturbation analysis for multiple genes.
        
        Args:
            gene_list: List of gene symbols
            organ: Optional organ focus
            perturbation_strength: Perturbation strength
            tf_organ: Optional organ filter for AlphaGenome TF predictions
        
        Returns:
            Dictionary mapping gene symbols to results
        """
        logger.info(f"🧬 Starting batch perturbation analysis for {len(gene_list)} genes")
        logger.info(f"📋 Genes: {', '.join(gene_list)}")
        
        results_dict = {}
        batch_start_time = datetime.now()
        
        for i, gene in enumerate(gene_list):
            logger.info(f"\\n🧪 Analyzing gene {i+1}/{len(gene_list)}: {gene}")
            
            try:
                results = self.analyze_gene_perturbation(
                    gene, organ=organ, perturbation_strength=perturbation_strength, tf_organ=tf_organ
                )
                results_dict[gene] = results
                
            except Exception as e:
                logger.error(f"Failed to analyze {gene}: {e}")
                self.pipeline_stats['failed_analyses'] += 1
                results_dict[gene] = {'error': str(e)}
        
        # Create batch summary
        batch_duration = datetime.now() - batch_start_time
        successful_genes = [gene for gene, result in results_dict.items() 
                          if 'error' not in result]
        
        logger.info("\\n" + "=" * 80)
        logger.info(f"📊 BATCH PERTURBATION ANALYSIS SUMMARY")
        logger.info(f"   Total genes: {len(gene_list)}")
        logger.info(f"   Successful: {len(successful_genes)}")
        logger.info(f"   Failed: {len(gene_list) - len(successful_genes)}")
        logger.info(f"   Total duration: {batch_duration}")
        logger.info(f"   Successful genes: {', '.join(successful_genes)}")
        logger.info("=" * 80)
        
        # Save batch summary
        self._save_batch_summary(gene_list, results_dict, batch_duration)
        
        return results_dict
    
    def _export_comprehensive_results(self, results: Dict[str, Any], significant_degs: int = 0) -> Dict[str, str]:
        """Export comprehensive analysis results."""
        logger.info("💾 Exporting comprehensive results...")
        
        gene_symbol = results['gene_symbol']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_paths = {}
        
        try:
            # Create main results directory
            results_dir = self.output_dir / f"{timestamp}_{gene_symbol}_results"
            results_dir.mkdir(exist_ok=True)
            
            # Export baseline results
            baseline_path = results_dir / f"{gene_symbol}_baseline_analysis.csv"
            results['baseline_analysis']['results_df'].to_csv(baseline_path, index=False)
            export_paths['baseline_analysis'] = str(baseline_path)
            logger.info(f"📊 Baseline analysis saved: {baseline_path}")
            
            # Export perturbation results
            if 'target_effects' in results['perturbation_analysis']:
                target_effects = results['perturbation_analysis']['target_effects']
                if isinstance(target_effects, pd.DataFrame) and not target_effects.empty:
                    target_path = results_dir / f"{gene_symbol}_target_effects.csv"
                    target_effects.to_csv(target_path, index=False)
                    export_paths['target_effects'] = str(target_path)
                    logger.info(f"🎯 Target effects saved: {target_path}")
            
            # Export DEG results manually to ensure they go to the right place
            try:
                if 'deg_analysis' in results:
                    deg_results = results['deg_analysis']
                    
                    # Export DEG list
                    if 'genome_wide_results' in deg_results:
                        deg_df = pd.DataFrame(deg_results['genome_wide_results'])
                        if not deg_df.empty:
                            degs_path = results_dir / f"{gene_symbol}_degs.csv"
                            deg_df.to_csv(degs_path, index=False)
                            export_paths['degs'] = str(degs_path)
                            logger.info(f"📈 DEGs saved: {degs_path}")
                    
                    # Export target gene changes
                    if 'target_gene_results' in deg_results:
                        target_changes_df = pd.DataFrame(deg_results['target_gene_results'])
                        if not target_changes_df.empty:
                            target_changes_path = results_dir / f"{gene_symbol}_target_changes.csv"
                            target_changes_df.to_csv(target_changes_path, index=False)
                            export_paths['target_changes'] = str(target_changes_path)
                            logger.info(f"🧬 Target gene changes saved: {target_changes_path}")
                    
                    # Export DEG summary
                    deg_summary_path = results_dir / f"{gene_symbol}_deg_summary.json"
                    deg_summary_data = {
                        'target_gene': gene_symbol,
                        'analysis_timestamp': timestamp,
                        'summary_statistics': deg_results.get('summary_statistics', {}),
                        'analysis_parameters': deg_results.get('analysis_parameters', {})
                    }
                    with open(deg_summary_path, 'w') as f:
                        json.dump(deg_summary_data, f, indent=2)
                    export_paths['deg_summary'] = str(deg_summary_path)
                    logger.info(f"📋 DEG summary saved: {deg_summary_path}")
            
            except Exception as e:
                logger.warning(f"DEG results export failed: {e}")
                export_paths['deg_export_error'] = str(e)
            
            # Export State model configuration and H5AD files to results directory
            try:
                # Copy State config
                state_config_files = list((self.output_dir / "state_cache" / "configs").glob("*.toml"))
                if state_config_files:
                    latest_config = max(state_config_files, key=lambda x: x.stat().st_mtime)
                    config_dest = results_dir / f"{gene_symbol}_state_config.toml"
                    import shutil
                    shutil.copy2(latest_config, config_dest)
                    export_paths['state_config'] = str(config_dest)
                    logger.info(f"⚙️ State config saved: {config_dest}")
                
                # Copy H5AD files
                h5ad_files = list((self.output_dir / "bridge_cache" / "h5ad_files").glob("*.h5ad"))
                if h5ad_files:
                    latest_h5ad = max(h5ad_files, key=lambda x: x.stat().st_mtime)
                    h5ad_dest = results_dir / f"{gene_symbol}_data.h5ad"
                    shutil.copy2(latest_h5ad, h5ad_dest)
                    export_paths['h5ad_data'] = str(h5ad_dest)
                    logger.info(f"💾 H5AD data saved: {h5ad_dest}")
            except Exception as e:
                logger.warning(f"State model files copy failed: {e}")
            
            # Export comprehensive summary
            summary_path = results_dir / f"{gene_symbol}_complete_analysis.json"
            summary_data = {
                'gene_symbol': gene_symbol,
                'analysis_timestamp': timestamp,
                'organ_focus': results['organ_focus'],
                'analysis_summary': {
                    'tfs_identified': results['baseline_analysis']['tf_count'],
                    'cell_lines_analyzed': results['baseline_analysis']['cell_lines'],
                    'perturbations_completed': results['baseline_analysis']['tf_count'],
                    'significant_degs': significant_degs,
                    'total_genes_tested': len(results.get('deg_analysis', {}).get('genome_wide_results', [])),
                    'files_exported': len(export_paths)
                },
                'file_locations': export_paths,
                'analysis_metadata': results['analysis_metadata'],
                'pipeline_statistics': self.get_pipeline_statistics()
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            export_paths['complete_summary'] = str(summary_path)
            logger.info(f"📋 Complete analysis summary saved: {summary_path}")
            
            # Also save a copy to main output directory for easy access
            main_summary_path = self.output_dir / f"{timestamp}_{gene_symbol}_summary.json"
            with open(main_summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            export_paths['main_summary'] = str(main_summary_path)
            
            logger.info(f"✅ Results exported to {len(export_paths)} files in {results_dir}")
            logger.info(f"📁 Main results directory: {results_dir}")
            
            return export_paths
            
        except Exception as e:
            logger.error(f"Results export failed: {e}")
            return {}
    
    def _save_batch_summary(self, gene_list: List[str], results_dict: Dict[str, Any], duration):
        """Save batch analysis summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"{timestamp}_batch_perturbation_summary.json"
        
        summary = {
            'batch_timestamp': timestamp,
            'genes_analyzed': gene_list,
            'total_genes': len(gene_list),
            'successful_analyses': len([g for g, r in results_dict.items() if 'error' not in r]),
            'failed_analyses': len([g for g, r in results_dict.items() if 'error' in r]),
            'duration_seconds': duration.total_seconds(),
            'pipeline_statistics': self.get_pipeline_statistics()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Batch summary saved to: {summary_path}")
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = self.pipeline_stats.copy()
        stats['pipeline_end_time'] = datetime.now().isoformat()
        return stats


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Perturbation Analysis Pipeline: AlphaGenome + Tahoe-100M + State Model + DEG Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single gene with organ focus
  python main.py --genes TP53 --organ stomach
  
  # Multiple genes with perturbation parameters
  python main.py --genes TP53,BRCA1,CLDN18 --organ lung --perturbation-strength 0.8
  
  # Filter TFs by organ context for AlphaGenome predictions
  python main.py --genes TP53 --tf-organ stomach --organ lung
  
  # Custom DEG thresholds
  python main.py --genes CLDN18 --deg-p-threshold 0.01 --deg-fc-threshold 1.0
        """
    )
    
    # Gene specification
    parser.add_argument(
        '--genes',
        type=str,
        required=True,
        help='Gene symbol(s) to analyze. Single gene: "TP53", Multiple genes: "TP53,BRCA1,CLDN18"'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--organ',
        type=str,
        help='Focus analysis on specific organ (e.g., "stomach", "lung"). For multiple organs, use comma separation (e.g., "stomach,lung,pancreas"). Filters both baseline and perturbation analysis.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging to console'
    )
    
    # Perturbation parameters
    parser.add_argument(
        '--perturbation-strength',
        type=float,
        default=0.5,
        help='Perturbation strength (0.1-1.0, default: 0.5)'
    )
    
    # DEG analysis parameters
    parser.add_argument(
        '--deg-p-threshold',
        type=float,
        default=0.05,
        help='P-value threshold for DEG significance (default: 0.05)'
    )
    
    parser.add_argument(
        '--deg-fc-threshold',
        type=float,
        default=0.5,
        help='Log2 fold-change threshold for DEG significance (default: 0.5)'
    )
    
    # TF prediction parameters
    parser.add_argument(
        '--tf-percent',
        type=int,
        default=10,
        help='Filter to top N%% of TF predictions when no organ specified (default: 10)'
    )
    
    parser.add_argument(
        '--tf-organ',
        type=str,
        help='Filter AlphaGenome TF predictions to specific organ context (e.g., "stomach", "lung"). This filters TFs based on their relevance to the specified organ, independent of the --organ parameter for perturbation analysis.'
    )
    
    # GCS/State configuration
    parser.add_argument(
        '--require-gcs',
        action='store_true',
        help='Require real Tahoe-100M H5AD from GCS; fail if not accessible (default: False)'
    )
    parser.add_argument(
        '--gcs-bucket',
        type=str,
        default='arc-ctc-tahoe100',
        help='GCS bucket name hosting Tahoe-100M H5AD (default: arc-ctc-tahoe100)'
    )
    parser.add_argument(
        '--tahoe-cache-dir',
        type=str,
        default='output/bridge_cache',
        help='Local cache directory for Tahoe H5AD downloads (default: output/bridge_cache)'
    )
    parser.add_argument(
        '--state-pert-col',
        type=str,
        default='drugname_drugconc',
        help='State model perturbation column in obs (default: drugname_drugconc)'
    )
    parser.add_argument(
        '--state-celltype-col',
        type=str,
        default='cell_name',
        help='State model cell type column in obs (default: cell_name)'
    )
    parser.add_argument(
        '--state-embed-key',
        type=str,
        default='X_hvg',
        help='Embedding key for State model inference (default: X_hvg)'
    )
    parser.add_argument(
        '--state-timeout',
        type=int,
        default=600,
        help='Timeout in seconds for each State TF inference (default: 600)'
    )
    
    return parser.parse_args()


def main():
    """Main pipeline entry point."""
    start_time = datetime.now()
    args = parse_arguments()
    
    # Set up logging with proper output directory and timestamp
    global log_file_path
    log_file_path = setup_logging(args.output, verbose=args.verbose)
    
    if args.verbose:
        print("🧬 Comprehensive Perturbation Analysis Pipeline")
        print("🔬 AlphaGenome + Tahoe-100M + State Model + DEG Analysis")
        print("🧫 Complete workflow from TF prediction to differential expression")
        print("=" * 80)
        print(f"📝 Logs will be saved to: {log_file_path}")
    
    try:
        # Build configuration
        config_overrides = {
            'perturbation_strength': args.perturbation_strength,
            'deg_p_threshold': args.deg_p_threshold,
            'deg_log2fc_threshold': args.deg_fc_threshold,
            # GCS/State configuration
            'require_gcs': args.require_gcs,
            'gcs_bucket': args.gcs_bucket,
            'tahoe_cache_dir': args.tahoe_cache_dir,
            'state_pert_col': args.state_pert_col,
            'state_celltype_col': args.state_celltype_col,
            'state_embed_key': args.state_embed_key,
            'state_timeout': args.state_timeout
        }
        
        if args.verbose:
            config_overrides['verbose'] = True
            print(f"🎛️  Perturbation strength: {args.perturbation_strength}")
            print(f"📊 DEG thresholds: p < {args.deg_p_threshold}, |log2FC| > {args.deg_fc_threshold}")
        
        # Initialize pipeline
        pipeline = PerturbationAnalysisPipeline(
            output_dir=args.output,
            config_overrides=config_overrides
        )
        
        # Parse genes (always use --genes argument)
        gene_list = [g.strip().upper() for g in args.genes.split(',') if g.strip()]
        
        # Parse organ parameter (handle comma-separated organs)
        organ_focus = None
        if args.organ:
            organs = [o.strip() for o in args.organ.split(',') if o.strip()]
            if len(organs) > 1:
                logger.warning(f"Multiple organs specified: {', '.join(organs)}. Using first organ: {organs[0]}")
                if args.verbose:
                    print(f"⚠️  Multiple organs specified. Using first organ: {organs[0]}")
            organ_focus = organs[0] if organs else None
        
        # Parse tf-organ parameter for AlphaGenome TF filtering
        tf_organ_focus = args.tf_organ.strip() if args.tf_organ else None
        
        if args.verbose and tf_organ_focus:
            print(f"🧬 TF organ filter: {tf_organ_focus}")
        if args.verbose and tf_organ_focus and organ_focus and tf_organ_focus != organ_focus:
            print(f"ℹ️  Using different organs for TF filtering ({tf_organ_focus}) and perturbation analysis ({organ_focus})")
        
        if len(gene_list) == 1:
            # Single gene analysis
            gene = gene_list[0]
            if args.verbose:
                organ_msg = f" with {organ_focus} focus" if organ_focus else ""
                print(f"🧬 Analyzing gene: {gene}{organ_msg}")
            
            # Apply top-N filtering when no organ specified
            top_n_to_use = None if organ_focus else args.tf_percent
            
            results = pipeline.analyze_gene_perturbation(
                gene, organ=organ_focus, perturbation_strength=args.perturbation_strength, 
                top_n_percent=top_n_to_use, tf_organ=tf_organ_focus
            )
            
            if args.verbose:
                print(f"\\n✅ Analysis completed successfully!")
                print(f"📊 TFs analyzed: {results['baseline_analysis']['tf_count']}")
                print(f"🧫 Cell lines: {results['baseline_analysis']['cell_lines']}")
                deg_count = 0
                if 'deg_analysis' in results and 'summary_statistics' in results['deg_analysis']:
                    deg_count = results['deg_analysis']['summary_statistics'].get('genome_wide_summary', {}).get('total_significant_degs', 0)
                print(f"📈 Significant DEGs: {deg_count}")
                
                files_exported = len(results.get('export_paths', {}))
                print(f"💾 Files exported: {files_exported}")
                
                if 'export_paths' in results and results['export_paths']:
                    # Show main results directory
                    results_dirs = [path for path in results['export_paths'].values() if 'results' in str(path)]
                    if results_dirs:
                        main_dir = Path(results_dirs[0]).parent
                        print(f"📁 Results directory: {main_dir}")
                
        else:
            # Multiple gene analysis
            if args.verbose:
                print(f"📋 Batch perturbation analysis for {len(gene_list)} genes: {', '.join(gene_list)}")
            
            results_dict = pipeline.analyze_multiple_genes(
                gene_list, organ=organ_focus, perturbation_strength=args.perturbation_strength, tf_organ=tf_organ_focus
            )
            
            successful = len([g for g, r in results_dict.items() if 'error' not in r])
            if args.verbose:
                print(f"\\n📊 Batch analysis completed: {successful}/{len(gene_list)} successful")
        
        # Show final pipeline statistics
        final_stats = pipeline.get_pipeline_statistics()
        if args.verbose:
            print(f"\\n📈 Pipeline Statistics:")
            print(f"   Genes analyzed: {final_stats['genes_analyzed']}")
            print(f"   Perturbations completed: {final_stats['perturbations_completed']}")
            print(f"   Successful analyses: {final_stats['successful_analyses']}")
            print(f"   Failed analyses: {final_stats['failed_analyses']}")
        
        # Log total execution time
        total_duration = datetime.now() - start_time
        logger.info(f"Total pipeline execution time: {total_duration}")
        if args.verbose:
            print(f"⏱️  Total analysis time: {total_duration}")
            print(f"\\n🎉 Perturbation Analysis Pipeline completed!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        if args.verbose:
            print(f"\\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
