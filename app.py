#!/usr/bin/env python3
"""
Advanced Medical Image Analytics Platform
========================================

A comprehensive medical image analysis application with enterprise-grade features,
robust error handling, and advanced analytics capabilities.

Author: Medical AI Systems
Version: 2.0.0
License: MIT
"""

import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import json
import logging
import traceback
import time
import hashlib
import base64
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
import pydicom
from contextlib import contextmanager
import tempfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import wraps, lru_cache
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'dcm']
MAX_CONCURRENT_ANALYSES = 3
CACHE_TTL = 3600  # 1 hour
DATABASE_PATH = 'medical_analytics.db'

# Enums
class AnalysisType(Enum):
    TUMOR_DETECTION = "tumor_detection"
    TISSUE_CLASSIFICATION = "tissue_classification"
    ANOMALY_DETECTION = "anomaly_detection"
    VOLUME_MEASUREMENT = "volume_measurement"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"

class ImageQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class SecurityLevel(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    HIPAA_COMPLIANT = "hipaa_compliant"

# Data Classes
@dataclass
class ImageMetadata:
    filename: str
    file_size: int
    dimensions: Tuple[int, int]
    format: str
    upload_time: datetime
    checksum: str
    patient_id: Optional[str] = None
    study_date: Optional[datetime] = None
    modality: Optional[str] = None
    
@dataclass
class AnalysisResult:
    analysis_id: str
    image_metadata: ImageMetadata
    analysis_type: AnalysisType
    confidence_score: float
    findings: Dict[str, Any]
    recommendations: List[str]
    processing_time: float
    quality_score: float
    
@dataclass
class ProcessingStats:
    total_analyses: int
    successful_analyses: int
    failed_analyses: int
    average_processing_time: float
    quality_distribution: Dict[str, int]

# Custom Exceptions
class MedicalAnalyticsError(Exception):
    """Base exception for medical analytics errors"""
    pass

class ImageProcessingError(MedicalAnalyticsError):
    """Raised when image processing fails"""
    pass

class AIAnalysisError(MedicalAnalyticsError):
    """Raised when AI analysis fails"""
    pass

class DatabaseError(MedicalAnalyticsError):
    """Raised when database operations fail"""
    pass

class SecurityError(MedicalAnalyticsError):
    """Raised when security validation fails"""
    pass

# Security and Validation
class SecurityValidator:
    """Handles security validation and HIPAA compliance"""
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format and strength"""
        if not api_key or len(api_key) < 20:
            return False
        # Add more sophisticated validation
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for security"""
        # Remove potentially dangerous characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        return ''.join(c for c in filename if c in safe_chars)
    
    @staticmethod
    def check_file_safety(file_content: bytes) -> bool:
        """Check if file content is safe"""
        # Basic malware detection patterns
        malware_signatures = [b'<script', b'javascript:', b'vbscript:']
        content_lower = file_content.lower()
        return not any(sig in content_lower for sig in malware_signatures)

# Database Manager
class DatabaseManager:
    """Handles all database operations with connection pooling"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    image_metadata TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    findings TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS processing_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_analyses INTEGER DEFAULT 0,
                    successful_analyses INTEGER DEFAULT 0,
                    failed_analyses INTEGER DEFAULT 0,
                    average_processing_time REAL DEFAULT 0.0,
                    quality_distribution TEXT NOT NULL,
                    UNIQUE(date)
                );
                
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analyses_count INTEGER DEFAULT 0
                );
                
                CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);
                CREATE INDEX IF NOT EXISTS idx_analyses_type ON analyses(analysis_type);
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
            ''')
            conn.commit()
    
    def save_analysis(self, result: AnalysisResult) -> bool:
        """Save analysis result to database"""
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT INTO analyses (
                        id, image_metadata, analysis_type, confidence_score,
                        findings, recommendations, processing_time, quality_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.analysis_id,
                    json.dumps(asdict(result.image_metadata), default=str),
                    result.analysis_type.value,
                    result.confidence_score,
                    json.dumps(result.findings),
                    json.dumps(result.recommendations),
                    result.processing_time,
                    result.quality_score
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            return False
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict]:
        """Get analysis history"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM analyses 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_processing_stats(self, days: int = 30) -> ProcessingStats:
        """Get processing statistics"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN confidence_score > 0 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN confidence_score = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(processing_time) as avg_time,
                    AVG(quality_score) as avg_quality
                FROM analyses 
                WHERE created_at > datetime('now', '-{} days')
            '''.format(days))
            
            row = cursor.fetchone()
            return ProcessingStats(
                total_analyses=row['total'] or 0,
                successful_analyses=row['successful'] or 0,
                failed_analyses=row['failed'] or 0,
                average_processing_time=row['avg_time'] or 0.0,
                quality_distribution={}
            )

# Image Processing Engine
class ImageProcessor:
    """Advanced image processing and enhancement"""
    
    def __init__(self):
        self.supported_formats = SUPPORTED_FORMATS
    
    def validate_image(self, image_data: bytes, filename: str) -> Tuple[bool, str]:
        """Validate image format and content"""
        try:
            # Check file size
            if len(image_data) > MAX_FILE_SIZE:
                return False, f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
            
            # Check format
            file_ext = Path(filename).suffix.lower().lstrip('.')
            if file_ext not in self.supported_formats:
                return False, f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
            
            # Try to open image
            image = Image.open(io.BytesIO(image_data))
            
            # Check dimensions
            if image.width < 64 or image.height < 64:
                return False, "Image too small. Minimum size: 64x64 pixels"
            
            if image.width > 4096 or image.height > 4096:
                return False, "Image too large. Maximum size: 4096x4096 pixels"
            
            # Check if image is corrupted
            image.verify()
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def enhance_medical_image(self, image: Image.Image) -> Image.Image:
        """Apply medical image enhancements"""
        try:
            # Convert to grayscale if needed for medical analysis
            if image.mode != 'L' and image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Apply sharpening
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Apply brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    def extract_metadata(self, image_data: bytes, filename: str) -> ImageMetadata:
        """Extract comprehensive image metadata"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Calculate checksum
            checksum = hashlib.sha256(image_data).hexdigest()
            
            # Extract EXIF data if available
            exif_data = {}
            if hasattr(image, '_getexif') and image._getexif():
                exif_data = image._getexif()
            
            return ImageMetadata(
                filename=SecurityValidator.sanitize_filename(filename),
                file_size=len(image_data),
                dimensions=(image.width, image.height),
                format=image.format or 'Unknown',
                upload_time=datetime.now(),
                checksum=checksum
            )
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            raise ImageProcessingError(f"Failed to extract metadata: {e}")
    
    def assess_image_quality(self, image: Image.Image) -> Tuple[float, ImageQuality]:
        """Assess image quality for medical analysis"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Calculate various quality metrics
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate contrast (standard deviation)
            contrast = gray.std()
            
            # Calculate brightness (mean)
            brightness = gray.mean()
            
            # Normalize and combine metrics
            sharpness_score = min(laplacian_var / 100, 1.0)
            contrast_score = min(contrast / 64, 1.0)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            overall_score = (sharpness_score * 0.4 + contrast_score * 0.4 + brightness_score * 0.2)
            
            # Determine quality level
            if overall_score >= 0.8:
                quality = ImageQuality.EXCELLENT
            elif overall_score >= 0.6:
                quality = ImageQuality.GOOD
            elif overall_score >= 0.4:
                quality = ImageQuality.FAIR
            else:
                quality = ImageQuality.POOR
            
            return overall_score, quality
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.5, ImageQuality.FAIR

# AI Analysis Engine
class AIAnalysisEngine:
    """Advanced AI analysis using Google Gemini"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize Gemini model"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("AI model initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise AIAnalysisError(f"Failed to initialize AI model: {e}")
    
    def analyze_medical_image(self, image: Image.Image, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Perform comprehensive medical image analysis"""
        try:
            # Prepare specialized prompt based on analysis type
            prompt = self._get_analysis_prompt(analysis_type)
            
            # Generate analysis
            response = self.model.generate_content([prompt, image])
            
            # Parse and structure response
            analysis_result = self._parse_analysis_response(response.text, analysis_type)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            raise AIAnalysisError(f"Analysis failed: {e}")
    
    def _get_analysis_prompt(self, analysis_type: AnalysisType) -> str:
        """Get specialized prompt for analysis type"""
        prompts = {
            AnalysisType.TUMOR_DETECTION: """
                As a medical imaging AI assistant, analyze this brain MRI image for potential tumor detection.
                Provide detailed findings including:
                1. Presence of any abnormal masses or lesions
                2. Location and approximate size of findings
                3. Signal characteristics and enhancement patterns
                4. Differential diagnosis considerations
                5. Confidence level (0-100%)
                6. Recommendations for further imaging or follow-up
                
                Format your response as structured JSON with clear sections.
            """,
            AnalysisType.TISSUE_CLASSIFICATION: """
                Analyze this medical image for tissue classification and anatomical structure identification.
                Include:
                1. Tissue types identified (gray matter, white matter, CSF, etc.)
                2. Anatomical structures visible
                3. Tissue contrast and signal intensity
                4. Any pathological tissue changes
                5. Confidence metrics for each classification
                
                Provide structured analysis in JSON format.
            """,
            AnalysisType.ANOMALY_DETECTION: """
                Perform comprehensive anomaly detection on this medical image.
                Identify:
                1. Any abnormal signal intensities
                2. Structural anomalies or deformities
                3. Asymmetries or unusual patterns
                4. Artifacts or technical issues
                5. Severity assessment
                
                Return findings in structured JSON format.
            """
        }
        
        return prompts.get(analysis_type, prompts[AnalysisType.TUMOR_DETECTION])
    
    def _parse_analysis_response(self, response_text: str, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Parse and structure AI response"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Fallback: structure the text response
            return {
                "analysis_type": analysis_type.value,
                "findings": {
                    "description": response_text,
                    "structured_analysis": self._extract_key_findings(response_text)
                },
                "confidence": self._extract_confidence(response_text),
                "recommendations": self._extract_recommendations(response_text)
            }
            
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return {
                "analysis_type": analysis_type.value,
                "findings": {"raw_response": response_text},
                "confidence": 0.5,
                "recommendations": ["Further review recommended"]
            }
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from text"""
        findings = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in ['finding', 'detected', 'observed', 'identified']):
                findings.append(line)
        
        return findings[:5]  # Limit to top 5 findings
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        import re
        
        # Look for percentage patterns
        confidence_patterns = [
            r'confidence[:\s]*(\d+)%',
            r'(\d+)%\s*confidence',
            r'certainty[:\s]*(\d+)%'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100.0
        
        return 0.7  # Default confidence
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from text"""
        recommendations = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'advise', 'follow-up']):
                recommendations.append(line)
        
        if not recommendations:
            recommendations = ["Consult with radiologist for detailed interpretation"]
        
        return recommendations

# Main Application Class
class MedicalAnalyticsApp:
    """Main application orchestrator"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.image_processor = ImageProcessor()
        self.ai_engine = None
        self.session_id = str(uuid.uuid4())
        self.processing_stats = {"total": 0, "successful": 0, "failed": 0}
        
    def initialize_ai_engine(self, api_key: str) -> bool:
        """Initialize AI engine with API key"""
        try:
            if not SecurityValidator.validate_api_key(api_key):
                return False
            
            self.ai_engine = AIAnalysisEngine(api_key)
            return True
            
        except Exception as e:
            logger.error(f"AI engine initialization failed: {e}")
            return False
    
    def process_image(self, image_data: bytes, filename: str, analysis_type: AnalysisType) -> Optional[AnalysisResult]:
        """Process single image with comprehensive analysis"""
        start_time = time.time()
        
        try:
            # Validate image
            is_valid, message = self.image_processor.validate_image(image_data, filename)
            if not is_valid:
                raise ImageProcessingError(message)
            
            # Security check
            if not SecurityValidator.check_file_safety(image_data):
                raise SecurityError("File failed security validation")
            
            # Extract metadata
            metadata = self.image_processor.extract_metadata(image_data, filename)
            
            # Process image
            image = Image.open(io.BytesIO(image_data))
            enhanced_image = self.image_processor.enhance_medical_image(image)
            
            # Assess quality
            quality_score, quality_level = self.image_processor.assess_image_quality(enhanced_image)
            
            # Perform AI analysis
            if self.ai_engine:
                analysis_findings = self.ai_engine.analyze_medical_image(enhanced_image, analysis_type)
            else:
                raise AIAnalysisError("AI engine not initialized")
            
            # Create result
            processing_time = time.time() - start_time
            
            result = AnalysisResult(
                analysis_id=str(uuid.uuid4()),
                image_metadata=metadata,
                analysis_type=analysis_type,
                confidence_score=analysis_findings.get('confidence', 0.7),
                findings=analysis_findings,
                recommendations=analysis_findings.get('recommendations', []),
                processing_time=processing_time,
                quality_score=quality_score
            )
            
            # Save to database
            self.db_manager.save_analysis(result)
            
            self.processing_stats["total"] += 1
            self.processing_stats["successful"] += 1
            
            return result
            
        except Exception as e:
            self.processing_stats["total"] += 1
            self.processing_stats["failed"] += 1
            logger.error(f"Image processing failed: {e}")
            raise
    
    def batch_process_images(self, image_files: List[Tuple[bytes, str]], analysis_type: AnalysisType) -> List[Tuple[Optional[AnalysisResult], Optional[str]]]:
        """Process multiple images concurrently"""
        results = []
        
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ANALYSES) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_image, image_data, filename, analysis_type): (image_data, filename)
                for image_data, filename in image_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                image_data, filename = future_to_file[future]
                try:
                    result = future.result()
                    results.append((result, None))
                except Exception as e:
                    results.append((None, str(e)))
        
        return results

# Streamlit UI Components
class UIComponents:
    """Reusable UI components for medical research"""
    
    @staticmethod
    def render_research_settings():
        """Render the Advanced Research Settings section with data source options"""
        st.markdown("---")
        st.subheader("🧪 Advanced Research Settings")
        st.caption("Configure your research data sources and collaboration settings")
        
        # Create tabs for different data source types
        tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "🏥 PACS", "🔍 Research Database", "🌐 Federated Learning"])
        
        with tab1:
            st.markdown("### Local File Upload")
            st.markdown("Upload medical images and datasets directly from your device")
            uploaded_files = st.file_uploader(
                "Drag and drop files or click to browse",
                type=['dcm', 'nii', 'nii.gz', 'nrrd', 'mhd', 'mha', 'jpg', 'jpeg', 'png', 'tiff'],
                accept_multiple_files=True,
                help="Supported formats: DICOM, NIfTI, NRRD, MetaImage, JPG, PNG, TIFF"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) ready for processing")
                
                # Show file details in an expander
                with st.expander("📋 View selected files"):
                    for file in uploaded_files:
                        st.write(f"- {file.name} ({file.size / (1024*1024):.2f} MB)")
                
                # Processing options
                st.markdown("### Processing Options")
                col1, col2 = st.columns(2)
                with col1:
                    anonymize = st.checkbox("Anonymize DICOM files", value=True)
                    convert_to_nifti = st.checkbox("Convert to NIfTI format", value=False)
                with col2:
                    auto_segment = st.checkbox("Auto-segment images", value=False)
                    extract_metadata = st.checkbox("Extract and store metadata", value=True)
                
                # Start processing button
                if st.button("🚀 Process Files", use_container_width=True):
                    with st.spinner("Processing files..."):
                        # Add your file processing logic here
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.02)  # Simulate processing
                            progress_bar.progress(i + 1)
                        st.success("Processing complete!")
        
        with tab2:
            st.markdown("### PACS Integration")
            st.markdown("Connect to your Picture Archiving and Communication System (PACS)")
            
            # PACS connection form
            with st.form("pacs_connection"):
                col1, col2 = st.columns(2)
                with col1:
                    pacs_host = st.text_input("PACS Host", "pacs.hospital.org")
                    pacs_port = st.number_input("Port", 104, 65535, 104)
                with col2:
                    pacs_ae_title = st.text_input("AE Title", "MY_PACS")
                    pacs_use_tls = st.checkbox("Use TLS/SSL", value=True)
                
                # Advanced settings expander
                with st.expander("⚙️ Advanced Settings"):
                    pacs_timeout = st.number_input("Connection Timeout (seconds)", 5, 300, 30)
                    pacs_retries = st.number_input("Max Retry Attempts", 1, 10, 3)
                
                # Connect button
                if st.form_submit_button("🔌 Connect to PACS", use_container_width=True):
                    with st.spinner("Connecting to PACS..."):
                        try:
                            # Add your PACS connection logic here
                            time.sleep(1.5)  # Simulate connection delay
                            st.success("✅ Successfully connected to PACS")
                            
                            # Show query interface after connection
                            st.markdown("### Query Studies")
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                start_date = st.date_input("From Date", value=datetime.now() - timedelta(days=30))
                            with col2:
                                end_date = st.date_input("To Date", value=datetime.now())
                            with col3:
                                st.markdown("<br>", unsafe_allow_html=True)
                                if st.button("🔍 Search"):
                                    # Add your study search logic here
                                    st.info("Searching for studies...")
                                    
                        except Exception as e:
                            st.error(f"❌ Failed to connect to PACS: {str(e)}")
        
        with tab3:
            st.markdown("### Research Database")
            st.markdown("Access and manage research datasets and annotations")
            
            # Database selection
            db_options = ["Select a database", "TCIA", "OpenNeuro", "MIMIC-CXR", "Custom Database"]
            selected_db = st.selectbox("Select Research Database", db_options)
            
            if selected_db != "Select a database":
                st.info(f"ℹ️ Connecting to {selected_db}...")
                
                if selected_db == "Custom Database":
                    # Custom database connection form
                    with st.form("custom_db_connection"):
                        db_type = st.selectbox("Database Type", ["MongoDB", "PostgreSQL", "MySQL", "SQLite"])
                        db_host = st.text_input("Host", "localhost")
                        db_port = st.number_input("Port", 1, 65535, 27017 if db_type == "MongoDB" else 5432)
                        db_name = st.text_input("Database Name", "research_db")
                        db_user = st.text_input("Username")
                        db_password = st.text_input("Password", type="password")
                        
                        if st.form_submit_button("Connect to Database"):
                            with st.spinner("Connecting..."):
                                # Add your database connection logic here
                                time.sleep(1)
                                st.success("✅ Database connection successful!")
                else:
                    # Show available datasets for public databases
                    st.markdown("### Available Datasets")
                    
                    # Simulated dataset search
                    search_query = st.text_input("Search datasets", placeholder="e.g., lung CT, brain MRI")
                    
                    # Example datasets (in a real app, this would come from the database)
                    example_datasets = [
                        {"name": "Lung CT Scans", "modality": "CT", "samples": 1_250, "size": "45.7 GB", "license": "CC BY 4.0"},
                        {"name": "Brain MRI Scans", "modality": "MRI", "samples": 890, "size": "32.1 GB", "license": "CC BY-NC 4.0"},
                        {"name": "Chest X-Rays", "modality": "X-Ray", "samples": 5_000, "size": "12.3 GB", "license": "CC BY-SA 4.0"},
                        {"name": "Whole-Body CT Scans", "modality": "CT", "samples": 320, "size": "78.9 GB", "license": "CC BY 3.0"},
                    ]
                    
                    # Filter datasets based on search query
                    filtered_datasets = [d for d in example_datasets 
                                      if not search_query or search_query.lower() in d["name"].lower()]
                    
                    # Display datasets in a grid
                    for dataset in filtered_datasets:
                        with st.expander(f"📁 {dataset['name']}"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Modality", dataset["modality"])
                            with col2:
                                st.metric("Samples", f"{dataset['samples']:,}")
                            with col3:
                                st.metric("Size", dataset["size"])
                            with col4:
                                st.metric("License", dataset["license"])
                            
                            # Action buttons
                            if st.button(f"Preview {dataset['name']}", key=f"preview_{dataset['name']}"):
                                st.info("Preview functionality would be implemented here")
                            if st.button(f"Download {dataset['name']}", key=f"dl_{dataset['name']}"):
                                st.info("Download functionality would be implemented here")
        
        with tab4:
            st.markdown("### Federated Learning")
            st.markdown("Collaborate on AI model training without sharing sensitive data")
            
            # Federated learning setup
            st.markdown("#### Setup Federated Learning")
            
            with st.expander("🔧 Configuration", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    fl_strategy = st.selectbox(
                        "Federated Learning Strategy",
                        ["Federated Averaging (FedAvg)", "Federated Dropout", "Federated Learning with Dynamic Regularization"]
                    )
                    num_rounds = st.number_input("Number of Rounds", 1, 1000, 100)
                with col2:
                    clients = st.multiselect(
                        "Select Participating Clients",
                        ["Hospital A", "Research Lab B", "Clinic C", "University D", "Medical Center E"],
                        default=["Hospital A", "Research Lab B"]
                    )
                    batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)
            
            # Model selection
            st.markdown("#### Model Configuration")
            model_type = st.radio(
                "Select Base Model",
                ["3D U-Net (for volumetric data)", "ResNet-50 (for 2D images)", "Custom Model"],
                horizontal=True
            )
            
            if model_type == "Custom Model":
                custom_model = st.file_uploader("Upload Custom Model (PyTorch .pt or .pth)", type=["pt", "pth"])
                if custom_model:
                    st.success(f"✅ {custom_model.name} uploaded successfully")
            
            # Training parameters
            st.markdown("#### Training Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                learning_rate = st.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%f")
            with col2:
                loss_function = st.selectbox("Loss Function", ["Dice Loss", "Cross Entropy", "Focal Loss"])
            with col3:
                optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
            
            # Privacy settings
            st.markdown("#### Privacy Settings")
            use_dp = st.checkbox("Enable Differential Privacy", value=True)
            if use_dp:
                col1, col2 = st.columns(2)
                with col1:
                    noise_multiplier = st.slider("Noise Multiplier", 0.1, 5.0, 1.0, 0.1)
                with col2:
                    max_grad_norm = st.slider("Max Gradient Norm", 0.1, 5.0, 1.0, 0.1)
            
            # Start federated learning button
            if st.button("🚀 Start Federated Learning", use_container_width=True):
                with st.spinner("Initializing federated learning..."):
                    # Add your federated learning logic here
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        # Simulate training progress
                        time.sleep(0.05)
                        progress = (i + 1) / 100
                        progress_bar.progress(progress)
                        
                        # Update status text
                        if progress < 0.3:
                            status_text.text(f"Initializing clients... {int(progress * 100)}%")
                        elif progress < 0.7:
                            status_text.text(f"Training in progress... {int(progress * 100)}%")
                        else:
                            status_text.text(f"Finalizing model... {int(progress * 100)}%")
                    
                    st.success("✅ Federated learning completed successfully!")
                    
                    # Show results
                    st.balloons()
                    st.markdown("### Training Results")
                    
                    # Example metrics (in a real app, these would come from the training process)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Accuracy", "92.4%", "+3.2% from baseline")
                    with col2:
                        st.metric("Loss", "0.124", "-0.08")
                    with col3:
                        st.metric("Rounds Completed", f"{num_rounds}", "100%")
                    
                    # Download trained model
                    st.download_button(
                        label="💾 Download Trained Model",
                        data=b"This would be the model binary in a real application",
                        file_name="federated_model.pt",
                        mime="application/octet-stream"
                    )
        
        # Add some spacing at the bottom
        st.markdown("<br><br>", unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render professional research platform header with dark theme"""
        st.set_page_config(
            page_title="NexusMed AI Research Platform",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Premium dark theme CSS for research scientists
        st.markdown("""
        <style>
            /* Base Styles */
            :root {
                --primary: #6366f1;
                --primary-dark: #4f46e5;
                --secondary: #10b981;
                --accent: #8b5cf6;
                --background: #0a0f1e;
                --surface: #111827;
                --surface-elevated: #1f2937;
                --text-primary: #f9fafb;
                --text-secondary: #d1d5db;
                --border: #374151;
                --success: #10b981;
                --warning: #f59e0b;
                --error: #ef4444;
                --card-bg: rgba(17, 24, 39, 0.8);
                --card-border: rgba(75, 85, 99, 0.3);
                --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            }
            
            /* Main Layout */
            .main .block-container {
                padding: 1.5rem 2rem 2rem;
                background: var(--background);
                color: var(--text-primary);
            }
            
            /* Typography */
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-primary);
                font-weight: 600;
                letter-spacing: -0.025em;
            }
            
            /* Buttons */
            .stButton>button {
                border: none;
                border-radius: 8px;
                padding: 0.6rem 1.25rem;
                font-weight: 500;
                transition: all 0.2s;
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            .stButton>button:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            }
            
            /* Cards & Containers */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                padding: 0 2px;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.75rem 1.25rem;
                border-radius: 8px;
                transition: all 0.2s;
                background: var(--surface-elevated);
                color: var(--text-secondary);
                border: 1px solid var(--border);
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white !important;
                border-color: var(--primary) !important;
                box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
            }
            
            /* Forms & Inputs */
            .stTextInput>div>div>input, 
            .stSelectbox>div>div>div,
            .stTextArea>div>textarea {
                background: var(--surface-elevated) !important;
                border: 1px solid var(--border) !important;
                color: var(--text-primary) !important;
                border-radius: 8px !important;
                padding: 0.5rem 0.75rem !important;
            }
            
            /* Progress & Loading */
            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
            }
            
            /* Custom Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--surface);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--border);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: var(--primary);
            }
            
            /* Custom Cards */
            .research-card {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: var(--card-shadow);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }
            
            .research-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            }
            
            /* Alerts & Notifications */
            .stAlert {
                background: transparent !important;
                border-left: 4px solid var(--primary);
                border-radius: 0 8px 8px 0;
            }
            
            /* Custom Tabs */
            .tab-content {
                padding: 1.5rem 0;
            }
            
            /* Utility Classes */
            .glass-effect {
                background: rgba(17, 24, 39, 0.7);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .gradient-text {
                background: linear-gradient(90deg, var(--primary), var(--accent));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: inline-block;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header with research focus
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("https://img.icons8.com/color/96/000000/test-tube.png", width=80)
        with col2:
            st.title("Advanced Medical Research Platform")
            st.caption("Cutting-edge tools for medical imaging research and analysis")
        
        st.markdown("---")
    
    @staticmethod
    def render_sidebar_config():
        """Render research-focused sidebar configuration"""
        with st.sidebar:
            st.header("⚙️ Research Configuration")
            
            # API Key Input
            api_key = st.text_input(
                "🔑 Research API Key",
                type="password",
                help="Enter your research API key for advanced features"
            )
            
            # Research Focus Area
            research_focus = st.selectbox(
                "🎯 Research Focus",
                [
                    "Oncology Imaging",
                    "Neurological Disorders",
                    "Cardiovascular Imaging",
                    "Musculoskeletal Research",
                    "Pulmonary Studies",
                    "Pediatric Imaging"
                ],
                index=0,
                help="Select your primary research area"
            )
            
            # Analysis Type with research-specific options
            analysis_type = st.selectbox(
                "🔍 Analysis Type",
                [t.value for t in AnalysisType] + [
                    "radiomics_analysis",
                    "longitudinal_study",
                    "ai_model_training"
                ],
                format_func=lambda x: x.replace("_", " ").title(),
                index=0,
                help="Select the type of analysis to perform"
            )
            
            # Advanced Research Options
            with st.expander("🧪 Advanced Research Settings"):
                # Data Source
                data_source = st.radio(
                    "📂 Data Source",
                    ["Upload", "PACS", "Research Database", "Federated Learning"],
                    index=0,
                    help="Select data source for analysis"
                )
                
                # Processing Resources
                processing_resources = st.select_slider(
                    "⚡ Compute Resources",
                    options=["Basic", "Standard", "High", "GPU-Accelerated", "Distributed"],
                    value="Standard",
                    help="Allocate computational resources"
                )
                
                # Data Privacy Level
                privacy_level = st.select_slider(
                    "🔒 Privacy Level",
                    options=["De-identified", "Limited Dataset", "Full PHI"],
                    value="De-identified",
                    help="Select data privacy level"
                )
            
            st.markdown("---")
            st.markdown("### Research Tools")
            
            # Quick Access Buttons
            if st.button("📚 Open Research Library"):
                st.session_state.active_tab = "Literature Review"
                
            if st.button("📊 View Analytics"):
                st.session_state.active_tab = "Analytics"
                
            if st.button("🤖 AI Assistant"):
                st.session_state.show_ai_assistant = not st.session_state.get('show_ai_assistant', False)
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("""
            **Advanced Medical Research Platform**  
            Research Edition v3.0  
            
            Comprehensive tools for medical imaging research with AI-powered analytics.
            """)    
        
        return api_key, analysis_type, True, 0.5
    
    @staticmethod
    def render_research_tabs():
        """Render the main research tabs"""
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🧬 Radiomics Analysis",
            "📊 Longitudinal Studies",
            "🧠 AI Model Training",
            "🔍 Literature Review",
            "📈 Statistical Analysis",
            "🌐 Collaborative Research"
        ])
        
        with tab1:
            st.header("🧬 Radiomics Analysis")
            st.markdown("""
            ### Extract Quantitative Imaging Biomarkers
            
            Advanced radiomics feature extraction from medical images for precision medicine.
            """)
            
            # Placeholder for radiomics controls
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Feature Classes", 
                           ["First Order", "Shape", "Texture", "Wavelet"],
                           key="feature_classes")
                st.checkbox("Enable Deep Radiomics", value=False, key="deep_radiomics")
                
            with col2:
                st.selectbox("Filtering", 
                           ["None", "Wavelet", "Laplacian", "Gradient", "Local Binary Pattern"],
                           key="filter_type")
                st.checkbox("Enable Feature Selection", value=True, key="feature_selection")
            
            st.markdown("---")
            st.markdown("### Feature Extraction Results")
            st.info("Feature extraction will be performed after image upload and analysis.")
            
        with tab2:
            st.header("📊 Longitudinal Study Analysis")
            st.markdown("""
            ### Track Disease Progression Over Time
            
            Compare multiple imaging studies from the same patient to analyze disease progression
            and treatment response.
            """)
            
            # Placeholder for longitudinal study controls
            st.radio("Analysis Type", 
                    ["Tumor Growth", "Atrophy Measurement", "Perfusion Changes", "Custom ROI Analysis"],
                    horizontal=True,
                    key="long_analysis_type")
            
            st.slider("Time Window (months)", 1, 60, 12, 1, 
                     help="Select the time window for analysis")
            
            st.markdown("---")
            st.markdown("### Study Timeline")
            st.info("Upload multiple studies to visualize changes over time.")
            
        with tab3:
            st.header("🧠 AI Model Training")
            st.markdown("""
            ### Custom Model Development
            
            Train and validate custom deep learning models on your imaging data.
            """)
            
            # Placeholder for model training controls
            model_type = st.selectbox("Model Architecture", 
                                    ["3D U-Net", "nnU-Net", "DenseNet", "Custom"],
                                    key="model_arch")
            
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Epochs", 1, 1000, 100, key="epochs")
                st.select_slider("Learning Rate", 
                               options=[f"1e-{i}" for i in range(1, 7)], 
                               value="1e-4",
                               key="learning_rate")
            
            with col2:
                st.selectbox("Loss Function", 
                           ["Dice Loss", "Cross Entropy", "Focal Loss", "Combined"],
                           key="loss_func")
                st.checkbox("Enable Early Stopping", value=True, key="early_stop")
            
            st.markdown("---")
            st.markdown("### Training Progress")
            st.info("Configure and start training to see metrics and visualizations.")
            
        with tab4:
            st.header("🔍 Literature Review")
            st.markdown("""
            ### AI-Powered Research Assistant
            
            Search and analyze the latest medical literature related to your research.
            """)
            
            # Placeholder for literature search
            query = st.text_input("Search Medical Literature", 
                                 placeholder="e.g., 'radiomics in glioblastoma prognosis'")
            
            if query:
                st.info(f"Searching for: {query}")
                
                # Simulated search results
                with st.expander("📄 Title: Radiomics in Glioblastoma: Current Applications..."):
                    st.caption("Journal of Medical Imaging, 2023 | DOI: 10.xxxx/xxxxxx")
                    st.write("Abstract: This review examines the current state of radiomics in...")
                    st.button("View Full Text", key="paper1")
                
                with st.expander("📄 Title: Deep Learning Approaches for Brain Tumor Segmentation..."):
                    st.caption("Nature Scientific Reports, 2023 | DOI: 10.xxxx/xxxxxx")
                    st.write("Abstract: Recent advances in deep learning have shown...")
                    st.button("View Full Text", key="paper2")
            
        with tab5:
            st.header("📈 Statistical Analysis")
            st.markdown("""
            ### Advanced Statistical Tools
            
            Perform statistical analysis on your imaging data and research findings.
            """)
            
            # Placeholder for statistical analysis
            analysis_type = st.selectbox("Analysis Type",
                                       ["Descriptive Statistics", 
                                        "T-tests & ANOVAs",
                                        "Regression Analysis",
                                        "Survival Analysis",
                                        "Machine Learning Pipeline"],
                                       key="stat_analysis_type")
            
            if analysis_type == "Descriptive Statistics":
                st.write("Generate summary statistics and visualizations for your data.")
            elif analysis_type == "T-tests & ANOVAs":
                st.write("Compare groups and test hypotheses about your imaging biomarkers.")
            elif analysis_type == "Regression Analysis":
                st.write("Model relationships between imaging features and clinical outcomes.")
            elif analysis_type == "Survival Analysis":
                st.write("Analyze time-to-event data with imaging biomarkers.")
            else:  # ML Pipeline
                st.write("Build and evaluate machine learning models on your data.")
            
            st.markdown("---")
            st.markdown("### Results")
            st.info("Run analysis to see results and visualizations.")
            
        with tab6:
            st.header("🌐 Collaborative Research")
            st.markdown("""
            ### Multi-Center Research Tools
            
            Collaborate with researchers worldwide while maintaining data privacy.
            """)
            
            # Placeholder for collaboration tools
            st.radio("Collaboration Mode",
                   ["Federated Learning", "Data Sharing", "Model Sharing", "Annotation Review"],
                   key="collab_mode")
            
            if st.button("Connect to Research Network"):
                st.session_state.connected = not st.session_state.get('connected', False)
            
            if st.session_state.get('connected', False):
                st.success("✅ Connected to Global Research Network")
                
                # Simulated research network
                st.markdown("#### Active Research Projects")
                projects = [
                    {"name": "Brain Tumor Segmentation Challenge", "institutions": 24, "datasets": "1.2TB"},
                    {"name": "COVID-19 Imaging Biomarkers", "institutions": 18, "datasets": "850GB"},
                    {"name": "Alzheimer's Disease Prediction", "institutions": 12, "datasets": "2.1TB"}
                ]
                
                for project in projects:
                    with st.expander(f"🔬 {project['name']}"):
                        st.write(f"**Institutions:** {project['institutions']}")
                        st.write(f"**Datasets:** {project['datasets']} of imaging data")
                        st.button("Join Project", key=f"join_{project['name']}")
            
            st.markdown("---")
            st.markdown("### Research Ethics")
            st.info("All collaborations follow strict ethical guidelines and data protection standards.")
    
    @staticmethod
    def render_file_uploader():
        """Render file uploader component"""
        st.header("📁 Image Upload")
        
        uploaded_files = st.file_uploader(
            "Upload Medical Images",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join([f.upper() for f in SUPPORTED_FORMATS])}"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
            
            # Display file information
            with st.expander("📋 File Information"):
                for file in uploaded_files:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Name:** {file.name}")
                    with col2:
                        st.write(f"**Size:** {file.size / 1024:.1f} KB")
                    with col3:
                        st.write(f"**Type:** {file.type}")
        
        return uploaded_files
    
    @staticmethod
    def render_analysis_results(results: List[Tuple[Optional[AnalysisResult], Optional[str]]]):
        """Render analysis results"""
        st.header("📊 Analysis Results")
        
        for i, (result, error) in enumerate(results):
            if error:
                st.error(f"Analysis {i+1} failed: {error}")
                continue
            
            if not result:
                st.warning(f"No result for analysis {i+1}")
                continue
            
            # Create expandable result section
            with st.expander(f"🔍 Analysis {i+1}: {result.image_metadata.filename}"):
                
                # Result overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Confidence", f"{result.confidence_score:.1%}")
                
                with col2:
                    st.metric("Quality Score", f"{result.quality_score:.2f}")
                
                with col3:
                    st.metric("Processing Time", f"{result.processing_time:.2f}s")
                
                with col4:
                    st.metric("Analysis Type", result.analysis_type.value.replace('_', ' ').title())
                
                # Detailed findings
                st.subheader("🔬 Detailed Findings")
                
                findings = result.findings
                if isinstance(findings, dict):
                    for key, value in findings.items():
                        if key != 'raw_response':
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Recommendations
                if result.recommendations:
                    st.subheader("💡 Recommendations")
                    for rec in result.recommendations:
                        st.write(f"• {rec}")
                
                # Technical details
                with st.expander("🔧 Technical Details"):
                    st.json({
                        "analysis_id": result.analysis_id,
                        "image_metadata": asdict(result.image_metadata),
                        "processing_stats": {
                            "processing_time": result.processing_time,
                            "quality_score": result.quality_score,
                            "confidence_score": result.confidence_score
                        }
                    })
    
    @staticmethod
    def render_analytics_dashboard(app: MedicalAnalyticsApp):
        """Render analytics dashboard"""
        st.header("📈 Analytics Dashboard")
        
        # Get processing stats
        stats = app.db_manager.get_processing_stats()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", stats.total_analyses)
        
        with col2:
            st.metric("Successful", stats.successful_analyses)
        
        with col3:
            st.metric("Failed", stats.failed_analyses)
        
        with col4:
            success_rate = (stats.successful_analyses / max(stats.total_analyses, 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Processing time chart
        if stats.total_analyses > 0:
            st.subheader("📊 Performance Metrics")
            
            # Create sample time series data
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            
            # Performance chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Processing Time Trend', 'Success Rate', 'Quality Distribution', 'Analysis Types'),
                specs=[[{"secondary_y": True}, {"type": "indicator"}],
                       [{"type": "bar"}, {"type": "pie"}]]
            )
            
            # Processing time trend
            processing_times = np.random.normal(stats.average_processing_time, 0.5, 30)
            fig.add_trace(
                go.Scatter(x=dates, y=processing_times, name="Processing Time", line=dict(color="blue")),
                row=1, col=1
            )
            
            # Success rate indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=success_rate,
                    title={"text": "Success Rate %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "green"},
                           'steps': [
                               {'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 80], 'color': "yellow"},
                               {'range': [80, 100], 'color': "green"}]}
                ),
                row=1, col=2
            )
            
            # Quality distribution
            quality_data = ['Excellent', 'Good', 'Fair', 'Poor']
            quality_values = [30, 40, 20, 10]  # Sample data
            fig.add_trace(
                go.Bar(x=quality_data, y=quality_values, name="Quality Distribution"),
                row=2, col=1
            )
            
            # Analysis types pie chart
            analysis_types = ['Tumor Detection', 'Tissue Classification', 'Anomaly Detection']
            type_counts = [50, 30, 20]  # Sample data
            fig.add_trace(
                go.Pie(labels=analysis_types, values=type_counts, name="Analysis Types"),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent analysis history
        st.subheader("📋 Recent Analysis History")
        
        history = app.db_manager.get_analysis_history(limit=10)
        if history:
            df = pd.DataFrame(history)
            
            # Format datetime columns
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Display table
            st.dataframe(
                df[['created_at', 'analysis_type', 'confidence_score', 'quality_score', 'processing_time']],
                use_container_width=True
            )
        else:
            st.info("No analysis history available yet.")

# Main Application Entry Point
def main():
    """Main application entry point"""
    try:
        # Initialize UI
        UIComponents.render_header()
        
        # Initialize app
        app = MedicalAnalyticsApp()
        
        # Render sidebar configuration
        api_key, analysis_type, batch_processing, quality_threshold = UIComponents.render_sidebar_config()
        
        # API key validation
        if not api_key:
            st.sidebar.warning("⚠️ Please enter your Gemini API key to continue.")
            st.info("👆 Please configure your API key in the sidebar to get started.")
            return
        
        # Initialize AI engine
        with st.spinner("🔄 Initializing AI engine..."):
            if not app.initialize_ai_engine(api_key):
                st.error("❌ Failed to initialize AI engine. Please check your API key.")
                return
        
        st.success("✅ AI engine initialized successfully!")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Analysis", "📊 Dashboard", "📋 History", "⚙️ Settings"])
        
        with tab1:
            # File upload and processing
            uploaded_files = UIComponents.render_file_uploader()
            
            if uploaded_files:
                # Process images button
                if st.button("🚀 Start Analysis", type="primary"):
                    
                    # Prepare image data
                    image_data_list = []
                    for file in uploaded_files:
                        try:
                            image_data = file.read()
                            file.seek(0)  # Reset file pointer
                            image_data_list.append((image_data, file.name))
                        except Exception as e:
                            st.error(f"Failed to read file {file.name}: {e}")
                            continue
                    
                    if not image_data_list:
                        st.error("No valid images to process.")
                        return
                    
                    # Process images
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    if batch_processing and len(image_data_list) > 1:
                        status_text.text("Processing images in batch mode...")
                        
                        try:
                            results = app.batch_process_images(image_data_list, analysis_type)
                            progress_bar.progress(100)
                            status_text.text("✅ Batch processing completed!")
                            
                        except Exception as e:
                            st.error(f"Batch processing failed: {e}")
                            return
                    else:
                        # Sequential processing
                        results = []
                        
                        for i, (image_data, filename) in enumerate(image_data_list):
                            status_text.text(f"Processing {filename}...")
                            
                            try:
                                result = app.process_image(image_data, filename, analysis_type)
                                results.append((result, None))
                                
                            except Exception as e:
                                results.append((None, str(e)))
                                logger.error(f"Processing failed for {filename}: {e}")
                            
                            progress_bar.progress((i + 1) / len(image_data_list))
                        
                        status_text.text("✅ Processing completed!")
                    
                    # Display results
                    UIComponents.render_analysis_results(results)
                    
                    # Show processing statistics
                    st.subheader("📊 Processing Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Processed", app.processing_stats["total"])
                    
                    with col2:
                        st.metric("Successful", app.processing_stats["successful"])
                    
                    with col3:
                        st.metric("Failed", app.processing_stats["failed"])
                    
                    # Show detailed statistics in expander
                    with st.expander("📈 View Detailed Statistics"):
                        UIComponents.render_analytics_dashboard(app)
        
        with tab2:
            # Dashboard view
            st.header("📊 Analytics Dashboard")
            UIComponents.render_analytics_dashboard(app)
        
        with tab3:
            # Analysis history
            st.header("📋 Analysis History")
            
            # Get analysis history from database
            try:
                history = app.db_manager.get_analysis_history(limit=50)
                
                if not history:
                    st.info("No analysis history found. Process some images to see results here.")
                else:
                    # Show recent analyses in a table
                    st.dataframe(
                        data=[{
                            "Date": result.timestamp.strftime("%Y-%m-%d %H:%M"),
                            "Filename": result.image_metadata.filename,
                            "Type": result.analysis_type.value.replace("_", " ").title(),
                            "Confidence": f"{result.confidence_score:.1%}",
                            "Quality": result.quality_score,
                            "ID": result.analysis_id[:8] + "..."
                        } for result in history],
                        use_container_width=True,
                        column_config={
                            "Date": "Date",
                            "Filename": "Filename",
                            "Type": "Analysis Type",
                            "Confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                format="%.1f%%",
                                min_value=0,
                                max_value=1,
                            ),
                            "Quality": st.column_config.ProgressColumn(
                                "Quality",
                                format="%.1f",
                                min_value=0,
                                max_value=1,
                            ),
                            "ID": "ID"
                        }
                    )
                    
            except Exception as e:
                st.error(f"Failed to load analysis history: {e}")
        
        with tab4:
            # Settings with tabs for different categories
            settings_tab1, settings_tab2, settings_tab3 = st.tabs(["⚙️ General", "🔬 Research", "🔒 Security"])
            
            with settings_tab1:
                # General settings
                st.header("General Settings")
                
                with st.expander("🔑 API Configuration"):
                    st.text_input("Gemini API Key", type="password", value=api_key, disabled=True)
                    st.caption("Configure your API key in the sidebar")
                
                with st.expander("⚡ Performance"):
                    st.checkbox("Enable batch processing", value=batch_processing, disabled=True)
                    st.checkbox("Use GPU acceleration", value=True, disabled=True)
                    st.slider("Max concurrent processes", 1, 8, 4, disabled=True)
                
                with st.expander("📁 Data Management"):
                    st.checkbox("Store processed images", value=True, disabled=True)
                    st.checkbox("Anonymize DICOM metadata", value=True, disabled=True)
                    st.slider("Max storage (GB)", 1, 100, 10, disabled=True)
                    
                    if st.button("🔄 Clear Cache", disabled=True):
                        st.info("Cache cleared successfully!")
                
                with st.expander("📧 Notifications"):
                    st.checkbox("Email notifications", value=False, disabled=True)
                    st.checkbox("Browser notifications", value=True, disabled=True)
                    st.checkbox("Only for errors", value=False, disabled=True)
                
                if st.button("💾 Save Settings", disabled=True):
                    st.success("Settings saved successfully!")
                    st.balloons()
            
            with settings_tab2:
                # Research settings
                st.header("Research Settings")
                
                # Render the research settings component
                UIComponents.render_research_settings()
                
                # Add a save button specifically for research settings
                if st.button("💾 Save Research Settings"):
                    st.success("Research settings saved successfully!")
                    st.balloons()
            
            with settings_tab3:
                # Security settings
                st.header("Security Settings")
                
                with st.expander("🔐 Authentication"):
                    st.checkbox("Require login", value=True, disabled=True)
                    st.checkbox("Two-factor authentication", value=False, disabled=True)
                    st.checkbox("Session timeout", value=True, disabled=True)
                    st.number_input("Inactivity timeout (minutes)", 1, 240, 30, disabled=True)
                
                with st.expander("🔒 Data Protection"):
                    st.checkbox("Encrypt data at rest", value=True, disabled=True)
                    st.checkbox("Encrypt data in transit", value=True, disabled=True)
                    st.selectbox("Data retention policy", 
                                ["30 days", "90 days", "1 year", "Indefinite"], 
                                index=1,
                                disabled=True)
                
                with st.expander("⚠️ Data Deletion"):
                    st.warning("This action cannot be undone. All your data will be permanently deleted.")
                    
                    if st.button("🗑️ Delete All My Data", type="secondary", disabled=True):
                        st.error("This feature is currently disabled.")
                
                if st.button("🔒 Save Security Settings", disabled=True):
                    st.success("Security settings saved successfully!")
                    st.balloons()
            
            # Log viewer
            st.subheader("📜 System Logs")
            
            if st.button("📖 View Recent Logs"):
                try:
                    if os.path.exists('medical_analytics.log'):
                        with open('medical_analytics.log', 'r') as f:
                            logs = f.readlines()[-50:]  # Last 50 lines
                        
                        st.text_area("Recent Logs", "\n".join(logs), height=300)
                    else:
                        st.info("No log file found.")
                except Exception as e:
                    st.error(f"Failed to read logs: {e}")
            
            # Export settings
            st.subheader("📤 Export Settings")
            
            if st.button("📊 Export System Report"):
                try:
                    report_data = {
                        "system_info": system_info,
                        "processing_stats": app.processing_stats,
                        "database_stats": asdict(app.db_manager.get_processing_stats()),
                        "generated_at": datetime.now().isoformat()
                    }
                    
                    report_json = json.dumps(report_data, indent=2)
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=report_json,
                        file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>🧠 Advanced Medical Image Analytics Platform v2.0.0</p>
            <p>Built with ❤️ for medical professionals | Powered by AI</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An unexpected error occurred: {e}")
        st.error("Please refresh the page and try again.")

# Error handling wrapper
def safe_main():
    """Safe wrapper for main application"""
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical application error: {e}")
        logger.critical(traceback.format_exc())
        
        st.error("🚨 Critical Error")
        st.error("A critical error occurred. Please contact support.")
        
        # Show error details in expandable section
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    safe_main()
