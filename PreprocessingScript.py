#FACT-CHECK AI MULTIMODAL PREPROCESSOR

import re
import html
import unicodedata
import os
import mimetypes
import tempfile
import shutil
import subprocess
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import easyocr
from faster_whisper import WhisperModel
from matplotlib import text
import torch
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_MB = 25
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}


# DATA STRUCTURES

@dataclass
class ProcessingMetadata:
    # Metadata about the processing result
    source_type: str  # 'text', 'image', 'audio', 'video'
    processing_time: float
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    warnings: List[str] = None
    confidence_score: Optional[float] = None
    
    # Misinformation signals
    all_caps_words: Optional[List[str]] = None
    excessive_punctuation: bool = False
    suspicious_patterns: List[str] = None
    
    # OCR specific
    ocr_confidence: Optional[float] = None
    text_regions: Optional[int] = None
    
    # Audio/Video specific
    duration_seconds: Optional[float] = None
    language_detected: Optional[str] = None
    speaker_changes: Optional[int] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.suspicious_patterns is None:
            self.suspicious_patterns = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingResult:
    # Complete processing result with text and metadata
    status: str  # 'success', 'error', 'warning'
    text: str
    metadata: ProcessingMetadata
    raw_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'text': self.text,
            'metadata': self.metadata.to_dict(),
            'raw_data': self.raw_data,
            'error_message': self.error_message
        }

# ENHANCED TEXT CLEANER

class EnhancedFactCheckTextCleaner:
    # Advanced text cleaning with misinformation detection
    
    def __init__(self):
        # Existing patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        self.html_pattern = re.compile(r'<.*?>', re.DOTALL)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        # New patterns for fact-checking
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.excessive_punct = re.compile(r'([!?.]){3,}')
        self.number_with_commas = re.compile(r'(\d+),(\d+)')
        self.citation_pattern = re.compile(r'\[\d+\]|\(\w+,?\s*\d{4}\)')
        
        # Quote normalization
        self.quote_pairs = [
            ('"', '"'), ('"', '"'),  # Curly double quotes
            (''', "'"), (''', "'"),  # Curly single quotes
            ('«', '"'), ('»', '"'),  # French quotes
        ]
        
        # Common misinformation markers
        self.clickbait_phrases = [
            'you won\'t believe', 'doctors hate', 'this one trick',
            'breaking news', 'urgent', 'must read', 'shocking', r'\b(shocking|amazing|incredible|unbelievable)\b',
             r'\bnumber \d+ will\b',
             r'\bwhat happened next\b',
             r'\b(they|he|she) doesn\'t want you to know\b'
        ]
    
    def remove_invisible_chars(self, text: str) -> str:
        # Remove invisible Unicode characters
        return ''.join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    
    def normalize_quotes(self, text: str) -> str:
        # Convert all quote types to standard ASCII quotes
        for fancy, plain in self.quote_pairs:
            text = text.replace(fancy, plain)
        return text
    
    def normalize_numbers(self, text: str) -> str:
        # Normalize number formatting for consistency
        # Remove commas from numbers: 1,000,000 -> 1000000
        text = self.number_with_commas.sub(r'\1\2', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        # Fix spacing and newlines
        # Normalize tabs and spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize multiple newlines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing spaces on lines
        text = re.sub(r' *\n *', '\n', text)
        return text.strip()
    
    #Check for all-caps words 
    def detect_all_caps_words(self, text: str) -> List[str]:
        words = text.split()
        return [w for w in words if len(w) > 3 and w.isupper()]
    
    def detect_excessive_punctuation(self, text: str) -> bool:
        # Detect excessive punctuation (!!!, ???, etc.)
        return bool(self.excessive_punct.search(text))
    
    def detect_clickbait(self, text: str) -> List[str]:
        # Detect clickbait phrases
        text_lower = text.lower()
        found = []
        for phrase in self.clickbait_phrases:
            if phrase in text_lower:
                found.append(phrase)
        return found
    
    def preserve_entities(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        # Extract and preserve important entities
        entities = {
            'mentions': self.mention_pattern.findall(text),
            'hashtags': self.hashtag_pattern.findall(text),
            'citations': self.citation_pattern.findall(text),
            'urls': self.url_pattern.findall(text)
        }
        return text, entities
    
    def clean(
        self,
        text: str,
        preserve_mentions: bool = False,
        preserve_hashtags: bool = False,
        detect_misinformation: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
       
        if not isinstance(text, str):
            return "", {"error": "Input is not a string"}
        
        original_text = text
        metadata = {}
        
        # Extract entities before cleaning
        text, entities = self.preserve_entities(text)
        metadata['entities'] = entities
        
        # Detect misinformation signals (on original text)
        if detect_misinformation:
            metadata['all_caps_words'] = self.detect_all_caps_words(original_text)
            metadata['excessive_punctuation'] = self.detect_excessive_punctuation(original_text)
            metadata['clickbait_phrases'] = self.detect_clickbait(original_text)
            metadata['suspicious_patterns'] = []
            
            if len(metadata['all_caps_words']) > 2 :
                metadata['suspicious_patterns'].append('excessive_caps')
            if metadata['excessive_punctuation']:
                metadata['suspicious_patterns'].append('excessive_punctuation')
            if metadata['clickbait_phrases']:
                metadata['suspicious_patterns'].append('clickbait')
        
        # HTML and encoding
        text = html.unescape(text)
        text = self.html_pattern.sub(" ", text)
        
        # Remove URLs (but already extracted)
        text = self.url_pattern.sub(" ", text)
        
        # Remove emojis
        text = self.emoji_pattern.sub(" ", text)
        
        # Conditionally remove mentions and hashtags
        if not preserve_mentions:
            text = self.mention_pattern.sub(" ", text)
        if not preserve_hashtags:
            text = self.hashtag_pattern.sub(" ", text)
        
        # Normalize quotes
        text = self.normalize_quotes(text)
        
        # Normalize numbers
        text = self.normalize_numbers(text)
        
        # Remove invisible characters
        text = self.remove_invisible_chars(text)
        
        # Fix spacing
        text = self.normalize_whitespace(text)

        return text, metadata
    
# ENHANCED PREPROCESSOR

class EnhancedFactCheckPreprocessor:
    def __init__(
        self,
        ocr_languages: List[str] = None,
        whisper_model_size: str = "base",
        lazy_load: bool = True,
        max_file_size_mb: int = 50,
        allowed_base_dir: Optional[str] = None,
        enable_video_frames: bool = True
    ):
        self.ocr_languages = ocr_languages or ['en']
        self.whisper_model_size = whisper_model_size
        self.lazy_load = lazy_load
        self.max_file_size_mb = max_file_size_mb
        self.allowed_base_dir = Path(allowed_base_dir) if allowed_base_dir else None
        self.enable_video_frames = enable_video_frames
        
        self.cleaner = EnhancedFactCheckTextCleaner()
        self._ocr = None
        self._speech_model = None
        
        if not lazy_load:
            self._load_ocr()
            self._load_speech()
    
    # MODEL LOADERS
        
    def _load_ocr(self):
        # Load OCR model with error handling
        if self._ocr is None:
            try:
                logger.info("Loading EasyOCR model...")
                self._ocr = easyocr.Reader(
                    self.ocr_languages,
                    gpu=torch.cuda.is_available()
                )
                logger.info("OCR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load OCR model: {e}")
                raise
    
    def _load_speech(self):
        # Load Whisper model with error handling
        if self._speech_model is None:
            try:
                logger.info(f"Loading Whisper {self.whisper_model_size} model...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "int8" if device == "cpu" else "float16"
                
                self._speech_model = WhisperModel(
                    self.whisper_model_size,
                    device=device,
                    compute_type=compute_type
                )
                logger.info(f"Whisper model loaded on {device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    @property
    def ocr(self):
        self._load_ocr()
        return self._ocr
    
    @property
    def speech_model(self):
        self._load_speech()
        return self._speech_model
    
    def cleanup(self):
        if self._ocr:
            del self._ocr
            self._ocr = None
        if self._speech_model:
            del self._speech_model
            self._speech_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Resources cleaned up")
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    # VALIDATION & SECURITY
    def _sanitize_path(self, file_path: Union[str, Path]) -> Path:
        try:
            path = Path(file_path).resolve()
        except Exception as e:
            raise ValueError(f"Invalid file path: {e}")
        
        # Check if path is within allowed base directory
        if self.allowed_base_dir:
            try:
                path.relative_to(self.allowed_base_dir)
            except ValueError:
                raise ValueError(f"Path traversal detected: {file_path}")
        
        return path
    
    def _validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        try:
            path = self._sanitize_path(file_path)
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        
        if not path.exists():
            return {"status": "error", "message": "File not found"}
        
        if not path.is_file():
            return {"status": "error", "message": "Path is not a file"}
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_file_size_mb * 1024 * 1024:
            return {
                "status": "error",
                "message": f"File too large: {file_size / 1024 / 1024:.2f}MB > {self.max_file_size_mb}MB"
            }
        
        # Detect MIME type
        try:
            import magic
            mime = magic.from_file(str(path), mime=True)
        except ImportError:
            logger.warning("python-magic not installed, falling back to mimetypes")
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        except Exception as e:
            logger.warning(f"MIME detection failed: {e}")
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        
        # Validate extension matches MIME type
        extension = path.suffix.lower()
        if mime.startswith("image/") and extension not in ALLOWED_IMAGE_EXTENSIONS:
            return {"status": "error", "message": f"Unsupported image extension: {extension}"}
        elif mime.startswith("audio/") and extension not in ALLOWED_AUDIO_EXTENSIONS:
            return {"status": "error", "message": f"Unsupported audio extension: {extension}"}
        elif mime.startswith("video/") and extension not in ALLOWED_VIDEO_EXTENSIONS:
            return {"status": "error", "message": f"Unsupported video extension: {extension}"}
        
        return {
            "status": "ok",
            "mime": mime,
            "size": file_size,
            "path": path
        }
    
    def _get_file_hash(self, file_path: Path) -> str:
        # Generate SHA256 hash of file for caching
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    # TEXT PROCESSING
        
    def process_text(
        self,
        text: str,
        preserve_mentions: bool = False,
        preserve_hashtags: bool = False
    ) -> ProcessingResult:
        
        start_time = datetime.now()
        
        try:
            cleaned, cleaning_metadata = self.cleaner.clean(
                text,
                preserve_mentions=preserve_mentions,
                preserve_hashtags=preserve_hashtags,
                detect_misinformation=True
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metadata = ProcessingMetadata(
                source_type='text',
                processing_time=processing_time,
                all_caps_words=cleaning_metadata.get('all_caps_words'),
                excessive_punctuation=cleaning_metadata.get('excessive_punctuation', False),
                suspicious_patterns=cleaning_metadata.get('suspicious_patterns', [])
            )
            
            return ProcessingResult(
                status='success',
                text=cleaned,
                metadata=metadata,
                raw_data={'cleaning_metadata': cleaning_metadata}
            )
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return ProcessingResult(
                status='error',
                text='',
                metadata=ProcessingMetadata(
                    source_type='text',
                    processing_time=(datetime.now() - start_time).total_seconds()
                ),
                error_message=str(e)
            )
    
        # IMAGE PROCESSING
        
    def process_image(
        self,
        image_path: Union[str, Path],
        return_positions: bool = True
    ) -> ProcessingResult:
        
        start_time = datetime.now()
        
        # Validate file
        validation = self._validate_file(image_path)
        if validation["status"] != "ok":
            return ProcessingResult(
                status='error',
                text='',
                metadata=ProcessingMetadata(
                    source_type='image',
                    processing_time=(datetime.now() - start_time).total_seconds()
                ),
                error_message=validation["message"]
            )
        
        path = validation["path"]
        
        try:
            # Run OCR with detailed output
            logger.info(f"Running OCR on {path}")
            results = self.ocr.readtext(str(path), detail=1)
            
            # Extract text and confidence scores
            text_blocks = []
            positions = []
            confidences = []
            
            for bbox, text, confidence in results:
                text_blocks.append(text)
                confidences.append(confidence)
                if return_positions:
                    positions.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence
                    })
            
            # Join text blocks
            raw_text = "\n".join(text_blocks)
            
            # Clean the text
            cleaned, cleaning_metadata = self.cleaner.clean(raw_text, detect_misinformation=True)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metadata = ProcessingMetadata(
                source_type='image',
                processing_time=processing_time,
                file_size=validation["size"],
                mime_type=validation["mime"],
                ocr_confidence=avg_confidence,
                text_regions=len(text_blocks),
                all_caps_words=cleaning_metadata.get('all_caps_words', 0),
                excessive_punctuation=cleaning_metadata.get('excessive_punctuation', False),
                suspicious_patterns=cleaning_metadata.get('suspicious_patterns', [])
            )
            
            if avg_confidence < 0.5:
                metadata.warnings.append("Low OCR confidence - text may be unreliable")
            
            raw_data = {
                'text_blocks': text_blocks,
                'positions': positions if return_positions else None,
                'confidences': confidences,
                'cleaning_metadata': cleaning_metadata
            }
            
            return ProcessingResult(
                status='success',
                text=cleaned,
                metadata=metadata,
                raw_data=raw_data
            )
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return ProcessingResult(
                status='error',
                text='',
                metadata=ProcessingMetadata(
                    source_type='image',
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    file_size=validation.get("size"),
                    mime_type=validation.get("mime")
                ),
                error_message=str(e)
            )
    
    # AUDIO PROCESSING
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        word_timestamps: bool = True
    ) -> ProcessingResult:
      
        start_time = datetime.now()
        
        # Validate file
        validation = self._validate_file(audio_path)
        if validation["status"] != "ok":
            return ProcessingResult(
                status='error',
                text='',
                metadata=ProcessingMetadata(
                    source_type='audio',
                    processing_time=(datetime.now() - start_time).total_seconds()
                ),
                error_message=validation["message"]
            )
        
        path = validation["path"]
        
        try:
            logger.info(f"Transcribing audio: {path}")
            
            # Transcribe with enhanced options
            segments, info = self.speech_model.transcribe(
                str(path),
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=True,  # Voice activity detection
                condition_on_previous_text=True
            )
            
            # Collect segments
            segment_list = list(segments)
            
            # Extract text
            raw_text = " ".join(segment.text for segment in segment_list)
            
            # Clean text
            cleaned, cleaning_metadata = self.cleaner.clean(raw_text, detect_misinformation=True)
            
            # Calculate average confidence
            avg_confidence = sum(
                segment.avg_logprob for segment in segment_list
            ) / len(segment_list) if segment_list else 0.0
            
            # Estimate speaker changes (simple heuristic: long pauses)
            speaker_changes = sum(
                1 for i in range(1, len(segment_list))
                if segment_list[i].start - segment_list[i-1].end > 2.0
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metadata = ProcessingMetadata(
                source_type='audio',
                processing_time=processing_time,
                file_size=validation["size"],
                mime_type=validation["mime"],
                confidence_score=avg_confidence,
                duration_seconds=info.duration,
                language_detected=info.language,
                speaker_changes=speaker_changes,
                all_caps_words=cleaning_metadata.get('all_caps_words'),
                excessive_punctuation=cleaning_metadata.get('excessive_punctuation', False),
                suspicious_patterns=cleaning_metadata.get('suspicious_patterns', [])
            )
            
            if avg_confidence < -1.0:  # Whisper logprobs are negative
                metadata.warnings.append("Low transcription confidence")
            
            # Prepare detailed segment data
            detailed_segments = []
            for segment in segment_list:
                seg_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'confidence': segment.avg_logprob
                }
                if word_timestamps and hasattr(segment, 'words'):
                    seg_data['words'] = [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        }
                        for word in segment.words
                    ]
                detailed_segments.append(seg_data)
            
            raw_data = {
                'segments': detailed_segments,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'cleaning_metadata': cleaning_metadata
            }
            
            return ProcessingResult(
                status='success',
                text=cleaned,
                metadata=metadata,
                raw_data=raw_data
            )
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return ProcessingResult(
                status='error',
                text='',
                metadata=ProcessingMetadata(
                    source_type='audio',
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    file_size=validation.get("size"),
                    mime_type=validation.get("mime")
                ),
                error_message=str(e)
            )
    
    # VIDEO PROCESSING
        
    def _extract_audio_ffmpeg(
        self,
        video_path: Path,
        temp_audio: Path,
        timeout: int = 300
    ) -> Tuple[bool, Optional[str]]:
        
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            return False, "ffmpeg not found in system PATH"
        
        # Build command with security considerations
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM audio codec
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",  # Mono
            str(temp_audio)
        ]
        
        try:
            logger.info(f"Extracting audio from video: {video_path}")
            result = subprocess.run(
                cmd,
                check=True,
                timeout=timeout,
                capture_output=True,
                text=True
            )
            return True, None
        except subprocess.TimeoutExpired:
            return False, f"Audio extraction timed out after {timeout}s"
        except subprocess.CalledProcessError as e:
            return False, f"ffmpeg error: {e.stderr}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def _extract_video_frames(
        self,
        video_path: Path,
        num_frames: int = 5
    ) -> List[np.ndarray]:
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.warning("Could not read video frames")
                return []
            
            # Extract frames at regular intervals
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    def _ocr_on_frames(self, frames: List[np.ndarray]) -> str:
       
        all_text = []
        
        for i, frame in enumerate(frames):
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run OCR
                results = self.ocr.readtext(frame_rgb, detail=0)
                if results:
                    all_text.append(f"[Frame {i+1}] " + " ".join(results))
            except Exception as e:
                logger.warning(f"OCR failed on frame {i}: {e}")
        
        return "\n".join(all_text)
    
    def process_video(
        self,
        video_path: Union[str, Path],
        extract_frames: Optional[bool] = None,
        language: Optional[str] = None,
        timeout: int = 600
    ) -> ProcessingResult:
        
        start_time = datetime.now()
        
        if extract_frames is None:
            extract_frames = self.enable_video_frames
        
        # Validate file
        validation = self._validate_file(video_path)
        if validation["status"] != "ok":
            return ProcessingResult(
                status='error',
                text='',
                metadata=ProcessingMetadata(
                    source_type='video',
                    processing_time=(datetime.now() - start_time).total_seconds()
                ),
                error_message=validation["message"]
            )
        
        path = validation["path"]
        
        # Create temporary audio file
        temp_audio = Path(tempfile.mktemp(suffix='.wav'))
        
        try:
            # Extract audio
            success, error_msg = self._extract_audio_ffmpeg(path, temp_audio)
            if not success:
                return ProcessingResult(
                    status='error',
                    text='',
                    metadata=ProcessingMetadata(
                        source_type='video',
                        processing_time=(datetime.now() - start_time).total_seconds(),
                        file_size=validation["size"],
                        mime_type=validation["mime"]
                    ),
                    error_message=error_msg
                )
            
            # Transcribe audio
            audio_result = self.process_audio(temp_audio, language=language)
            
            # Extract text from frames if enabled
            frame_text = ""
            if extract_frames:
                logger.info("Extracting frames for OCR...")
                frames = self._extract_video_frames(path)
                if frames:
                    frame_text = self._ocr_on_frames(frames)
            
            # Combine audio transcription and frame text
            combined_text = audio_result.text
            if frame_text:
                combined_text += "\n\n[On-screen text]\n" + frame_text
            
            # Clean combined text
            cleaned, cleaning_metadata = self.cleaner.clean(
                combined_text,
                detect_misinformation=True
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Merge metadata from audio processing
            metadata = audio_result.metadata
            metadata.source_type = 'video'
            metadata.processing_time = processing_time
            metadata.file_size = validation["size"]
            metadata.mime_type = validation["mime"]
            
            # Update suspicious patterns from combined text
            metadata.all_caps_words = cleaning_metadata.get('all_caps_words')
            metadata.excessive_punctuation = cleaning_metadata.get('excessive_punctuation', False)
            metadata.suspicious_patterns = cleaning_metadata.get('suspicious_patterns', [])
            
            raw_data = {
                'audio_transcription': audio_result.text,
                'frame_text': frame_text if frame_text else None,
                'audio_metadata': audio_result.raw_data,
                'cleaning_metadata': cleaning_metadata
            }
            
            return ProcessingResult(
                status='success',
                text=cleaned,
                metadata=metadata,
                raw_data=raw_data
            )
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return ProcessingResult(
                status='error',
                text='',
                metadata=ProcessingMetadata(
                    source_type='video',
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    file_size=validation.get("size"),
                    mime_type=validation.get("mime")
                ),
                error_message=str(e)
            )
        finally:
            # Clean up temporary audio file
            if temp_audio.exists():
                temp_audio.unlink()
    
    # MAIN ROUTER
        
    def process_input(
        self,
        user_input: Union[str, Path],
        **kwargs
    ) -> ProcessingResult:
        
        input_str = str(user_input)
        
        # Check if it's a file path
        if not os.path.exists(input_str):
            # Treat as raw text
            return self.process_text(input_str, **kwargs)
        
        # Validate file
        validation = self._validate_file(input_str)
        if validation["status"] != "ok":
            return ProcessingResult(
                status='error',
                text='',
                metadata=ProcessingMetadata(
                    source_type='unknown',
                    processing_time=0.0
                ),
                error_message=validation["message"]
            )
        
        mime = validation["mime"]
        
        # Route to appropriate processor
        if mime.startswith("image/"):
            return self.process_image(input_str, **kwargs)
        elif mime.startswith("audio/"):
            return self.process_audio(input_str, **kwargs)
        elif mime.startswith("video/"):
            return self.process_video(input_str, **kwargs)
        else:
            # Try to read as text file
            try:
                with open(input_str, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                return self.process_text(text, **kwargs)
            except Exception as e:
                return ProcessingResult(
                    status='error',
                    text='',
                    metadata=ProcessingMetadata(
                        source_type='file',
                        processing_time=0.0,
                        mime_type=mime
                    ),
                    error_message=f"Could not read file as text: {e}"
                )
    
    def process_batch(
        self,
        inputs: List[Union[str, Path]],
        **kwargs
    ) -> List[ProcessingResult]:
       
        results = []
        for i, user_input in enumerate(inputs):
            logger.info(f"Processing input {i+1}/{len(inputs)}")
            result = self.process_input(user_input, **kwargs)
            results.append(result)
        return results