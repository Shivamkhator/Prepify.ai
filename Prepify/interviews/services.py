# services.py

import os
import tempfile
import json
import numpy as np
import warnings
import logging
from typing import Dict, List, Any
from pathlib import Path

# Audio Processing
from faster_whisper import WhisperModel
import librosa
import soundfile as sf

# AI & Language Processing
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer, util
import language_tool_python
from langcodes import Language

# Text-to-Speech (if needed)
import pyttsx3
from gtts import gTTS

# Django
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

# Models
from .models import Interview, InterviewExchange, AnalysisResult

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', message='.*audioread.*')
warnings.filterwarnings('ignore', category=UserWarning, module='soundfile')

logger = logging.getLogger(__name__)


def load_audio_file(audio_file_path):
    """Load audio file with multiple fallback options"""
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
    try:
        y, sr = sf.read(audio_file_path)
        y = y.astype(np.float32)
        return y, sr
    except Exception as e1:
        try:
            y, sr = librosa.load(audio_file_path, sr=None)
            return y, sr
        except Exception as e2:
            raise Exception(f"Failed to load audio file. SoundFile error: {str(e1)}, Librosa error: {str(e2)}")


class AdvancedAudioProcessor:
    """Audio processing using Faster Whisper and tone analysis"""
    
    def __init__(self):
        try:
            self.whisper_model = WhisperModel("base", compute_type="int8")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            self.whisper_model = None

    def transcribe_audio(self, audio_file) -> Dict[str, Any]:
        """Transcribe audio using Faster Whisper with detailed debugging"""
        try:
            if not self.whisper_model:
                logger.error("Whisper model not available for transcription")
                return {
                    "text": "",
                    "error": "Whisper model not available",
                    "confidence": 0.0,
                    "language_code": "en",
                    "language_name": "English"
                }

            # Get audio file info
            audio_size = audio_file.size if hasattr(audio_file, 'size') else 0
            audio_name = audio_file.name if hasattr(audio_file, 'name') else 'unknown'
            logger.info(f"Starting transcription for audio: {audio_name}, size: {audio_size} bytes")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                chunk_count = 0
                total_written = 0
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                    chunk_count += 1
                    total_written += len(chunk)
                tmp_path = tmp_file.name
            
            logger.info(f"Audio saved to temp file: {tmp_path}, chunks: {chunk_count}, total_bytes: {total_written}")
            
            # Verify temp file was created properly
            if not os.path.exists(tmp_path):
                logger.error(f"Temp file was not created: {tmp_path}")
                return {
                    "text": "",
                    "error": "Failed to create temporary file",
                    "confidence": 0.0,
                    "language_code": "en",
                    "language_name": "English"
                }
            
            temp_file_size = os.path.getsize(tmp_path)
            logger.info(f"Temp file verification: exists={os.path.exists(tmp_path)}, size={temp_file_size}")

            try:
                logger.info("Starting Whisper transcription...")
                segments, info = self.whisper_model.transcribe(tmp_path)
                
                # Process segments
                segment_texts = []
                segment_count = 0
                for segment in segments:
                    segment_texts.append(segment.text.strip())
                    segment_count += 1
                
                text = " ".join(segment_texts)
                logger.info(f"Transcription completed: segments={segment_count}, text_length={len(text)}")
                logger.info(f"Transcription preview: '{text[:100]}...' " if len(text) > 100 else f"Full transcription: '{text}'")
                
            except Exception as whisper_error:
                logger.error(f"Whisper transcription failed: {whisper_error}")
                import traceback
                logger.error(f"Whisper traceback: {traceback.format_exc()}")
                return {
                    "text": "",
                    "error": f"Whisper transcription failed: {str(whisper_error)}",
                    "confidence": 0.0,
                    "language_code": "en",
                    "language_name": "English"
                }
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
                logger.info(f"Temp file cleaned up: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file: {cleanup_error}")

            # Process language info
            lang_code = info.language or "en"
            try:
                language_name = Language.get(lang_code).display_name() if Language.get(lang_code) else "Unknown"
            except Exception as lang_error:
                logger.warning(f"Language processing failed: {lang_error}")
                language_name = "English"

            result = {
                "text": text.strip(),
                "confidence": 0.8,
                "language_code": lang_code,
                "language_name": language_name
            }
            
            logger.info(f"Transcription result: success={bool(text)}, language={language_name}")
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            import traceback
            logger.error(f"Transcription traceback: {traceback.format_exc()}")
            return {
                "text": "",
                "error": str(e),
                "confidence": 0.0,
                "language_code": "en",
                "language_name": "English"
            }

    def analyze_tone_simple(self, transcription: str, confidence: float = 0.8) -> float:
        """Simple tone analysis based on text characteristics and transcription confidence"""
        try:
            if not transcription.strip():
                return 0.5
            
            # Text-based tone indicators
            words = transcription.lower().split()
            word_count = len(words)
            
            # Confidence indicators
            confidence_score = confidence  # From Whisper
            
            # Length indicates engagement (but not too verbose)
            length_score = min(word_count / 20.0, 1.0) if word_count > 0 else 0
            
            # Positive language indicators
            positive_words = ['excellent', 'great', 'good', 'confident', 'successful', 'achieved', 'accomplished', 'effective', 'strong']
            positive_count = sum(1 for word in words if word in positive_words)
            positivity_score = min(positive_count / 3.0, 1.0)
            
            # Hesitation indicators (lower score)
            hesitation_words = ['um', 'uh', 'er', 'like', 'you know', 'basically', 'sort of', 'kind of']
            hesitation_count = sum(1 for word in words if word in hesitation_words)
            hesitation_penalty = min(hesitation_count / 5.0, 0.3)
            
            # Combine factors
            tone_score = (confidence_score * 0.4 + 
                         length_score * 0.3 + 
                         positivity_score * 0.2 + 
                         (1.0 - hesitation_penalty) * 0.1)
            
            # Ensure reasonable bounds
            final_score = max(0.1, min(tone_score, 1.0))
            
            logger.info(f"Simple tone analysis: confidence={confidence_score:.3f}, length={length_score:.3f}, positivity={positivity_score:.3f}, hesitation_penalty={hesitation_penalty:.3f}, final={final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Simple tone analysis failed: {e}")
            return 0.5

    def analyze_tone(self, audio_path: str, transcription: str = "", confidence: float = 0.8) -> float:
        """Tone analysis using smart text-based approach (audio processing disabled for compatibility)"""
        try:
            logger.info(f"Starting tone analysis with text-based approach")
            
            # Use text-based analysis (much more reliable)
            text_score = self.analyze_tone_simple(transcription, confidence)
            
            logger.info(f"Tone analysis completed using text-based method: {text_score:.3f}")
            return text_score
            
        except Exception as e:
            logger.error(f"Text-based tone analysis failed: {e}")
            return 0.5


class AdvancedAIAnalyzer:
    """AI analysis using sentence embeddings, grammar tools, and Groq LLM"""

    def __init__(self):
        try:
            self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.st_model = None
            
        try:
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
        except Exception as e:
            logger.error(f"Failed to load grammar tool: {e}")
            self.grammar_tool = None

        self.questions = [
            "Tell me about yourself and your background.",
            "What are your greatest strengths and how do they apply to this role?",
            "Describe a challenging situation you faced and how you handled it.",
            "What is your biggest weakness and how are you working to improve it?",
            "Why do you want this position and why should we hire you?",
            "Where do you see yourself in 5 years?",
            "Do you have any questions for us?"
        ]

        self.ideal_answers = {
            self.questions[0]: "A concise summary of your background, skills, and goals relevant to the role.",
            self.questions[1]: "Specific strengths with concrete examples.",
            self.questions[2]: "A real-life example of a challenge, actions taken, and results.",
            self.questions[3]: "A genuine weakness and the steps you've taken to improve.",
            self.questions[4]: "Alignment of your goals and the company's mission.",
            self.questions[5]: "A realistic, growth-oriented vision for your career.",
            self.questions[6]: "Thoughtful questions about the company or role.",
        }

        try:
            self.groq_chat = ChatGroq(model="llama3-8b-8192")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
            self.groq_chat = None

    def get_current_question(self, interview: Interview) -> str:
        """Get the current question based on number of exchanges"""
        exchanges_count = interview.exchanges.count()
        if exchanges_count < len(self.questions):
            return self.questions[exchanges_count]
        return "Thank you for your responses. Any final thoughts?"

    def generate_response(self, user_input: str, interview: Interview) -> str:
        """Use Groq LLM to generate a follow-up reply."""
        try:
            if not self.groq_chat:
                return "Thank you for your response. Could you provide more details?"
                
            messages = [
                SystemMessage(content="You're a professional interviewer. Provide helpful, polite, follow-up feedback in 1-2 sentences."),
                HumanMessage(content=f"Candidate responded: {user_input}")
            ]
            groq_response = self.groq_chat.invoke(messages)
            return groq_response.content.strip()
        except Exception as e:
            logger.error(f"Groq response generation failed: {e}")
            return "Thank you for sharing. Could you give an example from your experience?"

    def analyze_grammar(self, transcription: str) -> Dict[str, Any]:
        """Analyze grammar using LanguageTool"""
        try:
            if not self.grammar_tool or not transcription.strip():
                return {
                    "errors": 0,
                    "feedback": ["Grammar analysis not available"],
                    "score": 0.8
                }
                
            matches = self.grammar_tool.check(transcription)
            error_count = len(matches)
            
            return {
                "errors": error_count,
                "feedback": [str(match) for match in matches[:5]] or ["No grammar issues detected"],  # Limit feedback
                "score": max(0.0, 1.0 - (error_count / max(len(transcription.split()), 1)))
            }
        except Exception as e:
            logger.error(f"Grammar analysis failed: {e}")
            return {
                "errors": 0,
                "feedback": [f"Grammar analysis failed: {str(e)}"],
                "score": 0.8
            }

    def analyze_relevance(self, transcription: str, question: str) -> Dict[str, Any]:
        """Analyze answer relevance using sentence embeddings"""
        try:
            if not self.st_model or not transcription.strip():
                return {
                    "score": 0.5,
                    "feedback": "Could not analyze relevance"
                }
                
            ideal_answer = self.ideal_answers.get(question, "A comprehensive answer that addresses the question directly.")
            embeddings = self.st_model.encode([transcription, ideal_answer])
            score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
            
            return {
                "score": score,
                "feedback": "Highly relevant response" if score > 0.7 else "Could include more relevant details"
            }
        except Exception as e:
            logger.error(f"Relevance analysis failed: {e}")
            return {
                "score": 0.5,
                "feedback": f"Relevance analysis failed: {str(e)}"
            }

    def calculate_overall_score(self, tone_analysis, grammar_analysis, relevance_analysis) -> float:
        """Calculate overall score from individual analysis components"""
        try:
            tone_score = tone_analysis.get('score', 0.5) if isinstance(tone_analysis, dict) else (tone_analysis if tone_analysis else 0.5)
            grammar_score = grammar_analysis.get('score', 0.5) if isinstance(grammar_analysis, dict) else 0.5
            relevance_score = relevance_analysis.get('score', 0.5) if isinstance(relevance_analysis, dict) else 0.5
            
            overall = (tone_score + grammar_score + relevance_score) / 3.0
            return round(overall, 3)
        except Exception as e:
            logger.error(f"Score calculation failed: {e}")
            return 0.5

    def generate_final_report(self, interview: Interview) -> Dict[str, Any]:
        """Generate comprehensive final interview report"""
        try:
            exchanges = interview.exchanges.all()
            
            if not exchanges:
                return {
                    'Tone_Analysis': {
                        'pitch': 0.5,
                        'intensity': 0.4,
                        'feedback': 'No data available for tone analysis'
                    },
                    'Grammar_Summary': {
                        'total_errors': 0,
                        'average_score': 0.5,
                        'feedback_samples': ['No data available']
                    },
                    'Relevance_Summary': {
                        'average_score': 0.5,
                        'individual_feedback': []
                    },
                    'Overall_Performance': {
                        'average_score': 0.5,
                        'total_responses': 0,
                        'completion_status': 'no_data'
                    }
                }

            # Collect analysis data safely
            tone_scores = []
            grammar_scores = []
            relevance_scores = []
            overall_scores = []
            total_grammar_errors = 0
            individual_feedback = []

            for exchange in exchanges:
                try:
                    # Use the correct relationship name 'analysis' instead of 'analysisresult'
                    if hasattr(exchange, 'analysis'):
                        analysis = exchange.analysis
                        tone_scores.append(getattr(analysis, 'tone_score', 0.5) or 0.5)
                        grammar_scores.append(getattr(analysis, 'grammar_score', 0.5) or 0.5)
                        relevance_scores.append(getattr(analysis, 'relevance_score', 0.5) or 0.5)
                        overall_scores.append(getattr(analysis, 'overall_score', 0.5) or 0.5)
                        
                        # Extract grammar errors from feedback if available
                        feedback_attr = getattr(analysis, 'feedback', None)
                        if feedback_attr:
                            try:
                                feedback_data = json.loads(feedback_attr) if isinstance(feedback_attr, str) else feedback_attr
                                grammar_data = feedback_data.get('grammar', {})
                                total_grammar_errors += grammar_data.get('errors', 0)
                            except:
                                pass
                        
                        individual_feedback.append({
                            'question': getattr(exchange, 'question', 'Unknown question') or 'Unknown question',
                            'score': getattr(analysis, 'relevance_score', 0.5) or 0.5,
                            'feedback': 'Good response' if (getattr(analysis, 'relevance_score', 0.5) or 0) > 0.7 else 'Could be more specific'
                        })
                except Exception as e:
                    logger.error(f"Error processing exchange {exchange.id}: {e}")
                    continue

            # Calculate averages
            avg_tone = sum(tone_scores) / len(tone_scores) if tone_scores else 0.5
            avg_grammar = sum(grammar_scores) / len(grammar_scores) if grammar_scores else 0.5
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
            avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.5

            return {
                'Tone_Analysis': {
                    'pitch': round(avg_tone, 2),
                    'intensity': round(avg_tone * 0.8, 2),
                    'feedback': 'Good overall tone and confidence' if avg_tone > 0.6 else 'Consider speaking with more energy and confidence'
                },
                'Grammar_Summary': {
                    'total_errors': total_grammar_errors,
                    'average_score': round(avg_grammar, 2),
                    'feedback_samples': ['Excellent grammar' if total_grammar_errors == 0 else f'{total_grammar_errors} grammar issues detected']
                },
                'Relevance_Summary': {
                    'average_score': round(avg_relevance, 2),
                    'individual_feedback': individual_feedback
                },
                'Overall_Performance': {
                    'average_score': round(avg_overall, 2),
                    'total_responses': len(exchanges),
                    'completion_status': 'completed'
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
            return {
                'error': f'Could not generate report: {str(e)}',
                'Tone_Analysis': {'pitch': 0.5, 'intensity': 0.4, 'feedback': 'Analysis failed'},
                'Grammar_Summary': {'total_errors': 0, 'average_score': 0.5, 'feedback_samples': ['Analysis failed']},
                'Relevance_Summary': {'average_score': 0.5, 'individual_feedback': []},
                'Overall_Performance': {'average_score': 0.5, 'total_responses': 0, 'completion_status': 'failed'}
            }

    def get_interview_progress(self, interview: Interview) -> Dict[str, Any]:
        """Get current interview progress information"""
        exchanges_count = interview.exchanges.count()
        total_questions = len(self.questions)
        
        return {
            'current_question_number': exchanges_count + 1,
            'total_questions': total_questions,
            'progress_percentage': round((exchanges_count / total_questions) * 100, 1),
            'questions_remaining': total_questions - exchanges_count,
            'is_final_question': exchanges_count >= total_questions - 1
        }

    def get_question_by_number(self, question_number: int) -> str:
        """Get a specific question by its number (1-indexed)"""
        if 1 <= question_number <= len(self.questions):
            return self.questions[question_number - 1]
        return "Invalid question number"


# Service instances (to be imported by views)
def get_audio_processor():
    """Factory function to get audio processor instance"""
    return AdvancedAudioProcessor()


def get_ai_analyzer():
    """Factory function to get AI analyzer instance"""
    return AdvancedAIAnalyzer()


# Global instances for the application
audio_processor = AdvancedAudioProcessor()
ai_analyzer = AdvancedAIAnalyzer()