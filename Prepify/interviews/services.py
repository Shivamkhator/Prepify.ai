import os
import tempfile
import json
import numpy as np
import warnings
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

# Text-to-Speech
import pyttsx3
from gtts import gTTS

# Django
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone
from .models import Interview, InterviewExchange, AnalysisResult

# Suppress warnings (from your existing code)
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', message='.*audioread.*')
warnings.filterwarnings('ignore', category=UserWarning, module='soundfile')


def load_audio_file(audio_file_path):
    """Load audio file with multiple fallback options (from your existing code)"""
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
        self.whisper_model = WhisperModel("base", compute_type="int8")  # or "float32" if needed

    def transcribe_audio(self, audio_file) -> Dict[str, Any]:
        """Transcribe audio using Faster Whisper"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name

            segments, info = self.whisper_model.transcribe(tmp_path)
            text = " ".join([segment.text.strip() for segment in segments])
            os.unlink(tmp_path)

            lang_code = info.language or "en"
            language_name = Language.get(lang_code).display_name() if Language.get(lang_code) else "Unknown"

            return {
                "text": text.strip(),
                "confidence": 0.8,
                "language_code": lang_code,
                "language_name": language_name
            }

        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "confidence": 0.0,
                "language_code": "en",
                "language_name": "English"
            }

    def analyze_tone(self, audio_path: str) -> float:
        """Basic tone analysis using average pitch"""
        try:
            y, sr = librosa.load(audio_path)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
            normalized = min(avg_pitch / 500.0, 1.0)  # Normalize assuming 500 Hz max pitch
            return normalized
        except Exception as e:
            print(f"Tone analysis failed: {e}")
            return 0.5  # Neutral score if analysis fails


class AdvancedAIAnalyzer:
    """AI analysis using sentence embeddings and grammar tools"""
    
    def __init__(self):
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.grammar_tool = language_tool_python.LanguageTool('en-US')

        self.ideal_answers = {
            "Tell me about yourself.": "A concise summary of your background, skills, and goals relevant to the role.",
            "What are your greatest strengths?": "Specific strengths with concrete examples.",
            "What is your biggest weakness?": "A genuine weakness with improvement steps.",
        }

        self.questions = [
            "Tell me about yourself and your background.",
            "What are your greatest strengths and how do they apply to this role?",
            "Describe a challenging situation you faced and how you handled it.",
            "What is your biggest weakness and how are you working to improve it?",
            "Why do you want this position and why should we hire you?",
            "Where do you see yourself in 5 years?",
            "Do you have any questions for us?"
        ]

    def get_current_question(self, interview: Interview) -> str:
        exchanges_count = interview.exchanges.count()
        if exchanges_count < len(self.questions):
            return self.questions[exchanges_count]
        return "Thank you for your responses. Any final thoughts?"
    
    

    def generate_response(self, user_input: str, interview: Interview) -> str:
        responses = [
            "That's a great answer! Can you elaborate on that experience?",
            "Interesting perspective. How did that shape your approach?",
            "I see. Can you give me a specific example?",
            "Thank you for sharing. What was the key learning from that?",
        ]
        import random
        return random.choice(responses)

    def analyze_grammar(self, transcription: str) -> Dict[str, Any]:
        try:
            matches = self.grammar_tool.check(transcription)
            return {
                "errors": len(matches),
                "feedback": [str(match) for match in matches] or ["No grammar issues detected"],
                "score": max(0.0, 1.0 - (len(matches) / max(len(transcription.split()), 1)))
            }
        except Exception as e:
            return {
                "errors": 0,
                "feedback": [f"Grammar analysis failed: {str(e)}"],
                "score": 0.8
            }

    def analyze_relevance(self, transcription: str, question: str) -> Dict[str, Any]:
        try:
            ideal_answer = self.ideal_answers.get(question, "A comprehensive answer that addresses the question directly.")
            embeddings = self.st_model.encode([transcription, ideal_answer])
            score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
            return {
                "score": score,
                "feedback": "Highly relevant" if score > 0.7 else "Include more relevant details"
            }
        except Exception as e:
            return {
                "score": 0.5,
                "feedback": f"Relevance analysis failed: {str(e)}"
            }

    def generate_final_report(self, interview: Interview) -> Dict[str, Any]:
        exchanges = interview.exchanges.all()
        if not exchanges:
            return {}

        avg_tone = sum(e.tone_score for e in exchanges) / len(exchanges)
        total_grammar_errors = sum(e.grammar_errors for e in exchanges)
        relevance_scores = [e.relevance_score or 0.0 for e in exchanges]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        return {
            'Tone_Analysis': {
                'pitch': avg_tone,
                'intensity': avg_tone * 0.8,
                'feedback': 'Good overall tone' if avg_tone > 0.6 else 'Work on confidence'
            },
            'Grammar_Summary': {
                'total_errors': total_grammar_errors,
                'feedback_samples': ['Good grammar' if total_grammar_errors == 0 else 'Some grammar issues detected']
            },
            'Relevance_Summary': {
                'average_score': avg_relevance,
                'individual_feedback': [
                    {
                        'question': exchange.question,
                        'score': exchange.relevance_score,
                        'feedback': 'Good response' if exchange.relevance_score > 0.7 else 'Could be more specific'
                    }
                    for exchange in exchanges
                ]
            }
        }
        
    def calculate_overall_score(self, tone_analysis, grammar_analysis, relevance_analysis) -> float:
        try:
            tone_score = tone_analysis.get('score', 0.5) if isinstance(tone_analysis, dict) else tone_analysis
            grammar_score = grammar_analysis.get('score', 0.5)
            relevance_score = relevance_analysis.get('score', 0.5)
            return round((tone_score + grammar_score + relevance_score) / 3.0, 3)
        except Exception as e:
            print(f"Score calculation failed: {e}")
            return 0.5

