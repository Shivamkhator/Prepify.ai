# views.py

import os
import tempfile
import json
import logging

# Django imports
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction
from django.utils import timezone

# REST Framework imports
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# Local imports
from .models import Interview, InterviewExchange, AnalysisResult
from .services import audio_processor, ai_analyzer

# Set up logging
logger = logging.getLogger(__name__)


# ========================
# TEMPLATE VIEWS
# ========================

def home(request):
    """Home page view"""
    return render(request, 'interviews/home.html')


def register_view(request):
    """User registration view"""
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')

        # Basic validation
        if not username:
            messages.error(request, 'Username is required!')
            return render(request, 'registration/register.html')

        if len(password) < 6:
            messages.error(request, 'Password must be at least 6 characters long!')
            return render(request, 'registration/register.html')

        if password != confirm_password:
            messages.error(request, 'Passwords do not match!')
            return render(request, 'registration/register.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists!')
            return render(request, 'registration/register.html')

        try:
            user = User.objects.create_user(username=username, password=password)
            login(request, user)
            return redirect('dashboard')
        except Exception as e:
            logger.error(f"Registration failed for user {username}: {e}")
            messages.error(request, f'Registration failed: {str(e)}')
            return render(request, 'registration/register.html')

    return render(request, 'registration/register.html')


def login_view(request):
    """User login view"""
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        if not username or not password:
            messages.error(request, 'Both username and password are required!')
            return render(request, 'registration/login.html')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password!')

    return render(request, 'registration/login.html')


def logout_view(request):
    """User logout view"""
    logout(request)
    return redirect('home')


@login_required
def dashboard(request):
    """User dashboard with interview history"""
    try:
        interviews = Interview.objects.filter(user=request.user).order_by('-date')
        return render(request, 'interviews/dashboard.html', {'interviews': interviews})
    except Exception as e:
        logger.error(f"Dashboard error for user {request.user.username}: {e}")
        messages.error(request, 'Error loading dashboard. Please try again.')
        return render(request, 'interviews/dashboard.html', {'interviews': []})


@login_required
def interview_session(request):
    """Interview session page"""
    return render(request, 'interviews/interview.html')


@login_required
def view_report(request, interview_id):
    """View interview report"""
    try:
        interview = get_object_or_404(Interview, id=interview_id, user=request.user)
        
        if interview.status != 'completed':
            messages.warning(request, 'This interview is not yet completed.')
            return redirect('dashboard')
        
        context = {
            'interview': interview,
            'report': interview.report
        }
        
        return render(request, 'interviews/report.html', context)
        
    except Exception as e:
        logger.error(f"Error viewing report {interview_id}: {e}")
        messages.error(request, 'Error loading report. Please try again.')
        return redirect('dashboard')


# ========================
# API VIEWS
# ========================

@api_view(['GET'])
@login_required
def check_database_state(request):
    """Check the current state of interviews, exchanges, and analysis results"""
    try:
        # Get user's interviews
        interviews = Interview.objects.filter(user=request.user).order_by('-date')[:5]
        
        database_state = []
        
        for interview in interviews:
            interview_data = {
                'interview_id': interview.id,
                'status': interview.status,
                'date': interview.date.isoformat() if interview.date else None,
                'exchanges': []
            }
            
            exchanges = interview.exchanges.all().order_by('timestamp')
            for exchange in exchanges:
                exchange_data = {
                    'exchange_id': exchange.id,
                    'question': exchange.question[:50] + '...' if len(exchange.question) > 50 else exchange.question,
                    'answer': exchange.answer[:50] + '...' if len(exchange.answer) > 50 else exchange.answer,
                    'timestamp': exchange.timestamp.isoformat() if exchange.timestamp else None,
                    'has_analysis': hasattr(exchange, 'analysis'),
                    'analysis_data': None
                }
                
                # Check if analysis exists
                if hasattr(exchange, 'analysis'):
                    try:
                        analysis = exchange.analysis
                        exchange_data['analysis_data'] = {
                            'analysis_id': analysis.id,
                            'tone_score': analysis.tone_score,
                            'grammar_score': analysis.grammar_score,
                            'relevance_score': analysis.relevance_score,
                            'overall_score': analysis.overall_score,
                            'has_feedback': bool(analysis.feedback),
                            'created_at': analysis.created_at.isoformat() if analysis.created_at else None
                        }
                    except Exception as e:
                        exchange_data['analysis_error'] = str(e)
                
                interview_data['exchanges'].append(exchange_data)
            
            database_state.append(interview_data)
        
        # Also get total counts
        total_interviews = Interview.objects.filter(user=request.user).count()
        total_exchanges = InterviewExchange.objects.filter(interview__user=request.user).count()
        total_analysis = AnalysisResult.objects.filter(exchange__interview__user=request.user).count()
        
        return Response({
            'status': 'success',
            'summary': {
                'total_interviews': total_interviews,
                'total_exchanges': total_exchanges,
                'total_analysis_results': total_analysis,
                'analysis_completion_rate': f"{(total_analysis/total_exchanges*100):.1f}%" if total_exchanges > 0 else "0%"
            },
            'recent_interviews': database_state
        })
        
    except Exception as e:
        return Response({
            'status': 'error',
            'error': str(e)
        })


@api_view(['GET'])
@login_required
def check_services(request):
    """Check if all analysis services are working properly"""
    try:
        service_status = {}
        
        # Check audio processor
        try:
            service_status['whisper_model'] = bool(audio_processor.whisper_model)
            service_status['whisper_model_loaded'] = audio_processor.whisper_model is not None
        except Exception as e:
            service_status['whisper_error'] = str(e)
        
        # Check AI analyzer components
        try:
            service_status['sentence_transformer'] = bool(ai_analyzer.st_model)
            service_status['grammar_tool'] = bool(ai_analyzer.grammar_tool)
            service_status['groq_chat'] = bool(ai_analyzer.groq_chat)
        except Exception as e:
            service_status['ai_analyzer_error'] = str(e)
        
        # Test basic functionality
        try:
            # Test grammar analysis with sample text
            test_grammar = ai_analyzer.analyze_grammar("This is a test sentence.")
            service_status['grammar_test'] = test_grammar.get('score', 0)
        except Exception as e:
            service_status['grammar_test_error'] = str(e)
        
        try:
            # Test relevance analysis with sample text
            test_relevance = ai_analyzer.analyze_relevance("I am a software engineer", "Tell me about yourself")
            service_status['relevance_test'] = test_relevance.get('score', 0)
        except Exception as e:
            service_status['relevance_test_error'] = str(e)
        
        # Check model questions
        service_status['total_questions'] = len(ai_analyzer.questions)
        service_status['sample_question'] = ai_analyzer.questions[0] if ai_analyzer.questions else None
        
        return Response({
            'status': 'success',
            'services': service_status,
            'recommendations': [
                'If whisper_model is False, check if faster-whisper is installed correctly',
                'If sentence_transformer is False, check if sentence-transformers is installed',
                'If grammar_tool is False, check if language-tool-python is installed',
                'If tests fail, check internet connection for model downloads'
            ]
        })
        
    except Exception as e:
        return Response({
            'status': 'error',
            'error': str(e)
        })


@api_view(['GET'])
@login_required
def debug_models(request):
    """Debug endpoint to check model structure"""
    try:
        # Check Interview model fields
        from django.db import models
        interview_fields = [f.name for f in Interview._meta.get_fields()]
        
        # Check if we have any interviews
        sample_interview = Interview.objects.filter(user=request.user).first()
        interview_data = None
        if sample_interview:
            interview_data = {
                'id': sample_interview.id,
                'status': sample_interview.status,
                'has_exchanges': sample_interview.exchanges.count(),
            }
            
            # Check exchanges and analysis results
            sample_exchange = sample_interview.exchanges.first()
            if sample_exchange:
                exchange_fields = [f.name for f in InterviewExchange._meta.get_fields()]
                analysis_fields = [f.name for f in AnalysisResult._meta.get_fields()]
                
                return Response({
                    'interview_fields': interview_fields,
                    'exchange_fields': exchange_fields,
                    'analysis_fields': analysis_fields,
                    'sample_interview': interview_data,
                    'has_analysis': hasattr(sample_exchange, 'analysisresult')
                })
        
        return Response({
            'interview_fields': interview_fields,
            'sample_interview': interview_data
        })
        
    except Exception as e:
        return Response({'error': str(e)})


@api_view(['POST'])
@login_required
def start_interview(request):
    """Start a new interview session"""
    try:
        interview = Interview.objects.create(
            user=request.user,
            status='active'
        )
        
        first_question = ai_analyzer.get_current_question(interview)
        
        return Response({
            'interview_id': interview.id,
            'status': 'started',
            'first_question': first_question
        })
        
    except Exception as e:
        logger.error(f"Error starting interview for user {request.user.username}: {e}")
        return Response({
            'error': f'Could not start interview: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@login_required
@csrf_exempt
def process_audio(request):
    """Process uploaded audio and generate analysis"""
    debug_info = {}
    
    try:
        # Validate input
        interview_id = request.data.get('interview_id')
        audio_file = request.FILES.get('audio')

        if not audio_file or not interview_id:
            return Response({
                'error': 'Missing audio file or interview ID'
            }, status=status.HTTP_400_BAD_REQUEST)

        interview = get_object_or_404(Interview, id=interview_id, user=request.user)
        current_question = ai_analyzer.get_current_question(interview)
        
        # Debug: Check audio file info
        debug_info['audio_file_size'] = audio_file.size
        debug_info['audio_file_name'] = audio_file.name
        debug_info['audio_content_type'] = getattr(audio_file, 'content_type', 'unknown')

        # Save uploaded audio to temp file for tone analysis
        tmp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                tmp_audio_path = tmp_file.name
            
            debug_info['temp_file_created'] = os.path.exists(tmp_audio_path)
            debug_info['temp_file_size'] = os.path.getsize(tmp_audio_path) if os.path.exists(tmp_audio_path) else 0

            # Transcribe audio
            logger.info(f"Starting transcription for audio file: {audio_file.name}")
            transcription_result = audio_processor.transcribe_audio(audio_file)
            transcription = transcription_result.get('text', '').strip()
            detected_language = transcription_result.get('language_code', 'en')
            language_name = transcription_result.get('language_name', 'English')
            
            debug_info['transcription_success'] = bool(transcription)
            debug_info['transcription_length'] = len(transcription)
            debug_info['transcription_error'] = transcription_result.get('error', None)

            # Perform analyses with debugging
            logger.info(f"Starting tone analysis for temp file: {tmp_audio_path}")
            tone_score = audio_processor.analyze_tone(tmp_audio_path)
            debug_info['tone_score'] = tone_score
            
            tone_analysis = {
                'score': tone_score,
                'feedback': 'Good tone and confidence' if tone_score > 0.6 else 'Consider speaking with more energy and confidence'
            }

            logger.info(f"Starting grammar analysis for text: {transcription[:50]}...")
            grammar_analysis = ai_analyzer.analyze_grammar(transcription)
            debug_info['grammar_score'] = grammar_analysis.get('score', 0)
            debug_info['grammar_errors'] = grammar_analysis.get('errors', 0)

            logger.info(f"Starting relevance analysis...")
            relevance_analysis = ai_analyzer.analyze_relevance(transcription, current_question)
            debug_info['relevance_score'] = relevance_analysis.get('score', 0)

            overall_score = ai_analyzer.calculate_overall_score(
                tone_analysis, grammar_analysis, relevance_analysis
            )
            debug_info['overall_score'] = overall_score

            # Generate AI response
            logger.info(f"Generating AI response...")
            ai_response = ai_analyzer.generate_response(transcription, interview)
            debug_info['ai_response_generated'] = bool(ai_response)

            # Save exchange and analysis
            with transaction.atomic():
                exchange = InterviewExchange.objects.create(
                    interview=interview,
                    question=current_question,
                    answer=transcription,
                    audio_file=audio_file
                )
                debug_info['exchange_created'] = exchange.id

                # Save analysis result with proper data structure
                analysis_data = {
                    'tone': tone_analysis,
                    'grammar': grammar_analysis,
                    'relevance': relevance_analysis,
                    'overall_score': overall_score,
                    'debug_info': debug_info
                }

                # Create AnalysisResult - the relationship name in your model is 'analysis'
                analysis_result = AnalysisResult.objects.create(
                    exchange=exchange,
                    tone_score=tone_analysis.get('score', 0.5),
                    grammar_score=grammar_analysis.get('score', 0.5),
                    relevance_score=relevance_analysis.get('score', 0.5),
                    overall_score=overall_score,
                    feedback=json.dumps(analysis_data)
                )
                debug_info['analysis_result_created'] = analysis_result.id
                
                # Verify the relationship was created
                exchange.refresh_from_db()
                debug_info['analysis_relationship_exists'] = hasattr(exchange, 'analysis')
                if hasattr(exchange, 'analysis'):
                    debug_info['analysis_relationship_id'] = exchange.analysis.id

            logger.info(f"Audio processing completed successfully for exchange {exchange.id}")

            return Response({
                'transcription': transcription,
                'language': language_name,
                'response': ai_response,
                'debug_info': debug_info,  # Include debug info in response
                'analysis': {
                    'tone_score': tone_analysis.get('score', 0.5),
                    'grammar_score': grammar_analysis.get('score', 0.5),
                    'relevance_score': relevance_analysis.get('score', 0.5),
                    'overall_score': overall_score,
                    'feedback': {
                        'tone': tone_analysis.get('feedback', 'No feedback available'),
                        'grammar': grammar_analysis.get('feedback', ['No feedback available']),
                        'relevance': relevance_analysis.get('feedback', 'No feedback available')
                    }
                }
            })

        finally:
            # Clean up temp file
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                try:
                    os.unlink(tmp_audio_path)
                    debug_info['temp_file_cleaned'] = True
                except Exception as cleanup_error:
                    debug_info['temp_file_cleanup_error'] = str(cleanup_error)

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        debug_info['main_error'] = str(e)
        return Response({
            'error': f'Processing failed: {str(e)}',
            'debug_info': debug_info
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@login_required
def end_interview(request, interview_id):
    """End interview and generate final report"""
    try:
        interview = get_object_or_404(Interview, id=interview_id, user=request.user)

        # Generate final report
        report = ai_analyzer.generate_final_report(interview)

        # Update interview status
        interview.status = 'completed'
        interview.report = report
        interview.save()

        return Response({
            'status': 'completed',
            'report_url': f'/report/{interview.id}/',
            'summary': {
                'total_questions': interview.exchanges.count(),
                'overall_performance': report.get('Overall_Performance', {}).get('average_score', 0.5)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in end_interview: {str(e)}")
        return Response({
            'error': f'Could not end interview: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)