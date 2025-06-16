from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import logging
import tempfile
import os

from .models import Interview, InterviewExchange, AnalysisResult
from .services import AdvancedAudioProcessor, AdvancedAIAnalyzer

logger = logging.getLogger(__name__)

# Initialize services
audio_processor = AdvancedAudioProcessor()
ai_analyzer = AdvancedAIAnalyzer()


def home(request):
    return render(request, 'interviews/home.html')


def register_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password != confirm_password:
            messages.error(request, 'Passwords do not match!')
            return render(request, 'registration/register.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists!')
            return render(request, 'registration/register.html')

        user = User.objects.create_user(username=username, password=password)
        login(request, user)
        return redirect('dashboard')

    return render(request, 'registration/register.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password!')

    return render(request, 'registration/login.html')


def logout_view(request):
    logout(request)
    return redirect('home')


@login_required
def dashboard(request):
    interviews = Interview.objects.filter(user=request.user)
    return render(request, 'interviews/dashboard.html', {
        'interviews': interviews
    })


@login_required
def interview_session(request):
    return render(request, 'interviews/interview.html')


@login_required
def view_report(request, interview_id):
    interview = get_object_or_404(Interview, id=interview_id, user=request.user)
    return render(request, 'interviews/report.html', {
        'interview': interview,
        'report': interview.report
    })


# API Views
@api_view(['POST'])
@login_required
def start_interview(request):
    interview = Interview.objects.create(
        user=request.user,
        status='active'
    )
    return Response({
        'interview_id': interview.id,
        'status': 'started'
    })


@api_view(['POST'])
@login_required
@csrf_exempt
def process_audio(request):
    try:
        interview_id = request.data.get('interview_id')
        audio_file = request.FILES.get('audio')

        if not audio_file or not interview_id:
            return Response({'error': 'Missing audio file or interview ID'}, status=status.HTTP_400_BAD_REQUEST)

        interview = get_object_or_404(Interview, id=interview_id, user=request.user)
        current_question = ai_analyzer.get_current_question(interview)

        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            for chunk in audio_file.chunks():
                tmp_file.write(chunk)
            tmp_audio_path = tmp_file.name

        # Transcribe
        transcription_result = audio_processor.transcribe_audio(audio_file)
        transcription = transcription_result.get('text', '')
        detected_language = transcription_result.get('language_code')
        language_name = transcription_result.get('language_name', 'Unknown')

        # AI Analyses
        tone_score = audio_processor.analyze_tone(tmp_audio_path)
        tone_analysis = {
            'score': tone_score,
            'feedback': 'Good tone' if tone_score > 0.6 else 'Could speak with more energy'
        }

        grammar_analysis = ai_analyzer.analyze_grammar(transcription)
        relevance_analysis = ai_analyzer.analyze_relevance(transcription, current_question)
        overall_score = ai_analyzer.calculate_overall_score(tone_analysis, grammar_analysis, relevance_analysis)

        # AI-generated response
        ai_response = ai_analyzer.generate_response(transcription, interview)

        # Save exchange
        exchange = InterviewExchange.objects.create(
            interview=interview,
            question=current_question,
            answer=transcription,
            audio_file=audio_file
        )

        # Save analysis result
        analysis_result = AnalysisResult.create_from_services_output(exchange, {
            'tone': tone_analysis,
            'grammar': grammar_analysis,
            'relevance': relevance_analysis,
            'overall_score': overall_score
        })

        # Clean up temp file
        os.unlink(tmp_audio_path)

        return Response({
            'transcription': transcription,
            'language': language_name,
            'response': ai_response,
            'analysis': {
                'tone_score': analysis_result.tone_score,
                'grammar_score': analysis_result.grammar_score,
                'relevance_score': analysis_result.relevance_score,
                'overall_score': analysis_result.overall_score,
                'feedback': analysis_result.get_detailed_feedback()
            }
        })

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return Response({'error': f'Processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@login_required
def end_interview(request, interview_id):
    interview = get_object_or_404(Interview, id=interview_id, user=request.user)

    report = ai_analyzer.generate_final_report(interview)

    interview.status = 'completed'
    interview.report = report
    interview.save()

    return Response({
        'status': 'completed',
        'report_url': f'/report/{interview.id}/'
    })
