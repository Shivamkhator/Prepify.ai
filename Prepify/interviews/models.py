# interviews/models.py - FIXED VERSION WITH COMPATIBILITY
from django.db import models
from django.contrib.auth.models import User
import json

class Interview(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='active')  # 'active', 'completed'
    report = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"Interview {self.id} - {self.user.username}"
    
    def get_average_scores(self):
        """Calculate average scores from all exchanges"""
        analyses = AnalysisResult.objects.filter(exchange__interview=self)
        if not analyses.exists():
            return {
                'tone': 0.0,
                'grammar': 0.0, 
                'relevance': 0.0,
                'overall': 0.0
            }
        
        return {
            'tone': analyses.aggregate(avg=models.Avg('tone_score'))['avg'] or 0.0,
            'grammar': analyses.aggregate(avg=models.Avg('grammar_score'))['avg'] or 0.0,
            'relevance': analyses.aggregate(avg=models.Avg('relevance_score'))['avg'] or 0.0,
            'overall': analyses.aggregate(avg=models.Avg('overall_score'))['avg'] or 0.0,
        }

class InterviewExchange(models.Model):
    """Exchange between interviewer and candidate"""
    interview = models.ForeignKey(Interview, on_delete=models.CASCADE, related_name='exchanges')
    question = models.TextField()
    answer = models.TextField(blank=True, null=True)
    audio_file = models.FileField(upload_to='interview_audio/', null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Exchange {self.id} - Interview {self.interview.id}"

class AnalysisResult(models.Model):
    """Analysis results for each exchange - FIXED to match services.py output"""
    exchange = models.OneToOneField(InterviewExchange, on_delete=models.CASCADE, related_name='analysis')
    
    # Individual scores (0.0 to 1.0)
    tone_score = models.FloatField(default=0.5)
    grammar_score = models.FloatField(default=0.5)
    relevance_score = models.FloatField(default=0.5)
    overall_score = models.FloatField(default=0.5)
    
    # Store the full analysis JSON from services
    raw_analysis = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    @classmethod
    def create_from_services_output(cls, exchange, services_analysis):
        """
        Create AnalysisResult from services.py output
        This bridges the gap between services and models
        """
        # Extract individual scores from nested dictionaries
        tone_data = services_analysis.get('tone', {})
        relevance_data = services_analysis.get('relevance', {})
        grammar_data = services_analysis.get('grammar', {})
        
        # Calculate tone score from pitch and intensity
        tone_score = (
            tone_data.get('pitch', 0.5) + 
            tone_data.get('intensity', 0.5)
        ) / 2
        
        # Get relevance score directly
        relevance_score = relevance_data.get('score', 0.5)
        
        # Calculate grammar score from error count
        error_count = grammar_data.get('error_count', 0)
        grammar_score = max(0.0, 1.0 - (error_count * 0.1))
        
        # Get overall score
        overall_score = services_analysis.get('overall_score', 0.5)
        
        # Create the analysis result
        analysis = cls.objects.create(
            exchange=exchange,
            tone_score=tone_score,
            grammar_score=grammar_score,
            relevance_score=relevance_score,
            overall_score=overall_score,
            raw_analysis=services_analysis  # Store full analysis for reference
        )
        
        return analysis
    
    def get_detailed_feedback(self):
        """Get human-readable feedback from raw analysis"""
        if not self.raw_analysis:
            return "No detailed analysis available"
        
        feedback_parts = []
        
        # Tone feedback
        tone_feedback = self.raw_analysis.get('tone', {}).get('feedback', '')
        if tone_feedback:
            feedback_parts.append(f"Speaking Style: {tone_feedback}")
        
        # Relevance feedback  
        relevance_feedback = self.raw_analysis.get('relevance', {}).get('feedback', '')
        if relevance_feedback:
            feedback_parts.append(f"Content Relevance: {relevance_feedback}")
        
        # Grammar feedback
        grammar_feedback = self.raw_analysis.get('grammar', {}).get('feedback', '')
        if grammar_feedback:
            feedback_parts.append(f"Language Quality: {grammar_feedback}")
        
        return " | ".join(feedback_parts) if feedback_parts else "Analysis completed"
    
    def __str__(self):
        return f"Analysis for Exchange {self.exchange.id} (Score: {self.overall_score:.2f})"

class InterviewReport(models.Model):
    """Overall interview report"""
    interview = models.OneToOneField(Interview, on_delete=models.CASCADE, related_name='detailed_report')
    
    # Overall scores - calculated from individual analyses
    average_tone_score = models.FloatField(default=0.0)
    average_relevance_score = models.FloatField(default=0.0)
    average_grammar_score = models.FloatField(default=0.0)
    overall_score = models.FloatField(default=0.0)
    
    # Detailed feedback
    strengths = models.TextField(blank=True)
    areas_for_improvement = models.TextField(blank=True)
    detailed_feedback = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    @classmethod
    def generate_from_interview(cls, interview):
        """Generate report from completed interview"""
        scores = interview.get_average_scores()
        
        # Generate strengths and improvements based on scores
        strengths = []
        improvements = []
        
        if scores['tone'] > 0.7:
            strengths.append("Excellent speaking confidence and tone")
        elif scores['tone'] < 0.4:
            improvements.append("Work on speaking with more confidence and clarity")
        
        if scores['relevance'] > 0.7:
            strengths.append("Highly relevant and well-structured responses")
        elif scores['relevance'] < 0.4:
            improvements.append("Focus more on answering the specific question asked")
        
        if scores['grammar'] > 0.8:
            strengths.append("Excellent communication and language skills")
        elif scores['grammar'] < 0.5:
            improvements.append("Review grammar and sentence structure")
        
        # Default messages if no specific feedback
        if not strengths:
            strengths.append("Completed the interview with good effort")
        if not improvements:
            improvements.append("Continue practicing interview skills")
        
        report, created = cls.objects.get_or_create(
            interview=interview,
            defaults={
                'average_tone_score': scores['tone'],
                'average_relevance_score': scores['relevance'], 
                'average_grammar_score': scores['grammar'],
                'overall_score': scores['overall'],
                'strengths': " • ".join(strengths),
                'areas_for_improvement': " • ".join(improvements),
                'detailed_feedback': f"Interview completed with {interview.exchanges.count()} questions answered."
            }
        )
        
        return report
    
    def __str__(self):
        return f"Report for Interview {self.interview.id} (Score: {self.overall_score:.2f})"