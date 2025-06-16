# interviews/admin.py - FIXED VERSION
from django.contrib import admin
from .models import Interview, InterviewExchange, AnalysisResult, InterviewReport

@admin.register(Interview)
class InterviewAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'date', 'status']
    list_filter = ['status', 'date']
    search_fields = ['user__username']
    readonly_fields = ['date']

@admin.register(InterviewExchange)
class InterviewExchangeAdmin(admin.ModelAdmin):
    # ✅ FIXED: Removed tone_score and relevance_score (they're in AnalysisResult now)
    list_display = ['id', 'interview', 'question_preview', 'answer_preview', 'timestamp']
    list_filter = ['timestamp']
    search_fields = ['question', 'answer']
    readonly_fields = ['timestamp']
    
    def question_preview(self, obj):
        """Show first 50 characters of question"""
        return obj.question[:50] + "..." if len(obj.question) > 50 else obj.question
    question_preview.short_description = 'Question'
    
    def answer_preview(self, obj):
        """Show first 50 characters of answer"""
        return obj.answer[:50] + "..." if obj.answer and len(obj.answer) > 50 else obj.answer or "No answer"
    answer_preview.short_description = 'Answer'

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    # ✅ NEW: Admin for analysis results with all the scores
    list_display = ['id', 'exchange', 'overall_score', 'tone_score', 'relevance_score', 'grammar_score', 'created_at']
    list_filter = ['created_at']
    readonly_fields = ['created_at']
    
    def exchange(self, obj):
        return f"Exchange {obj.exchange.id} - Interview {obj.exchange.interview.id}"
    exchange.short_description = 'Exchange'

@admin.register(InterviewReport)
class InterviewReportAdmin(admin.ModelAdmin):
    list_display = ['id', 'interview', 'overall_score', 'created_at']
    readonly_fields = ['created_at']
    
    def interview(self, obj):
        return f"Interview {obj.interview.id} - {obj.interview.user.username}"
    interview.short_description = 'Interview'