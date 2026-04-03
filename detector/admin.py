from django.contrib import admin
from .models import DetectionHistory


@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = ('filename', 'input_type', 'is_fake',
                    'confidence', 'timestamp')
    list_filter = ('input_type', 'is_fake')
    search_fields = ('filename',)
    ordering = ('-timestamp',)
