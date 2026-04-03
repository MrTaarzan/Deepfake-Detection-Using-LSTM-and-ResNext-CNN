from django.db import models
from django.utils import timezone


class DetectionHistory(models.Model):
    MEDIA_TYPES = [
        ('image', 'Image'),
        ('video', 'Video'),
    ]

    file = models.FileField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    input_type = models.CharField(max_length=5, choices=MEDIA_TYPES)
    timestamp = models.DateTimeField(default=timezone.now)
    is_fake = models.BooleanField()
    confidence = models.FloatField()

    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Detection histories'

    def __str__(self):
        return f"{self.filename} - {'Fake' if self.is_fake else 'Real'} ({self.confidence:.2%})"
