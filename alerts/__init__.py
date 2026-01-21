"""
Alerts Package
==============
Alert and notification system for violence detection events.
"""

from .alert_system import (
    AlertEvent,
    SoundAlert,
    EmailAlert,
    WebhookAlert,
    VideoClipRecorder,
    AlertManager
)

__all__ = [
    'AlertEvent',
    'SoundAlert',
    'EmailAlert',
    'WebhookAlert',
    'VideoClipRecorder',
    'AlertManager'
]
