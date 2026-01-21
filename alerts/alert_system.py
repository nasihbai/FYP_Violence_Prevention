"""
Alert Notification System
=========================
Comprehensive alert system for violence detection events.

Supports:
- Sound alerts
- Email notifications
- Webhook notifications (Slack, Discord, etc.)
- Screenshot capture
- Video clip recording
- Logging
"""

import os
import cv2
import time
import logging
import smtplib
import threading
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from dataclasses import dataclass
from collections import deque
from queue import Queue
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)


@dataclass
class AlertEvent:
    """Data class for alert events."""
    timestamp: datetime
    frame: Optional[Any]  # np.ndarray
    detections: List[Dict]
    confidence: float
    source: str
    event_id: str


class SoundAlert:
    """Sound alert handler using system audio."""

    def __init__(self, sound_file: Optional[str] = None):
        """
        Initialize sound alert.

        Args:
            sound_file: Path to WAV file for alert sound
        """
        self.sound_file = sound_file
        self._available = False

        # Check for available audio backend
        try:
            import winsound
            self._backend = 'winsound'
            self._available = True
        except ImportError:
            try:
                import pygame
                pygame.mixer.init()
                self._backend = 'pygame'
                self._available = True
            except ImportError:
                try:
                    # Linux/Mac fallback
                    self._backend = 'system'
                    self._available = True
                except Exception:
                    logger.warning("No audio backend available for sound alerts")

    def play(self):
        """Play alert sound."""
        if not self._available:
            return

        try:
            if self._backend == 'winsound':
                import winsound
                if self.sound_file and os.path.exists(self.sound_file):
                    winsound.PlaySound(self.sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    winsound.Beep(1000, 500)  # Fallback beep

            elif self._backend == 'pygame':
                import pygame
                if self.sound_file and os.path.exists(self.sound_file):
                    pygame.mixer.Sound(self.sound_file).play()

            elif self._backend == 'system':
                if self.sound_file and os.path.exists(self.sound_file):
                    os.system(f'aplay {self.sound_file} &' if os.name != 'nt' else '')
                else:
                    print('\a')  # Terminal bell

        except Exception as e:
            logger.error(f"Failed to play sound: {e}")


class EmailAlert:
    """Email notification handler."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipients: List[str]
    ):
        """
        Initialize email alert.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port
            sender_email: Sender email address
            sender_password: Sender password (app-specific)
            recipients: List of recipient emails
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipients = recipients

    def send(self, event: AlertEvent, screenshot_path: Optional[str] = None):
        """
        Send email alert.

        Args:
            event: Alert event
            screenshot_path: Optional path to screenshot
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"🚨 Violence Alert - {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

            # Email body
            body = f"""
Violence Detection Alert
========================

Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Source: {event.source}
Confidence: {event.confidence:.2%}
Event ID: {event.event_id}

Detections:
"""
            for det in event.detections:
                body += f"  - Person {det.get('person_id', 'N/A')}: {det.get('class_name', 'N/A')} ({det.get('confidence', 0):.2%})\n"

            body += """

This is an automated alert from the Violence Detection System.
Please review the incident immediately.
"""
            msg.attach(MIMEText(body, 'plain'))

            # Attach screenshot if available
            if screenshot_path and os.path.exists(screenshot_path):
                with open(screenshot_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(screenshot_path))
                    msg.attach(img)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"Email alert sent to {len(self.recipients)} recipients")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


class WebhookAlert:
    """Webhook notification handler for Slack, Discord, etc."""

    def __init__(self, webhook_url: str, platform: str = 'generic'):
        """
        Initialize webhook alert.

        Args:
            webhook_url: Webhook URL
            platform: Platform type ('slack', 'discord', 'generic')
        """
        self.webhook_url = webhook_url
        self.platform = platform

    def send(self, event: AlertEvent):
        """Send webhook notification."""
        try:
            if self.platform == 'slack':
                payload = self._format_slack(event)
            elif self.platform == 'discord':
                payload = self._format_discord(event)
            else:
                payload = self._format_generic(event)

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=10)

            logger.info(f"Webhook alert sent to {self.platform}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def _format_slack(self, event: AlertEvent) -> dict:
        """Format payload for Slack."""
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚨 Violence Detection Alert",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Time:*\n{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"},
                        {"type": "mrkdwn", "text": f"*Confidence:*\n{event.confidence:.2%}"},
                        {"type": "mrkdwn", "text": f"*Source:*\n{event.source}"},
                        {"type": "mrkdwn", "text": f"*Event ID:*\n{event.event_id}"}
                    ]
                }
            ]
        }

    def _format_discord(self, event: AlertEvent) -> dict:
        """Format payload for Discord."""
        return {
            "embeds": [{
                "title": "🚨 Violence Detection Alert",
                "color": 16711680,  # Red
                "fields": [
                    {"name": "Time", "value": event.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "inline": True},
                    {"name": "Confidence", "value": f"{event.confidence:.2%}", "inline": True},
                    {"name": "Source", "value": event.source, "inline": True},
                    {"name": "Event ID", "value": event.event_id, "inline": False}
                ],
                "timestamp": event.timestamp.isoformat()
            }]
        }

    def _format_generic(self, event: AlertEvent) -> dict:
        """Format generic JSON payload."""
        return {
            "alert_type": "violence_detection",
            "timestamp": event.timestamp.isoformat(),
            "confidence": event.confidence,
            "source": event.source,
            "event_id": event.event_id,
            "detections": event.detections
        }


class VideoClipRecorder:
    """Records video clips around alert events."""

    def __init__(
        self,
        output_dir: str,
        buffer_seconds: int = 10,
        fps: int = 30
    ):
        """
        Initialize video clip recorder.

        Args:
            output_dir: Directory to save clips
            buffer_seconds: Seconds to buffer before alert
            fps: Frame rate for recording
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_seconds = buffer_seconds
        self.fps = fps

        self._frame_buffer = deque(maxlen=buffer_seconds * fps)
        self._lock = threading.Lock()

    def add_frame(self, frame):
        """Add frame to buffer."""
        with self._lock:
            self._frame_buffer.append(frame.copy())

    def save_clip(self, event: AlertEvent, post_alert_frames: List = None) -> str:
        """
        Save video clip.

        Args:
            event: Alert event
            post_alert_frames: Frames recorded after alert

        Returns:
            Path to saved clip
        """
        filename = f"alert_{event.event_id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = self.output_dir / filename

        with self._lock:
            frames = list(self._frame_buffer)

        if post_alert_frames:
            frames.extend(post_alert_frames)

        if not frames:
            return ""

        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (w, h))

            for frame in frames:
                writer.write(frame)

            writer.release()
            logger.info(f"Video clip saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save video clip: {e}")
            return ""


class AlertManager:
    """
    Central alert management system.

    Coordinates all alert types with proper debouncing
    and async notification handling.
    """

    def __init__(
        self,
        cooldown_seconds: int = 10,
        sound_file: Optional[str] = None,
        screenshot_dir: Optional[str] = None,
        video_clip_dir: Optional[str] = None
    ):
        """
        Initialize alert manager.

        Args:
            cooldown_seconds: Minimum time between alerts
            sound_file: Path to alert sound file
            screenshot_dir: Directory to save screenshots
            video_clip_dir: Directory to save video clips
        """
        self.cooldown_seconds = cooldown_seconds
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        self._last_alert_time = 0
        self._event_counter = 0
        self._lock = threading.Lock()

        # Create directories
        if self.screenshot_dir:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Alert handlers
        self.sound_alert = SoundAlert(sound_file) if sound_file else None
        self.email_alert: Optional[EmailAlert] = None
        self.webhook_alert: Optional[WebhookAlert] = None
        self.video_recorder = VideoClipRecorder(video_clip_dir) if video_clip_dir else None

        # Alert queue for async processing
        self._alert_queue: Queue = Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Callbacks
        self._callbacks: List[Callable[[AlertEvent], None]] = []

    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipients: List[str]
    ):
        """Configure email alerts."""
        self.email_alert = EmailAlert(
            smtp_server, smtp_port, sender_email, sender_password, recipients
        )

    def configure_webhook(self, webhook_url: str, platform: str = 'generic'):
        """Configure webhook alerts."""
        self.webhook_alert = WebhookAlert(webhook_url, platform)

    def add_callback(self, callback: Callable[[AlertEvent], None]):
        """Add custom alert callback."""
        self._callbacks.append(callback)

    def start(self):
        """Start alert processing thread."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Stop alert processing."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def trigger_alert(
        self,
        frame,
        detections: List[Dict],
        confidence: float,
        source: str = "camera"
    ) -> bool:
        """
        Trigger an alert if cooldown has passed.

        Args:
            frame: Current video frame
            detections: List of detection dictionaries
            confidence: Overall confidence score
            source: Video source identifier

        Returns:
            True if alert was triggered, False if in cooldown
        """
        current_time = time.time()

        with self._lock:
            if current_time - self._last_alert_time < self.cooldown_seconds:
                return False

            self._last_alert_time = current_time
            self._event_counter += 1
            event_id = f"EVT{self._event_counter:06d}"

        # Create alert event
        event = AlertEvent(
            timestamp=datetime.now(),
            frame=frame.copy() if frame is not None else None,
            detections=detections,
            confidence=confidence,
            source=source,
            event_id=event_id
        )

        # Queue for processing
        self._alert_queue.put(event)
        return True

    def _process_alerts(self):
        """Process alert queue."""
        while self._running:
            try:
                event = self._alert_queue.get(timeout=0.5)
            except Exception:
                continue

            # Play sound
            if self.sound_alert:
                self.sound_alert.play()

            # Save screenshot
            screenshot_path = None
            if self.screenshot_dir and event.frame is not None:
                screenshot_path = self._save_screenshot(event)

            # Send email
            if self.email_alert:
                threading.Thread(
                    target=self.email_alert.send,
                    args=(event, screenshot_path),
                    daemon=True
                ).start()

            # Send webhook
            if self.webhook_alert:
                threading.Thread(
                    target=self.webhook_alert.send,
                    args=(event,),
                    daemon=True
                ).start()

            # Custom callbacks
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

            logger.info(f"Alert processed: {event.event_id}")

    def _save_screenshot(self, event: AlertEvent) -> str:
        """Save screenshot of alert frame."""
        filename = f"alert_{event.event_id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.screenshot_dir / filename

        try:
            cv2.imwrite(str(filepath), event.frame)
            logger.info(f"Screenshot saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return ""

    def add_frame_to_buffer(self, frame):
        """Add frame to video buffer."""
        if self.video_recorder:
            self.video_recorder.add_frame(frame)
