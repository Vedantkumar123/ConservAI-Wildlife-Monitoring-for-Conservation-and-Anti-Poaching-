# smtp_server.py
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv
from datetime import datetime, timedelta # Import datetime and timedelta

# Load environment variables from .env file
load_dotenv()

# --- Email Configuration ---
# It's highly recommended to use environment variables for security
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "mail")
SENDER_PASSWORD = os.getenv("SENDER_APP_PASSWORD", "password")
RECIPIENT_EMAIL = "add email"

# --- NEW: Rate Limiting / Cooldown Configuration ---
# Set the cooldown period to 1 minute
ALERT_COOLDOWN = timedelta(minutes=1) 
# This variable will store the timestamp of the last sent alert
last_alert_time = None
def send_alert_email(image_bytes: bytes, label: str, confidence: float, timestamp):
    """
    Connects to the SMTP server and sends an email with the alert details and image,
    respecting a 1-minute cooldown period.
    """
    global last_alert_time # Use the global variable to track the last alert time
    
    current_time = datetime.now()

    # --- NEW: Check if we are in the cooldown period ---
    if last_alert_time and (current_time - last_alert_time) < ALERT_COOLDOWN:
        print(f"Cooldown active. Suppressing new alert for '{label}'.")
        return # Exit the function immediately without sending an email

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("ERROR: Email credentials are not set.")
        return

    try:
        # Create the root message
        msg = MIMEMultipart()
        msg['Subject'] = f"🚨 SECURITY ALERT: '{label.upper()}' Detected!"
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL

        # --- Create the HTML body (no changes here) ---
        body = f"""
        <html>
            <body>
                <h2>Security Alert</h2>
                <p>A potential threat has been detected by the surveillance system.</p>
                <ul>
                    <li><strong>Detected Object:</strong> {label.capitalize()}</li>
                    <li><strong>Confidence Score:</strong> {confidence:.2%}</li>
                    <li><strong>Timestamp:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</li>
                </ul>
                <p>The annotated image is attached to this email.</p>
            </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        # --- Attach the image (no changes here) ---
        image = MIMEImage(image_bytes, name=f"{label}_detection.jpg")
        msg.attach(image)

        # --- Send the email via Gmail's SMTP server (no changes here) ---
        print(f"Connecting to SMTP server to send alert to {RECIPIENT_EMAIL}...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp_server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())

        # --- NEW: Update the last alert time after a successful send ---
        last_alert_time = current_time
        print(f"✅ Alert email sent successfully. Cooldown of 1 minute has started.")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")
