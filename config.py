import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sarvam AI Configuration
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

# Supported languages by Sarvam AI
SUPPORTED_LANGUAGES = {
    'Hindi': 'hi-IN',
    'Bengali': 'bn-IN',
    'Tamil': 'ta-IN',
    'Telugu': 'te-IN',
    'Gujarati': 'gu-IN',
    'Kannada': 'kn-IN',
    'Malayalam': 'ml-IN',
    'Marathi': 'mr-IN',
    'Punjabi': 'pa-IN',
    'Odia': 'od-IN',
    'English': 'en-IN',
    'Auto-detect': 'unknown'
}

# Audio recording settings
AUDIO_FORMAT = 16  # 16-bit
AUDIO_CHANNELS = 1  # Mono
AUDIO_RATE = 16000  # 16kHz sample rate
CHUNK_SIZE = 1024
