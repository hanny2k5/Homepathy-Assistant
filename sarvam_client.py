import requests
import requests
import json
from config import SARVAM_API_KEY

class SarvamSTT:
    def __init__(self):
        self.api_key = SARVAM_API_KEY
        self.base_url = "https://api.sarvam.ai"
        self.headers = {
            "api-subscription-key": self.api_key
        }
    
    def transcribe_audio(self, audio_file_path, language_code="unknown", model="saarika:v2", translate_to_english=False):
        """
        Transcribe audio file to text using Sarvam AI
        
        Args:
            audio_file_path (str): Path to the audio file
            language_code (str): Language code (e.g., 'hi-IN', 'en-IN', 'unknown' for auto-detect)
            model (str): Model to use ('saarika:v2' or 'saaras')
            translate_to_english (bool): If True, uses Saaras model to directly translate to English
        
        Returns:
            dict: Transcription result
        """
        if not self.api_key:
            raise ValueError("Sarvam API key not found. Please set SARVAM_API_KEY in your .env file")
        
        # Use translation workflow for English output
        if translate_to_english:
            return self.transcribe_and_translate(audio_file_path, language_code)
        
        url = f"{self.base_url}/speech-to-text"
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    'file': (audio_file_path, audio_file, 'audio/wav')
                }
                
                data = {
                    'model': model,
                    'language_code': language_code
                }
                
                response = requests.post(
                    url,
                    headers=self.headers,
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        'success': True,
                        'transcript': result.get('transcript', ''),
                        'language_detected': result.get('language_code', language_code),
                        'confidence': result.get('confidence', 0),
                        'translated_to_english': translate_to_english,
                        'full_response': result
                    }
                else:
                    return {
                        'success': False,
                        'error': f"API Error: {response.status_code} - {response.text}",
                        'transcript': ''
                    }
                    
        except FileNotFoundError:
            return {
                'success': False,
                'error': f"Audio file not found: {audio_file_path}",
                'transcript': ''
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {str(e)}",
                'transcript': ''
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'transcript': ''
            }
    
    def transcribe_with_diarization(self, audio_file_path, language_code="unknown", num_speakers=2):
        """
        Transcribe audio with speaker diarization
        
        Args:
            audio_file_path (str): Path to the audio file
            language_code (str): Language code
            num_speakers (int): Number of speakers
        
        Returns:
            dict: Transcription result with speaker information
        """
        if not self.api_key:
            raise ValueError("Sarvam API key not found. Please set SARVAM_API_KEY in your .env file")
        
        url = f"{self.base_url}/speech-to-text"
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    'file': (audio_file_path, audio_file, 'audio/wav')
                }
                
                data = {
                    'model': 'saarika:v2',
                    'language_code': language_code,
                    'with_diarization': 'true',
                    'num_speakers': str(num_speakers)
                }
                
                response = requests.post(
                    url,
                    headers=self.headers,
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        'success': True,
                        'transcript': result.get('transcript', ''),
                        'speakers': result.get('speakers', []),
                        'language_detected': result.get('language_code', language_code),
                        'full_response': result
                    }
                else:
                    return {
                        'success': False,
                        'error': f"API Error: {response.status_code} - {response.text}",
                        'transcript': ''
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': f"Error: {str(e)}",
                'transcript': ''
            }
    
    def transcribe_and_translate(self, audio_file_path, source_language="unknown"):
        """
        Transcribe audio and translate to English using two-step process
        
        Args:
            audio_file_path (str): Path to the audio file
            source_language (str): Source language code (optional, auto-detected if unknown)
        
        Returns:
            dict: Translation result with English text
        """
        # First, transcribe the audio normally
        transcribe_result = self.transcribe_audio(audio_file_path, source_language, model="saarika:v2", translate_to_english=False)
        
        if not transcribe_result['success']:
            return transcribe_result
        
        transcript = transcribe_result['transcript']
        detected_language = transcribe_result['language_detected']
        
        # If already in English, return as is
        if detected_language == 'en-IN' or not transcript.strip():
            transcribe_result['translated_to_english'] = True
            transcribe_result['source_language'] = detected_language
            return transcribe_result
        
        # Translate to English using Sarvam's translate API
        return self.translate_text_to_english(transcript, detected_language, transcribe_result)
    
    def translate_text_to_english(self, text, source_language, original_result):
        """
        Translate text to English using Sarvam AI translation API
        
        Args:
            text (str): Text to translate
            source_language (str): Source language code
            original_result (dict): Original transcription result
        
        Returns:
            dict: Translation result
        """
        if not self.api_key:
            raise ValueError("Sarvam API key not found. Please set SARVAM_API_KEY in your .env file")
        
        # Try multiple translation approaches
        translation_methods = [
            self._try_translate_api,
            self._try_basic_translation,
            self._try_simple_translation
        ]
        
        for method in translation_methods:
            try:
                result = method(text, source_language, original_result)
                if result and result.get('success') and 'Translation failed' not in result.get('transcript', ''):
                    return result
            except Exception as e:
                print(f"Translation method failed: {e}")
                continue
        
        # If all methods fail, return original text with fallback message
        return {
            'success': True,
            'transcript': f"{text} (Translation not available for {source_language})",
            'source_language': source_language,
            'language_detected': source_language,
            'translated_to_english': True,
            'confidence': original_result.get('confidence', 0),
            'original_transcript': text,
            'translation_method': 'Fallback - no translation available'
        }
    
    def _try_translate_api(self, text, source_language, original_result):
        """Try the main translate API"""
        url = f"{self.base_url}/translate"
        
        payload = {
            "input": text,
            "source_language_code": source_language,
            "target_language_code": "en-IN",
            "speaker_gender": "male",
            "mode": "formal",
            "model": "mayura:v1"
        }
        
        response = requests.post(
            url,
            headers={
                **self.headers,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'transcript': result.get('translated_text', text),
                'source_language': source_language,
                'language_detected': source_language,
                'target_language': 'en-IN',
                'translated_to_english': True,
                'confidence': original_result.get('confidence', 0),
                'full_response': result,
                'original_transcript': text
            }
        return None
    
    def _try_basic_translation(self, text, source_language, original_result):
        """Try basic translation with minimal parameters"""
        url = f"{self.base_url}/translate"
        
        payload = {
            "input": text,
            "source_language_code": source_language,
            "target_language_code": "en-IN"
        }
        
        response = requests.post(
            url,
            headers={
                **self.headers,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'transcript': result.get('translated_text', result.get('output', text)),
                'source_language': source_language,
                'language_detected': source_language,
                'target_language': 'en-IN',
                'translated_to_english': True,
                'confidence': original_result.get('confidence', 0),
                'full_response': result,
                'original_transcript': text
            }
        return None
    
    def _try_simple_translation(self, text, source_language, original_result):
        """Try simplest translation approach"""
        # Basic language mapping for common phrases
        simple_translations = {
            'hi-IN': {'नमस्ते': 'Hello', 'धन्यवाद': 'Thank you'},
            'te-IN': {'నమస్కారం': 'Hello', 'ధన్యవాదాలు': 'Thank you'},
            'ta-IN': {'வணக்கம்': 'Hello', 'நன்றி': 'Thank you'},
        }
        
        if source_language in simple_translations:
            for original, english in simple_translations[source_language].items():
                if original in text:
                    translated = text.replace(original, english)
                    return {
                        'success': True,
                        'transcript': f"{translated} (Basic translation)",
                        'source_language': source_language,
                        'language_detected': source_language,
                        'target_language': 'en-IN',
                        'translated_to_english': True,
                        'confidence': original_result.get('confidence', 0),
                        'original_transcript': text
                    }
        
        return None
