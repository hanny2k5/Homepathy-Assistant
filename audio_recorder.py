import threading
import time
from config import AUDIO_FORMAT, AUDIO_CHANNELS, AUDIO_RATE, CHUNK_SIZE

try:
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ PyAudio not available. Audio recording will be disabled.")
    print("💡 To enable audio recording, install PyAudio:")
    print("   pip install PyAudio")

class AudioRecorder:
    def __init__(self):
        if not AUDIO_AVAILABLE:
            raise ImportError("PyAudio is not available. Please install it to use audio recording.")
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.frames = []
        self.stream = None
        
    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return False
            
        self.frames = []
        self.is_recording = True
        
        # Configure audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        
        return True
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        
        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        return self.frames
    
    def _record(self):
        """Internal method to record audio frames"""
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"Error during recording: {e}")
                break
    
    def save_audio(self, filename="temp_audio.wav"):
        """Save recorded audio to file"""
        if not self.frames:
            return None
            
        wf = wave.open(filename, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        return filename
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.close()
        self.audio.terminate()
