#!/usr/bin/env python3

import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
import whisper
import sounddevice as sd
import threading
import queue
import signal
import sys
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import pyttsx3

load_dotenv()

stop_flag = threading.Event()
tts_playing = threading.Event()

def cleanup():
    stop_flag.set()
    sd.stop()

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)
def INI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Load Whisper with auto-download
        try:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base", device="cuda")
            print("Whisper model loaded with CUDA")
        except Exception as e:
            print(f"CUDA not available, using CPU: {e}")
            self.whisper_model = whisper.load_model("base", device="cpu")
        
        # Initialize TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)
        
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.conversation_history = []
        self.recording = False
        self.audio_buffer = []
    
    def get_gemini_response(self, text):
        try:
            self.conversation_history.append({"role": "user", "parts": [text]})
            
            response = self.model.generate_content(
                self.conversation_history,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=150
                )
            )
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
            
            self.conversation_history.append({"role": "model", "parts": [full_response]})
            
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return full_response
        except Exception as e:
            return f"Error: {e}"
    
    def tts_worker(self):
        while not stop_flag.is_set():
            try:
                text = self.tts_queue.get(timeout=0.5)
                if text and not stop_flag.is_set():
                    tts_playing.set()
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    tts_playing.clear()
                self.tts
        energy = np.sum(audio_chunk ** 2) / len(audio_chunk)
        return energy > 0.001
    
    def run(self):
        signal.signal(signal.SIGINT, signal_handler)
        
        threading.Thread(target=self.tts_worker, daemon=True).start()
        threading.Thread(target=self.whisper_worker, daemon=True).start()
        
        print("Whisper Voice Assistant started. Speak to interact...")
        
        silence_count = 0
        
        
