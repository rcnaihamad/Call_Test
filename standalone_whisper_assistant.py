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
        
        with sd.InputStream(channels=1, dtype="float32", samplerate=16000, blocksize=1600) as s:
            while not stop_flag.is_set():
                samples, _ = s.read(1600)
                audio_chunk = samples.reshape(-1)
                
                if not tts_playing.is_set():
                    if self.detect_speech(audio_chunk):
                        if not self.recording:
                            self.recording = True
                            self.audio_buffer = []
                            print("\nRecording...", end="", flush=True)
                        
                        self.audio_buffer.extend(audio_chunk)
                        silence_count = 0
                    else:
                        if self.recording:
                            silence_count += 1
                            if silence_count > 30:
                                self.recording = False
                                if len(self.audio_buffer) > 16000:
                                    audio_array = np.array(self.audio_buffer, dtype=np.float32)
                                    self.audio_queue.put(audio_array)
                                print("\nProcessing...", end="", flush=True)
                                silence_count = 0

