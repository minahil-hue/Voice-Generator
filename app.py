import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import numpy as np
from streamlit_mic_recorder import mic_recorder
import io
import tempfile
import os
from scipy.io import wavfile

# Page configuration
st.set_page_config(
    page_title="Voice & Text Converter",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Voice-to-Text & Text-to-Voice Converter")
st.markdown("Convert speech to text and text to speech using AI models!")

# Cache model loading to avoid reloading
@st.cache_resource
def load_stt_model():
    """Load Speech-to-Text model (Whisper)"""
    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.config.forced_decoder_ids = None
        return processor, model
    except Exception as e:
        st.error(f"Error loading STT model: {e}")
        return None, None

@st.cache_resource
def load_tts_model():
    """Load Text-to-Speech model (VITS)"""
    try:
        model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading TTS model: {e}")
        return None, None

def transcribe_audio(audio_data, sample_rate, stt_processor, stt_model):
    """Convert audio to text using Whisper"""
    try:
        # Convert audio to float32
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            duration = len(audio_data) / sample_rate
            target_length = int(duration * 16000)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_length),
                np.arange(len(audio_data)),
                audio_data
            )
            sample_rate = 16000
        
        # Process and transcribe
        inputs = stt_processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
        
        with torch.no_grad():
            predicted_ids = stt_model.generate(inputs["input_features"])
        
        transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription, None
    except Exception as e:
        return None, f"Error during transcription: {str(e)}"

def text_to_speech(text, tts_model, tts_tokenizer):
    """Convert text to speech using VITS"""
    try:
        # Tokenize text
        inputs = tts_tokenizer(text, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            output = tts_model(**inputs).waveform
        
        # Convert to numpy array
        audio_data = output.squeeze().cpu().numpy()
        sample_rate = tts_model.config.sampling_rate
        
        return audio_data, sample_rate, None
    except Exception as e:
        return None, None, f"Error during TTS: {str(e)}"

def load_audio_from_bytes(audio_bytes):
    """Load audio from bytes"""
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        return audio_data, sample_rate, None
    except Exception as e:
        # Try different formats
        for ext in ['.wav', '.webm', '.ogg', '.mp3']:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                audio_data, sample_rate = sf.read(tmp_path)
                os.unlink(tmp_path)
                return audio_data, sample_rate, None
            except:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                continue
        
        return None, None, f"Error loading audio: {str(e)}"

def save_audio_to_bytes(audio_data, sample_rate):
    """Save audio numpy array to bytes"""
    try:
        # Normalize to int16 range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Write to bytes buffer
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)
        
        return buffer.getvalue(), None
    except Exception as e:
        return None, f"Error saving audio: {str(e)}"

# Load models
with st.spinner("🔄 Loading AI models... Please wait."):
    stt_processor, stt_model = load_stt_model()
    tts_model, tts_tokenizer = load_tts_model()

if stt_processor is None or stt_model is None:
    st.error("❌ Failed to load Speech-to-Text model.")
    st.stop()

if tts_model is None or tts_tokenizer is None:
    st.error("❌ Failed to load Text-to-Speech model.")
    st.stop()

st.success("✅ All models loaded successfully!")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📁 Upload Audio (STT → TTS)", "🎙️ Record Audio (STT → TTS)", "✍️ Text Input (TTS)"])

# Tab 1: Upload Audio File
with tab1:
    st.header("📁 Upload Audio File")
    st.markdown("Upload an audio file to convert speech to text, then automatically convert back to speech.")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        key="audio_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Display uploaded audio
            st.audio(uploaded_file, format="audio/wav")
            
            # Load and transcribe audio
            with st.spinner("🔄 Transcribing audio..."):
                audio_bytes = uploaded_file.read()
                audio_data, sample_rate, error = load_audio_from_bytes(audio_bytes)
                
                if error:
                    st.error(error)
                else:
                    transcription, error = transcribe_audio(audio_data, sample_rate, stt_processor, stt_model)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success("✅ Transcription complete!")
                        st.markdown("### 📝 Transcribed Text:")
                        st.info(transcription)
                        
                        # Automatically perform TTS
                        with st.spinner("🔄 Converting text back to speech..."):
                            tts_audio, tts_sample_rate, tts_error = text_to_speech(transcription, tts_model, tts_tokenizer)
                            
                            if tts_error:
                                st.error(tts_error)
                            else:
                                # Save and play generated audio
                                audio_bytes, save_error = save_audio_to_bytes(tts_audio, tts_sample_rate)
                                
                                if save_error:
                                    st.error(save_error)
                                else:
                                    st.success("✅ Speech generation complete!")
                                    st.markdown("### 🔊 Generated Audio:")
                                    st.audio(audio_bytes, format="audio/wav")
        
        except Exception as e:
            st.error(f"Error processing audio file: {str(e)}")

# Tab 2: Record Audio
with tab2:
    st.header("🎙️ Record Audio from Microphone")
    st.markdown("Record your voice to convert speech to text, then automatically convert back to speech.")
    
    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        key="mic_recorder"
    )
    
    if audio:
        try:
            # Display recorded audio
            st.audio(audio['bytes'], format="audio/wav")
            
            # Load and transcribe audio
            with st.spinner("🔄 Transcribing audio..."):
                audio_data, sample_rate, error = load_audio_from_bytes(audio['bytes'])
                
                if error:
                    st.error(error)
                else:
                    transcription, error = transcribe_audio(audio_data, sample_rate, stt_processor, stt_model)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success("✅ Transcription complete!")
                        st.markdown("### 📝 Transcribed Text:")
                        st.info(transcription)
                        
                        # Automatically perform TTS
                        with st.spinner("🔄 Converting text back to speech..."):
                            tts_audio, tts_sample_rate, tts_error = text_to_speech(transcription, tts_model, tts_tokenizer)
                            
                            if tts_error:
                                st.error(tts_error)
                            else:
                                # Save and play generated audio
                                audio_bytes, save_error = save_audio_to_bytes(tts_audio, tts_sample_rate)
                                
                                if save_error:
                                    st.error(save_error)
                                else:
                                    st.success("✅ Speech generation complete!")
                                    st.markdown("### 🔊 Generated Audio:")
                                    st.audio(audio_bytes, format="audio/wav")
        
        except Exception as e:
            st.error(f"Error processing recorded audio: {str(e)}")

# Tab 3: Text Input
with tab3:
    st.header("✍️ Text Input for Text-to-Speech")
    st.markdown("Enter text to convert it to speech.")
    
    text_input = st.text_area(
        "Enter text here:",
        placeholder="Type or paste your text here...",
        height=150,
        key="text_input"
    )
    
    if st.button("🔊 Generate Speech", key="tts_button"):
        if text_input.strip():
            try:
                with st.spinner("🔄 Converting text to speech..."):
                    tts_audio, tts_sample_rate, tts_error = text_to_speech(text_input, tts_model, tts_tokenizer)
                    
                    if tts_error:
                        st.error(tts_error)
                    else:
                        # Save and play generated audio
                        audio_bytes, save_error = save_audio_to_bytes(tts_audio, tts_sample_rate)
                        
                        if save_error:
                            st.error(save_error)
                        else:
                            st.success("✅ Speech generation complete!")
                            st.markdown("### 🔊 Generated Audio:")
                            st.audio(audio_bytes, format="audio/wav")
            
            except Exception as e:
                st.error(f"Error generating speech: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text to convert to speech.")

# Footer
st.markdown("---")
st.markdown("""
**About this app:**
- **Speech-to-Text**: Uses OpenAI's Whisper Tiny model
- **Text-to-Speech**: Uses Facebook's MMS-TTS English model
- All processing runs locally without external APIs
""")
