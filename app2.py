import torch 
# 1. Permanent fix for the PyTorch unpickler security wall
from TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])

from TTS.api import TTS
import time
import json
import random
from pathlib import Path
import streamlit as st
import html 
import uuid
import librosa
from scipy.io.wavfile import write
import os 
import io # <-- Memory buffers to help browser playback

params = {
    "activate": True,
    "autoplay": True,
    "show_text": False,
    "remove_trailing_dots": False,
    "voice": "Rogger.wav",
    "language": "English",
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
}

SUPPORTED_FORMATS = ['wav']
SAMPLE_RATE = 16000

@st.cache_data
def get_cached_speakers():
    os.makedirs(os.path.join(".", "targets"), exist_ok=True)
    os.makedirs(os.path.join(".", "outputs"), exist_ok=True)
    return {p.stem: str(p) for p in list(Path('targets').iterdir())}

speakers = get_cached_speakers()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner="Loading XTTS Model into GPU Memory...")
def load_model():
    print("[XTTS] Loading XTTS permanently into cache...")
    model = TTS(model_name=params["model_name"]).to(device)
    return model

tts = load_model()
this_dir = str(Path(__file__).parent.resolve())

def get_available_voices():
    return sorted([voice.name for voice in Path(f"{this_dir}/targets").glob("*.wav")])

def gen_voice(string, spk, speed_rate, english_lang, language_dict):
    string = html.unescape(string)
    short_uuid = str(uuid.uuid4())[:8]
    fl_name = 'outputs/' + spk + "-" + short_uuid + '.wav'
    output_file = Path(fl_name)
    tts.tts_to_file(
        text=string,
        speed=speed_rate,
        file_path=output_file,
        speaker_wav=[f"{this_dir}/targets/" + spk + ".wav"],
        language=language_dict[english_lang]
    )
    return output_file

st.title("TTS based Voice Cloning in 16 Languages.")
st.header('Text to speech generation')

languages = None 
with open(Path(f"{this_dir}/languages.json"), encoding='utf8') as f:
    languages = json.load(f)

# Sidebar layout
with st.sidebar:
    voice_list = get_available_voices()
    st.title("Text to Voice")
    english = st.radio(label="Choose your language", options=languages, index=0, horizontal=True)
   
    default_speaker_name = "Rogger"
    options_list = [None] + list(speakers.keys())
    try:
        default_index = options_list.index(default_speaker_name)
    except ValueError:
        default_index = 0

    speaker_name = st.selectbox('Select target speaker:', options=options_list, index=default_index)

    wav_tgt = None
    if speaker_name is not None and speaker_name in speakers:
        wav_tgt, _ = librosa.load(speakers[speaker_name], sr=22000)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        
        # --- Strictly 32-bit float memory buffer ---
        preview_buffer = io.BytesIO()
        write(preview_buffer, 22000, wav_tgt)
        
        st.write('Selected Target Preview:')
        st.audio(preview_buffer.getvalue(), format='audio/wav')
   
    text = st.text_area('Enter text to convert to audio format', value="Hello")
    speed = st.slider('Speed', 0.1, 1.99, 0.8, 0.01)

    st.write("---")
    st.markdown("### 📥 Add Custom Voice Sample")
    new_tgt = st.file_uploader('Upload new TARGET audio:', type=SUPPORTED_FORMATS, accept_multiple_files=False)
    
    if new_tgt is not None:
        file_name = new_tgt.name
        file_path = os.path.join("./targets/", file_name)
        st.info(f"Uploading: {file_name}")
        
        file_name_without_extension = os.path.splitext(file_name)[0]

        wav_uploaded, _ = librosa.load(new_tgt, sr=22000)
        wav_uploaded, _ = librosa.effects.trim(wav_uploaded, top_db=20)

        # --- Strictly writing the original 32-bit float array ---
        write('./targets/' + file_name_without_extension + '.wav', 22000, wav_uploaded)
        st.success(f"Saved to targets folder!")
        st.cache_data.clear() 
        st.rerun()

# Global execution state tracking
if "last_generated_audio" not in st.session_state:
    st.session_state.last_generated_audio = None
if "last_speaker" not in st.session_state:
    st.session_state.last_speaker = None

st.write("---")
if st.button('Convert Text to Cloned Speech', use_container_width=True):
    if speaker_name is None:
        st.error("Please select a target speaker from the sidebar first!")
    else:
        with st.spinner('Generating audio on your GPU... please wait...'):
            output_file = gen_voice(text, speaker_name, speed, english, languages)
            st.session_state.last_generated_audio = str(output_file)
            st.session_state.last_speaker = speaker_name
        st.success('Conversion Complete!')

# Render final generated file output blocks
if st.session_state.last_generated_audio and os.path.exists(st.session_state.last_generated_audio):
    st.write(f'Generated Output (Target Voice: **{st.session_state.last_speaker}**)')
    
    # Read the 32-bit XTTS generated file exactly as it is 
    with open(st.session_state.last_generated_audio, 'rb') as audio_file:
        audio_bytes_output = audio_file.read()
    
    # Serve to Streamlit natively 
    st.audio(audio_bytes_output, format='audio/wav')