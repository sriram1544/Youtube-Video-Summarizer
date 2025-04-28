# import streamlit as st
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# from youtube_transcript_api import YouTubeTranscriptApi
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# st.set_page_config(page_title="Multilingual YouTube Summarizer", layout="centered")

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Load NLLB model + tokenizer
# @st.cache_resource
# def load_nllb_model():
#     model_name = "facebook/nllb-200-distilled-600M"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     return tokenizer, model

# nllb_tokenizer, nllb_model = load_nllb_model()

# # NLLB language codes (Facebook format)
# nllb_lang_codes = {
#     "English": "eng_Latn",
#     "Hindi": "hin_Deva",
#     "Telugu": "tel_Telu",
#     "Tamil": "tam_Taml",
#     "Bengali": "ben_Beng",
#     "Kannada": "kan_Knda",
#     "Malayalam": "mal_Mlym",
#     "Marathi": "mar_Deva",
#     "Gujarati": "guj_Gujr",
#     "Odia": "ory_Orya",
#     "Urdu": "urd_Arab"
# }

# # UI Styling
# st.markdown("""
#     <style>
#     .stApp { background-color: #b68f40; color: #f8f4ec; font-family: 'Segoe UI', sans-serif; }
#     h1, label { color: #fffaf0 !important; }
#     .stTextInput input, .stSelectbox div div {
#         background-color: #fffaf0 !important; color: #000 !important;
#     }
#     .stButton button {
#         background-color: #000000 !important; color: #fffaf0 !important; border: 1px solid #fffaf0;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Step 1: Extract transcript
# def extract_transcript(video_url):
#     try:
#         video_id = video_url.split("v=")[1].split("&")[0]
#         transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
#         return " ".join([t['text'] for t in transcript_data])
#     except Exception as e:
#         st.error(f"Transcript error: {e}")
#         st.stop()

# # Step 2: Summarize using Gemini
# def summarize_with_gemini(transcript):
#     prompt = "Summarize the following transcript into clear bullet points under 300 words:\n\n"
#     try:
#         model = genai.GenerativeModel("gemini-1.5-pro")
#         response = model.generate_content(prompt + transcript)
#         return response.text.strip()
#     except Exception as e:
#         st.error(f"Gemini Error: {e}")
#         st.stop()

# # Step 3: Translate using NLLB
# def translate_with_nllb(text, target_lang_code):
#     try:
#         inputs = nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         translated_tokens = nllb_model.generate(
#             **inputs,
#             forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(target_lang_code),
#             max_length=512
#         )
#         return nllb_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
#     except Exception as e:
#         return f"[NLLB Translation Error: {e}]\n\n{text}"

# # UI Layout
# st.title("🎬 YouTube Transcript to Multilingual Summary")

# youtube_url = st.text_input("Enter YouTube Video URL:")
# col1, col2 = st.columns(2)
# with col1:
#     selected_lang = st.selectbox("Select Output Language", list(nllb_lang_codes.keys()))
# with col2:
#     generate = st.button("Generate Summary")

# # Show thumbnail
# if youtube_url:
#     try:
#         video_id = youtube_url.split("v=")[1].split("&")[0]
#         st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
#     except:
#         pass

# # Generate Summary
# if generate and youtube_url:
#     transcript = extract_transcript(youtube_url)
#     english_summary = summarize_with_gemini(transcript)

#     if selected_lang != "English":
#         lang_code = nllb_lang_codes[selected_lang]
#         translated_summary = translate_with_nllb(english_summary, lang_code)
#     else:
#         translated_summary = english_summary

#     st.markdown("## 📋 Summary:")
#     st.write(translated_summary)



import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
st.set_page_config(page_title="🎥 YouTube Multilingual Summarizer", layout="wide")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_resource
def load_nllb_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

nllb_tokenizer, nllb_model = load_nllb_model()

nllb_lang_codes = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Bengali": "ben_Beng",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Odia": "ory_Orya",
    "Urdu": "urd_Arab"
}

languages = list(nllb_lang_codes.keys())

# Inject dynamic CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Manrope', sans-serif;
        background: linear-gradient(135deg, #1e1e1e 0%, #3e3e3e 100%);
        color: #fefefe;
    }
    .stButton > button {
        background-color: #d4af37;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        transition: 0.3s;
        border: none;
    }
    .stButton > button:hover {
        background-color: #fff176;
        color: #000;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #f5f5dc;
        color: #000;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(255,255,255,0.2);
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Utility functions
def extract_transcript(video_url):
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        except NoTranscriptFound:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list:
                if transcript.is_translatable:
                    transcript_data = transcript.translate('en').fetch()
                    break
            else:
                raise NoTranscriptFound(video_id)
        return " ".join([t['text'] for t in transcript_data])
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript) as e:
        st.error("❌ Transcript could not be retrieved for this video. It may not be available in English or the video has disabled transcripts.")
        st.stop()
    except Exception as e:
        st.error(f"Transcript error: {e}")
        st.stop()

def summarize_with_gemini(transcript):
    prompt = "Summarize the following transcript into clear bullet points under 300 words:\n\n"
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt + transcript)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        st.stop()

def translate_with_nllb(text, target_lang_code):
    try:
        inputs = nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(target_lang_code),
            max_length=512
        )
        return nllb_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        return f"[NLLB Translation Error: {e}]\n\n{text}"

# UI Elements
st.title("📽️ YouTube Video Multilingual Summarizer")

with st.container():
    col1, col2 = st.columns([2, 1])

    with col1:
        youtube_url = st.text_input("Enter YouTube Video URL:")
    with col2:
        selected_lang = st.selectbox("Select Output Language", languages)

if youtube_url:
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
    except:
        st.warning("Could not display video thumbnail.")

if st.button("🎯 Generate Summary") and youtube_url:
    transcript = extract_transcript(youtube_url)
    english_summary = summarize_with_gemini(transcript)

    if selected_lang != "English":
        lang_code = nllb_lang_codes[selected_lang]
        final_summary = translate_with_nllb(english_summary, lang_code)
    else:
        final_summary = english_summary

    st.markdown("## 📝 Summary")
    st.write(final_summary)
    st.success("✅ Summary generated successfully!")


