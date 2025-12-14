import streamlit as st
import os
import re
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from deep_translator import GoogleTranslator
from gtts import gTTS
from PIL import Image
import io

# --- PAGE SETUP ---
st.set_page_config(page_title="NTPC Sahayak Pro", layout="wide", page_icon="🏭")

# --- LOAD RESOURCES (Cached) ---
@st.cache_resource
def load_resources():
    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Load the Vector Store
    if os.path.exists("faiss_index_store"):
        path = "faiss_index_store"
    else:
        path = "." 
        
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    
    # 3. Load the Re-Ranker
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return vector_store, cross_encoder

# --- VISION HELPER ---
def analyze_image_with_gemini(image, api_key):
    """Uses Gemini to see the image."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') 
        response = model.generate_content([
            "Analyze this industrial image. Describe the equipment condition, look for leaks, corrosion, fire, or reading values on gauges. Be technical.", 
            image
        ])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# --- AUDIO CLEANER ---
def clean_text_for_audio(text):
    # Remove asterisks, hashes, and dashes used for formatting
    clean = re.sub(r'[\*\#\-]', '', text)
    # Remove extra whitespace
    clean = " ".join(clean.split())
    return clean

# --- TRANSLATION HELPER ---
def translate_and_audio(text, lang):
    lang_map = {"Hindi": "hi", "Marathi": "mr", "Telugu": "te", "Tamil": "ta", "Odia": "or"}
    target_code = lang_map.get(lang, "en")
    
    final_text = text
    if target_code != "en":
        try:
            final_text = GoogleTranslator(source='auto', target=target_code).translate(text[:4500])
        except:
            pass
            
    # Audio Generation
    audio_bytes = None
    try:
        audio_text = clean_text_for_audio(final_text)
        if audio_text.strip():
            tts = gTTS(text=audio_text, lang=target_code, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.getvalue() # Get raw bytes
    except Exception as e:
        print(f"Audio Error: {e}")
        audio_bytes = None
        
    return final_text, audio_bytes

# --- UI LAYOUT ---
st.title("🏭 NTPC Intelligent Plant Sahayak")
st.caption("Multi-Modal AI: Vision + RAG + Cross-Encoder Re-Ranking")

# Sidebar
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
if not api_key:
    st.sidebar.warning("API Key required.")

try:
    vector_store, cross_encoder = load_resources()
    st.sidebar.success("✅ Knowledge Base Loaded")
except Exception as e:
    st.error(f"Error loading Knowledge Base: {e}")
    st.stop()

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    # 1. Inputs
    query = st.text_area("Enter Technical Query (e.g., 'Procedure for BFP isolation'):")
    uploaded_file = st.file_uploader("Upload Site Photo (Optional - e.g., Leakage/Meter)", type=["jpg", "jpeg", "png"])
    lang = st.selectbox("Output Language", ["English", "Hindi", "Marathi", "Odia", "Telugu"])
    
    if st.button("Analyze & Answer"):
        if not api_key:
            st.error("Please enter API Key in the sidebar.")
        else:
            with st.spinner("Running AI Pipeline (Vision -> Retrieval -> Re-Ranking)..."):
                
                # A. VISION LAYER
                visual_context = ""
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Site Image", width=300)
                    with st.status("Analyzing Image..."):
                        visual_context = analyze_image_with_gemini(image, api_key)
                        st.write(f"**AI Vision detected:** {visual_context}")
                
                # B. RETRIEVAL LAYER (RAG)
                search_query = query
                if visual_context:
                    search_query += f" {visual_context}"
                
                initial_docs = vector_store.similarity_search(search_query, k=15)
                
                # C. RE-RANKING LAYER (Cross-Encoder)
                doc_contents = [doc.page_content for doc in initial_docs]
                pairs = [[search_query, doc] for doc in doc_contents]
                scores = cross_encoder.predict(pairs)
                scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
                top_docs = [doc for score, doc in scored_docs[:5]] 
                
                # D. SOURCE EXTRACTION
                unique_sources = set()
                context_text = ""
                for doc in top_docs:
                    context_text += doc.page_content + "\n\n"
                    src = doc.metadata.get('source', 'Unknown Manual')
                    unique_sources.add(os.path.basename(src))

                # E. GENERATION LAYER
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                
                prompt = f"""
                You are an expert NTPC Senior Engineer. Answer the user question using the Context provided.
                
                CONTEXT (Manuals):
                {context_text}
                
                VISUAL CONTEXT:
                {visual_context if visual_context else "N/A"}
                
                USER QUESTION:
                {query}
                
                Instructions:
                1. Answer in clear bullet points.
                2. If the exact answer is missing in the context, use general thermal power plant engineering knowledge to answer, but mention "Based on general engineering practices".
                3. Prioritize safety steps.
                """
                response = llm.invoke(prompt)
                
                # F. TRANSLATION & AUDIO
                trans_text, audio_bytes = translate_and_audio(response.content, lang)
                
                st.markdown("### ✅ Expert Response")
                st.write(trans_text)
                
                st.markdown("---")
                st.markdown("#### 📚 Reference Documents:")
                if unique_sources:
                    for src in unique_sources:
                        st.info(f"📄 {src}")
                else:
                    st.warning("No specific manual found. Answer generated using general engineering knowledge.")

                # AUDIO PLAYER & DOWNLOAD BUTTON
                if audio_bytes:
                    st.markdown("#### 🔊 Audio Response")
                    st.audio(audio_bytes, format="audio/mp3")
                    
                    # Dedicate Download Button
                    st.download_button(
                        label="⬇️ Download Audio (MP3)",
                        data=audio_bytes,
                        file_name="ntpc_sahayak_response.mp3",
                        mime="audio/mp3"
                    )

with col2:
    st.info("System Architecture")
    st.markdown("""
    1. **Visual Encoder**: Gemini Flash
    2. **Retrieval**: FAISS (Vector Search)
    3. **Re-Ranking**: Cross-Encoder (MS-Marco)
    4. **Generator**: LLM
    5. **Localization**: NMT + TTS
    """)
