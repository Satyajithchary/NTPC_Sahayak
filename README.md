# 🏭 NTPC Sahayak: AI-Based Multi-Modal Decision Support System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Satyajithchary/NTPC_RAG)
[![Python](https://img.shields.io/badge/Python-3.10-yellow)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Gradio-orange)](https://gradio.app/)

**NTPC Sahayak** is a Generative AI prototype developed for the **Indian Power Stations (IPS-2026)** conference. It is designed to assist Operation & Maintenance (O&M) personnel in thermal power plants by retrieving troubleshooting procedures from technical manuals and analyzing site images for hazards.

---

## 🚀 Key Features

*   **Multi-Modal Analysis:** Integrates **Computer Vision** (Gemini Flash) to analyze site photos (e.g., fire, leakage) and generate context-aware safety responses.
*   **Advanced RAG Architecture:** Utilizes **HyDE (Hypothetical Document Embeddings)** and **Cross-Encoder Re-Ranking** to ensure high-precision retrieval from technical manuals.
*   **Source Traceability:** Every AI response cites the specific PDF manual used, ensuring trust and accountability.
*   **Multilingual Support:** Provides real-time **Neural Machine Translation** (Hindi, Telugu, Marathi, Odia) and **Text-to-Speech (TTS)** audio for accessible field guidance.

---

## 🛠️ System Architecture

The system pipeline consists of four distinct layers:
1.  **Visual Encoder:** Converts site images into technical textual descriptions.
2.  **Retrieval Engine:** Uses FAISS with HNSW indexing to search relevant Standard Operating Procedures (SOPs).
3.  **Re-Ranking:** A Cross-Encoder model grades retrieved documents to filter out hallucinations.
4.  **Localization:** Google Translate and gTTS provide vernacular support.

---

## 📂 Project Structure

```bash
.
├── app.py                  # Main application logic (Gradio UI + RAG Pipeline)
├── requirements.txt        # Python dependencies
├── faiss_index_store/      # Pre-trained Vector Database (The "Brain")
└── README.md               # Project documentation
