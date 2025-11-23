import streamlit as st
import pandas as pd
import random
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- 1. CONFIGURATION & STYLING (Enhanced) ---

# Set page configuration for a modern, professional look (dark mode recommended)
st.set_page_config(
    page_title="News Shield | AI Fact-Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, dark-themed interface and responsiveness
st.markdown("""
    <style>
    /* Main container styling for dark theme */
    .stApp {
        background-color: #0d1117; /* GitHub Dark Mode color */
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    /* Restrict max width of content on large screens for readability */
    .main .block-container {
        max-width: 1000px;
        padding-top: 3rem;
        padding-right: 2rem;
        padding-left: 2rem;
    }

    /* Header and Title Styling */
    h1, .st-emotion-cache-12fmw37 {
        color: #58a6ff; /* Soft blue for emphasis */
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #c9d1d9;
        font-weight: 600;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
        margin-top: 25px;
    }
    
    /* Text Area Styling */
    textarea {
        border: 2px solid #30363d !important;
        border-radius: 10px !important;
        background-color: #161b22 !important;
        color: #ffffff !important;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
        font-family: monospace;
    }

    /* Primary Button Styling (Analyze) */
    .stButton button {
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        padding: 10px 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .stButton button[kind="primary"] {
        background-color: #58a6ff; /* GitHub blue */
        color: white;
        border: none;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #79c0ff;
        transform: translateY(-2px);
    }

    /* Secondary Buttons (Examples) */
    .stButton button:not([kind="primary"]) {
        background-color: #30363d;
        color: #c9d1d9;
        border: 1px solid #484f58;
    }
    .stButton button:not([kind="primary"]):hover {
        background-color: #484f58;
        transform: translateY(-1px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #161b22;
        padding: 1.5rem;
    }

    /* Correction Link Styling */
    .correction-link a {
        color: #79c0ff !important;
        text-decoration: underline;
        font-weight: 500;
    }

    </style>
    """, unsafe_allow_html=True)


# --- 2. MOCK MODEL SETUP (The "Engine" of the Detector) ---

@st.cache_resource
def load_mock_model():
    """Mocks the loading of a pre-trained model and vectorizer."""
    class MockVectorizer:
        def transform(self, data):
            return pd.Series([1, 0, 1])
    class MockModel:
        def predict(self, features):
            return features[0]
    return {'vectorizer': MockVectorizer(), 'model': MockModel()}

mock_assets = load_mock_model()


def detect_fake_news_mock(text):
    """
    Simulates the model prediction process and generates mock corrective links.
    (0 = Fake News, 1 = Real News)
    """
    if not text:
        return 0, 0.5, [] # Default probability for empty text

    sensational_words = ["shocking", "bombshell", "must-see", "disaster", "urgent alert", "massive cover-up", "secretly", "experts agree", "share this immediately", "reveals all", "exposed", "scandal", "breaking now", "lie", "fraud"]
    sensational_count = sum(1 for word in sensational_words if word in text.lower())
    all_caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
    word_count = len(text.split())
    
    prob_real = 0.5
    prob_real -= (sensational_count * 0.08)
    
    if word_count > 0:
        caps_ratio = all_caps_words / word_count
        prob_real -= (caps_ratio * 0.3)
    
    if word_count < 100:
        prob_real -= 0.15 
    elif word_count > 300:
        prob_real += 0.10 

    prob_real = max(0.05, min(0.95, prob_real))
    prediction = 1 if prob_real >= 0.5 else 0
    
    corrective_links = []
    
    if prediction == 0:
        # Generate mock corrective links when the news is fake
        # Note: These are based on the example texts used for demonstration.
        corrective_links = [
            {"title": "Fact-Check: Separating Science from Misinformation (Science Journal)", "url": "https://trustedscience.org/fact-check"},
            {"title": "Official Statement on Economic Outlook (Federal Reserve)", "url": "https://federalreserve.gov/latest-reports"},
            {"title": "Understanding Clickbait Language: A Guide to Media Literacy", "url": "https://medialiteracy.org/clickbait-guide"}
        ]
    
    return prediction, prob_real, corrective_links


# --- 3. UI COMPONENTS & LOGIC ---

def display_corrective_links(links):
    """Displays the list of corrective, non-fake news links."""
    if not links:
        return

    st.markdown("### 🔗 Corrective & Relevant Information")
    st.markdown(
        """
        <p style='font-size: 1rem; color: #a0a0a0;'>
        Based on the predicted topic, here are verified sources and articles providing accurate context or official information.
        </p>
        """, unsafe_allow_html=True
    )
    
    # Use an HTML list with custom styling for mock links
    link_html = "<ul style='list-style-type: none; padding-left: 0;'>"
    for link in links:
        link_html += f"""
        <li style="margin-bottom: 10px;" class="correction-link">
            <span style="color: #484f58;">•</span> 
            <a href="{link['url']}" target="_blank">{link['title']}</a>
        </li>
        """
    link_html += "</ul>"
    
    st.markdown(link_html, unsafe_allow_html=True)


def display_result(prediction, confidence, corrective_links):
    """Displays the result badge, Fakiness Score, and calls for corrective links."""
    
    label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
    
    if prediction == 1:
        color = "#28a745"  # Green
        emoji = "✅"
        score_label = "Confidence Score"
        score_value = f"{confidence * 100:.2f}%"
        message = "The model suggests this article is likely **authentic and factual**. It exhibits characteristics consistent with verified reports."
    else:
        color = "#dc3545"  # Red
        emoji = "❌"
        score_label = "Fakiness Score"
        # Calculate Fakiness Score
        fakiness_score = (1 - confidence) * 100
        score_value = f"{fakiness_score:.2f}%"
        message = "The model suggests this article contains **misinformation or is fabricated**. It shows signs of sensationalism or stylistic inconsistencies."

    
    # Streamlit Markdown with HTML/CSS for the Result Card
    st.markdown(f"""
    <div style="background-color: #161b22; border: 1px solid {color}; padding: 25px; border-radius: 12px; margin-top: 30px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.6);">
        <h2 style="color: {color}; border-bottom: none; margin-top: 0; padding-bottom: 0; text-align: center;">
            {emoji} FINAL VERDICT: {label} {emoji}
        </h2>
        <div style="text-align: center; margin-top: 15px;">
            <span style="font-size: 1.5rem; font-weight: bold; color: {color};">
                {score_label}: {score_value}
            </span>
        </div>
        <p style="font-size: 1.05rem; margin-top: 20px; text-align: center; font-style: italic; color: #a0a0a0;">
            {message} **Always verify critical information through multiple trusted, independent sources.**
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display corrective links only if it is FAKE news
    if prediction == 0:
        display_corrective_links(corrective_links)


def setup_sidebar():
    """Sets up the left sidebar for instructions and details."""
    st.sidebar.markdown(
        """
        # ℹ️ How It Works
        
        **News Shield** uses a simplified machine learning approach to analyze linguistic and stylistic features of news text.
        
        1.  **Paste Text:** Copy the full article text into the main box.
        2.  **Analyze:** Click 'Analyze Article'.
        3.  **Receive Verdict & Correction:** The model checks for patterns common in misinformation. If detected, a **Fakiness Score** and **Corrective Links** are provided.
        
        ---
        
        ## 📝 Input Analysis
        
        This Streamlit app is designed to analyze **Text Content** (the body of the article). 
        
        *The multimodal inputs (Links/Images/Videos) seen in the standalone HTML version require a dedicated, non-Streamlit server-side API (like the Gemini API) for effective analysis.*
        
        ---
        
        ## 🧠 Model Details
        
        **Classification:** Binary (Real/Fake)
        
        **Technique (Simulated):** Linguistic Heuristics based on features like sensational keywords and capitalization.
        """
    )


def main_app():
    """Main Streamlit application function."""
    
    setup_sidebar()

    st.title("News Shield: AI-Powered News Authenticity Scanner 🔍")
    st.markdown("### The Modern Defense Against Misinformation")
    st.markdown("Paste the full text of any news article into the box below for an immediate, AI-driven assessment of its authenticity.")

    st.markdown("---")

    # --- Input Area (Single Column for better responsiveness) ---
    
    current_news_text = st.text_area(
        "📰 Enter News Article Text Here:",
        key="news_input_key", 
        height=300,
        placeholder="E.g., 'SHOCKING new report reveals that all local politicians are secretly aliens! The evidence, which is completely unverified, was found in a dusty pamphlet last Tuesday...'"
    )

    # --- Action Buttons ---
    
    col_analyze, col_fake, col_real = st.columns([1.5, 1, 1])

    with col_analyze:
        detect_button = st.button("🚀 Analyze Article", type="primary", use_container_width=True)
    
    with col_fake:
        example_button_fake = st.button("Load Fake Example", key="load_fake_example", use_container_width=True)
    
    with col_real:
        example_button_real = st.button("Load Real Example", key="load_real_example", use_container_width=True)
        
    # Handle Example Loading (Update the text input widget via st.session_state)
    if example_button_fake:
        st.session_state.news_input_key = "URGENT ALERT: Scientists confirm that drinking lemon water at exactly 3:00 AM reverses aging completely. The findings were based on a study of a single mouse, but experts agree this is the MOST SHOCKING discovery of the decade! Share this immediately!"
        st.rerun()
    
    if example_button_real:
        st.session_state.news_input_key = "On Thursday, the Federal Reserve announced it would keep its benchmark interest rate target unchanged, holding steady in the range of 5.25% to 5.50%. This decision follows a period of stable inflation data and a tightening labor market, suggesting a cautious but optimistic outlook on economic recovery."
        st.rerun()

    # --- Analysis Logic ---
    if detect_button:
        if not current_news_text:
            st.error("⚠️ Please paste an article into the text box before analyzing.")
            return

        # 1. Show Loading/Progress
        st.info("Analyzing content... please wait.")
        progress_bar = st.progress(0)
        
        # Simulate processing time
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        progress_bar.empty()
        
        # 2. Get Prediction, Confidence, and Corrective Links
        prediction, confidence, corrective_links = detect_fake_news_mock(current_news_text)

        # 3. Display Result and Corrective Links
        display_result(prediction, confidence, corrective_links)

        # 4. Display Technical Details in an Expander
        st.markdown("---")
        with st.expander("🔬 Detailed Model Interpretation", expanded=False):
            st.markdown(f"""
            This section provides the underlying data used for the verdict.
            
            * **Word Count:** {len(current_news_text.split())} words
            * **Sensational Keyword Density (Enhanced Check):** High counts of keywords like 'SHOCKING', 'URGENT', or 'BOMBSHELL' correlates with lower authenticity.
            * **Capitalization Index (Enhanced Check):** Checks for excessive ALL-CAPS words (e.g., 'URGENT ALERT') which strongly suggests clickbait or ragebait.
            
            The current prediction is based on the combination of these computed features.
            
            **Model Note:** The deployed model (simulated Logistic Regression) looks for stylistic deviations from established, fact-checked news corpora.
            """)


# --- Run the application ---
if __name__ == "__main__":
    main_app()