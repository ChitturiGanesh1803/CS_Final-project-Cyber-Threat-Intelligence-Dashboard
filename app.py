# import streamlit as st
# import pandas as pd
# import numpy as np
# import tldextract
# import re
# import joblib
# import os
# from urllib.parse import urlparse
# from datetime import datetime
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # ------------------ CONFIG ------------------
# st.set_page_config(
#     page_title="PhishShield Pro - AI Security",
#     page_icon="🛡️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ------------------ STYLES ------------------
# st.markdown("""
# <style>
# .main-title { font-size: 2.8rem; font-weight: 800; text-align: center; color: #1E3A8A; }
# .risk-card { padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
# .safe-card { background: rgba(16,185,129,.1); border:1px solid #10B981; }
# .phishing-card { background: rgba(239,68,68,.1); border:1px solid #EF4444; }
# .warning-card { background: rgba(245,158,11,.1); border:1px solid #F59E0B; }
# .disclaimer { font-size:.9rem; color:#6B7280; }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ FEATURE EXTRACTOR ------------------
# class URLAnalyzer:
#     def extract_features(self, url):
#         parsed = urlparse(url)
#         ext = tldextract.extract(url)
#         f = {}

#         f['url_length'] = len(url)
#         f['domain_length'] = len(parsed.netloc)
#         f['num_subdomains'] = parsed.netloc.count('.')
#         f['has_https'] = int(parsed.scheme == 'https')
#         f['special_chars'] = sum(url.count(c) for c in ['@','!','$','%','&'])
#         f['suspicious_keywords'] = sum(k in url.lower() for k in
#             ['login','verify','secure','account','update','password','bank'])
#         f['nb_dots'] = url.count('.')
#         f['nb_hyphens'] = url.count('-')

#         try:
#             info = whois.whois(parsed.netloc, timeout=5)
#             cd = info.creation_date
#             if isinstance(cd, list): cd = cd[0]
#             f['domain_age_days'] = (datetime.now() - cd).days if cd else -1
#         except Exception:
#             f['domain_age_days'] = -1

#         return f

# # ------------------ DATASET ------------------
# @st.cache_data
# def load_csv_dataset():
#     if not os.path.exists("dataset_phishing.csv"):
#         st.warning("dataset_phishing.csv not found. Using synthetic data (5 URLs).")
#         # Synthetic dataset fallback
#         data = pd.DataFrame({
#             'url': [
#                 'https://github.com', 'https://www.nytimes.com', 'https://microsoft.com',
#                 'http://faceb00k-login.net', 'https://paypal-security-update.com'
#             ],
#             'status': ['legitimate','legitimate','legitimate','phishing','phishing']
#         })
#         data['label'] = data['status'].map({'legitimate':0,'phishing':1})
#         analyzer = URLAnalyzer()
#         features = data['url'].apply(analyzer.extract_features)
#         features_df = pd.DataFrame(list(features))
#         return pd.concat([features_df, data[['label','status']]], axis=1)

#     df = pd.read_csv("dataset_phishing.csv")
#     df['status'] = df['status'].str.lower().str.strip()
#     df['label'] = df['status'].map({'legitimate': 0, 'phishing': 1})
#     analyzer = URLAnalyzer()
#     features = df['url'].apply(analyzer.extract_features)
#     features_df = pd.DataFrame(list(features))
#     return pd.concat([features_df, df[['label','status']]], axis=1)

# # ------------------ MODEL ------------------
# def train_and_save_model(df):
#     X = df.drop(['label','status'], axis=1)
#     y = df['label']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     model = GradientBoostingClassifier(
#         n_estimators=300, learning_rate=0.05, random_state=42
#     )
#     model.fit(X_train, y_train)

#     model_data = {
#         'model': model,
#         'features': X.columns.tolist(),
#         'train_accuracy': accuracy_score(y_train, model.predict(X_train)),
#         'test_accuracy': accuracy_score(y_test, model.predict(X_test))
#     }

#     joblib.dump(model_data, "phishshield_model.pkl")
#     return model_data

# @st.cache_resource
# def load_model():
#     df = load_csv_dataset()

#     # Automatically create the model if not exists
#     if os.path.exists("phishshield_model.pkl"):
#         model_data = joblib.load("phishshield_model.pkl")
#     else:
#         st.info("Model file not found. Training new model...")
#         model_data = train_and_save_model(df)
#         st.success("Model trained and saved as phishshield_model.pkl")

#     return model_data

# # ------------------ MAIN APP ------------------
# def main():
#     st.markdown("<div class='main-title'>🛡️ PhishShield Pro</div>", unsafe_allow_html=True)

#     model_data = load_model()
#     analyzer = URLAnalyzer()

#     df = load_csv_dataset()

#     # Sidebar info
#     st.sidebar.markdown(f"**Dataset loaded:** {len(df)} URLs")
#     st.sidebar.markdown(f"**Legitimate URLs:** {(df['status']=='legitimate').sum()}")
#     st.sidebar.markdown(f"**Phishing URLs:** {(df['status']=='phishing').sum()}")
#     st.sidebar.markdown("---")
#     st.sidebar.markdown(f"Train accuracy: {model_data['train_accuracy']*100:.1f}%")
#     st.sidebar.markdown(f"Test accuracy: {model_data['test_accuracy']*100:.1f}%")

#     url = st.text_input("Enter URL", placeholder="https://example.com")

#     if st.button("Analyze") and url:
#         if not url.startswith(('http://','https://')):
#             url = 'http://' + url

#         features = analyzer.extract_features(url)
#         X = pd.DataFrame([{k: features.get(k,0) for k in model_data['features']}])

#         pred = model_data['model'].predict(X)[0]
#         proba = model_data['model'].predict_proba(X)[0][1]

#         if pred == 0:
#             st.success(f"✅ Safe URL (Phishing probability {proba*100:.1f}%)")
#         else:
#             st.error(f"⚠️ Phishing URL (Confidence {proba*100:.1f}%)")

# if __name__ == "__main__":
#     main()



import streamlit as st
import pandas as pd
import numpy as np
import tldextract
import joblib
import os
import requests
from urllib.parse import urlparse
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="PhishShield Pro - AI Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ STYLES ------------------
st.markdown("""
<style>
.main-title { font-size: 2.8rem; font-weight: 800; text-align: center; color: #1E3A8A; }
.risk-card { padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
.safe-card { background: rgba(16,185,129,.1); border:1px solid #10B981; }
.phishing-card { background: rgba(239,68,68,.1); border:1px solid #EF4444; }
.warning-card { background: rgba(245,158,11,.1); border:1px solid #F59E0B; }
.disclaimer { font-size:.9rem; color:#6B7280; }
</style>
""", unsafe_allow_html=True)

# ------------------ FEATURE EXTRACTOR ------------------
class URLAnalyzer:
    def __init__(self, use_whois=False):
        self.use_whois = use_whois

    def extract_features(self, url):
        parsed = urlparse(url)
        f = {}

        f['url_length'] = len(url)
        f['domain_length'] = len(parsed.netloc)
        f['num_subdomains'] = parsed.netloc.count('.')
        f['has_https'] = int(parsed.scheme == 'https')
        f['special_chars'] = sum(url.count(c) for c in ['@','!','$','%','&'])
        f['suspicious_keywords'] = sum(k in url.lower() for k in
            ['login','verify','secure','account','update','password','bank'])
        f['nb_dots'] = url.count('.')
        f['nb_hyphens'] = url.count('-')

        # WHOIS optional (can slow things down)
        if self.use_whois:
            try:
                import whois
                info = whois.whois(parsed.netloc, timeout=5)
                cd = info.creation_date
                if isinstance(cd, list): cd = cd[0]
                f['domain_age_days'] = (datetime.now() - cd).days if cd else -1
            except Exception:
                f['domain_age_days'] = -1
        else:
            f['domain_age_days'] = 1000  # placeholder

        return f

# ------------------ DATASET ------------------
@st.cache_data
def load_csv_dataset():
    if not os.path.exists("dataset_phishing.csv"):
        st.warning("dataset_phishing.csv not found. Using synthetic data (5 URLs).")
        data = pd.DataFrame({
            'url': [
                'https://github.com', 'https://www.nytimes.com', 'https://microsoft.com',
                'http://faceb00k-login.net', 'https://paypal-security-update.com'
            ],
            'status': ['legitimate','legitimate','legitimate','phishing','phishing']
        })
        data['label'] = data['status'].map({'legitimate':0,'phishing':1})
        analyzer = URLAnalyzer()
        features = data['url'].apply(analyzer.extract_features)
        features_df = pd.DataFrame(list(features))
        return pd.concat([features_df, data[['label','status']]], axis=1)

    df = pd.read_csv("dataset_phishing.csv")
    df['status'] = df['status'].str.lower().str.strip()
    df['label'] = df['status'].map({'legitimate': 0, 'phishing': 1})
    analyzer = URLAnalyzer()
    features = df['url'].apply(analyzer.extract_features)
    features_df = pd.DataFrame(list(features))
    return pd.concat([features_df, df[['label','status']]], axis=1)

# ------------------ MODEL ------------------
def train_and_save_model(df):
    X = df.drop(['label','status'], axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)

    model_data = {
        'model': model,
        'features': X.columns.tolist(),
        'train_accuracy': accuracy_score(y_train, model.predict(X_train)),
        'test_accuracy': accuracy_score(y_test, model.predict(X_test))
    }

    joblib.dump(model_data, "phishshield_model.pkl")
    return model_data

@st.cache_resource
def load_model():
    df = load_csv_dataset()

    if os.path.exists("phishshield_model.pkl"):
        model_data = joblib.load("phishshield_model.pkl")
    else:
        st.info("Model not found. Training new model...")
        model_data = train_and_save_model(df)
        st.success("Model trained and saved as phishshield_model.pkl")

    return model_data

# ------------------ VIRUSTOTAL ------------------
def check_virustotal(url, api_key):
    headers = {"x-apikey": api_key}
    vt_url = f"https://www.virustotal.com/api/v3/urls"
    try:
        # Encode URL as per VT API
        import base64
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        response = requests.get(f"{vt_url}/{url_id}", headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json()
            # Example: check if VT labels it malicious
            malicious = result.get('data', {}).get('attributes', {}).get('last_analysis_stats', {}).get('malicious', 0)
            return malicious
        else:
            return None
    except Exception as e:
        return None

# ------------------ MAIN APP ------------------
def main():
    st.markdown("<div class='main-title'>🛡️ Cyber Threat Intelligence Dashboard</div>", unsafe_allow_html=True)

    model_data = load_model()
    analyzer = URLAnalyzer()

    df = load_csv_dataset()

    # Sidebar info
    st.sidebar.markdown(f"**Dataset loaded:** {len(df)} URLs")
    st.sidebar.markdown(f"**Legitimate URLs:** {(df['status']=='legitimate').sum()}")
    st.sidebar.markdown(f"**Phishing URLs:** {(df['status']=='phishing').sum()}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Train accuracy: {model_data['train_accuracy']*100:.1f}%")
    st.sidebar.markdown(f"Test accuracy: {model_data['test_accuracy']*100:.1f}%")

    # VirusTotal API Key
    VT_API_KEY = st.secrets.get("virustotal", {}).get("api_key", "")
    if not VT_API_KEY:
        st.sidebar.warning("VirusTotal API key not set in Streamlit secrets.")

    url = st.text_input("Enter URL", placeholder="https://example.com")

    if st.button("Analyze") and url:
        if not url.startswith(('http://','https://')):
            url = 'http://' + url

        # --- Local Model ---
        features = analyzer.extract_features(url)
        X = pd.DataFrame([{k: features.get(k,0) for k in model_data['features']}])
        pred = model_data['model'].predict(X)[0]
        proba = model_data['model'].predict_proba(X)[0][1]

        if pred == 0:
            st.success(f"✅ Local Model: Safe URL (Phishing probability {proba*100:.1f}%)")
        else:
            st.error(f"⚠️ Local Model: Phishing URL (Confidence {proba*100:.1f}%)")

        # --- VirusTotal API ---
        if VT_API_KEY:
            vt_result = check_virustotal(url, VT_API_KEY)
            if vt_result is None:
                st.warning("VirusTotal: Could not fetch data or URL not found.")
            elif vt_result > 0:
                st.error(f"⚠️ VirusTotal: Detected as malicious ({vt_result} engines flagged)")
            else:
                st.success("✅ VirusTotal: No engines flagged this URL")

if __name__ == "__main__":
    main()





