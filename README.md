PhishShield Pro 🛡️
=================

PhishShield Pro is a phishing URL detection web application built using Streamlit.  
It provides dual analysis by combining a local machine learning model with real-time threat intelligence from VirusTotal.

The application uses:
- A local Gradient Boosting Classifier trained on phishing and legitimate URLs
- VirusTotal API to fetch live URL reputation data

Users can input a URL and receive:
- Local ML model prediction
- VirusTotal scan results


Features
--------
- Detect phishing URLs using a trained local ML model
- Check URLs against the VirusTotal database
- Display sidebar metrics (dataset size, class distribution, model accuracy)
- Gracefully handles missing datasets or unscanned URLs
- Automatically retrains model if not found


Installation
------------

Clone the repository:
git clone https://github.com/yourusername/phishshield-pro.git
cd phishshield-pro

Create a Python virtual environment:
python -m venv venv
source venv/bin/activate      (Linux/macOS)
venv\Scripts\activate         (Windows)

Install required packages:
pip install -r requirements.txt


requirements.txt (example)
--------------------------
streamlit
pandas
numpy
scikit-learn
tldextract
joblib
requests
python-whois

Note:
python-whois is optional and only required if WHOIS features are enabled.


Setup VirusTotal API Key
------------------------

1. Create an account at https://www.virustotal.com
2. Generate a personal API key
3. Add the key to Streamlit secrets file:

Create file:
.streamlit/secrets.toml

Add:
[virustotal]
api_key = "YOUR_VIRUSTOTAL_API_KEY"


Running the App
---------------

Run the Streamlit application:
streamlit run app.py

Open the browser at the URL shown in the terminal
(usually http://localhost:8501)


Usage
-----

1. Enter a URL in the input box
2. Click "Analyze"
3. View results:
   - Local ML prediction (Safe or Phishing) with confidence score
   - VirusTotal results showing how many engines flagged the URL

If the URL has not been scanned by VirusTotal:
"URL not scanned yet. Consider waiting a few minutes for analysis."


Notes
-----
- The ML model uses features such as:
  - URL length
  - Number of subdomains
  - HTTPS usage
  - Suspicious keywords
  - Special characters
  - Optional WHOIS domain age

- The dataset file "dataset_phishing.csv" should be located in the same directory as app.py
- If the dataset is missing, a small synthetic dataset is used for demonstration
- VirusTotal API has rate limits; excessive requests may be blocked
- The model is retrained automatically if phishshield_model.pkl is not found


Deployment
----------
- Can be deployed on Streamlit Cloud, Heroku, or similar platforms
- Ensure .streamlit/secrets.toml is properly configured
- For production use:
  - Enable WHOIS features
  - Use a full, updated phishing dataset for better accuracy


License
-------
MIT License  
Free to use, modify, and distribute.
