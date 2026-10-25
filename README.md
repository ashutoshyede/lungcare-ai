# 🫁 LungCare AI

A lung cancer information chatbot and survival estimator, built with Streamlit.

**🔗 Live app:** https://lungcare-ai-qtmn47oahebuk6ngkdmsjb.streamlit.app/

> ⚠️ **Disclaimer:** This is a student research project. It is **not** a medical device and does **not** give diagnosis or treatment advice. Always consult a qualified doctor.

## What it does

- **💬 Chatbot:** answers questions about lung cancer and common minor illnesses, using a Google Gemini model. It never gives medicine doses and redirects danger signs to emergency care.
- **📚 Knowledge base:** a 22-entry knowledge base (lung cancer types, symptoms, staging, treatment overview, screening, common illnesses). Answers show the sources used, retrieved with TF-IDF similarity search.
- **📈 Survival estimate:** a Cox proportional hazards model gives estimated 1, 3 and 5-year survival from age, sex, stage and cancer type (LUAD or LUSC).
- **🔗 Chatbot + model:** if a user asks about survival in chat, the bot collects the missing details, runs the model, and explains the result with caveats.

## The model

- **Data:** TCGA-LUAD and TCGA-LUSC clinical data from the GDC (1,005 patients, 397 deaths).
- **Features:** age, stage, sex, cancer type.
- **Models compared:** Cox PH, Random Survival Forest, Gradient Boosting Survival Analysis.

| Model | Test C-index (n=201) | External C-index (GSE72094, n=398) |
|---|---|---|
| Cox PH | 0.682 | 0.643 |
| Gradient Boosting | 0.676 | 0.621 |
| Random Survival Forest | 0.641 | 0.617 |

Cox PH was never worse than the complex models, so it is used in the app. On the test set its 95% CI was 0.616 to 0.747.

**Limitations:** the model uses only 4 clinical features, so its discrimination is modest. Calibration checks showed that all models overestimate survival in the highest-risk patients. The estimates are for learning and research only.

## Run locally

1. Install the requirements: `pip install -r requirements.txt`
2. Start the app: `streamlit run app.py`
3. For the chatbot, get a free Gemini API key from https://aistudio.google.com/apikey and paste it in the sidebar. The survival page works without a key.

## Project structure

- `app.py` is the Streamlit app (chatbot, knowledge base, survival page)
- `models/cox_baseline.joblib` is the trained Cox model bundle
- `requirements.txt` lists the Python dependencies

## Tech stack

Python, Streamlit, scikit-survival, scikit-learn, pandas, Google Gemini API

## Author

Built by Ashutosh, B.Tech AI/ML student.
