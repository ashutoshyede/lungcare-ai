import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="LungCare AI", page_icon="🫁", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cox_baseline.joblib")

DEFAULT_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# =====================================================================
# MEDICAL KNOWLEDGE BASE
# Source labels:
#   "NCI PDQ"  = core facts checked against NCI (cancer.gov) patient information
#   "General medical knowledge" = written in simple words from standard references,
#                                  NOT individually verified. Verify before citing in a paper.
# To add more knowledge later: add a new dict to this list.
# =====================================================================
NCI = "NCI PDQ patient information (cancer.gov)"
GEN = "General medical knowledge (verify before citing)"

KB = [
    {
        "title": "Lung cancer: the two main types",
        "keywords": "lung cancer types kitne prakar fefde ka cancer kainsar NSCLC SCLC",
        "text": "There are two main types of lung cancer: non-small cell lung cancer (NSCLC) and small cell lung cancer (SCLC). NSCLC is more common. Small cell lung cancer is less common and grows and spreads fast. The names come from how the cancer cells look under a microscope. The type matters because treatment plans are different.",
        "source": NCI,
    },
    {
        "title": "Non-small cell lung cancer (NSCLC) subtypes",
        "keywords": "adenocarcinoma squamous cell large cell carcinoma LUAD LUSC subtype",
        "text": "NSCLC has several subtypes named after the cells involved. The three main ones are adenocarcinoma (the most common, often starts in the outer parts of the lung, and is the most common type in people who never smoked), squamous cell carcinoma (usually starts in the larger airways in the middle of the lung and is strongly linked to smoking) and large cell carcinoma (less common). Less common ones include adenosquamous carcinoma, sarcomatoid carcinoma and carcinoid tumor. In this project, LUAD means adenocarcinoma and LUSC means squamous cell carcinoma.",
        "source": NCI,
    },
    {
        "title": "Small cell lung cancer (SCLC)",
        "keywords": "small cell lung cancer limited extensive stage fast growing",
        "text": "Small cell lung cancer is fast-growing and mostly linked to smoking. Its main types are small cell carcinoma and combined small cell carcinoma. It is often described as limited stage (cancer in one side of the chest, within an area that can be treated with radiation) or extensive stage (spread widely in the lung or to other parts of the body). Treatment usually relies on chemotherapy, radiation and sometimes immunotherapy. Surgery is used in only a small number of patients.",
        "source": NCI,
    },
    {
        "title": "Risk factors for lung cancer",
        "keywords": "risk factors causes kaaran smoking cigarette bidi tambaku radon asbestos pollution pradushan chulha",
        "text": "Smoking is the most important risk factor for lung cancer, and risk goes up further when smoking is combined with other risk factors. Other risk factors include secondhand smoke, radon gas in homes, asbestos and other workplace exposures (arsenic, chromium, nickel, diesel exhaust), air pollution, earlier radiation to the chest, and a family history of lung cancer. In India, smoke from cooking with wood or biomass and heavy air pollution are also discussed as risks. Having a risk factor does not mean a person will definitely get cancer, and people with no risk factors can also get it. Some risk factors, like smoking, can be changed.",
        "source": NCI,
    },
    {
        "title": "Symptoms of lung cancer",
        "keywords": "symptoms lakshan signs khansi cough saans phoolna chest pain seene mein dard khoon weight loss",
        "text": "Lung cancer may cause no symptoms at all in early stages. Signs can include a cough that does not go away, shortness of breath, chest pain, wheezing, coughing up blood, hoarse voice, feeling very tired, loss of appetite and unexplained weight loss. These symptoms can also be caused by other conditions such as infections or TB, so symptoms alone do not mean cancer. A cough lasting more than 3 weeks should be checked by a doctor.",
        "source": NCI,
    },
    {
        "title": "Danger signs: when to see a doctor urgently",
        "keywords": "emergency red flag danger khatra turant doctor coughing blood trouble breathing saans lene mein dikkat",
        "text": "Go to a doctor or emergency service immediately for: trouble breathing, coughing up a lot of blood, severe chest pain, fainting, confusion, blue lips, very high fever that does not come down, or sudden weakness of the face, arm or leg. In India the emergency number is 112.",
        "source": GEN,
    },
    {
        "title": "How lung cancer is diagnosed",
        "keywords": "diagnosis test jaanch CT scan biopsy X-ray PET bronchoscopy kaise pata chalta hai",
        "text": "Tests that look at the lungs are used to diagnose and stage lung cancer. Common ones are chest X-ray, CT scan, PET-CT scan, and a biopsy (taking a small piece of tissue to check under a microscope). Biopsy can be done with a needle, with a camera through the airways (bronchoscopy) or by surgery. Blood tests alone are not used to diagnose lung cancer. Tests on the tumor tissue (for example EGFR, ALK, PD-L1) are done to choose the best treatment. Diagnosis and staging tests are often done at the same time.",
        "source": NCI,
    },
    {
        "title": "Stages of lung cancer (NSCLC)",
        "keywords": "stage stages I II III IV staging TNM charan matlab stage 3 stage 4",
        "text": "NSCLC stages run from 0 to IV. Stage I is divided into IA and IB, stage II into IIA and IIB, and stage IV into IVA and IVB. In simple words: Stage I means a small cancer only in the lung. Stage II means a larger cancer or spread to nearby lymph nodes. Stage III means more spread to lymph nodes in the middle of the chest or nearby structures. Stage IV means the cancer has spread to other parts of the body, such as the other lung, fluid around the lung, brain, bones or liver. Doctors decide stage using the TNM system (tumor size, lymph nodes, metastasis). If lung cancer spreads, for example to the brain, the cells there are still lung cancer cells.",
        "source": NCI,
    },
    {
        "title": "Treatment options for lung cancer (overview)",
        "keywords": "treatment ilaj surgery radiation chemotherapy immunotherapy targeted therapy operation",
        "text": "Treatment depends on the type (NSCLC or SCLC), stage, tumor test results and the person's general health. Main options are surgery (best chance of cure in early stages), radiation therapy, chemotherapy, targeted therapy (drugs aimed at specific changes in the cancer cell) and immunotherapy (helps the immune system fight cancer). Often these are combined. Palliative care helps with symptoms and quality of life at any stage. The treatment plan must always be made by the patient's cancer team (oncologist).",
        "source": NCI,
    },
    {
        "title": "Names of lung cancer medicines (information only, doctor must prescribe)",
        "keywords": "cancer medicine dawai drug names osimertinib erlotinib gefitinib afatinib alectinib crizotinib pembrolizumab nivolumab atezolizumab durvalumab cisplatin carboplatin pemetrexed paclitaxel etoposide",
        "text": "Examples of medicine groups used in lung cancer (names for information only): Targeted drugs for EGFR changes: osimertinib, erlotinib, gefitinib, afatinib. Targeted drugs for ALK changes: alectinib, crizotinib, brigatinib, lorlatinib. Immunotherapy: pembrolizumab, nivolumab, atezolizumab, durvalumab. Chemotherapy: cisplatin, carboplatin, pemetrexed, paclitaxel, gemcitabine, and etoposide (often used in small cell lung cancer). Which drug is right depends on tumor test results, stage and health. These are strong medicines with serious side effects. They must NEVER be taken on one's own: only if the patient's own oncologist prescribes them, and the plan should be verified with the doctor.",
        "source": GEN,
    },
    {
        "title": "Lung cancer screening",
        "keywords": "screening low dose CT LDCT jaanch kaun karwaye high risk smoker",
        "text": "Screening means testing people with no symptoms. For people at high risk, a yearly low-dose CT scan (LDCT) can find lung cancer early. In the US, guidelines suggest it for adults roughly 50 to 80 years old with a heavy smoking history (about 20 pack-years) who still smoke or quit within the past 15 years. Guidelines differ between countries, and chest X-ray is not recommended for routine screening. A doctor can say whether screening is right for a particular person.",
        "source": GEN,
    },
    {
        "title": "Prevention: lowering lung cancer risk",
        "keywords": "prevention bachav kaise bachen quit smoking chhodna tambaku secondhand radon pollution",
        "text": "The most effective step is to not smoke, or to quit. Quitting helps at any age, and doctors can help with counseling and quit-smoking medicines. Also avoid secondhand smoke, test homes for radon where it is a known problem, use protective equipment at jobs with dust or chemicals, use cleaner cooking fuel and good ventilation, and reduce exposure to heavy air pollution (for example masks on very polluted days). Healthy food and regular exercise support overall health.",
        "source": GEN,
    },
    {
        "title": "Survival, prognosis and what the numbers mean",
        "keywords": "survival prognosis life expectancy kitna jeeyega chances survival rate umeed",
        "text": "Survival numbers are statistics from groups of patients, not predictions for one person. In general, the earlier the stage at diagnosis, the better the outlook, and stage IV has the lowest survival. Age, general health, tumor type, tumor test results and response to treatment also matter. Newer treatments (targeted therapy, immunotherapy) have improved outcomes for many people, so old numbers can be pessimistic. Only the treating doctor can discuss an individual's outlook.",
        "source": GEN,
    },
    {
        "title": "About the LungCare AI survival model (this project)",
        "keywords": "survival model project LungCare AI Cox TCGA accuracy C-index kaise kaam karta hai",
        "text": "The Survival Estimate page uses a penalized Cox proportional hazards model trained on about 1,000 patients from TCGA-LUAD and TCGA-LUSC. It uses only four inputs: age, stage, sex and cancer type. On a locked test set its C-index was about 0.68, and on an outside LUAD cohort (GSE72094) about 0.64, which is moderate. A C-index of 0.5 is random guessing and 1.0 is perfect. The model is weaker for stage III-IV patients, and in the outside cohort it was over-optimistic for the highest-risk patients. It is a research prototype and not for medical decisions.",
        "source": "LungCare AI project results",
    },
    {
        "title": "Fever: what to do",
        "keywords": "fever bukhaar temperature tez bukhar paracetamol viral dengue malaria typhoid",
        "text": "Fever means body temperature of about 38 C (100.4 F) or higher. Most fevers come from viral infections and settle in a few days. Home care: rest, plenty of fluids (water, ORS, soups), light food, light clothing and a lukewarm sponge. Paracetamol is the usual first medicine for fever and pain in adults. The dose depends on age and weight, so follow the packet or ask a doctor or pharmacist, and do not give aspirin to children or teenagers. See a doctor if fever lasts more than 3 days, is very high, or comes with severe headache, rash, bleeding, stiff neck, trouble breathing, repeated vomiting, severe weakness or confusion. In India, high fever with body ache needs a doctor's check for dengue, malaria or typhoid. Avoid taking antibiotics on your own.",
        "source": GEN,
    },
    {
        "title": "Common cold and flu",
        "keywords": "cold flu sardi zukaam nazla runny nose chheenk sneezing",
        "text": "Colds and flu are usually caused by viruses and get better by themselves in about a week. Antibiotics do not work on viruses. Care: rest, warm fluids, steam inhalation, warm salt-water gargle for sore throat, saline nose drops and enough sleep. Paracetamol can ease fever and body pain, and an antihistamine like cetirizine is commonly used for a runny nose or sneezing, though it can cause sleepiness. Check with a pharmacist or doctor first if you have other illnesses, are pregnant, or are giving medicine to a child. See a doctor if symptoms last more than 10 days, get worse, or you have trouble breathing or chest pain.",
        "source": GEN,
    },
    {
        "title": "Cough",
        "keywords": "cough khansi dry cough balgam phlegm TB tuberculosis lasting cough",
        "text": "Most coughs after a cold settle in 1 to 3 weeks. Warm fluids, honey with warm water (not for babies under 1 year), steam and avoiding smoke and dust can help. A cough that lasts more than 3 weeks, or comes with blood, weight loss, night sweats, fever or breathlessness needs a doctor's check, because TB, asthma and lung cancer can also cause long cough. Cough syrups should be chosen with a pharmacist or doctor, especially for children.",
        "source": GEN,
    },
    {
        "title": "Headache",
        "keywords": "headache sar dard migraine tension head pain",
        "text": "Most headaches are tension-type or related to tiredness, dehydration, hunger, eye strain or poor sleep. Rest, water, a light meal and a break from screens can help. Paracetamol is a common first choice for mild headache in adults. Get urgent medical help for a sudden, very severe headache, headache after a head injury, or headache with fever and stiff neck, weakness, confusion, vision changes or fainting. See a doctor if headaches are frequent or getting worse.",
        "source": GEN,
    },
    {
        "title": "Sore throat",
        "keywords": "sore throat gala kharash gale mein dard tonsils",
        "text": "Sore throat is usually viral. Warm salt-water gargles, warm fluids, honey with warm water and rest help. Paracetamol can ease pain. See a doctor if it lasts more than a week, if swallowing or breathing is hard, or if there is high fever or white patches in the throat, because a bacterial infection might need a doctor-prescribed antibiotic.",
        "source": GEN,
    },
    {
        "title": "Acidity and heartburn",
        "keywords": "acidity heartburn gas jalan pet dard antacid indigestion",
        "text": "Acidity often improves by eating smaller meals, avoiding very spicy or oily food, not lying down right after eating, limiting tea, coffee, alcohol and smoking. Antacids from a pharmacy can give short relief. See a doctor if it happens often, or comes with trouble swallowing, black stools, vomiting blood, weight loss or chest pain, because chest pain can also be a heart problem.",
        "source": GEN,
    },
    {
        "title": "Diarrhea and dehydration",
        "keywords": "diarrhea loose motion dast ORS dehydration pet kharab vomiting ulti",
        "text": "The main danger of diarrhea and vomiting is losing too much water. Drink ORS (oral rehydration solution), water, coconut water and light foods like rice, banana and curd. See a doctor quickly if there is blood in stool, high fever, diarrhea for more than 2 days, signs of dehydration (very dry mouth, little urine, dizziness), or if a child, an elderly person or a pregnant woman is affected.",
        "source": GEN,
    },
    {
        "title": "Body pain and muscle ache",
        "keywords": "body pain muscle ache badan dard joint pain kamar dard weakness",
        "text": "Body aches after exertion or with a viral illness usually improve with rest, fluids and warm compresses. Paracetamol is a common first medicine for mild pain. Pain relievers like ibuprofen can upset the stomach and affect kidneys and are not suitable for everyone, so check with a doctor or pharmacist first. See a doctor if pain is severe, lasts more than a week, follows an injury, or comes with fever and rash.",
        "source": GEN,
    },
]

SYSTEM_PROMPT = """You are LungCare Assistant, a friendly health-information chatbot inside a student research project.

LANGUAGE: Reply in the same language/style the user writes in (Hindi, English or Hinglish). Keep answers clear, simple and not too long.

WHAT YOU CAN DO:
1. Lung cancer and other cancers: explain types, stages, symptoms, risk factors, screening, precautions, lifestyle, what tests mean, how treatment works in general.
2. Common minor illnesses (fever, cold, cough, headache, mild body pain, acidity, etc.): explain causes, home care, and you MAY name common over-the-counter medicines (for example paracetamol for fever).
3. Cancer treatment drugs: you MAY name them and explain in general what they are for, BUT you must clearly say that such a medicine must only be used if the patient's own doctor prescribes it, and that they should ask their doctor to verify it for their case.

HARD RULES:
- NEVER give doses, how many tablets, how often, or for how many days. If asked, say the dose depends on age, weight and health, and they should follow the doctor or pharmacist or the packet instructions.
- NEVER say you are diagnosing. You give general information only, not a diagnosis.
- Do not name prescription-only drugs (antibiotics, steroids, injections, etc.) for minor illnesses. Tell them to see a doctor instead.
- If the user mentions danger signs (trouble breathing, chest pain, coughing blood, fainting, very high fever that does not come down, confusion, severe bleeding, thoughts of self-harm), tell them to go to a doctor or emergency service immediately (in India call 112).
- If asked about survival chances of a specific person, explain that numbers are rough statistics and cannot predict one person, and use the SURVIVAL MODEL RESULT block if one is given below; if none is given, point to the Survival Estimate page in the sidebar as a research-only tool.
- End EVERY answer that mentions any medicine or treatment with a short reminder: please verify with your doctor before using it.
- If you are not sure about something, say so honestly. Do not invent facts, studies or numbers.

KNOWLEDGE BASE USE:
- Below the rules you may receive KNOWLEDGE BASE EXCERPTS. Use them as your main source and stay consistent with them.
- If the excerpts do not cover the question, say briefly that it is not in your knowledge base and answer carefully from general knowledge.
- Do not mention excerpt numbers. The app shows the sources separately.
"""


# ---------------------------------------------------------------- retrieval
@st.cache_resource
def build_index():
    docs = [f'{e["title"]} {e["keywords"]} {e["title"]} {e["keywords"]} {e["text"]}' for e in KB]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
    mat = vec.fit_transform(docs)
    return vec, mat


def retrieve(history, k=3, min_score=0.10):
    """Find the knowledge entries closest to the last two user messages."""
    user_msgs = [m["content"] for m in history if m["role"] == "user"][-2:]
    query = " ".join(user_msgs)
    if not query.strip():
        return []
    vec, mat = build_index()
    scores = cosine_similarity(vec.transform([query]), mat)[0]
    order = np.argsort(scores)[::-1][:k]
    return [KB[i] for i in order if scores[i] >= min_score]


def make_context(entries):
    if not entries:
        return "\n\nKNOWLEDGE BASE EXCERPTS: (none matched this question)"
    blocks = []
    for e in entries:
        blocks.append(f'- {e["title"]} [source: {e["source"]}]\n  {e["text"]}')
    return "\n\nKNOWLEDGE BASE EXCERPTS:\n" + "\n".join(blocks)


# ---------------------------------------------------------------- model loading
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------- survival engine (shared)
try:
    _b = load_bundle()
    imputer, scaler, model = _b["imputer"], _b["scaler"], _b["model"]
    features, stage_map = _b["features"], _b["stage_map"]
    _times = np.asarray(model.unique_times_, dtype=float)
    max_t, min_t = float(_times.max()), float(_times.min())
    per_year = 365.25 if max_t > 1000 else 1.0  # days vs years guess
    SURV_OK, SURV_ERR = True, ""
except Exception as _e:  # model file missing/broken: chatbot still works
    SURV_OK, SURV_ERR = False, str(_e)


def predict_curve(age, sex, stage, cancer_type):
    row = {
        "age": float(age),
        "stage": float(stage_map[stage]),
        "male": 1.0 if sex == "Male" else 0.0,
        "lusc": 1.0 if cancer_type.startswith("Squamous") else 0.0,
    }
    X = pd.DataFrame([row])[features]
    X = scaler.transform(imputer.transform(X))
    return model.predict_survival_function(X)[0]


def surv_at(fn, years):
    t = years * per_year
    if t < min_t:
        return 1.0
    return float(fn(min(t, max_t)))


SURV_TRIGGER = re.compile(
    r"(surviv|jee?y|jiy|bachn|bacheg|chance|prognos|estimate|andaz|umeed|life expect|kitne saal|how long)",
    re.I,
)

EXTRACT_PROMPT = """Read the conversation and extract patient details for a lung-cancer survival estimate.
Return ONLY a JSON object, nothing else:
{"wants_estimate": true or false, "age": integer or null, "sex": "Male" or "Female" or null, "stage": "I" or "II" or "III" or "IV" or null, "cancer_type": "LUAD" or "LUSC" or null}
Rules:
- wants_estimate is true only if the user asks for a survival / prognosis / chances estimate for a specific patient (themselves or someone else), or is giving patient details to get one (including answering an earlier question asking for age, sex or stage).
- Stage 1/2/3/4 or with letters A/B map to I, II, III, IV.
- adenocarcinoma = LUAD, squamous = LUSC.
- Use null for anything not stated. Use details from earlier user messages too.
"""


def extract_patient(api_key, history, custom_model=""):
    lines = []
    for m in history[-6:]:
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{who}: {m['content'][:600]}")
    body = {
        "systemInstruction": {"parts": [{"text": EXTRACT_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": "Conversation:\n" + "\n".join(lines)}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    models = [custom_model.strip()] if custom_model.strip() else DEFAULT_MODELS
    for name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent"
        try:
            r = requests.post(url, headers=headers, json=body, timeout=60)
            if r.status_code != 200:
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def survival_context(info):
    """Returns (context_text, source_dict_or_None, still_waiting_for_details)."""
    missing = []
    age = info.get("age")
    try:
        age = int(age)
        if not 18 <= age <= 100:
            raise ValueError
    except (TypeError, ValueError):
        age = None
        missing.append("age (years)")
    sex = info.get("sex") if info.get("sex") in ("Male", "Female") else None
    if not sex:
        missing.append("sex (male/female)")
    stage = info.get("stage") if info.get("stage") in stage_map else None
    if not stage:
        missing.append("stage (I, II, III or IV)")

    if missing:
        ctx = (
            "\n\nSURVIVAL MODEL: The user wants a survival estimate but these details are missing or invalid: "
            + ", ".join(missing)
            + ". Ask for ONLY these details in one short friendly message (cancer type adenocarcinoma/squamous is optional). "
            "Do NOT invent any survival numbers."
        )
        return ctx, None, True

    ctype = info.get("cancer_type")
    names = {"LUAD": "Adenocarcinoma (LUAD)", "LUSC": "Squamous cell (LUSC)"}
    types = [ctype] if ctype in names else ["LUAD", "LUSC"]
    lines = []
    for t in types:
        fn = predict_curve(age, sex, stage, names[t])
        vals = ", ".join(f"{y}-year {surv_at(fn, y) * 100:.0f}%" for y in (1, 3, 5))
        lines.append(f"- {names[t]}: {vals}")
    given = "cancer type given" if ctype in names else "cancer type NOT given, so both types are listed"
    ctx = (
        "\n\nSURVIVAL MODEL RESULT (computed by the app's Cox model; research only):\n"
        f"Patient: age {age}, {sex}, stage {stage} ({given}).\n"
        "Estimated survival probability:\n" + "\n".join(lines) + "\n"
        "INSTRUCTIONS: Show these numbers exactly as given (a small list or table). In simple words explain: "
        "they are rough group statistics from a model that uses only 4 inputs (age, stage, sex, cancer type); "
        "its test C-index was about 0.68 (moderate) and it is weaker for stage III-IV; it cannot predict one "
        "person's outcome; newer treatments may do better than these historical numbers. Do not add other numbers. "
        "Encourage discussing it with the treating oncologist."
    )
    src = {
        "title": "LungCare Cox survival model",
        "source": "Penalized Cox on TCGA-LUAD + TCGA-LUSC; test C-index about 0.68; research only",
    }
    return ctx, src, False


# ---------------------------------------------------------------- Gemini call
def ask_gemini(api_key, history, custom_model="", context=""):
    contents = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT + context}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.4},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    models = [custom_model.strip()] if custom_model.strip() else DEFAULT_MODELS
    last_error = ""
    for name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent"
        try:
            r = requests.post(url, headers=headers, json=body, timeout=60)
        except requests.RequestException as e:
            return None, f"Internet/connection problem: {e}"

        if r.status_code == 200:
            data = r.json()
            try:
                parts = data["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                if text.strip():
                    return text, name
                last_error = f"{name}: empty reply (maybe blocked by safety filter)"
            except (KeyError, IndexError):
                last_error = f"{name}: empty reply (maybe blocked by safety filter)"
            continue

        if r.status_code in (401, 403) or "API key not valid" in r.text:
            return None, "The API key is invalid or not working. Please check the key in the sidebar."

        last_error = f"{name}: error {r.status_code} - {r.text[:200]}"

    return None, last_error or "No model could answer."


def show_sources(sources):
    if sources:
        with st.expander("📚 Sources used"):
            for s in sources:
                st.markdown(f"- **{s['title']}**  \n  _{s['source']}_")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Page")
    page = st.radio("Page", ["💬 Chatbot", "📈 Survival Estimate"], label_visibility="collapsed")
    st.divider()
    st.header("Settings")
    _typed_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Needed only for the chatbot. Do not share it with anyone.",
    )
    # If the owner stored a key in Streamlit secrets / environment, visitors can use the chatbot
    # without typing a key. The stored key is never shown on the page.
    _env_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = _typed_key.strip() or _env_key
    if _env_key and not _typed_key.strip():
        st.caption("This app has a shared key, so you do not need to enter your own.")
    custom_model = st.text_input("Model name (optional)", value="", help="Leave blank and the app will try its default models automatically.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Knowledge base: {len(KB)} entries")
    st.caption("Research prototype. Not a substitute for a doctor's advice.")

st.markdown(
    """
    <style>
    /* ---------- page ---------- */
    .block-container {padding-top: 1.5rem; padding-bottom: 7rem; max-width: 880px;}

    /* ---------- hero banner ---------- */
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #0e7490 50%, #1d4ed8 100%);
        border-radius: 20px; padding: 26px 30px; margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(14,116,144,0.30);
    }
    .hero h1 {color: #ffffff; margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: 0.3px;}
    .hero p  {color: rgba(255,255,255,0.88); margin: 6px 0 0 0; font-size: 1.05rem;}

    /* ---------- sidebar page selector (big buttons) ---------- */
    section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 10px;}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        width: 100%; padding: 16px 18px; border-radius: 14px;
        border: 1.5px solid rgba(128,128,128,0.35);
        background: rgba(128,128,128,0.08);
        transition: all 0.15s ease; cursor: pointer;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        border-color: #14b8a6; background: rgba(20,184,166,0.12);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 1.2rem !important; font-weight: 700 !important; margin: 0;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #0f766e, #1d4ed8);
        border-color: transparent;
        box-shadow: 0 4px 14px rgba(29,78,216,0.35);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {color: #ffffff !important;}
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {display: none;}

    /* ---------- chat bubbles ---------- */
    div[data-testid="stChatMessage"] {
        border-radius: 16px; padding: 14px 16px; margin-bottom: 10px;
        border: 1px solid rgba(128,128,128,0.22);
        background: rgba(128,128,128,0.06);
    }
    div[data-testid="stChatInput"] {border-radius: 16px;}

    /* ---------- metric cards (survival %) ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15,118,110,0.18), rgba(29,78,216,0.18));
        border: 1px solid rgba(20,184,166,0.45);
        border-radius: 16px; padding: 16px 18px;
    }
    div[data-testid="stMetricValue"] {font-size: 2.2rem; font-weight: 800;}
    div[data-testid="stMetricLabel"] p {font-size: 1rem; font-weight: 600;}

    /* ---------- buttons ---------- */
    div.stButton > button {
        border-radius: 12px; font-weight: 700; padding: 0.55rem 1.2rem;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0f766e, #1d4ed8); border: none; color: white;
    }

    /* ---------- misc ---------- */
    h2, h3 {font-weight: 700;}
    </style>
    <div class="hero">
        <h1>🫁 LungCare AI</h1>
        <p>Lung cancer information chatbot and survival estimate (research prototype)</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "This is for learning and research only. It is not a diagnosis or medical advice. "
    "Always verify any medicine or treatment with your doctor before using it."
)


# ---------------------------------------------------------------- chatbot tab
if page == "💬 Chatbot":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.info(
            "Ask me anything, for example:\n\n"
            "- What does stage 3 lung cancer mean?\n"
            "- I have a fever, what should I do?\n"
            "- What are the symptoms of lung cancer?"
        )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_sources(m.get("sources"))

    user_text = st.chat_input("Type your question...")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            if not api_key:
                reply = "Please enter your Gemini API key in the left sidebar first."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                entries = retrieve(st.session_state.messages)
                surv_ctx, surv_src = "", None
                if SURV_OK and (st.session_state.get("awaiting_patient") or SURV_TRIGGER.search(user_text)):
                    with st.spinner("Reading patient details..."):
                        pinfo = extract_patient(api_key, st.session_state.messages, custom_model)
                    if pinfo and pinfo.get("wants_estimate"):
                        surv_ctx, surv_src, waiting = survival_context(pinfo)
                        st.session_state.awaiting_patient = waiting
                    else:
                        st.session_state.awaiting_patient = False
                with st.spinner("Thinking..."):
                    text, info = ask_gemini(
                        api_key, st.session_state.messages, custom_model, make_context(entries) + surv_ctx
                    )
                if text:
                    st.markdown(text)
                    srcs = [{"title": e["title"], "source": e["source"]} for e in entries]
                    if surv_src:
                        srcs.insert(0, surv_src)
                    show_sources(srcs)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": text, "sources": srcs}
                    )
                else:
                    st.error(info)
                    st.session_state.messages.pop()

# ---------------------------------------------------------------- survival tab
if page == "📈 Survival Estimate":
    if not SURV_OK:
        st.error(f"Could not load the survival model: {SURV_ERR}")
        st.stop()

    st.subheader("Patient details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=65)
        sex = st.selectbox("Sex", ["Male", "Female"])
    with c2:
        stage = st.selectbox("Stage", ["I", "II", "III", "IV"])
        cancer_type = st.selectbox("Cancer type", ["Adenocarcinoma (LUAD)", "Squamous cell (LUSC)"])

    if st.button("Estimate survival", type="primary"):
        fn = predict_curve(age, sex, stage, cancer_type)

        st.subheader("Estimated survival probability")
        m1, m2, m3 = st.columns(3)
        for col, yrs in zip((m1, m2, m3), (1, 3, 5)):
            col.metric(f"{yrs}-year", f"{surv_at(fn, yrs) * 100:.0f}%")

        xs_years = np.linspace(0, min(10.0, max_t / per_year), 60)
        ys = [surv_at(fn, y) for y in xs_years]
        chart = pd.DataFrame({"Years": xs_years, "Survival probability": ys}).set_index("Years")
        st.line_chart(chart)

        st.subheader("Same patient, different stages")
        rows = []
        for s in ["I", "II", "III", "IV"]:
            f2 = predict_curve(age, sex, s, cancer_type)
            rows.append(
                {
                    "Stage": s,
                    "1-year": f"{surv_at(f2, 1) * 100:.0f}%",
                    "3-year": f"{surv_at(f2, 3) * 100:.0f}%",
                    "5-year": f"{surv_at(f2, 5) * 100:.0f}%",
                }
            )
        st.table(pd.DataFrame(rows).set_index("Stage"))

        st.info(
            "This model uses only 4 inputs (age, stage, sex and cancer type), so it gives a very "
            "rough estimate and cannot predict the outcome for one person. The model is weaker for "
            "stage III-IV. Its C-index on the test data was 0.68 (moderate). "
            "For research only."
        )

st.divider()
st.caption("Survival model: penalized Cox, trained on TCGA-LUAD + TCGA-LUSC clinical data. Chatbot: Google Gemini + built-in knowledge base.")
