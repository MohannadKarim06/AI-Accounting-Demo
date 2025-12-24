import streamlit as st
import pandas as pd
import requests
import base64
import json
import os
import time
import hashlib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="نظام محاسبي ذكي",
    layout="wide"
)

# =========================
# Arabic font + UI styling
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
}

h1, h2, h3, h4 {
    font-weight: 700;
}

.stDataFrame {
    direction: rtl;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Constants & paths
# =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_VISION = "anthropic/claude-3.5-sonnet"

BASE_CACHE = "/data/cache"
DOC_CACHE = f"{BASE_CACHE}/documents"
QUERY_CACHE = f"{BASE_CACHE}/queries"
REPORT_CACHE = f"{BASE_CACHE}/reports"

for path in [DOC_CACHE, QUERY_CACHE, REPORT_CACHE]:
    os.makedirs(path, exist_ok=True)

ALLOWED_CATEGORIES = [
    "مبيعات",
    "مشتريات",
    "مصروفات تشغيل",
    "رواتب",
    "ضرائب",
    "خدمات",
    "أخرى"
]

# =========================
# Helpers
# =========================
def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def encode_file(file):
    return base64.b64encode(file.read()).decode("utf-8")

def call_openrouter(messages, max_tokens=1500):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_VISION,
        "messages": messages,
        "max_tokens": max_tokens
    }
    res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# =========================
# Document extraction
# =========================
def extract_document(file):
    raw = file.read()
    file_hash = hash_bytes(raw)
    cache_path = f"{DOC_CACHE}/{file_hash}.json"

    if os.path.exists(cache_path):
        time.sleep(2)
        return json.load(open(cache_path, "r", encoding="utf-8"))

    base64_file = base64.b64encode(raw).decode("utf-8")
    mime = file.type

    prompt = f"""
أنت محاسب محترف في السوق المصري.

استخرج البيانات التالية بدقة:
- نوع المستند
- التاريخ (YYYY-MM-DD)
- العميل أو المورد
- المبلغ الإجمالي (رقم فقط)
- ضريبة القيمة المضافة (رقم فقط)
- التصنيف (واحد فقط من القائمة):
{", ".join(ALLOWED_CATEGORIES)}

أرجع النتيجة بصيغة JSON فقط.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{base64_file}"
                    }
                }
            ]
        }
    ]

    result = call_openrouter(messages)
    data = json.loads(result)
    data["filename"] = file.name

    json.dump(data, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    return data

# =========================
# Semantic query
# =========================
def semantic_query(df, question):
    key = hashlib.sha256((df.to_csv() + question).encode()).hexdigest()
    path = f"{QUERY_CACHE}/{key}.json"

    if os.path.exists(path):
        time.sleep(1.5)
        return json.load(open(path, "r", encoding="utf-8"))

    prompt = f"""
البيانات التالية:
{df.to_string(index=False)}

السؤال:
{question}

أجب بصيغة JSON:
{{
  "answer": "",
  "rows": []
}}
"""

    result = call_openrouter([{"role": "user", "content": prompt}], max_tokens=2000)
    parsed = json.loads(result)

    json.dump(parsed, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return parsed

# =========================
# Report generation
# =========================
def generate_report(df):
    key = hashlib.sha256(df.to_csv().encode()).hexdigest()
    path = f"{REPORT_CACHE}/{key}.json"

    if os.path.exists(path):
        time.sleep(2)
        return json.load(open(path, "r", encoding="utf-8"))

    prompt = f"""
البيانات:
{df.to_string(index=False)}

أنشئ تقريرًا ماليًا احترافيًا بصيغة JSON:
{{
 "العنوان": "",
 "إجمالي_الإيرادات": 0,
 "إجمالي_المصروفات": 0,
 "صافي_الربح": 0,
 "الملخص": ""
}}
"""

    result = call_openrouter([{"role": "user", "content": prompt}], max_tokens=1500)
    parsed = json.loads(result)

    json.dump(parsed, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return parsed

# =========================
# PDF export
# =========================
def export_pdf(report):
    file_path = "/tmp/report.pdf"
    c = canvas.Canvas(file_path, pagesize=A4)
    text = c.beginText(40, 800)

    for k, v in report.items():
        text.textLine(f"{k}: {v}")

    c.drawText(text)
    c.save()
    return file_path

# =========================
# UI
# =========================
st.title("📊 نظام محاسبي ذكي متكامل")

st.markdown("""
هذا عرض تجريبي لنظام **يدخل الذكاء الاصطناعي داخل العمليات المحاسبية**  
وليس مجرد أداة منفصلة.
""")

st.divider()

# -------- Step 1
st.header("1️⃣ رفع المستندات المحاسبية")

files = st.file_uploader(
    "ارفع فواتير، إيصالات، مصروفات (صور أو PDF)",
    accept_multiple_files=True
)

records = []
if files:
    with st.spinner("جاري تحليل المستندات..."):
        for f in files:
            records.append(extract_document(f))

if records:
    df = pd.DataFrame(records)
    st.success("تم استخراج البيانات بنجاح")
    st.dataframe(df)

    # -------- Step 2
    st.header("2️⃣ الاستعلام الذكي")
    question = st.text_input("اسأل عن البيانات (مثال: إجمالي المبيعات في يوم معين)")

    if question:
        with st.spinner("جاري التحليل..."):
            answer = semantic_query(df, question)
        st.markdown(f"**الإجابة:** {answer['answer']}")
        if answer["rows"]:
            st.dataframe(pd.DataFrame(answer["rows"]))

    # -------- Step 3
    st.header("3️⃣ تقرير مالي تلقائي")

    if st.button("إنشاء تقرير"):
        with st.spinner("جاري إعداد التقرير..."):
            report = generate_report(df)
        st.json(report)

        pdf_path = export_pdf(report)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 تحميل التقرير PDF",
                f,
                file_name="financial_report.pdf"
            )

st.divider()
st.caption("نموذج تجريبي — يوضح الإمكانيات وليس المنتج النهائي")
