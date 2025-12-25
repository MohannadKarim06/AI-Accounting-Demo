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
    page_title="نظام محاسبة ذكي",
    layout="wide"
)

# =========================
# Arabic RTL Styling
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
}

.stDataFrame table {
    direction: rtl;
    text-align: right;
}

.stDataFrame th {
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Constants
# =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_VISION = "anthropic/claude-3.5-sonnet"

BASE_CACHE = "./data/cache"
DOC_CACHE = f"{BASE_CACHE}/documents"
QUERY_CACHE = f"{BASE_CACHE}/queries"
REPORT_CACHE = f"{BASE_CACHE}/reports"

for p in [DOC_CACHE, QUERY_CACHE, REPORT_CACHE]:
    os.makedirs(p, exist_ok=True)

# =========================
# Helpers
# =========================
def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

def call_llm(messages, max_tokens=1500):
    payload = {
        "model": MODEL_VISION,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# =========================
# Document Extraction
# =========================
def extract_document(file):
    raw = file.read()
    h = hash_bytes(raw)
    cache = f"{DOC_CACHE}/{h}.json"

    if os.path.exists(cache):
        time.sleep(1.5)
        return json.load(open(cache, encoding="utf-8"))

    prompt = """
أنت محاسب محترف.

حلل المستند وحدد هل هو:
- income (دخل)
- expense (مصروف)

استخرج البيانات التالية.
إذا لم تجد أي معلومة، ضع null ولا تخمن.

أرجع JSON فقط:

{
 "transaction_type": "income أو expense",
 "document_type": null,
 "invoice_number": null,
 "date": null,
 "party_name": null,
 "category": null,
 "description": null,
 "subtotal": null,
 "tax_amount": null,
 "total_amount": null,
 "payment_method": null,
 "currency": "EGP",
 "confidence_score": 0.0
}
"""

    b64 = base64.b64encode(raw).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{file.type};base64,{b64}"}}
        ]
    }]

    result = json.loads(call_llm(messages))
    result["source_document"] = file.name

    json.dump(result, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return result

# =========================
# Semantic Query
# =========================
def semantic_query(df, question):
    prompt = f"""
البيانات:
{df.to_string(index=False)}

السؤال:
{question}

أجب بصيغة JSON:
{{
 "answer_text": "",
 "answer_numeric": null,
 "rows": []
}}
"""
    return json.loads(call_llm([{"role": "user", "content": prompt}], 2000))

# =========================
# Report
# =========================
def generate_report(income_df, expense_df):
    total_income = income_df["إجمالي المبلغ"].sum()
    total_expense = expense_df["إجمالي المبلغ"].sum()
    net = total_income - total_expense

    return {
        "العنوان": "تقرير مالي تحليلي",
        "إجمالي الإيرادات": total_income,
        "إجمالي المصروفات": total_expense,
        "صافي الربح": net,
        "الملخص": (
            "يعرض هذا التقرير نظرة شاملة على الأداء المالي. "
            "تم استخراج البيانات تلقائيًا من المستندات باستخدام الذكاء الاصطناعي، "
            "مما يساعد على تقليل الأخطاء اليدوية وتحسين سرعة اتخاذ القرار."
        )
    }

# =========================
# PDF Export
# =========================
def export_pdf(report):
    path = "/tmp/report.pdf"
    c = canvas.Canvas(path, pagesize=A4)
    text = c.beginText(450, 800)
    for k, v in report.items():
        text.textLine(f"{k}: {v}")
    c.drawText(text)
    c.save()
    return path

# =========================
# UI
# =========================
st.title("📊 نظام محاسبة ذكي متكامل")

files = st.file_uploader("ارفع المستندات", accept_multiple_files=True)

income, expense = [], []

if files:
    for f in files:
        d = extract_document(f)
        if d["transaction_type"] == "income":
            income.append(d)
        else:
            expense.append(d)

if income or expense:
    income_df = pd.DataFrame(income)
    expense_df = pd.DataFrame(expense)

    rename_map = {
        "document_type": "نوع المستند",
        "invoice_number": "رقم الفاتورة",
        "date": "التاريخ",
        "party_name": "العميل / المورد",
        "category": "التصنيف",
        "description": "الوصف",
        "subtotal": "قبل الضريبة",
        "tax_amount": "الضريبة",
        "total_amount": "إجمالي المبلغ",
        "payment_method": "طريقة الدفع"
    }

    if not income_df.empty:
        st.subheader("📈 الإيرادات")
        st.dataframe(income_df.rename(columns=rename_map))

    if not expense_df.empty:
        st.subheader("📉 المصروفات")
        st.dataframe(expense_df.rename(columns=rename_map))

    q = st.text_input("اسأل عن البيانات")
    if q:
        res = semantic_query(pd.concat([income_df, expense_df]), q)
        st.markdown(f"**الإجابة:** {res['answer_text']}")
        if res["rows"]:
            st.dataframe(pd.DataFrame(res["rows"]))

    if st.button("إنشاء تقرير"):
        report = generate_report(
            income_df.rename(columns=rename_map),
            expense_df.rename(columns=rename_map)
        )
        st.json(report)
        with open(export_pdf(report), "rb") as f:
            st.download_button("تحميل PDF", f, "report.pdf")

st.caption("عرض تجريبي — يوضح الإمكانيات")
