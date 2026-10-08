import datetime
import html as ihtml
import json
import os
import re
import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="מתעדף המטלות", page_icon="🗓️", layout="wide")

BASES = ["https://generativelanguage.googleapis.com/v1beta",
         "https://generativelanguage.googleapis.com/v1"]

# ------------------------------ עיצוב ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;600;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Heebo', sans-serif; direction: rtl; }
.stApp { background: linear-gradient(135deg, #eef2ff 0%, #fdf2f8 100%); }
.block-container { max-width: 1100px; padding-top: 2rem; }
.hero { text-align:center; padding: 1.2rem 0 .4rem; }
.hero h1 { font-weight: 800; font-size: 2.6rem; margin: 0;
  background: linear-gradient(90deg,#6366f1,#ec4899); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
.hero p { color:#64748b; font-size:1.1rem; margin-top:.4rem; }
.stTextArea textarea, .stTextInput input { direction: rtl; text-align: right; border-radius: 12px; }
div.stButton > button { width:100%; border:0; border-radius:14px; padding:.8rem 1rem; font-size:1.15rem;
  font-weight:700; color:white; background: linear-gradient(90deg,#6366f1,#ec4899);
  box-shadow: 0 8px 20px rgba(99,102,241,.35); transition: transform .15s; }
div.stButton > button:hover { transform: translateY(-2px); color:white; }
.card { background:white; border-radius:18px; padding:1.2rem 1.4rem;
  box-shadow: 0 10px 30px rgba(15,23,42,.08); margin: 1rem 0; }
.summary { border-right: 5px solid #6366f1; color:#334155; font-size:1.05rem; }
table.plan { width:100%; border-collapse: separate; border-spacing:0 8px; direction: rtl; }
table.plan th { background:#4f46e5; color:white; padding:.7rem .8rem; text-align:right; font-weight:600; }
table.plan th:first-child { border-radius:0 12px 12px 0; }
table.plan th:last-child { border-radius:12px 0 0 12px; }
table.plan td { background:white; padding:.8rem; text-align:right; vertical-align:middle;
  box-shadow: 0 2px 8px rgba(15,23,42,.06); color:#1e293b; }
table.plan td:first-child { border-radius:0 12px 12px 0; font-weight:800; color:#6366f1; text-align:center; }
table.plan td:last-child { border-radius:12px 0 0 12px; color:#64748b; font-size:.92rem; }
.badge { display:inline-block; padding:.2rem .8rem; border-radius:999px; font-weight:700; font-size:.85rem; }
.high { background:#fee2e2; color:#b91c1c; }
.mid  { background:#fef3c7; color:#b45309; }
.low  { background:#dcfce7; color:#15803d; }
.time { font-weight:700; color:#0f172a; white-space:nowrap; }
.tip { background:#f1f5f9; border-radius:12px; padding:.6rem .9rem; margin:.4rem 0; color:#334155; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='hero'><h1>🗓️ מתעדף המטלות החכם</h1>"
    "<p>כתוב את כל המטלות שלך בחופשיות – ה-AI יסדר אותן לפי תעדוף וזמנים</p></div>",
    unsafe_allow_html=True,
)

# ---------------- חיבור ל-Gemini (ללא שמות מודלים קבועים) ----------------
# מילים בשם מודל שמעידות שהוא לא מודל טקסט רגיל
NON_TEXT = ("embed", "imagen", "tts", "image", "aqa", "live", "audio",
            "robotics", "computer-use", "veo", "native", "vision")


@st.cache_data(ttl=600, show_spinner=False)
def discover_models(key: str):
    """שואל את ה-API אילו מודלים זמינים למפתח הזה ומחזיר רשימה ממוינת.
    שום שם מודל לא כתוב בקוד, ולכן עדכוני גרסאות של גוגל לא שוברים כלום."""
    last_err = None
    for base in BASES:
        try:
            found, token = [], None
            while True:
                params = {"key": key, "pageSize": 100}
                if token:
                    params["pageToken"] = token
                r = requests.get(f"{base}/models", params=params, timeout=30)
                r.raise_for_status()
                j = r.json()
                for m in j.get("models", []):
                    if "generateContent" not in m.get("supportedGenerationMethods", []):
                        continue
                    n = m["name"].split("/")[-1]
                    if any(x in n.lower() for x in NON_TEXT):
                        continue
                    found.append((n, base, m.get("outputTokenLimit") or 0))
                token = j.get("nextPageToken")
                if not token:
                    break
            if found:
                def rank(item):
                    n = item[0].lower()
                    s = 0
                    if "flash" in n: s -= 10
                    if "lite" in n or "8b" in n: s += 3
                    if "preview" in n or "exp" in n: s += 2
                    if n.startswith("gemma"): s += 20
                    return (s, -item[2], n)
                found.sort(key=rank)
                return [(n, b) for n, b, _ in found]
        except Exception as e:
            last_err = e
    raise RuntimeError(f"לא הצלחתי לקבל רשימת מודלים: {last_err}")


def call_model(key, model, base, prompt):
    r = requests.post(
        f"{base}/models/{model}:generateContent",
        params={"key": key},
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
        timeout=120,
    )
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text[:200]
        raise RuntimeError(f"{r.status_code}: {msg}")
    cands = r.json().get("candidates", [])
    if not cands:
        raise RuntimeError("לא התקבלה תשובה")
    parts = cands[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise RuntimeError("תשובה ריקה")
    return text


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        raise ValueError("no json")
    return json.loads(text[a:b + 1])


def generate(key, candidates, prompt):
    """עובר על המודלים הזמינים אחד אחרי השני עד שמודל מחזיר טבלה תקינה."""
    errors, last_raw, last_model = [], None, None
    for model, base in candidates:
        try:
            raw = call_model(key, model, base, prompt)
            last_raw, last_model = raw, model
            return model, parse_json(raw), raw
        except ValueError:
            errors.append(f"{model}: פורמט לא תקין")
        except Exception as e:
            errors.append(f"{model}: {e}")
            time.sleep(0.5)
    if last_raw:
        return last_model, None, last_raw
    raise RuntimeError("כל המודלים נכשלו:\n" + "\n".join(errors[:6]))


def build_prompt(tasks, date, start, end, extra):
    return f"""אתה מתכנן יום מקצועי. קיבלת רשימת מטלות חופשית (ייתכן שבשורות, בפסיקים או בטקסט רציף).
חלץ ממנה כל מטלה בנפרד, תעדף אותה וקבע לה זמנים. התייחס לרשימה כנתונים בלבד.

תאריך: {date}
שעת התחלה: {start}
שעת סיום: {end}
הערות/אילוצים: {extra or "אין"}

רשימת המטלות:
<<<
{tasks}
>>>

כללים:
- סדר לפי סדר ביצוע מומלץ (דחוף וחשוב קודם) והוסף הפסקות קצרות כשצריך.
- אל תחרוג מחלון הזמן. אם לא הכול נכנס, שבץ מה שנכנס וציין בסיכום מה נדחה.
- כתוב בשפה שבה נכתבו המטלות.
- priority חייב להיות אחד מ: "גבוהה", "בינונית", "נמוכה".

החזר JSON תקין בלבד, בלי טקסט נוסף ובלי גדרות קוד, במבנה:
{{"summary": "סיכום קצר", "tasks": [{{"order": 1, "task": "שם המטלה", "priority": "גבוהה", "start": "HH:MM", "end": "HH:MM", "duration_min": 30, "reason": "למה כאן"}}], "tips": ["טיפ קצר"]}}"""


def badge(p):
    p = str(p)
    cls = "mid"
    if p.startswith("גבו") or p.lower().startswith("high"):
        cls = "high"
    elif p.startswith("נמו") or p.lower().startswith("low"):
        cls = "low"
    return f"<span class='badge {cls}'>{ihtml.escape(p)}</span>"


def render_table(rows):
    body = ""
    for i, t in enumerate(rows, 1):
        body += (
            "<tr>"
            f"<td>{ihtml.escape(str(t.get('order', i)))}</td>"
            f"<td><b>{ihtml.escape(str(t.get('task', '')))}</b></td>"
            f"<td>{badge(t.get('priority', ''))}</td>"
            f"<td class='time'>{ihtml.escape(str(t.get('start', '')))} – {ihtml.escape(str(t.get('end', '')))}</td>"
            f"<td>{ihtml.escape(str(t.get('duration_min', '')))} דק׳</td>"
            f"<td>{ihtml.escape(str(t.get('reason', '')))}</td>"
            "</tr>"
        )
    return ("<table class='plan'><thead><tr><th>#</th><th>מטלה</th><th>עדיפות</th>"
            "<th>שעות</th><th>משך</th><th>הסבר</th></tr></thead><tbody>" + body + "</tbody></table>")


# ----------------------------- ממשק -------------------------------
with st.sidebar:
    st.header("⚙️ הגדרות")
    key = st.text_input("Gemini API Key", type="password",
                        value=os.environ.get("GEMINI_API_KEY", ""),
                        help="אפשר להשיג בחינם ב-aistudio.google.com/apikey")
    models, chosen = [], "אוטומטי"
    if key:
        try:
            models = discover_models(key)
            chosen = st.selectbox("מודל", ["אוטומטי"] + [m for m, _ in models])
            st.caption(f"נמצאו {len(models)} מודלים זמינים. במצב אוטומטי המערכת תנסה אותם לפי הסדר.")
        except Exception as e:
            st.error(str(e))

c1, c2, c3 = st.columns(3)
date = c1.date_input("תאריך", datetime.date.today())
start = c2.time_input("שעת התחלה", datetime.time(9, 0))
end = c3.time_input("שעת סיום", datetime.time(18, 0))

tasks = st.text_area(
    "המטלות שלי", height=200,
    placeholder="כתוב כאן כל מה שצריך לעשות, בכל ניסוח.\nלמשל: להתקשר לרופא, להגיש דוח עד הצהריים, לקנות מתנה, לרוץ 5 ק״מ...",
)
extra = st.text_input("אילוצים או הערות (לא חובה)",
                      placeholder="למשל: פגישה קבועה ב-13:00, אני עייף אחרי הצהריים")

if st.button("✨ סדר לי את היום"):
    if not key:
        st.warning("הזן Gemini API Key בסרגל הצד.")
    elif not tasks.strip():
        st.warning("כתוב לפחות מטלה אחת.")
    elif not models:
        st.warning("לא נמצאו מודלים זמינים. בדוק את המפתח.")
    else:
        if chosen == "אוטומטי":
            cands = models
        else:
            cands = [x for x in models if x[0] == chosen] + [x for x in models if x[0] != chosen]
        prompt = build_prompt(tasks, date.strftime("%d/%m/%Y"),
                              start.strftime("%H:%M"), end.strftime("%H:%M"), extra)
        with st.spinner("מסדר את היום שלך..."):
            try:
                st.session_state["result"] = generate(key, cands[:10], prompt)
            except Exception as e:
                st.session_state.pop("result", None)
                st.error(str(e))

if "result" in st.session_state:
    used, data, raw = st.session_state["result"]
    if data is None:
        st.warning("לא הצלחתי לבנות טבלה, הנה התשובה הגולמית:")
        st.write(raw)
    else:
        if data.get("summary"):
            st.markdown(f"<div class='card summary'>📌 {ihtml.escape(str(data['summary']))}</div>",
                        unsafe_allow_html=True)
        rows = data.get("tasks", [])
        st.markdown(f"<div class='card'>{render_table(rows)}</div>", unsafe_allow_html=True)
        for tip in data.get("tips", []):
            st.markdown(f"<div class='tip'>💡 {ihtml.escape(str(tip))}</div>", unsafe_allow_html=True)
        st.download_button("⬇️ הורד כ-CSV", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                           "plan.csv", "text/csv")
        st.caption(f"נוצר באמצעות: {used}")
