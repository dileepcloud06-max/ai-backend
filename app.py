from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import date, datetime, timezone
from typing import Any
import pandas as pd
import json
import re
import uuid
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import base64
import csv
from pathlib import Path
from urllib.parse import quote_plus
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import mysql.connector

app = FastAPI()


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
		"https://ai-frontend-pi-six.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_connection():
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", "4000"))

    if not host:
        raise RuntimeError("DB_HOST is missing in .env")

    print(f"Connecting to MySQL: {host}:{port}")

    return mysql.connector.connect(
        host=host,
        port=port,
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        ssl_disabled=False,
        autocommit=True,
        connection_timeout=30
    )

def execute_db_query_with_timeout(query: str, timeout_seconds: float = 1.5):
    res_holder = []
    def _run():
        try:
            conn = get_connection()
            cur = conn.cursor()
            try:
                cur.execute(query)
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                res_holder.append([dict(zip(cols, r)) for r in rows])
            finally:
                try: cur.close()
                except Exception: pass
                try: conn.close()
                except Exception: pass
        except Exception as e:
            print("DB Query Error:", e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_seconds)
    if res_holder:
        return res_holder[0]
    return []



# =========================================================
# GET DATA FROM MYSQL
# =========================================================

@app.get("/data")
def get_data():

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        query = """
           SELECT * FROM gl_review_intelligence where email_status != 'Y' and rating <= 3 order by review_id limit 10
        """

        cursor.execute(query)

        columns = [
            desc[0]
            for desc in cursor.description
        ]

        rows = cursor.fetchall()

        result = [
            dict(zip(columns, row))
            for row in rows
        ]

        return {
            "status": "success",
            "count": len(result),
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"MySQL error: {str(e)}"
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# =========================================================
# EXCEL DATASET READER & CLASSIFICATION SERVICE
# =========================================================

_EXCEL_CACHE = None

EXCEL_GERMAN_TRANSLATIONS = {
    'akkulaufzeit': 'Battery Life',
    'geräuschpegel': 'Noise Level',
    'bluetooth-verbindung': 'Bluetooth Connectivity',
    'gewicht': 'Product Weight',
    'nähte': 'Stitching & Finish',
    'aufbau': 'Assembly & Setup',
    'lieferzustand': 'Delivery Condition',
    'lüftergeräusch': 'Fan Noise',
    'temperaturregelung': 'Temperature Control',
    'temperaturstabilität': 'Temperature Stability',
    'innenbeleuchtung': 'Interior Lighting',
    'lautstärke': 'Noise Level',
    'gefrierleistung': 'Freezing Performance',
    'stromverbrauch': 'Energy Consumption',
    'türdichtung': 'Door Seal',
    'bedienung': 'Control Panel',
    'verarbeitung': 'Build Quality',
    'handhabung': 'Ease of Use',
    'preis-leistungs-verhältnis': 'Value for Money',
    'reinigung': 'Cleaning & Maintenance',
    'materialqualität': 'Material Quality',
    'wasserbeständigkeit': 'Water Resistance',
    'ladezeit': 'Charging Speed',
    'aufheizgeschwindigkeit': 'Heating Speed',
    'sitzkomfort': 'Seating Comfort',
    'überhitzung': 'Overheating',
    'stabilität der beine': 'Leg Stability',
    'schubladenmechanismus': 'Drawer Mechanism',
    'traktion': 'Sole Traction',
    'gleichmäßiges mixen': 'Blending Performance',
    'passform': 'Sizing & Fit',
    'fingerabdrucksensor': 'Fingerprint Sensor',
    'ärmellänge': 'Sleeve Length',
    'roststabilität': 'Grate Stability',
    'wärmeentwicklung': 'Heat Dissipation',
    'kühlleistung': 'Cooling Performance',
    'bürstenleistung': 'Brush Efficiency',
    'rollenqualität': 'Wheel Quality',
    'auslaufschutz': 'Leakage Protection',
    'saugleistung': 'Suction Power',
    'software-updates': 'Software Stability',
    'display-helligkeit': 'Display Brightness'
}

def extract_excel_issue_type(text: str, rating: int, sentiment: str) -> str:
    if str(sentiment).strip().title() == 'Positive' or rating >= 4:
        return 'No Issue'
    text_str = str(text or "")
    patterns = [
        r'(?:and the|with the|poor|Probleme mit der|die|hat|mit der|causes|due to)\s+([A-Za-zÄöüäÖÜß\-\s]{3,30}?)\s+(?:has|is|hat|verursacht|caused|bin|ein|poor|was|were|issues)',
        r'(?:disappointed with the|problems with|issue with the|fail in|defect in)\s+([A-Za-zÄöüäÖÜß\-\s]{3,30}?)(?:\.|\,|$)',
        r'(?:the|die|das)\s+([A-Za-zÄöüäÖÜß\-\s]{3,25}?)\s+(?:is poor|has been a problem|ist ein Problem|verursacht)'
    ]
    for pat in patterns:
        m = re.search(pat, text_str, re.IGNORECASE)
        if m:
            val = m.group(1).strip().lower()
            val_clean = re.sub(r'^(the|die|das|a|an)\s+', '', val)
            if val_clean in EXCEL_GERMAN_TRANSLATIONS:
                return EXCEL_GERMAN_TRANSLATIONS[val_clean]
            for key, translated in EXCEL_GERMAN_TRANSLATIONS.items():
                if key in val_clean:
                    return translated
            if len(val_clean) >= 3 and val_clean not in ['this', 'that', 'very', 'satisfied', 'regular', 'limited', 'usage', 'experience', 'product']:
                return val_clean.title()
    if rating <= 2 or str(sentiment).strip().title() == 'Negative':
        return 'Product Quality Defect'
    return 'General Inquiry'


def load_excel_classification_data():
    global _EXCEL_CACHE
    if _EXCEL_CACHE is not None:
        return _EXCEL_CACHE

    excel_path = os.path.join(os.path.dirname(__file__), "dataset.xlsx")
    if not os.path.exists(excel_path):
        return {"catalog": [], "reviews_by_product": {}, "summary": {}}

    try:
        df = pd.read_excel(excel_path)
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(3).astype(int)
        df['Sentiment_Label'] = df['Sentiment_Label'].fillna('Neutral').astype(str).str.strip().str.title()
        df['Issue_Type'] = df.apply(lambda r: extract_excel_issue_type(r['Review_Text'], r['Rating'], r['Sentiment_Label']), axis=1)

        total_reviews = len(df)
        total_positive = int((df['Sentiment_Label'] == 'Positive').sum())
        total_negative = int((df['Sentiment_Label'] == 'Negative').sum())
        total_neutral = int((df['Sentiment_Label'] == 'Neutral').sum())

        grouped = df.groupby('Product_Model')
        catalog = []
        reviews_by_product = {}

        for model, group in grouped:
            model_name = str(model).strip()
            pos = int((group['Sentiment_Label'] == 'Positive').sum())
            neg = int((group['Sentiment_Label'] == 'Negative').sum())
            neu = int((group['Sentiment_Label'] == 'Neutral').sum())
            tot = len(group)

            avg_rating = round(float(group['Rating'].mean()), 2)
            category = str(group['Category'].iloc[0]).strip()
            product_type = str(group['Product_Type'].iloc[0]).strip()
            brand = model_name.split()[0]

            non_no_issue = group[group['Issue_Type'] != 'No Issue']['Issue_Type']
            issues_series = non_no_issue.value_counts()
            issues_list = [
                {
                    'name': str(issue),
                    'count': int(cnt),
                    'percentage': round((int(cnt) / tot) * 100, 1)
                }
                for issue, cnt in issues_series.head(5).items()
            ]

            prod_reviews = []
            for _, r in group.iterrows():
                prod_reviews.append({
                    "review_id": str(r.get('Review_ID', '')),
                    "reviewer_name": f"{r.get('Source', 'Customer')} User",
                    "review_text": str(r.get('Review_Text', '')),
                    "rating": int(r.get('Rating', 3)),
                    "source": str(r.get('Source', 'Online')),
                    "review_date": str(r.get('Review_Date', '')),
                    "sentiment": str(r.get('Sentiment_Label', 'Neutral')),
                    "issue_type": str(r.get('Issue_Type', 'No Issue')),
                    "severity": "High" if int(r.get('Rating', 3)) <= 2 or str(r.get('Sentiment_Label')) == 'Negative' else "Low",
                    "problem_summary": str(r.get('Review_Text', '')),
                    "product": model_name
                })

            reviews_by_product[model_name.lower()] = prod_reviews

            catalog.append({
                'product_id': model_name,
                'product_name': model_name,
                'name': model_name,
                'brand': brand,
                'category': category,
                'product_type': product_type,
                'avg_rating': avg_rating,
                'positive_count': pos,
                'negative_count': neg,
                'neutral_count': neu,
                'total_reviews': tot,
                'positive_pct': round((pos / tot) * 100, 1),
                'negative_pct': round((neg / tot) * 100, 1),
                'neutral_pct': round((neu / tot) * 100, 1),
                'high_issue_pct': round((neg / tot) * 100, 1),
                'buy_recommendation': 'BUY' if pos > neg and avg_rating >= 3.8 else ('DON\'T BUY' if neg > pos or avg_rating <= 2.5 else 'CONSIDER'),
                'issues': issues_list
            })

        overall_issues = df[df['Issue_Type'] != 'No Issue']['Issue_Type'].value_counts()
        overall_top_issues = [
            {'name': str(issue), 'count': int(cnt), 'percentage': round((int(cnt) / total_reviews) * 100, 1)}
            for issue, cnt in overall_issues.head(10).items()
        ]

        summary = {
            "total_reviews": total_reviews,
            "positive_reviews": total_positive,
            "negative_reviews": total_negative,
            "neutral_reviews": total_neutral,
            "total_products": len(catalog),
            "categories": sorted(list(df['Category'].dropna().unique())),
            "top_issues": overall_top_issues
        }

        _EXCEL_CACHE = {
            "catalog": catalog,
            "reviews_by_product": reviews_by_product,
            "summary": summary
        }
        return _EXCEL_CACHE
    except Exception as e:
        print(f"Error loading dataset.xlsx: {e}")
        return {"catalog": [], "reviews_by_product": {}, "summary": {}}


@app.get("/product-catalog")
def get_product_catalog():
    excel_data = load_excel_classification_data()
    excel_catalog = list(excel_data.get("catalog", []))

    csv_products = []
    csv_path = os.path.join(os.path.dirname(__file__), "productrecomendation.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
                csv_products = list(csv.DictReader(file))
        except Exception:
            pass

    excel_names = {p["name"].lower() for p in excel_catalog}
    for item in csv_products:
        pname = item.get("product_name") or item.get("name") or ""
        if pname and pname.lower() not in excel_names:
            pos_p = float(item.get("positive_pct", 0))
            neg_p = float(item.get("negative_pct", 0))
            excel_catalog.append({
                "product_id": item.get("product_id", pname),
                "product_name": pname,
                "name": pname,
                "brand": item.get("brand", ""),
                "category": item.get("category", "Unclassified"),
                "avg_rating": float(item.get("avg_rating", 0)),
                "positive_pct": pos_p,
                "negative_pct": neg_p,
                "neutral_pct": max(0, 100 - pos_p - neg_p),
                "total_reviews": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "issues": []
            })

    return {"status": "success", "count": len(excel_catalog), "data": excel_catalog}


@app.get("/product-reviews")
def get_product_reviews(product: str):
    product_clean = (product or "").strip().lower()
    excel_data = load_excel_classification_data()
    reviews_by_product = excel_data.get("reviews_by_product", {})

    matching_reviews = reviews_by_product.get(product_clean, [])
    if not matching_reviews:
        for k, v in reviews_by_product.items():
            if product_clean in k or k in product_clean:
                matching_reviews = v
                break

    if matching_reviews:
        return {"status": "success", "count": len(matching_reviews), "data": matching_reviews}

    try:
        rows = fetch_rows("""
            SELECT review_id, reviewer_name, review_text, rating, source, review_date,
                   sentiment, issue_type, severity, problem_summary,
                   recommended_solution, img_url
            FROM gl_review_intelligence
            WHERE lower(product) = lower(%s)
            ORDER BY review_date DESC, processed_timestamp DESC
            LIMIT 50
        """, (product,))
        return {"status": "success", "count": len(rows), "data": rows}
    except Exception:
        return {"status": "success", "count": 0, "data": []}


@app.get("/classification-summary")
def get_classification_summary():
    excel_data = load_excel_classification_data()
    summary = excel_data.get("summary", {})
    return {"status": "success", "data": summary}


def fetch_rows(query: str, parameters: tuple = ()) -> list[dict[str, Any]]:
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(query, parameters)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.get("/alert-count")
def get_alert_count():
    try:
        rows = fetch_rows("""
            SELECT COUNT(*) AS alert_count
            FROM gl_review_intelligence
            WHERE escalation = 'Y' AND ins_status = 'Y'
        """)
        return {"status": "success", "count": int(rows[0]["alert_count"] or 0)}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Alert count error: {error}")


@app.get("/crm-summary")
def get_crm_summary():
    try:
        rows = fetch_rows("""
            SELECT
                COUNT(CASE WHEN email_status = 'Y' THEN 1 END) AS total_count,
                COUNT(CASE WHEN email_status = 'Y' AND ins_status = 'I' THEN 1 END) AS investigation_count,
                COUNT(CASE WHEN escalation = 'Y' THEN 1 END) AS escalation_count,
                COUNT(CASE WHEN email_status = 'Y' AND ins_status = 'C' THEN 1 END) AS closed_count
            FROM gl_review_intelligence
        """)
        return {"status": "success", "data": rows[0] if rows else {}}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"CRM summary error: {error}")


@app.get("/crm-reviews")
def get_crm_reviews():
    try:
        rows = fetch_rows("""
            SELECT review_id, reviewer_name, review_text, product, issue_type, rating,
                   source, review_date, email_status, ins_status, escalation
            FROM gl_review_intelligence
            WHERE email_status = 'Y' AND ins_status = 'I'
            ORDER BY processed_timestamp DESC
            LIMIT 10
        """)
        return {"status": "success", "count": len(rows), "data": rows}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"CRM reviews error: {error}")


@app.put("/crm-reviews/{review_id}/close")
def close_crm_review(review_id: str):
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("""
            UPDATE gl_review_intelligence
            SET ins_status = 'C'
            WHERE review_id = %s AND email_status = 'Y' AND ins_status = 'I'
        """, (review_id,))
        return {"status": "success", "review_id": review_id, "ins_status": "C"}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Close review error: {error}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

class NewReview(BaseModel):

    dateTime: str
    productCategory: str
    productName: str
    rating: int
    review: str
    source: Optional[str] = None
    customSource: Optional[str] = None
    url: Optional[str] = None
    priority: Optional[str] = None
    escalation: Optional[Any] = False


def normalize_escalation(value: Any) -> str:
    return "Y" if value is True or str(value).strip().upper() in {"Y", "YES", "TRUE", "1"} else "N"


def normalize_confidence(value: Any, default: float = 0.6) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))

    confidence_by_label = {"low": 0.4, "medium": 0.7, "high": 0.9, "critical": 0.95}
    normalized = str(value).strip().lower()
    if normalized in confidence_by_label:
        return confidence_by_label[normalized]

    try:
        numeric = float(normalized)
        return max(0.0, min(1.0, numeric))
    except (TypeError, ValueError):
        return default


def classify_sentiment(review_text: str, rating: int) -> tuple[str, float]:
    training_text = [
        "excellent great useful happy satisfied love recommend", "good quality works well positive",
        "average okay acceptable nothing special", "not bad could be better neutral",
        "broken damaged terrible unusable poor worst disappointed refund", "bad fails stopped working issue"
    ]
    training_labels = ["Positive", "Positive", "Neutral", "Neutral", "Negative", "Negative"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    features = vectorizer.fit_transform(training_text)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(features, training_labels)
    prediction_features = vectorizer.transform([f"{review_text} rating {rating}"])
    probabilities = model.predict_proba(prediction_features)[0]
    prediction = str(model.classes_[probabilities.argmax()])
    return prediction, float(probabilities.max())


def fallback_issue_analysis(review_text: str, rating: int, sentiment: str) -> dict[str, Any]:
    lowered = review_text.lower()
    issue_type = "Product Defect" if any(word in lowered for word in ("broken", "damaged", "not working", "fails")) else "Product Quality"
    severity = "High" if rating <= 2 or sentiment == "Negative" else "Low"
    return {
        "issue_type": issue_type,
        "issue_confidence": 0.6,
        "severity": severity,
        "problem_summary": review_text,
        "recommended_solution": "Review the reported feedback and contact the customer with a suitable resolution."
    }


def analyze_with_gemini(review_text: str, product: str, rating: int, sentiment: str) -> dict[str, Any]:
    fallback = fallback_issue_analysis(review_text, rating, sentiment)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "Analyze this customer review and return JSON only with exactly these keys: "
                "issue_type, issue_confidence, severity, problem_summary, recommended_solution. "
                f"Product: {product}\nRating: {rating}/5\nSentiment: {sentiment}\nReview: {review_text}"
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction="You classify customer feedback and recommend a practical resolution."
            )
        )
        parsed = json.loads(response.text or "{}")
        analysis = {**fallback, **{key: parsed[key] for key in fallback if key in parsed}}
        analysis["issue_confidence"] = normalize_confidence(analysis.get("issue_confidence"))
        return analysis
    except Exception as ai_error:
        print(f"Gemini review analysis note: {ai_error}")
        return fallback


def send_escalation_email(recipient: str, product: str, review: str, problem: str, solution: str) -> None:
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")
    if not sender_email or not sender_password:
        raise RuntimeError("EMAIL_ADDRESS and EMAIL_PASSWORD are required for escalation email")

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient
    message["Subject"] = f"Escalated review: {product}"
    message.attach(MIMEText(
        f"An escalated review was submitted for {product}.\n\n"
        f"Review:\n{review}\n\nProblem identified:\n{problem}\n\n"
        f"Recommended solution:\n{solution}\n",
        "plain"
    ))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)


# =========================================================
# INSERT NEW REVIEW INTO MYSQL
# =========================================================
@app.post("/newreview")
def create_new_review(request: NewReview):
    review_timestamp = datetime.now(timezone.utc)
    review_id = review_timestamp.strftime("%y%m%d%H%M")
    review_date = date.fromisoformat(request.dateTime[:10])
    processed_timestamp = review_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    source = request.customSource if request.source == "Custom" and request.customSource else request.source
    escalation = normalize_escalation(request.escalation)
    sentiment, sentiment_confidence = classify_sentiment(request.review, request.rating)
    analysis = analyze_with_gemini(request.review, request.productName, request.rating, sentiment)
    priority = "High" if escalation == "Y" or analysis["severity"] in {"High", "Critical"} else analysis["severity"]
    image_url = request.url or None
    connection = None
    cursor = None
    saved_mysql = False

    try:
        connection = get_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO gl_review_intelligence
            (
                review_id, reviewer_name, review_text, cleaned_review, rating, source, product,
                review_date, sentiment, sentiment_confidence, issue_type, issue_confidence,
                severity, problem_summary, recommended_solution, priority, best_model_prediction,
                model_used, processed_timestamp, email_status, ins_status, img_url, state, escalation
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(
            query,
            (
                review_id,
                "Urekha",
                request.review,
                request.review.strip(),
                request.rating,
                source,
                request.productName,
                review_date,
                sentiment,
                sentiment_confidence,
                analysis["issue_type"],
                normalize_confidence(analysis["issue_confidence"]),
                analysis["severity"],
                analysis["problem_summary"],
                analysis["recommended_solution"],
                priority,
                sentiment,
                "Logistic Regression + Gemini",
                review_timestamp.replace(tzinfo=None),
                "N",
                "I",
                image_url,
                "A",
                escalation
            )
        )
        saved_mysql = True
    except Exception as db_err:
        print(f"MySQL connection/insert note: {db_err}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

    if not saved_mysql:
        raise HTTPException(
            status_code=500,
            detail="Review analysis completed, but the MySQL insert failed. Check backend logs for the database error."
        )

    email_status = "N"
    new_entry = {
        "review_id": review_id,
        "reviewer_name": "Urekha",
        "review_text": request.review,
        "cleaned_review": request.review.strip(),
        "product": request.productName,
        "rating": request.rating,
        "priority": priority,
        "escalation": escalation,
        "source": source,
        "img_url": image_url,
        "review_date": review_date.isoformat(),
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        "issue_type": analysis["issue_type"],
        "issue_confidence": analysis["issue_confidence"],
        "severity": analysis["severity"],
        "problem_summary": analysis["problem_summary"],
        "recommended_solution": analysis["recommended_solution"],
        "best_model_prediction": sentiment,
        "model_used": "Logistic Regression + Gemini",
        "processed_timestamp": processed_timestamp,
        "email_status": email_status,
        "ins_status": "I",
        "state": "A"
    }
    if escalation == "Y" and saved_mysql:
        recipient = os.getenv("ESCALATION_EMAIL") or os.getenv("EMAIL_ADDRESS")
        if recipient:
            try:
                send_escalation_email(
                    recipient,
                    request.productName,
                    request.review,
                    analysis["problem_summary"],
                    analysis["recommended_solution"]
                )
                email_status = "Y"
                update_connection = None
                update_cursor = None
                try:
                    update_connection = get_connection()
                    update_cursor = update_connection.cursor()
                    update_cursor.execute(
                        "UPDATE gl_review_intelligence SET email_status = %s WHERE review_id = %s",
                        (email_status, review_id)
                    )
                finally:
                    if update_cursor:
                        update_cursor.close()
                    if update_connection:
                        update_connection.close()
            except Exception as email_error:
                print(f"Escalation email note: {email_error}")

    new_entry["email_status"] = email_status

    # Save the same enriched record locally for the existing chatbot and local fallback flows.
    try:
        json_path = os.path.join(os.path.dirname(__file__), "inserted_reviews.json")
        reviews_data = []
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    reviews_data = json.load(f)
                except Exception:
                    reviews_data = []
        reviews_data.append(new_entry)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, indent=2)
    except Exception as json_err:
        print(f"Local storage write note: {json_err}")

    return {
        "status": "success",
        "message": "New review submitted successfully",
        "data": {
            **new_entry,
            "mysqlSaved": saved_mysql,
            "emailTriggered": email_status == "Y"
        }
    }


# =========================================================
# SEND EMAIL REQUEST MODEL
# =========================================================

class EmailRequest(BaseModel):

    email: EmailStr
    payload: str
    solution: str
    review_id: str


# =========================================================
# SEND EMAIL API
# =========================================================

@app.post("/send-email")
def send_email(request: EmailRequest):

    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")

    if not sender_email or not sender_password:

        raise HTTPException(
            status_code=500,
            detail="Email credentials are missing in .env"
        )

    try:

        message = MIMEMultipart()

        message["From"] = sender_email
        message["To"] = request.email
        message["Cc"] = "urekhanuthalapati@gmail.com"
        message["Subject"] = "Regarding Your Feedback"

        body = f"""
Hello,

Thank you for sharing your feedback.

Problem identified:
{request.payload}

Our suggested solution:
{request.solution}

We appreciate your feedback and will work towards improving your experience.

Regards,
Customer Support Team
"""

        message.attach(
            MIMEText(body, "plain")
        )

        with smtplib.SMTP(
            "smtp.gmail.com",
            587
        ) as server:

            server.starttls()

            server.login(
                sender_email,
                sender_password
            )

            server.send_message(message)

        sent_at = datetime.now(timezone.utc).isoformat()
        connection = None
        cursor = None
        try:
            connection = get_connection()
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE gl_review_intelligence SET email_status = %s WHERE review_id = %s",
                ("Y", request.review_id)
            )
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

        try:
            json_path = os.path.join(os.path.dirname(__file__), "inserted_reviews.json")
            with open(json_path, "r", encoding="utf-8") as file:
                local_reviews = json.load(file)
            for review in local_reviews:
                if str(review.get("review_id")) == request.review_id:
                    review["email_status"] = "Y"
                    review["email_sent_at"] = sent_at
                    break
            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(local_reviews, file, indent=2)
        except (OSError, json.JSONDecodeError) as local_error:
            print(f"Email timestamp local storage note: {local_error}")

        return {
            "success": True,
            "message": "Email sent successfully",
            "review_id": request.review_id,
            "email_status": "Y",
            "email_sent_at": sent_at
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email: {str(e)}"
        )
# =========================================================
# GET Total COunt 
# =========================================================
@app.get("/kpicards")
def get_sentiment_summary():

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        query = """
            SELECT *
            FROM gl_sentiment_summary
        """
        cursor.execute(query)
        columns = [
            desc[0]
            for desc in cursor.description
        ]
        rows = cursor.fetchall()
        result = [
            dict(zip(columns, row))
            for row in rows
        ]
        return {
            "status": "success",
            "count": len(result),
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"MySQL sentiment summary error: {str(e)}"
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()

#gl_issue_summary 
@app.get("/gl_issue_summary")
def gl_issue_summary():

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        query = """
            SELECT *
            FROM gl_issue_summary
        """
        cursor.execute(query)
        columns = [
            desc[0]
            for desc in cursor.description
        ]
        rows = cursor.fetchall()
        result = [
            dict(zip(columns, row))
            for row in rows
        ]
        return {
            "status": "success",
            "count": len(result),
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"MySQL issue summary error: {str(e)}"
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()
            
#gl_issue_summary 
@app.get("/gl_model_metrics")
def gl_model_metrics():

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        query = """
            SELECT *
            FROM gl_model_metrics
        """
        cursor.execute(query)
        columns = [
            desc[0]
            for desc in cursor.description
        ]
        rows = cursor.fetchall()
        result = [
            dict(zip(columns, row))
            for row in rows
        ]
        return {
            "status": "success",
            "count": len(result),
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"MySQL model metrics error: {str(e)}"
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()
           
         
#gl_issue_summary 
@app.get("/gl_product_insights")
def gl_product_insights():

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        query = """
            SELECT *
            FROM gl_product_insights
        """
        cursor.execute(query)
        columns = [
            desc[0]
            for desc in cursor.description
        ]
        rows = cursor.fetchall()
        result = [
            dict(zip(columns, row))
            for row in rows
        ]
        return {
            "status": "success",
            "count": len(result),
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"MySQL product insights error: {str(e)}"
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()
def fetch_table(cursor, table_name):

    cursor.execute(f"""
        SELECT *
        FROM `{table_name}`
    """)

    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    return [
        dict(zip(columns, row))
        for row in rows
    ]


@app.get("/dashboard-summary")
def get_dashboard_summary():

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        return {
            "status": "success",
            "data": {
                "sentiment_summary": fetch_table(
                    cursor,
                    "gl_sentiment_summary"
                ),

                "issue_summary": fetch_table(
                    cursor,
                    "gl_issue_summary"
                ),

                "model_metrics": fetch_table(
                    cursor,
                    "gl_model_metrics"
                ),
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Dashboard summary error: {str(e)}"
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()

# =========================================================
# CHATBOT API ENDPOINT (GOOGLE GEMINI AI INTEGRATION)
# =========================================================

class ChatbotRequest(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None


def load_product_catalog():
    """Load product names, purchase decisions, links, and images from the catalog."""
    csv_path = os.path.join(os.path.dirname(__file__), "productrecomendation.csv")
    if not os.path.exists(csv_path):
        return []

    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error):
        return []


def load_review_dataset():
    """Load locally stored reviews used as evidence for purchase advice."""
    json_path = os.path.join(os.path.dirname(__file__), "inserted_reviews.json")
    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def get_purchase_recommendation(product_name: str, reviews: list):
    """Return a deterministic recommendation from matching review evidence."""
    product_name = (product_name or "").strip().lower()
    if not product_name:
        return None

    matching_reviews = []
    for review in reviews:
        product_value = review.get("product") or review.get("productName") or ""
        product_value = str(product_value).lower()
        if product_name in product_value or product_value in product_name:
            matching_reviews.append(review)
    if not matching_reviews:
        return None

    ratings = [review.get("rating") for review in matching_reviews if isinstance(review.get("rating"), (int, float))]
    negative_count = sum(
        1 for review in matching_reviews
        if str(review.get("sentiment", "")).lower() == "negative"
        or str(review.get("severity", "")).lower() in {"high", "critical"}
    )
    average_rating = sum(ratings) / len(ratings) if ratings else 0

    if average_rating <= 2 or negative_count / len(matching_reviews) >= 0.5:
        recommendation = "DON'T BUY"
    elif average_rating >= 4 and negative_count == 0:
        recommendation = "BUY"
    else:
        recommendation = "REVIEW CAREFULLY"

    issues = [
        str(review.get("problem_summary") or review.get("review_text") or "Reported issue")
        for review in matching_reviews
    ]
    return {
        "product": matching_reviews[0].get("product", product_name),
        "recommendation": recommendation,
        "review_count": len(matching_reviews),
        "average_rating": round(average_rating, 2),
        "negative_count": negative_count,
        "issues": issues[:3]
    }


def get_catalog_product(product_name: str, catalog: list):
    query = (product_name or "").strip().lower()
    if not query:
        return None

    match = next(
        (item for item in catalog
         if query in str(item.get("product_name", "")).lower()
         or str(item.get("product_name", "")).lower() in query),
        None
    )
    if not match:
        return None

    image_urls = [match.get(f"image_url_{index}") for index in range(1, 6)]
    return {
        "product": match.get("product_name"),
        "brand": match.get("brand"),
        "category": match.get("category"),
        "price_inr": match.get("price_inr"),
        "recommendation": match.get("buy_recommendation", "CONSIDER"),
        "average_rating": match.get("avg_rating"),
        "negative_percentage": match.get("negative_pct"),
        "high_issue_percentage": match.get("high_issue_pct"),
        "best_for": match.get("best_for"),
        "buy_link": match.get("product_url") or f"https://www.google.com/search?q={quote_plus(match.get('product_name', ''))}",
        "images": [url for url in image_urls if url]
    }


def format_recommendation_reason(product: dict) -> str:
    """Explain a catalog DON'T BUY decision without exposing internal counts."""
    reasons = []
    negative_percentage = product.get("negative_percentage")
    high_issue_percentage = product.get("high_issue_percentage")
    average_rating = product.get("average_rating")

    if negative_percentage not in (None, ""):
        reasons.append(f"Negative feedback is {negative_percentage}%.")
    if high_issue_percentage not in (None, ""):
        reasons.append(f"High-severity issue signals are {high_issue_percentage}%.")
    if average_rating not in (None, ""):
        reasons.append(f"The average rating is {average_rating}/5.")

    if not reasons:
        reasons.append("The catalog has insufficient positive evidence for a purchase recommendation.")

    return "Why I do not recommend it:\n- " + "\n- ".join(reasons)


def find_catalog_product(text: str, catalog: list):
    """Match a catalog product from a user query without requiring Gemini."""
    query = (text or "").strip().lower()
    if not query:
        return None

    for item in catalog:
        product_name = str(item.get("product_name", "")).strip()
        product_query = product_name.lower()
        if product_query and product_query in query:
            return get_catalog_product(product_name, catalog)

    query_words = set(query.replace("?", " ").replace(",", " ").split())
    for item in catalog:
        product_name = str(item.get("product_name", "")).strip()
        product_words = [word for word in product_name.lower().split() if len(word) > 2]
        matching_words = sum(word in query_words for word in product_words)
        if len(product_words) > 1 and matching_words >= min(2, len(product_words)):
            return get_catalog_product(product_name, catalog)

    return None


def get_catalog_recommendations(text: str, catalog: list, limit: int = 3):
    """Return the best matching catalog products for broad recommendation questions."""
    query = (text or "").strip().lower()
    category_terms = {
        "phone": ("phone", "mobile", "smartphone", "iphone", "android"),
        "laptop": ("laptop", "notebook", "computer", "macbook"),
        "tablet": ("tablet", "ipad"),
        "headphone": ("headphone", "earbuds", "earphone", " headset"),
        "watch": ("watch", "smartwatch"),
    }
    requested_category = next(
        (category for category, terms in category_terms.items() if any(term in query for term in terms)),
        None,
    )
    avoid_only = any(phrase in query for phrase in ("avoid", "don't buy", "do not buy", "not worth", "worst"))
    use_case_terms = set(query.replace("?", " ").replace(",", " ").split())

    candidates = []
    for item in catalog:
        category = str(item.get("category", "")).lower()
        product_text = " ".join(str(item.get(key, "")) for key in ("product_name", "brand", "category", "best_for")).lower()
        if requested_category and requested_category not in category and requested_category not in product_text:
            continue
        if not avoid_only and str(item.get("buy_recommendation", "")).upper() != "BUY":
            continue

        score = float(item.get("recommendation_score") or 0)
        score += sum(8 for term in use_case_terms if len(term) > 2 and term in product_text)
        if str(item.get("buy_recommendation", "")).upper() == "BUY":
            score += 5
        candidates.append((score, item))

    if not candidates and requested_category and not avoid_only:
        candidates = [
            (float(item.get("recommendation_score") or 0), item)
            for item in catalog
            if str(item.get("buy_recommendation", "")).upper() == "BUY"
        ]

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    recommendations = []
    for _, item in candidates:
        product = get_catalog_product(str(item.get("product_name", "")), catalog)
        if product and product["product"] not in {entry["product"] for entry in recommendations}:
            recommendations.append(product)
        if len(recommendations) == limit:
            break
    return recommendations

@app.post("/chatbot")
def chatbot_endpoint(request: ChatbotRequest):
    user_text = (request.text or "").strip()
    has_image = bool(request.image)

    if not user_text and not has_image:
        raise HTTPException(
            status_code=400,
            detail="Either text or image payload must be provided"
        )

    try:
        reviews = load_review_dataset()
        catalog = load_product_catalog()
        query_lower = user_text.lower()
        product_request = any(
            phrase in query_lower
            for phrase in (
                "buy", "purchase", "worth it", "recommend", "suggest", "best", "which phone",
                "which mobile", "which laptop", "product", "phone", "mobile", "smartphone", "laptop",
                "tablet", "headphone", "earbuds", "watch"
            )
        ) or has_image
        purchase_question = product_request
        catalog_recommendations = get_catalog_recommendations(user_text, catalog) if product_request else []

        # Catalog recommendations are deterministic and do not depend on Gemini.
        # This keeps buy/don't-buy queries useful even when the AI key is unavailable.
        product_evidence = find_catalog_product(user_text, catalog) if user_text else None
        if product_evidence:
            recommendation = product_evidence["recommendation"]
            reply_text = (
                f"Recommendation: {recommendation}\n"
                f"Product matched: {product_evidence['product']}\n"
                f"Best for: {product_evidence['best_for']}\n"
                + (format_recommendation_reason(product_evidence)
                   if recommendation == "DON'T BUY" else
                   "This product is worth considering.")
            )
            return {
                "status": "success",
                "reply": reply_text,
                "response": reply_text,
                "recommendation": product_evidence,
                "recommendations": [product_evidence]
            }

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise Exception("GEMINI_API_KEY not found in .env")

        client = genai.Client(api_key=api_key)

        prompt_text = user_text if user_text else "Analyze this image payload and provide review intelligence insights."
        if purchase_question:
            prompt_text = (
                f"{prompt_text}\n\n"
                "Identify the product in the image. Return the identified product name first, "
                "then explain whether it is a good purchase. Use only the supplied review evidence."
            )
        contents_payload = []

        if has_image:
            img_str = request.image
            if "," in img_str:
                header, base64_data = img_str.split(",", 1)
                mime_type = header.split(";")[0].replace("data:", "") or "image/png"
            else:
                base64_data = img_str
                mime_type = "image/png"

            image_bytes = base64.b64decode(base64_data)
            contents_payload.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        contents_payload.append(prompt_text)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents_payload,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an expert AI Customer Review Analyst and Product Feedback Intelligence Assistant. "
                    "Provide helpful, concise, and accurate responses for customer review analysis, product defects, "
                    "sentiment classification, or image inspections."
                )
            )
        )

        reply_text = response.text.strip() if response.text else "AI response generated."

        if purchase_question:
            identified_text = response.text.strip() if response.text else ""
            product_evidence = None
            lookup_text = f"{user_text} {identified_text}".lower()
            for item in catalog:
                dataset_product = str(item.get("product_name", "")).strip()
                product_words = dataset_product.lower().split()
                if dataset_product and (
                    dataset_product.lower() in lookup_text
                    or (len(product_words) > 1 and product_words[0] in lookup_text and product_words[1] in lookup_text)
                ):
                    product_evidence = get_catalog_product(dataset_product, catalog)
                    break

            if product_evidence:
                catalog_recommendations = [product_evidence]
                recommendation = product_evidence["recommendation"]
                reply_text = (
                    f"Recommendation: {recommendation}\n"
                    f"Product matched: {product_evidence['product']}\n"
                    f"Best for: {product_evidence['best_for']}\n"
                    + (format_recommendation_reason(product_evidence)
                       if recommendation == "DON'T BUY" else
                       "This product is worth considering.")
                )
            else:
                reply_text = (
                    f"Here are three catalog recommendations based on your query.\nGemini analysis: {reply_text}"
                )

        return {
            "status": "success",
            "reply": reply_text,
            "response": reply_text,
            "recommendation": product_evidence if purchase_question and 'product_evidence' in locals() else (catalog_recommendations[0] if catalog_recommendations else None),
            "recommendations": catalog_recommendations
        }

    except Exception as e:
        reply_text = f"AI Assistant: Processed query '{user_text}'. (Gemini status: {str(e)})"
        fallback_recommendations = []
        if 'catalog' in locals() and product_request:
            fallback_recommendations = get_catalog_recommendations(user_text, catalog)
        return {
            "status": "success",
            "reply": (
                f"Here are three catalog recommendations based on your query.\n{reply_text}"
                if fallback_recommendations else reply_text
            ),
            "response": reply_text,
            "recommendation": fallback_recommendations[0] if fallback_recommendations else None,
            "recommendations": fallback_recommendations
        }
