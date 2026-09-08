import pandas as pd
import json
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

INPUT_FILE = "negative_reviews.csv"
OUTPUT_FILE = "ai_negative_reviews.csv"

BATCH_SIZE = 50

# --------------------------------------------------
# READ DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Total negative reviews:", len(df))


# --------------------------------------------------
# CHECKPOINT / RESUME
# --------------------------------------------------

if os.path.exists(OUTPUT_FILE):

    existing_df = pd.read_csv(OUTPUT_FILE)

    results = existing_df.to_dict("records")

    processed_count = len(existing_df)

    print("Already processed:", processed_count)

else:

    results = []

    processed_count = 0


# --------------------------------------------------
# PROCESS REVIEWS
# --------------------------------------------------

for index in range(processed_count, len(df)):

    row = df.iloc[index]

    review = str(row.get("review", ""))
    reviewer_name = str(row.get("reviewerName", ""))
    summary = str(row.get("summary", ""))
    rating = row.get("rating", "")

    prompt = f"""
You are an expert customer review analyst.

Analyze the following customer review.

Customer Review:
{review}

Rating:
{rating}

Summary:
{summary}

Identify the following:

1. Main customer problem
2. Complaint category
3. Possible reason for the complaint
4. Practical solution that the company can implement

Return ONLY valid JSON.
"""

    try:

        response = client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt,

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                response_schema={
                    "type": "OBJECT",

                    "properties": {

                        "problem": {
                            "type": "STRING"
                        },

                        "category": {
                            "type": "STRING"
                        },

                        "reason": {
                            "type": "STRING"
                        },

                        "solution": {
                            "type": "STRING"
                        }
                    },

                    "required": [
                        "problem",
                        "category",
                        "reason",
                        "solution"
                    ]
                }
            )
        )

        result = json.loads(response.text)

        results.append({

            "S.No": index + 1,

            "reviewerName": reviewer_name,

            "review": review,

            "summary": summary,

            "rating": rating,

            "problem": result["problem"],

            "category": result["category"],

            "reason": result["reason"],

            "solution": result["solution"]
        })


        print(
            f"Processed {index + 1}/{len(df)}"
        )


        # --------------------------------------------------
        # SAVE EVERY BATCH
        # --------------------------------------------------

        if (index + 1) % BATCH_SIZE == 0:

            result_df = pd.DataFrame(results)

            result_df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8"
            )

            print(
                f"Saved checkpoint: {index + 1} records"
            )


        # Small delay

        time.sleep(0.5)


    except Exception as e:

        print(
            f"Error processing record {index + 1}: {e}"
        )

        # Save whatever is already processed

        result_df = pd.DataFrame(results)

        result_df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8"
        )

        print("Progress saved.")

        time.sleep(5)

        continue


# --------------------------------------------------
# FINAL SAVE
# --------------------------------------------------

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\n================================")
print("PROCESSING COMPLETED")
print("================================")

print("Total records:", len(df))
print("Processed:", len(results))
print("Output file:", OUTPUT_FILE)