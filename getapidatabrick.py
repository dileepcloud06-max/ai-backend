from fastapi import FastAPI, HTTPException
from databricks import sql
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Databricks Data API")


def get_connection():
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )


@app.get("/data")
def get_data():

    try:
        connection = get_connection()
        cursor = connection.cursor()

        query = """
            SELECT *
            FROM backspace_databricks.bronze.bl_otto
            LIMIT 0,10
        """

        cursor.execute(query)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        result = [
            dict(zip(columns, row))
            for row in rows
        ]

        cursor.close()
        connection.close()

        return {
            "status": "success",
            "count": len(result),
            "data": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )