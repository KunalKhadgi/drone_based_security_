import os
import sqlite3
from dotenv import load_dotenv
import pandas as pd
from google import genai

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
api_key = os.getenv("API_KEY")

# -----------------------------
# Configure Gemini API
# -----------------------------
client = genai.Client(api_key=api_key)  

# -----------------------------
# Function to fetch table schema
# -----------------------------
def get_table_schema(db_file, table_name="detections"):
    """Fetches the schema of a given table to help generate SQL queries."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    conn.close()
    
    schema = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
    return f"The table '{table_name}' has columns: {schema}"

# -----------------------------
# Function to generate SQL query using Gemini
# -----------------------------

def get_sql_query(user_query, schema):
    """Uses Google Gemini AI to generate an SQL query based on user input."""
    prompt = f"""
    You are an expert SQL assistant. Convert the following user request into an SQL query.

    User Request: "{user_query}"
    Table Schema: {schema}

    Respond with ONLY the SQL query, without any explanation, extra formatting, or markdown.
    """
    
    # Using the genai module directly to generate content
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # You can replace with the appropriate model
        contents=prompt
    )
    
    sql_query = response.text.strip()

    # Clean up formatting
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    return sql_query

# -----------------------------
# Function to execute SQL query
# -----------------------------
def execute_query(db_file, sql_query):
    """Executes a given SQL query and returns the result."""
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df

# -----------------------------
# Main query function
# -----------------------------
def run_query():
    """Handles user queries and executes SQL queries using Gemini AI."""
    db_file = "detection_log.db"

    while True:
        user_query = input("Ask a query (type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # Fetch table schema
        schema = get_table_schema(db_file)

        # Generate SQL query using Gemini AI
        sql_query = get_sql_query(user_query, schema)
        
        print(f"🔍 Generated SQL Query:\n{sql_query}")

        try:
            # Execute the generated SQL query
            result = execute_query(db_file, sql_query)
            if result.empty:
                print("No results found.")
            else:
                print("📊 Query Results:")
                print(result)
        except Exception as e:
            print(f"⚠️ SQL Execution Error: {e}")

# -----------------------------
# Run the query loop
# -----------------------------
if __name__ == "__main__":
    print("🔍 Welcome to AI Log Analyzer using Google Gemini AI!")
    run_query()
