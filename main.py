from flask import Flask, request, jsonify
from ibm_watsonx_ai.foundation_models.utils import Toolkit
from ibm_watsonx_ai import APIClient, Credentials
import os

# Watsonx config
project_id = os.environ.get("WATSONX_PROJECT_ID")
api_key = os.environ.get("WATSONX_API_KEY")
url = "https://us-south.ml.cloud.ibm.com"
vector_index_id = os.environ.get("WATSONX_VECTOR_INDEX_ID")

# Init Watsonx client
credentials = Credentials(api_key=api_key, url=url)
api_client = APIClient(project_id=project_id, credentials=credentials)
toolkit = Toolkit(api_client=api_client)
rag_tool = toolkit.get_tool("RAGQuery")

# Flask app
app = Flask(__name__)

@app.route('/')
def hello():
    return "Watsonx RAG API is running!"

@app.route('/rag', methods=['POST'])
def run_rag():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    config = {
        "vectorIndexId": vector_index_id,
        "projectId": project_id
    }

    try:
        result = rag_tool.run(input=query, config=config)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
