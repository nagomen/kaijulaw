from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# 環境変数を .env ファイルからロード
load_dotenv()

# 必要な環境変数を取得
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 環境変数が設定されているか確認
if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY]):
    raise ValueError("環境変数が正しく設定されていません。")

# Flask アプリケーションの初期化
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# LangChain LLMの設定
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# プロンプトテンプレート
template = """あなたは優秀な法律家AIです。下記のコンテキストを利用してユーザーの質問に丁寧に答えてください。必ず文脈からわかる情報のみを使用して回答を生成してください。また回答の際にはどの法令や判例を根拠にしたか示しなさい。また回答を行う際は断言は避け、「～できるかもしれません。」のような口調を使ってください。また質問者に寄り添うような丁寧な回答をしてください。また以下の点に注意してください➔１．取消しと無効は全く別の概念。２．内縁と事実婚は同じ
{context}

ユーザーの質問: {question}

回答：
"""
prompt = ChatPromptTemplate.from_template(template)

# グラフ検索用のNeo4jGraphオブジェクト
graph = Neo4jGraph()

# ベクトル検索用のNeo4jVectorオブジェクト
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(model="text-embedding-3-large"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# LangChainの処理チェーン
chain = (
    RunnableParallel(
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 質問リフレーズ用関数
def rephrase_question(question: str) -> str:
    """
    質問を検索に最適化された形式にリフレーズする。
    """
    prompt = ChatPromptTemplate.from_template(
        """
        あなたは法律家AIです。以下の質問を検索に最適化し、法律的な言い回しでリフレーズしてください。
        また以下の点に注意してください➔１．取消しと無効は全く別の概念。２．内縁と事実婚は同じ
        リフレーズされた質問は、「質問:」で始めてください。
        質問: {question}
        """
    )
    try:
        rephrased = llm(prompt.format_prompt(question=question).to_messages())
        return rephrased.content
    except Exception as e:
        print(f"質問リフレーズ中のエラー: {e}")
        return question

# グラフ検索関数
def fetch_graph_data(question: str) -> str:
    """
    グラフ検索で関連データを取得。
    """
    query = """
    MATCH (n)-[r]->(m)
    WHERE n.text CONTAINS $question OR m.text CONTAINS $question
    RETURN n.text AS source, type(r) AS relationship, m.text AS target
    LIMIT 10
    """
    try:
        results = graph.query(query, {"question": question})
        context = "\n".join([f"{record['source']} -[{record['relationship']}]-> {record['target']}" for record in results])
        return context
    except Exception as e:
        return f"グラフ検索エラー: {str(e)}"

# ベクトル検索関数
def fetch_vector_data(question: str) -> str:
    """
    ベクトル検索で関連文書を取得。
    """
    try:
        results = vector_index.similarity_search(question)
        context = "\n".join([result.page_content for result in results])
        return context
    except Exception as e:
        return f"ベクトル検索エラー: {str(e)}"

# Flask APIエンドポイント
@app.route('/')
def home():
    # templates/index.html をレンダリング
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    try:
        # リクエストヘッダーをログ出力
        print("Headers:", request.headers)
        print("Content-Type:", request.content_type)

        # Content-Type チェック
        if request.content_type != 'application/json':
            return jsonify({"error": "Unsupported Media Type. Please use 'application/json'."}), 415

        # JSON データの取得
        data = request.get_json()
        print("Request Body:", data)

        question = data.get("question", "")
        if not question:
            return jsonify({"error": "質問が空です。"}), 400

        # 処理（例: 質問のリフレーズなど）
        rephrased_question = rephrase_question(question)
        graph_context = fetch_graph_data(rephrased_question)
        vector_context = fetch_vector_data(rephrased_question)
        context = f"グラフ検索結果:\n{graph_context}\n\nベクトル検索結果:\n{vector_context}"
        response = chain.invoke({"context": context, "question": rephrased_question})

        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
