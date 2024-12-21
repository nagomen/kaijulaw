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
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from dotenv import load_dotenv
import os



# 環境変数が設定されているか確認# 環境変数を .env ファイルからロード
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

#komikomi

# ベクトル検索用のNeo4jVectorオブジェクト
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(model="text-embedding-3-large"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# グラフ検索用のNeo4jGraphオブジェクト
graph = Neo4jGraph()

graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")



# 質問リフレーズ用関数
def rephrase_question(question: str) -> str:
    """
    質問を検索に最適化された形式にリフレーズする。
    """
    prompt0 = ChatPromptTemplate.from_template(
        """
        あなたは法律家AIです。以下の質問を検索に最適化し、法律的な言い回しでリフレーズしてください。
        また以下の点に注意してください➔１．取消しと無効は全く別の概念。２．内縁と事実婚は同じ
        リフレーズされた質問は、「質問:」で始めてください。
        質問: {question}
        """
    )
    try:
        rephrased = llm(prompt0.format_prompt(question=question).to_messages())
        return rephrased.content
    except Exception as e:
        print(f"質問リフレーズ中のエラー: {e}")
        return question

# エンティティの抽出
class Entities(BaseModel):
    """エンティティに関する情報の識別"""

    names: List[str] = Field(
        ...,
        description="文章の中に登場する、すべての法的エンティティ",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "以下の質問を検索に最適化し、法律的な観点で明確にリフレーズして、その後、特定の法的エンティティを抽出してください。例えば、人、関係、義務、契約などをエンティティとして識別します。",
        ),
        (
            "human",
            "以下の質問を検索に最適化し、法律的な観点で明確にリフレーズして、リフレーズされた質問から情報を抽出します。"
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

# 検索クエリ作成関数
def generate_full_text_query(input: str) -> str:
    """
    指定された入力文字列に対する全文検索クエリを生成します。

    この関数は、全文検索に適したクエリ文字列を構築します。
    入力文字列を単語に分割し、
    各単語に対する類似性のしきい値 (変更された最大 2 文字) を結合します。
    AND 演算子を使用してそれらを演算します。ユーザーの質問からエンティティをマッピングするのに役立ちます
    データベースの値と一致しており、多少のスペルミスは許容されます。
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# グラフ検索関数
def fetch_graph_data(question: str) -> str:
    """
    グラフ検索で関連データを取得。
    """
    # 変数 result を初期化
    result = ""
    rephrased_question = rephrase_question(question)  # 質問をリフレーズ
    entities = entity_chain.invoke({"question": rephrased_question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit: 20})
              YIELD node, score
              CALL(node, score){
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
              }
              WITH output
              RETURN output
              LIMIT 1000
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

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

# プロンプトテンプレート
template = """あなたは優秀な法律家AIです。下記のコンテキストを利用してユーザーの質問に丁寧に答えてください。また回答の際にはどの法令や判例を根拠にしたか示しなさい。また回答を行う際は断言は避け、「～できるかもしれません。」のような口調を使ってください。また質問者に寄り添い、質問者に有利となるような情報を意識して丁寧な回答をしてください。また以下の点に注意してください➔１．取消しと無効は全く別の概念。２．内縁と事実婚は同じ。
{context}

ユーザーの質問: {question}

回答：
"""

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

# Flask APIエンドポイント
@app.route('/')
def home():
    # templates/index.html をレンダリング
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    # 質問と回答の履歴を保存するリスト
    history = []
    # 履歴を整形してプロンプト用のコンテキストに追加
    context = ""
    if history:
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])

    # プロンプトを作成（履歴があれば含める）
    prompt = f"{context}\nQ: {question}" if context else f"Q: {question}"
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

        # 現在の質問と回答を履歴に保存
        history.append((rephrased_question, response))

        # 履歴が多くなりすぎた場合は古いものを削除（例: 最大5件に制限）
        if len(history) > 5:
            history.pop(0)

        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
