# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Customer Support AI Agent - Overview
# MAGIC %md
# MAGIC > ⚠️ **2026年7月現在の推奨経路について**
# MAGIC > 本記事は ResponsesAgent を Unity Catalog へ登録し **Model Serving** へデプロイする方式を扱います。
# MAGIC > Databricks は 2026年7月以降、新規エージェント開発には **Databricks Apps ベースの Custom Agent** を推奨しています（[公式ドキュメント](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent)）。
# MAGIC > Model Serving 方式は、従来方式の仕組みを短距離で理解する教材、既存環境、または Apps を利用できない環境向けの選択肢として理解してください。
# MAGIC > このノートブックが扱うのは、AgentOps 全体のうち主に **Tracing、パッケージング、登録、デプロイ、手動レビュー** です。評価データセット、品質ゲート、Production Monitoring は別途実装が必要です。
# MAGIC
# MAGIC # 🤖 カスタマーサポート AI エージェント on Databricks
# MAGIC
# MAGIC このノートブックでは、以下の構成でカスタマーサポートAIエージェントを構築・デプロイします。
# MAGIC
# MAGIC ## アーキテクチャ
# MAGIC
# MAGIC ```
# MAGIC ユーザー問い合わせ
# MAGIC         ↓
# MAGIC [Model Serving Endpoint]  ← REST API (OpenAI互換)
# MAGIC         ↓
# MAGIC [CustomerSupportAgent]   ← LangGraph + MLflow ResponsesAgent
# MAGIC         ↓
# MAGIC   ┌─────────────────────────────────────┐
# MAGIC   │ Tools (LLMが自動で選択・実行)       │
# MAGIC   │  - lookup_order_status: 注文状況確認│
# MAGIC   │  - search_faq: FAQ検索              │
# MAGIC   │  - create_support_ticket: チケット作成│
# MAGIC   └─────────────────────────────────────┘
# MAGIC         ↓
# MAGIC [databricks-meta-llama-3-3-70b-instruct]
# MAGIC ```
# MAGIC
# MAGIC ## ステップ
# MAGIC 1. パッケージインストール
# MAGIC 2. エージェントファイル (`agent.py`) の作成
# MAGIC 3. ローカルテスト
# MAGIC 4. MLflow / Unity Catalog へのモデル登録
# MAGIC 5. Model Serving へのデプロイ
# MAGIC 6. エンドポイントへのクエリ

# COMMAND ----------

# DBTITLE 1,Step 1: Install Packages
# Step 1: パッケージインストール (初回のみ実行)
%pip install -U \
    mlflow==3.6.0 \
    databricks-langchain==0.8.2 \
    langgraph==0.3.4 \
    langchain-core==0.3.86 \
    databricks-agents \
    pydantic==2.12.5 \
    -q
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Step 2: Configuration
# Step 2: 設定
# UCのカタログ・スキーマを必要に応じて変更してください
CATALOG = "main"
SCHEMA = "your_schema"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.customer_support_agent"
AGENT_ENDPOINT_NAME = "customer-support-agent"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

print(f"Model will be registered as: {MODEL_NAME}")
print(f"Serving endpoint name: {AGENT_ENDPOINT_NAME}")
print(f"LLM: {LLM_ENDPOINT}")

# MLflow Experiment を明示設定――Step 4 の Trace と Step 5 の Run を同じ Experiment に集約する
import mlflow
try:
    _username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except Exception:
    _username = "your-email@databricks.com"  # 自分のメールアドレスに変更してください
MLFLOW_EXPERIMENT_NAME = f"/Users/{_username}/customer-support-agent"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Step 3: Copy agent.py to /tmp
# MAGIC %%writefile /tmp/agent.py
# MAGIC # agent.py - Customer Support AI Agent
# MAGIC # MLflow ResponsesAgent + LangGraph + Databricks Foundation Model API
# MAGIC
# MAGIC """--- 以下は単体で実行可能な自己完結ファイルです ---"""
# MAGIC import hashlib
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream, to_chat_completions_input,
# MAGIC )
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from langchain_core.messages import AIMessage, BaseMessage
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.tools import tool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from typing import Annotated, Generator, Sequence, TypedDict
# MAGIC
# MAGIC LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC
# MAGIC SYSTEM_PROMPT = (
# MAGIC     "あなたは優秀なカスタマーサポートAIエージェントです。"
# MAGIC     "Eコマースショッピングサイトのカスタマーサポート担当として、お客様の問題を丁寧かつ迅速に解決します。\n\n"
# MAGIC     "利用可能なツール：\n"
# MAGIC     "- lookup_order_status: 注文番号(ORD-XXX形式)で注文状況を確認\n"
# MAGIC     "- search_faq: 返品・配送・支払い・保証などのFAQを検索\n"
# MAGIC     "- create_support_ticket: 自力で解決できない問題のチケットを作成\n\n"
# MAGIC     "ルール： 1.必ず日本語で回答  2.注文関連は lookup_order_status  "
# MAGIC     "3.一般質問は search_faq  "
# MAGIC     "4.ユーザーが明示的にチケット作成を依頼した場合は内容を確認してから create_support_ticket。それ以外は解決不能時のみ  "
# MAGIC     "5.簡潔に要約して終わる"
# MAGIC )
# MAGIC
# MAGIC ORDER_DB = {
# MAGIC     "ORD-001": {"status": "配送中",   "item": "ノートPC",           "estimated_delivery": "2026-07-20"},
# MAGIC     "ORD-002": {"status": "配送完了", "item": "ワイヤレスイヤホン", "delivered_date":      "2026-07-15"},
# MAGIC     "ORD-003": {"status": "処理中",   "item": "スマートフォン",     "estimated_delivery": "2026-07-22"},
# MAGIC     "ORD-004": {"status": "配送完了", "item": "タブレット",         "delivered_date":      "2026-07-10"},
# MAGIC }
# MAGIC
# MAGIC FAQ_DB = [
# MAGIC     {"keywords": ["返品","返金","返却"],       "question": "返品・返金ポリシー",     "answer": "商品到着から30日以内、未使用品に限り返品可。返金は5〜7営業日。"},
# MAGIC     {"keywords": ["配送","配達","届く","所要","期間"], "question": "配送期間・配送方法",   "answer": "通常2〜5営業日でお届け。追跡番号をメールでお知らせ。"},
# MAGIC     {"keywords": ["支払","決済","payment","クレジット","振込"], "question": "支払い方法",           "answer": "クレジットカード・銀行振込・コンビニ払い・PayPay・LINE Pay・Amazon Pay対応。"},
# MAGIC     {"keywords": ["保証","修理","故障","不良品","壊れ"],   "question": "保証・修理対応",     "answer": "購入から1年間のメーカー保証。期間内の不具合は無償修理・交換。"},
# MAGIC     {"keywords": ["会員","登録","アカウント","パスワード","ログイン"], "question": "会員登録・アカウント", "answer": "メールアドレスとパスワードで無料登録。ポイント制度あり。"},
# MAGIC     {"keywords": ["キャンセル","注文取り消し"],         "question": "注文キャンセル",       "answer": "発送前はマイページからキャンセル可。発送後は返品手続きが必要。"},
# MAGIC ]
# MAGIC
# MAGIC
# MAGIC @tool
# MAGIC def lookup_order_status(order_id: str) -> str:
# MAGIC     """注文IDで注文状況を確認。注文IDはORD-XXX形式。
# MAGIC     ※ モック実装。本番では「認証済み顧客ID + 注文の所有権検証」が必須。
# MAGIC     　注文IDだけで取得できる設計は IDOR 脆弱性になる。顧客IDは LLM に渡さずサーバー側で注入すること。
# MAGIC     """
# MAGIC     order = ORDER_DB.get(order_id.strip().upper())
# MAGIC     if not order:
# MAGIC         return f"注文ID '{order_id}' は見つかりませんでした。"
# MAGIC     lines = [f"注文ID: {order_id}", f"商品: {order['item']}", f"状態: {order['status']}"]
# MAGIC     if "estimated_delivery" in order: lines.append(f"配達予定日: {order['estimated_delivery']}")
# MAGIC     if "delivered_date"    in order: lines.append(f"配達完了日: {order['delivered_date']}")
# MAGIC     return "\n".join(lines)
# MAGIC
# MAGIC
# MAGIC @tool
# MAGIC def search_faq(query: str) -> str:
# MAGIC     """質問に関連するFAQを検索。"""
# MAGIC     results = [
# MAGIC         f"《{faq['question']}》\n{faq['answer']}"
# MAGIC         for faq in FAQ_DB
# MAGIC         if any(kw in query for kw in faq["keywords"])
# MAGIC     ]
# MAGIC     if not results:
# MAGIC         return "該当するFAQは見つかりませんでした。別のキーワードで再度お試しください。"
# MAGIC     return "\n\n".join(results)
# MAGIC
# MAGIC
# MAGIC @tool
# MAGIC def create_support_ticket(customer_name: str, issue_summary: str, priority: str = "medium") -> str:
# MAGIC     """解決できない問題のサポートチケットを作成。優先度はlow/medium/high/urgent。
# MAGIC     ※ モック実装。本番では確認ステップ・冪等性キーの付与を推奨。
# MAGIC     　本番の冪等性: リクエスト単位のキーをチケットAPIへ渡し、再試行による二重登録を防ぐこと。
# MAGIC     """
# MAGIC     # セキュリティ用途ではなく再現性のため: 同じ入力から常に同じモックIDを生成する
# MAGIC     hash_val = int(hashlib.md5(f"{customer_name}:{issue_summary}".encode()).hexdigest(), 16) % 90000 + 10000
# MAGIC     tid = f"TKT-{hash_val}"
# MAGIC     return (
# MAGIC         f"チケットID: {tid}\n"
# MAGIC         f"お客様名: {customer_name}\n"
# MAGIC         f"問題概要: {issue_summary}\n"
# MAGIC         f"優先度: {priority}\n"
# MAGIC         f"担当者が24時間以内にご連絡いたします。"
# MAGIC     )
# MAGIC
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC
# MAGIC
# MAGIC class CustomerSupportAgent(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         self.tools = [lookup_order_status, search_faq, create_support_ticket]
# MAGIC         self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=2000)
# MAGIC         self.llm_with_tools = self.llm.bind_tools(self.tools)
# MAGIC         self.graph = self._build_graph()  # リクエストごとに再構築せず一度だけ初期化
# MAGIC
# MAGIC     def _build_graph(self):
# MAGIC         def should_continue(state: AgentState) -> str:
# MAGIC             last = state["messages"][-1]
# MAGIC             return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"
# MAGIC
# MAGIC         def call_model(state: AgentState) -> dict:
# MAGIC             messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(state["messages"])
# MAGIC             return {"messages": [self.llm_with_tools.invoke(messages)]}
# MAGIC
# MAGIC         g = StateGraph(AgentState)
# MAGIC         g.add_node("agent", RunnableLambda(call_model))
# MAGIC         g.add_node("tools", ToolNode(self.tools))
# MAGIC         g.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
# MAGIC         g.add_edge("tools", "agent")
# MAGIC         g.set_entry_point("agent")
# MAGIC         return g.compile()
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [e.item for e in self.predict_stream(request) if e.type == "response.output_item.done"]
# MAGIC         return ResponsesAgentResponse(output=outputs)
# MAGIC
# MAGIC     def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         messages = to_chat_completions_input([m.model_dump() for m in request.input])
# MAGIC         # recursion_limit: LLMが無限にツールを呼び続けた場合の安全用フェイルセーフ
# MAGIC         for event in self.graph.stream(
# MAGIC             {"messages": messages},
# MAGIC             stream_mode=["updates"],
# MAGIC             config={"recursion_limit": 10},
# MAGIC         ):
# MAGIC             if event[0] == "updates":
# MAGIC                 for node_data in event[1].values():
# MAGIC                     if node_data.get("messages"):
# MAGIC                         yield from output_to_responses_items_stream(node_data["messages"])
# MAGIC
# MAGIC
# MAGIC # ※ 本番では PII マスキング・トレース閲覧権限・保存期間の設定が必須。
# MAGIC # 　ツール引数・戻り値には氏名・注文情報などの機密情報が含まれる。
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = CustomerSupportAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# DBTITLE 1,Step 4: Local Test
# Step 4: エージェントのローカルテスト
# デプロイ前に動作を確認する

import sys
sys.path.insert(0, "/tmp")

# agent.py から AGENT をインポート
from agent import AGENT
from mlflow.types.responses import ResponsesAgentRequest

def demo_agent(question: str):
    """動作確認用デモ関数。期待ツールや引数の自動検証はしない。
    本番テストでは expected_tool_calls / expected_facts を検証するアサーションを別途追加すること。
    """
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": question}]
    )
    response = AGENT.predict(request)
    print(f"Q: {question}")
    for item in response.output:
        if hasattr(item, "content"):
            content = item.content
            if isinstance(content, list):
                for c in content:
                    if hasattr(c, "text"):
                        print(f"A: {c.text}")
            else:
                print(f"A: {content}")
    print("-" * 60)

# ※ 顧客名は合成ID (TEST-USER-001) を使用。実名をTraceへ残さないこと (PII対策)
# テストケース 1: 注文状況確認
demo_agent("注文ORD-001の配送状況を教えてください")

# テストケース 2: FAQ検索
demo_agent("返品ポリシーを教えてください")

# テストケース 3: チケット作成
demo_agent("届いた商品が壊れていました。TEST-USER-001としてサポートチケットを作成してください")

# COMMAND ----------

# DBTITLE 1,Step 4.5: Create UC Schema if not exists
# Step 4.5: UCスキーマが存在しない場合は作成
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

w = WorkspaceClient()
try:
    w.schemas.get(f"{CATALOG}.{SCHEMA}")
    print(f"✓ Schema {CATALOG}.{SCHEMA} already exists")
except NotFound:
    w.schemas.create(name=SCHEMA, catalog_name=CATALOG)
    print(f"✓ Schema {CATALOG}.{SCHEMA} created")
except Exception as e:
    raise RuntimeError(f"スキーマ確認中にエラー: {e}") from e

# COMMAND ----------

# DBTITLE 1,Step 5: Log Model to MLflow & Register to Unity Catalog
# Step 5: MLflow にモデルを登録し Unity Catalog にレジストリ登録
import mlflow
from mlflow.models.resources import DatabricksServingEndpoint

mlflow.set_registry_uri("databricks-uc")

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)]

input_example = {
    "input": [{"role": "user", "content": "注文ORD-001の状況を教えてください"}]
}

with mlflow.start_run(run_name="customer-support-agent"):
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="/tmp/agent.py",
        resources=resources,
        pip_requirements=[
            "mlflow==3.6.0",
            "databricks-langchain==0.8.2",   # 再現性のため全主要パッケージを固定
            "langgraph==0.3.4",
            "langchain-core==0.3.86",
            "pydantic==2.12.5",
        ],
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )

print(f"✓ モデル登録完了")
print(f"  Model URI : {model_info.model_uri}")
print(f"  Registered: {MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Step 6: Deploy to Model Serving
# Step 6: Model Serving エンドポイントにデプロイ
# デプロイには数分〜15分ほどかかります

from databricks import agents

print(f"デプロイ開始: {MODEL_NAME}")
print("完了までかかります...※セルを再実行しないでください")

deploy_info = agents.deploy(
    model_name=MODEL_NAME,
    model_version=model_info.registered_model_version,
    endpoint_name=AGENT_ENDPOINT_NAME,
    tags={"environment": "development", "use_case": "customer_support"},
)
print(f"✓ デプロイ完了")
print(f"  Endpoint: {deploy_info.endpoint_name}")
print(f"  Review App: {deploy_info.review_app_url}")

# COMMAND ----------

# DBTITLE 1,Step 7: Check Endpoint Status
# Step 7: エンドポイントの状況確認
# READY になるまで待機する (数分～15分)

import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

def wait_for_endpoint(name: str, timeout_min: int = 20):
    print(f"エンドポイント '{name}' の起動を待機中...")
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        ep = w.serving_endpoints.get(name=name)
        state = ep.state.config_update.value if ep.state and ep.state.config_update else "UNKNOWN"
        ready = ep.state.ready.value if ep.state and ep.state.ready else "NOT_READY"
        print(f"  state={state}, ready={ready}")
        if ready == "READY":
            print(f"\n✓ エンドポイントが準備完了しました: {name}")
            return True
        time.sleep(30)
    raise TimeoutError(
        f"エンドポイント '{name}' が {timeout_min} 分以内に READY になりませんでした。"
        "次のセルへ進む前に状態を確認してください。"
    )

wait_for_endpoint(AGENT_ENDPOINT_NAME)  # タイムアウト時は例外で停止

# COMMAND ----------

# DBTITLE 1,Step 8: Query the Deployed Endpoint
# Step 8: デプロイされたエンドポイントにクエリ
# REST API経由 (外部アプリからも同様に呼び出し可能)

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

def chat(question: str):
    response = client.predict(
        endpoint=AGENT_ENDPOINT_NAME,
        inputs={"input": [{"role": "user", "content": question}]},
    )
    for item in response.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    print(f"Q: {question}")
                    print(f"A: {c['text']}")
                    print("-" * 60)

# 実際の問い合わせ
chat("注文ORD-003はいつ届きますか？")
chat("返品ポリシーを教えてください")
chat("商品が壊れていた。TEST-USER-001としてチケットを作ってほしい")  # PII対策: 実名不使用

# COMMAND ----------

# DBTITLE 1,Step 9: Inspect MLflow Traces
# Step 9: MLflow Trace でエージェントの内部動作を確認
#
# Trace で見るべきポイント:
#   1. LLM がどのツールを選択したか  (lookup_order_status / search_faq / create_support_ticket)
#   2. ツールに渡した引数                (order_id: "ORD-001" など)
#   3. ツールの戻り値                   (注文情報・ FAQ が正しいか)
#   4. 各 Span のレイテンシ            (ボトルネックの特定)
#   5. エラー有無と最終回答内容
#
# Trace 画面の閲き方: 左ツリーで Span を選択 → 右パネルで入出力を確認

import mlflow

# Step 2 で設定した Experiment から Trace を取得
experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if experiment is None:
    print(f"Experiment が見つかりません: {MLFLOW_EXPERIMENT_NAME}")
    print("「Step 2」と「Step 4」を再実行してください。")
else:
    traces = mlflow.search_traces(
        experiment_ids=[experiment.experiment_id],
        max_results=10,
    )
    if traces.empty:
        print("トレースが見つかりません。Step 4 を再実行してください。")
    else:
        print(f"✓ 取得した Trace: {len(traces)} 件")
        print("↓ 左の request_id をクリックするとツール呼び出しの詳細が確認できます。")
        display(traces)
