# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///

# DBTITLE 1,Customer Support AI Agent - Overview
# MAGIC %md
# MAGIC > ⚠️ **2026年7月現在の推奨経路について**
# MAGIC >
# MAGIC > このNotebookは、MLflow `ResponsesAgent`をUnity Catalogへ登録し、Databricks Model Servingへデプロイする方式を扱います。
# MAGIC > 新規エージェント開発ではDatabricks AppsベースのCustom Agentが推奨されています。
# MAGIC > 詳細: https://docs.databricks.com/aws/en/agents/agent-framework/migrate-agent-to-apps
# MAGIC >
# MAGIC > このNotebookの実行範囲は、エージェント構築、ローカルデモ、Tracing、モデル登録、デプロイ、Endpoint確認です。
# MAGIC > 評価データセット、品質ゲート、Production Monitoring、改善ループは記事中のコード例を基に追加してください。
# MAGIC
# MAGIC # 🤖 カスタマーサポートAIエージェント on Databricks
# MAGIC
# MAGIC ```text
# MAGIC ユーザー問い合わせ
# MAGIC         ↓
# MAGIC Databricks Model Serving
# MAGIC         ↓
# MAGIC CustomerSupportAgent（LangGraph + ResponsesAgent）
# MAGIC         ↓
# MAGIC lookup_order_status / search_faq / create_support_ticket
# MAGIC         ↓
# MAGIC Databricks Foundation Model API + MLflow Tracing
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Step 1: Install Packages
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
import mlflow

CATALOG = "main"
SCHEMA = "your_schema"

MODEL_NAME = f"{CATALOG}.{SCHEMA}.customer_support_agent"
EVAL_DATASET_NAME = f"{CATALOG}.{SCHEMA}.customer_support_eval"
AGENT_ENDPOINT_NAME = "customer-support-agent"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
AGENT_FILE_PATH = "/tmp/customer_support_agent.py"

try:
    username = (
        dbutils.notebook.entry_point
        .getDbutils()
        .notebook()
        .getContext()
        .userName()
        .get()
    )
except Exception:
    username = "your-email@databricks.com"

MLFLOW_EXPERIMENT_NAME = f"/Users/{username}/customer-support-agent"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print(f"Model: {MODEL_NAME}")
print(f"Evaluation dataset: {EVAL_DATASET_NAME}")
print(f"Serving endpoint: {AGENT_ENDPOINT_NAME}")
print(f"LLM endpoint: {LLM_ENDPOINT}")
print(f"Agent file: {AGENT_FILE_PATH}")
print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Step 3: Prepare Dedicated Agent File
import os

try:
    os.chmod(AGENT_FILE_PATH, 0o666)
except FileNotFoundError:
    pass
except PermissionError:
    os.remove(AGENT_FILE_PATH)

# COMMAND ----------

# DBTITLE 1,Step 4: Write Agent Module
# MAGIC %%writefile /tmp/customer_support_agent.py
# MAGIC import hashlib
# MAGIC from typing import Annotated, Generator, Sequence, TypedDict
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from langchain_core.messages import AIMessage, BaseMessage
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.tools import tool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC
# MAGIC LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC
# MAGIC SYSTEM_PROMPT = """
# MAGIC あなたはEコマースのカスタマーサポートAIです。必ず日本語で回答してください。
# MAGIC
# MAGIC 利用可能なツール:
# MAGIC - lookup_order_status: 注文番号から注文状況を取得する
# MAGIC - search_faq: 返品、配送、支払い、保証などのFAQを検索する
# MAGIC - create_support_ticket: ユーザーが明示的に依頼した場合に問い合わせチケットを作成する
# MAGIC
# MAGIC ルール:
# MAGIC 1. 注文状態、日付、商品名、チケットIDはツール結果の表記を変換・省略しない。
# MAGIC 2. 日付はYYYY-MM-DDのまま記載する。
# MAGIC 3. ツール結果に基づく回答は「システムで確認しました。結果：」から始める。
# MAGIC 4. 挨拶や雑談ではツールを呼ばない。
# MAGIC 5. 注文番号がある返品・交換問い合わせでは、注文確認後に返品FAQも検索する。
# MAGIC 6. ツールで確認できない情報を推測して断定しない。
# MAGIC 7. チケット作成はユーザーが明示的に依頼した場合だけ実行する。
# MAGIC """.strip()
# MAGIC
# MAGIC ORDER_DB = {
# MAGIC     "ORD-001": {
# MAGIC         "status": "配送中",
# MAGIC         "item": "ノートPC",
# MAGIC         "estimated_delivery": "2026-07-20",
# MAGIC     },
# MAGIC     "ORD-002": {
# MAGIC         "status": "配送完了",
# MAGIC         "item": "ワイヤレスイヤホン",
# MAGIC         "delivered_date": "2026-07-15",
# MAGIC     },
# MAGIC     "ORD-003": {
# MAGIC         "status": "処理中",
# MAGIC         "item": "スマートフォン",
# MAGIC         "estimated_delivery": "2026-07-22",
# MAGIC     },
# MAGIC     "ORD-004": {
# MAGIC         "status": "配送完了",
# MAGIC         "item": "タブレット",
# MAGIC         "delivered_date": "2026-07-10",
# MAGIC     },
# MAGIC }
# MAGIC
# MAGIC FAQ_DB = [
# MAGIC     {
# MAGIC         "keywords": ["返品", "返金", "返却", "交換"],
# MAGIC         "question": "返品・返金ポリシー",
# MAGIC         "answer": "商品到着から30日以内、未使用品に限り返品可。返金は5〜7営業日。",
# MAGIC     },
# MAGIC     {
# MAGIC         "keywords": ["配送", "配達", "届く", "所要", "期間"],
# MAGIC         "question": "配送期間・配送方法",
# MAGIC         "answer": "通常2〜5営業日でお届け。追跡番号をメールでお知らせ。",
# MAGIC     },
# MAGIC     {
# MAGIC         "keywords": ["支払", "決済", "payment", "クレジット", "振込"],
# MAGIC         "question": "支払い方法",
# MAGIC         "answer": "クレジットカード・銀行振込・コンビニ払い・PayPay・LINE Pay・Amazon Pay対応。",
# MAGIC     },
# MAGIC     {
# MAGIC         "keywords": ["保証", "修理", "故障", "不良品", "壊れ"],
# MAGIC         "question": "保証・修理対応",
# MAGIC         "answer": "購入から1年間のメーカー保証。期間内の不具合は無償修理・交換。",
# MAGIC     },
# MAGIC     {
# MAGIC         "keywords": ["キャンセル", "注文取り消し"],
# MAGIC         "question": "注文キャンセル",
# MAGIC         "answer": "発送前はマイページからキャンセル可。発送後は返品手続きが必要。",
# MAGIC     },
# MAGIC ]
# MAGIC
# MAGIC
# MAGIC @tool
# MAGIC def lookup_order_status(order_id: str) -> str:
# MAGIC     """注文IDで注文状況を確認する。注文IDはORD-XXX形式。"""
# MAGIC     normalized_order_id = order_id.strip().upper()
# MAGIC     order = ORDER_DB.get(normalized_order_id)
# MAGIC     if not order:
# MAGIC         return f"注文ID '{normalized_order_id}' は見つかりませんでした。"
# MAGIC
# MAGIC     lines = [
# MAGIC         f"注文ID: {normalized_order_id}",
# MAGIC         f"商品: {order['item']}",
# MAGIC         f"状態: {order['status']}",
# MAGIC     ]
# MAGIC     if "estimated_delivery" in order:
# MAGIC         lines.append(f"配達予定日: {order['estimated_delivery']}")
# MAGIC     if "delivered_date" in order:
# MAGIC         lines.append(f"配達完了日: {order['delivered_date']}")
# MAGIC     return "\n".join(lines)
# MAGIC
# MAGIC
# MAGIC @tool
# MAGIC def search_faq(query: str) -> str:
# MAGIC     """質問に関連するFAQを検索する。"""
# MAGIC     results = [
# MAGIC         f"《{faq['question']}》\n{faq['answer']}"
# MAGIC         for faq in FAQ_DB
# MAGIC         if any(keyword in query for keyword in faq["keywords"])
# MAGIC     ]
# MAGIC     if not results:
# MAGIC         return "該当するFAQは見つかりませんでした。別のキーワードで再度お試しください。"
# MAGIC     return "\n\n".join(results)
# MAGIC
# MAGIC
# MAGIC @tool
# MAGIC def create_support_ticket(
# MAGIC     customer_name: str,
# MAGIC     issue_summary: str,
# MAGIC     priority: str = "medium",
# MAGIC ) -> str:
# MAGIC     """問い合わせチケットを作成するモック。"""
# MAGIC     allowed_priorities = {"low", "medium", "high", "urgent"}
# MAGIC     if priority not in allowed_priorities:
# MAGIC         priority = "medium"
# MAGIC
# MAGIC     digest = hashlib.sha256(
# MAGIC         f"{customer_name}:{issue_summary}".encode("utf-8")
# MAGIC     ).hexdigest()
# MAGIC     ticket_id = f"TKT-{int(digest[:8], 16) % 90000 + 10000}"
# MAGIC     return (
# MAGIC         f"チケットID: {ticket_id}\n"
# MAGIC         f"お客様名: {customer_name}\n"
# MAGIC         f"問題概要: {issue_summary}\n"
# MAGIC         f"優先度: {priority}\n"
# MAGIC         "担当者が24時間以内にご連絡いたします。"
# MAGIC     )
# MAGIC
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC
# MAGIC
# MAGIC class CustomerSupportAgent(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         self.tools = [
# MAGIC             lookup_order_status,
# MAGIC             search_faq,
# MAGIC             create_support_ticket,
# MAGIC         ]
# MAGIC         self.llm = ChatDatabricks(
# MAGIC             endpoint=LLM_ENDPOINT,
# MAGIC             temperature=0.1,
# MAGIC             max_tokens=2000,
# MAGIC         )
# MAGIC         self.llm_with_tools = self.llm.bind_tools(self.tools)
# MAGIC         self.graph = self._build_graph()
# MAGIC
# MAGIC     def _build_graph(self):
# MAGIC         def should_continue(state: AgentState) -> str:
# MAGIC             last_message = state["messages"][-1]
# MAGIC             if isinstance(last_message, AIMessage) and last_message.tool_calls:
# MAGIC                 return "tools"
# MAGIC             return "end"
# MAGIC
# MAGIC         def call_model(state: AgentState) -> dict:
# MAGIC             messages = [
# MAGIC                 {"role": "system", "content": SYSTEM_PROMPT},
# MAGIC                 *list(state["messages"]),
# MAGIC             ]
# MAGIC             return {"messages": [self.llm_with_tools.invoke(messages)]}
# MAGIC
# MAGIC         graph = StateGraph(AgentState)
# MAGIC         graph.add_node("agent", RunnableLambda(call_model))
# MAGIC         graph.add_node("tools", ToolNode(self.tools))
# MAGIC         graph.add_conditional_edges(
# MAGIC             "agent",
# MAGIC             should_continue,
# MAGIC             {"tools": "tools", "end": END},
# MAGIC         )
# MAGIC         graph.add_edge("tools", "agent")
# MAGIC         graph.set_entry_point("agent")
# MAGIC         return graph.compile()
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         messages = to_chat_completions_input(
# MAGIC             [message.model_dump() for message in request.input]
# MAGIC         )
# MAGIC         for event in self.graph.stream(
# MAGIC             {"messages": messages},
# MAGIC             stream_mode=["updates"],
# MAGIC             config={"recursion_limit": 10},
# MAGIC         ):
# MAGIC             if event[0] != "updates":
# MAGIC                 continue
# MAGIC             for node_data in event[1].values():
# MAGIC                 if node_data.get("messages"):
# MAGIC                     yield from output_to_responses_items_stream(
# MAGIC                         node_data["messages"]
# MAGIC                     )
# MAGIC
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = CustomerSupportAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# DBTITLE 1,Step 5: Load Agent Module Explicitly
import importlib.util
import os

if not os.path.exists(AGENT_FILE_PATH):
    raise FileNotFoundError(f"{AGENT_FILE_PATH} が見つかりません。")

spec = importlib.util.spec_from_file_location(
    "customer_support_agent_module",
    AGENT_FILE_PATH,
)
if spec is None or spec.loader is None:
    raise ImportError(f"{AGENT_FILE_PATH} の読み込み準備に失敗しました。")

agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)
AGENT = agent_module.AGENT

# COMMAND ----------

# DBTITLE 1,Step 6: Local Demo
from mlflow.types.responses import ResponsesAgentRequest


def _get_attr_or_key(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def demo_agent(question: str) -> None:
    """回答を表示するデモ。期待ツールや引数を検証する自動テストではない。"""
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": question}]
    )
    response = AGENT.predict(request)

    print(f"Q: {question}")
    for item in response.output:
        item_type = _get_attr_or_key(item, "type")
        if item_type not in (None, "message"):
            continue
        content_items = _get_attr_or_key(item, "content", []) or []
        if not isinstance(content_items, list):
            content_items = [content_items]
        for content_item in content_items:
            content_type = _get_attr_or_key(content_item, "type")
            text = _get_attr_or_key(content_item, "text")
            if text and content_type in (None, "output_text"):
                print(f"A: {text}")
    print("-" * 60)


demo_agent("注文ORD-001の配送状況を教えてください")
demo_agent("返品ポリシーを教えてください")
demo_agent(
    "注文商品に問題があります。"
    "TEST-USER-001として問い合わせチケットの登録をお願いします"
)

# COMMAND ----------

# DBTITLE 1,Step 7: Ensure Unity Catalog Schema Exists
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

w = WorkspaceClient()

try:
    w.schemas.get(f"{CATALOG}.{SCHEMA}")
    print(f"Schema already exists: {CATALOG}.{SCHEMA}")
except NotFound:
    w.schemas.create(name=SCHEMA, catalog_name=CATALOG)
    print(f"Schema created: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Step 8: Log and Register Model
from mlflow.models.resources import DatabricksServingEndpoint

mlflow.set_registry_uri("databricks-uc")

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)]
input_example = {
    "input": [
        {"role": "user", "content": "注文ORD-001の状況を教えてください"}
    ]
}

with mlflow.start_run(run_name="customer-support-agent"):
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=AGENT_FILE_PATH,
        resources=resources,
        pip_requirements=[
            "mlflow==3.6.0",
            "databricks-langchain==0.8.2",
            "langgraph==0.3.4",
            "langchain-core==0.3.86",
            "pydantic==2.12.5",
        ],
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )

print(f"Model URI: {model_info.model_uri}")
print(f"Registered model: {MODEL_NAME}")
print(f"Version: {model_info.registered_model_version}")

# COMMAND ----------

# DBTITLE 1,Step 9: Wait for Previous Endpoint Update
import time


def wait_for_config_update(name: str, timeout_min: int = 20) -> None:
    deadline = time.time() + timeout_min * 60

    while time.time() < deadline:
        try:
            endpoint = w.serving_endpoints.get(name=name)
        except NotFound:
            print(f"Endpoint does not exist yet: {name}")
            return

        state = endpoint.state
        config_update = (
            state.config_update.value
            if state and state.config_update
            else "UNKNOWN"
        )
        print(f"config_update={config_update}")

        if config_update in {"NOT_UPDATING", "UPDATE_CANCELED"}:
            return
        if config_update == "UPDATE_FAILED":
            raise RuntimeError(
                f"Endpoint {name} の前回更新が失敗しています。"
            )
        time.sleep(30)

    raise TimeoutError(
        f"Endpoint {name} の設定更新が{timeout_min}分以内に完了しませんでした。"
    )


wait_for_config_update(AGENT_ENDPOINT_NAME)

# COMMAND ----------

# DBTITLE 1,Step 10: Deploy to Model Serving
from databricks import agents

print(f"Deploying {MODEL_NAME} version {model_info.registered_model_version}")

deploy_info = agents.deploy(
    model_name=MODEL_NAME,
    model_version=model_info.registered_model_version,
    endpoint_name=AGENT_ENDPOINT_NAME,
    tags={"environment": "development", "use_case": "customer_support"},
)

print(f"Endpoint: {deploy_info.endpoint_name}")
review_app_url = getattr(deploy_info, "review_app_url", None)
if review_app_url:
    print(f"Review App: {review_app_url}")

# COMMAND ----------

# DBTITLE 1,Step 11: Wait Until Endpoint Is Ready

def wait_for_endpoint(name: str, timeout_min: int = 20) -> None:
    deadline = time.time() + timeout_min * 60

    while time.time() < deadline:
        endpoint = w.serving_endpoints.get(name=name)
        state = endpoint.state
        ready = state.ready.value if state and state.ready else "NOT_READY"
        config_update = (
            state.config_update.value
            if state and state.config_update
            else "UNKNOWN"
        )
        print(f"ready={ready}, config_update={config_update}")

        if ready == "READY":
            print(f"Endpoint is ready: {name}")
            return
        if config_update == "UPDATE_FAILED":
            raise RuntimeError(f"Endpoint {name} の更新に失敗しました。")
        time.sleep(30)

    raise TimeoutError(
        f"Endpoint {name} が{timeout_min}分以内にREADYになりませんでした。"
    )


wait_for_endpoint(AGENT_ENDPOINT_NAME)

# COMMAND ----------

# DBTITLE 1,Step 12: Query Deployed Endpoint
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")


def chat(question: str) -> None:
    response = client.predict(
        endpoint=AGENT_ENDPOINT_NAME,
        inputs={"input": [{"role": "user", "content": question}]},
    )

    print(f"Q: {question}")
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []):
            if content_item.get("type") == "output_text":
                print(f"A: {content_item.get('text', '')}")
    print("-" * 60)


chat("注文ORD-001の配送状況を教えてください")
chat("返品ポリシーを教えてください")
chat(
    "注文商品に問題があります。"
    "TEST-USER-001として問い合わせチケットの登録をお願いします"
)

# COMMAND ----------

# DBTITLE 1,Step 13: Inspect MLflow Traces
experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(
        f"Experimentが見つかりません: {MLFLOW_EXPERIMENT_NAME}"
    )

traces = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    max_results=20,
)

if traces.empty:
    print("Traceが見つかりません。ローカルデモまたはEndpoint問い合わせを実行してください。")
else:
    object_columns = list(traces.select_dtypes(include="object").columns)
    if object_columns:
        traces = traces.astype({column: str for column in object_columns})
    display(traces)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 次のステップ
# MAGIC
# MAGIC 記事では、このベースNotebookに次の要素を追加する設計を解説しています。
# MAGIC
# MAGIC - 評価データセット
# MAGIC - Traceを参照するカスタムScorer
# MAGIC - デプロイを停止する品質ゲート
# MAGIC - Production Monitoring（2026年7月現在Beta）
# MAGIC - 低品質候補Traceを人手レビューへ戻す改善ループ
