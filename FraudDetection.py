import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ---------------------------
# CONFIG - DATA PATH
# ---------------------------

dataset_path = r"D:/bigdata project/dataset/fraud_dataset.csv"

# ---------------------------
# SPARK SESSION
# ---------------------------

spark = SparkSession.builder \
    .appName("FraudDetectionDash") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------------------
# LOAD DATA
# ---------------------------

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

df_spark = spark.read.csv(dataset_path, header=True, inferSchema=True)

# Clean columns
for c in df_spark.columns:
    df_spark = df_spark.withColumnRenamed(c, c.strip())

# Fill missing
string_cols = [f.name for f in df_spark.schema.fields if "StringType" in str(f.dataType)]
num_cols = [f.name for f in df_spark.schema.fields if any(
    t in str(f.dataType) for t in ["IntegerType", "LongType", "DoubleType", "FloatType"]
)]

df_spark = df_spark.fillna({c: "Unknown" for c in string_cols})
df_spark = df_spark.fillna({c: 0 for c in num_cols})

# Ensure required columns
required_cols = [
    "TransactionID", "Amount", "IsFraud", "Merchant", "Channel",
    "Location", "Device", "PreviousTransactions", "AccountAgeYears"
]

for c in required_cols:
    if c not in df_spark.columns:
        df_spark = df_spark.withColumn(c, col(df_spark.columns[0]))

# Convert numeric
df_spark = df_spark.withColumn("Amount", col("Amount").cast(DoubleType()))
df_spark = df_spark.withColumn("PreviousTransactions", col("PreviousTransactions").cast(IntegerType()))
df_spark = df_spark.withColumn("AccountAgeYears", col("AccountAgeYears").cast(DoubleType()))
df_spark = df_spark.withColumn("IsFraud", col("IsFraud").cast(IntegerType()))

df_spark.cache()
df_spark.count()

# ---------------------------
# COLORS
# ---------------------------

COLORS = {
    "blue": "#4cc9f0",
    "green": "#80ed99",
    "red": "#ef476f",
    "orange": "#f9844a",
    "purple": "#9d4edd",
    "yellow": "#f9c74f",
    "teal": "#38a3a5",
    "background": "#f7f7f7"
}

# ---------------------------
# DASH APP UI
# ---------------------------

app = Dash(__name__)
app.title = "PySpark Fraud Dashboard"

app.layout = html.Div([

    html.H1("💳 Financial Fraud Detection Dashboard (PySpark)",
            style={'textAlign': 'center', 'marginBottom': 20}),

    html.Div([
        html.Div([
            html.Label("Enter TransactionID:"),
            dcc.Input(id='txn_input', type='text', value="T0001", style={'width': '200px'}),
        ], style={'marginBottom': '12px'}),

        html.Div([
            html.Label("Enter Minimum Amount (₹):"),
            dcc.Input(id='amount_input', type='number', value=50000, style={'width': '200px'}),
        ], style={'marginBottom': '12px'}),

        html.Div([
            html.Label("Enter Minimum Account Age (Years):"),
            dcc.Input(id='account_age_input', type='number', value=0, style={'width': '200px'}),
        ], style={'marginBottom': '12px'}),

        html.Div([
            html.Label("Enter Minimum Previous Transactions:"),
            dcc.Input(id='prev_txn_input', type='number', value=0, style={'width': '200px'}),
        ], style={'marginBottom': '12px'}),

        html.Div(id="info", style={'fontSize': '15px', 'fontWeight': 'bold', 'marginTop': 12}),
    ], style={'padding': '20px'}),

    html.Div([
        html.Div([dcc.Graph(id='graph1')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='graph2')], style={'width': '48%', 'display': 'inline-block'}),
    ]),
    html.Div([
        html.Div([dcc.Graph(id='graph3')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='graph4')], style={'width': '48%', 'display': 'inline-block'}),
    ]),
    html.Div([
        html.Div([dcc.Graph(id='graph5')], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='graph6')], style={'width': '48%', 'display': 'inline-block'}),
    ])
])

# ---------------------------
# CALLBACK
# ---------------------------

@app.callback(
    [
        Output('graph1', 'figure'),
        Output('graph2', 'figure'),
        Output('graph3', 'figure'),
        Output('graph4', 'figure'),
        Output('graph5', 'figure'),
        Output('graph6', 'figure'),
        Output('info', 'children')
    ],
    [
        Input('txn_input', 'value'),
        Input('amount_input', 'value'),
        Input('account_age_input', 'value'),
        Input('prev_txn_input', 'value')
    ]
)
def update_graphs(txn_id, min_amount, min_age, min_prev_txn):

    if min_amount is None: min_amount = 0
    if min_age is None: min_age = 0
    if min_prev_txn is None: min_prev_txn = 0

    filtered = df_spark.filter(
        ((col("TransactionID") == txn_id) |
         (col("Amount") >= float(min_amount))) &
        (col("AccountAgeYears") >= float(min_age)) &
        (col("PreviousTransactions") >= float(min_prev_txn))
    )

    count_rows = filtered.count()
    info = f"📊 Showing {count_rows} records"

    if count_rows == 0:
        empty = px.scatter(title="No Data Available")
        return empty, empty, empty, empty, empty, empty, "❌ No data found!"

    pdf = filtered.toPandas()

    # 1 PIE
    g1 = pdf.groupby("IsFraud").size().reset_index(name="Count")
    g1["Label"] = g1["IsFraud"].map({0: "Not Fraud", 1: "Fraud"})
    fig1 = px.pie(g1, names="Label", values="Count")

    # 2 BAR
    g2 = pdf.groupby("Merchant")["Amount"].mean().reset_index()
    fig2 = px.bar(g2, x="Merchant", y="Amount", title="Avg Amount by Merchant")

    # 3 HISTOGRAM
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=pdf["Amount"], nbinsx=30, name="Amount"))
    fig3.add_trace(go.Histogram(x=pdf["PreviousTransactions"], nbinsx=30, name="Previous Txn"))
    fig3.add_trace(go.Histogram(x=pdf["AccountAgeYears"], nbinsx=30, name="Account Age"))
    fig3.update_layout(barmode="overlay", title="Distributions")
    fig3.update_traces(opacity=0.7)

    # 4 INDICATOR
    fraud_rate = round((pdf["IsFraud"].mean() * 100), 2)
    fig4 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_rate,
        title={'text': "Fraud Percentage"},
        gauge={'axis': {'range': [0, 100]}}
    ))

    # 5 HEATMAP
    corr = pdf[["Amount", "PreviousTransactions", "AccountAgeYears"]].corr()
    fig5 = px.imshow(corr, text_auto=True, title="Heatmap")

    # 6 DONUT
    g6 = pdf.groupby("Channel").size().reset_index(name="Count")
    fig6 = px.pie(g6, names="Channel", values="Count", hole=0.5)

    return fig1, fig2, fig3, fig4, fig5, fig6, info

# ---------------------------
# RUN
# ---------------------------

if __name__ == "__main__":
    print("🚀 Dashboard running at: http://127.0.0.1:8050/")
    app.run(debug=False)