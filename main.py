import io
import base64
import json
import os
import traceback
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

COLORS = {
    "bg": "#0d1117",
    "card": "#161b22",
    "primary": "#58a6ff",
    "accent": "#f78166",
    "green": "#3fb950",
    "purple": "#bc8cff",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "bars": ["#58a6ff", "#f78166", "#3fb950", "#bc8cff", "#ffa657", "#79c0ff"],
}


def load_data():
    try:
        return pd.read_csv(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        )
    except Exception:
        return pd.DataFrame({
            "PassengerId": range(1, 21),
            "Survived": [0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,1],
            "Pclass": [3,1,3,1,3,3,1,3,3,2,3,1,3,3,3,2,3,2,3,3],
            "Name": [f"Passenger {i}" for i in range(1, 21)],
            "Sex": ["male","female","female","female","male","male","male","male",
                    "female","female","female","female","male","male","female",
                    "female","male","male","female","female"],
            "Age": [22,38,26,35,35,None,54,2,27,14,4,58,20,39,14,55,2,None,31,None],
            "SibSp": [1,1,0,1,0,0,0,3,0,1,1,0,0,1,0,0,4,0,1,0],
            "Parch": [0,0,0,0,0,0,0,1,2,0,1,0,0,5,0,0,1,0,0,0],
            "Ticket": [f"T{i}" for i in range(1, 21)],
            "Fare": [7.25,71.28,7.92,53.1,8.05,8.46,51.86,21.07,11.13,30.07,
                     16.7,26.55,8.05,31.27,7.85,16,29.12,13,18,7.22],
            "Cabin": [None]*20,
            "Embarked": ["S","C","S","S","S","Q","S","S","S","C",
                         "S","S","S","S","S","S","Q","S","S","C"],
        })


df = load_data()


def style_axes(fig, ax, title):
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["card"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["muted"])
    ax.yaxis.label.set_color(COLORS["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_title(title, color=COLORS["text"], fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="y", color="#21262d", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor=COLORS["bg"])
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return result


@tool
def query_data(expression: str) -> str:
    """
    Run a Pandas expression on the Titanic dataframe. The dataframe variable is called df.
    Use this to get counts, averages, percentages, and any other statistics.

    Examples:
    - Total passengers: len(df)
    - Male percentage: round(df[df.Sex=='male'].shape[0] / len(df) * 100, 2)
    - Average fare: round(df['Fare'].mean(), 2)
    - Port counts: df['Embarked'].value_counts().to_string()
    """
    try:
        result = eval(expression, {"df": df, "pd": pd, "np": np})
        if isinstance(result, pd.DataFrame):
            return result.to_string()
        if isinstance(result, pd.Series):
            return result.to_string()
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def create_chart(spec: str) -> str:
    """
    Draw a chart and return it as a base64 PNG image.

    Pass a JSON string with these fields:
    - type: histogram, bar, pie, scatter, box, or line
    - column: the main column to plot
    - title: chart title
    - xlabel: x axis label (optional)
    - ylabel: y axis label (optional)
    - hue: grouping column (optional)
    - bins: number of bins for histogram (optional, default 20)
    - top_n: show only top N categories (optional)
    - agg: count, mean, or sum for bar charts (optional, default count)
    - agg_col: column to aggregate when agg is mean or sum (optional)

    Returns JSON: {"image": "<base64 string>", "caption": "chart title"}
    """
    try:
        s = json.loads(spec)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Bad JSON: {e}"})

    kind    = s.get("type", "bar").lower()
    col     = s.get("column")
    hue     = s.get("hue")
    title   = s.get("title", "Chart")
    xlabel  = s.get("xlabel", col or "")
    ylabel  = s.get("ylabel", "Count")
    bins    = int(s.get("bins", 20))
    top_n   = s.get("top_n")
    agg     = s.get("agg", "count")
    agg_col = s.get("agg_col")

    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        style_axes(fig, ax, title)

        if kind == "histogram":
            ax.hist(df[col].dropna(), bins=bins,
                    color=COLORS["primary"], edgecolor=COLORS["bg"], linewidth=0.5)
            ax.set_xlabel(xlabel, color=COLORS["muted"])
            ax.set_ylabel(ylabel or "Frequency", color=COLORS["muted"])

        elif kind == "bar":
            if agg == "mean":
                counts = df.groupby(col)[agg_col].mean()
            elif agg == "sum":
                counts = df.groupby(col)[agg_col].sum()
            else:
                counts = df[col].value_counts()

            if top_n:
                counts = counts.head(int(top_n))

            bars = ax.bar(counts.index.astype(str), counts.values,
                          color=COLORS["bars"][:len(counts)],
                          edgecolor=COLORS["bg"], linewidth=0.5)
            ax.set_xlabel(xlabel, color=COLORS["muted"])
            ax.set_ylabel(ylabel, color=COLORS["muted"])
            for bar, val in zip(bars, counts.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts.values) * 0.01,
                    f"{val:,.1f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", color=COLORS["text"], fontsize=8
                )

        elif kind == "pie":
            counts = df[col].value_counts()
            if top_n:
                counts = counts.head(int(top_n))
            ax.pie(
                counts.values,
                labels=counts.index.astype(str),
                colors=COLORS["bars"][:len(counts)],
                autopct="%1.1f%%",
                pctdistance=0.82,
                wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
                textprops={"color": COLORS["text"], "fontsize": 9},
            )

        elif kind == "scatter":
            y_col = agg_col or "Fare"
            ax.scatter(df[col].dropna(), df[y_col].dropna(),
                       c=COLORS["primary"], alpha=0.6, s=14, edgecolors="none")
            ax.set_xlabel(xlabel, color=COLORS["muted"])
            ax.set_ylabel(ylabel or y_col, color=COLORS["muted"])

        elif kind == "box":
            group_col = hue or "Pclass"
            groups = [df[df[group_col] == g][col].dropna()
                      for g in sorted(df[group_col].unique())]
            bp = ax.boxplot(
                groups,
                patch_artist=True,
                medianprops={"color": COLORS["accent"], "linewidth": 2},
                whiskerprops={"color": COLORS["muted"]},
                capprops={"color": COLORS["muted"]},
                flierprops={"marker": "o", "markersize": 3,
                             "markerfacecolor": COLORS["muted"], "alpha": 0.5},
            )
            for patch, color in zip(bp["boxes"], COLORS["bars"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            ax.set_xticklabels(sorted(df[group_col].unique()), color=COLORS["muted"])
            ax.set_xlabel(xlabel or group_col, color=COLORS["muted"])
            ax.set_ylabel(ylabel or col, color=COLORS["muted"])

        elif kind == "line":
            counts = df[col].value_counts().sort_index()
            ax.plot(counts.index.astype(str), counts.values,
                    color=COLORS["primary"], linewidth=2, marker="o",
                    markersize=5, markerfacecolor=COLORS["accent"])
            ax.fill_between(range(len(counts)), counts.values,
                            alpha=0.15, color=COLORS["primary"])
            ax.set_xlabel(xlabel, color=COLORS["muted"])
            ax.set_ylabel(ylabel, color=COLORS["muted"])

        else:
            plt.close(fig)
            return json.dumps({"error": f"Unknown chart type: {kind}"})

        fig.tight_layout(pad=1.5)
        return json.dumps({"image": fig_to_base64(fig), "caption": title})

    except Exception as e:
        plt.close("all")
        return json.dumps({"error": str(e)})


tools = [query_data, create_chart]

prompt = PromptTemplate.from_template("""
You are TitanicBot, a friendly assistant that helps people explore the Titanic passenger dataset.

The dataset has these columns:
PassengerId, Survived (0=no, 1=yes), Pclass (1/2/3), Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked (S/C/Q)

Rules:
- Always use query_data to get real numbers before stating any facts
- When someone asks for a chart, histogram, or plot, use create_chart
- Be friendly and conversational in your answers
- Never make up numbers

{tools}

Use this format strictly:

Question: the question to answer
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the input for that action
Observation: the result
... (repeat as needed)
Thought: I now know the final answer
Final Answer: your answer

Question: {input}
{agent_scratchpad}
""".strip())

agent_executor = None


def get_agent():
    global agent_executor
    if agent_executor is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
        )
    return agent_executor


app = FastAPI(title="TitanicBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    image_b64: Optional[str] = None
    image_caption: Optional[str] = None


@app.get("/ping")
def ping():
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "rows": len(df)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = get_agent().invoke({"input": req.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    answer = result.get("output", "")
    image_b64 = None
    image_caption = None

    for step in result.get("intermediate_steps", []):
        tool_name = getattr(step[0], "tool", "")
        tool_output = step[1] if len(step) > 1 else ""
        if tool_name == "create_chart" and isinstance(tool_output, str):
            try:
                data = json.loads(tool_output)
                if "image" in data:
                    image_b64 = data["image"]
                    image_caption = data.get("caption", "")
            except json.JSONDecodeError:
                pass

    return ChatResponse(answer=answer, image_b64=image_b64, image_caption=image_caption)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)