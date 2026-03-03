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
from langchain_xai import ChatXAI
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

# ---------------------------------------------------------------------------
# Pre-built responses for sidebar suggestion buttons
# ---------------------------------------------------------------------------

PREBUILT: dict = {}


def _chart(spec_dict: dict) -> Optional[str]:
    """Generate a chart and return its base64 PNG, or None on failure."""
    try:
        return json.loads(create_chart.run(json.dumps(spec_dict))).get("image")
    except Exception:
        return None


def _build_prebuilt() -> None:
    """Compute text + chart responses for every sidebar quick-access question."""
    global PREBUILT

    total = len(df)
    survived_n = int(df["Survived"].sum())
    not_survived_n = total - survived_n
    survival_rate = round(df["Survived"].mean() * 100, 1)

    PREBUILT["What was the survival rate?"] = {
        "answer": (
            f"Out of {total} passengers, **{survived_n} survived** — a survival rate of "
            f"**{survival_rate}%**. The remaining {not_survived_n} "
            f"({100 - survival_rate:.1f}%) did not survive."
        ),
        "image_b64": _chart({
            "type": "bar", "column": "Survived",
            "title": "Survival Count",
            "xlabel": "Survived (0 = No, 1 = Yes)", "ylabel": "Passengers",
        }),
        "image_caption": "Survival Count",
    }

    avg_age = round(float(df["Age"].mean()), 1)
    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())
    PREBUILT["Show me a histogram of passenger ages"] = {
        "answer": (
            f"Passenger ages ranged from **{min_age}** to **{max_age}** years, "
            f"with an average of **{avg_age}** years. "
            f"Most passengers were between 20 and 40 years old."
        ),
        "image_b64": _chart({
            "type": "histogram", "column": "Age", "bins": 20,
            "title": "Distribution of Passenger Ages",
            "xlabel": "Age", "ylabel": "Frequency",
        }),
        "image_caption": "Distribution of Passenger Ages",
    }

    fare_by_class = df.groupby("Pclass")["Fare"].mean().round(2)
    PREBUILT["Average fare by passenger class"] = {
        "answer": (
            f"Average fares — "
            f"1st class: **${fare_by_class.get(1, 0):.2f}**, "
            f"2nd class: **${fare_by_class.get(2, 0):.2f}**, "
            f"3rd class: **${fare_by_class.get(3, 0):.2f}**. "
            f"1st class passengers paid significantly more."
        ),
        "image_b64": _chart({
            "type": "bar", "column": "Pclass", "agg": "mean", "agg_col": "Fare",
            "title": "Average Fare by Passenger Class",
            "xlabel": "Passenger Class", "ylabel": "Avg Fare ($)",
        }),
        "image_caption": "Average Fare by Passenger Class",
    }

    port_counts = df["Embarked"].value_counts()
    port_names = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    port_str = ", ".join(
        f"**{port_names.get(p, p)} ({p})**: {n}" for p, n in port_counts.items()
    )
    port_entry = {
        "answer": (
            f"Embarkation ports — {port_str}. "
            f"Southampton was by far the most common boarding point."
        ),
        "image_b64": _chart({
            "type": "bar", "column": "Embarked",
            "title": "Passengers by Embarkation Port",
            "xlabel": "Port (S=Southampton, C=Cherbourg, Q=Queenstown)",
            "ylabel": "Passengers",
        }),
        "image_caption": "Passengers by Embarkation Port",
    }
    PREBUILT["How many passengers from each port?"] = port_entry
    PREBUILT["How many passengers embarked from each port?"] = port_entry

    surv_by_sex = df.groupby("Sex")["Survived"].mean().mul(100).round(1)
    count_by_sex = df["Sex"].value_counts()
    PREBUILT["Survival rate by gender"] = {
        "answer": (
            f"Women had a much higher survival rate "
            f"(**{surv_by_sex.get('female', 0)}%**) than men "
            f"(**{surv_by_sex.get('male', 0)}%**). "
            f"There were {count_by_sex.get('female', 0)} women and "
            f"{count_by_sex.get('male', 0)} men aboard. "
            f'"Women and children first" was clearly practiced.'
        ),
        "image_b64": _chart({
            "type": "bar", "column": "Sex", "agg": "mean", "agg_col": "Survived",
            "title": "Survival Rate by Gender",
            "xlabel": "Gender", "ylabel": "Survival Rate (0–1)",
        }),
        "image_caption": "Survival Rate by Gender",
    }

    age_med_by_class = df.groupby("Pclass")["Age"].median().round(1)
    PREBUILT["Show age distribution by class as a box plot"] = {
        "answer": (
            f"Median ages by class — "
            f"1st class: **{age_med_by_class.get(1, 0)}** yrs, "
            f"2nd class: **{age_med_by_class.get(2, 0)}** yrs, "
            f"3rd class: **{age_med_by_class.get(3, 0)}** yrs. "
            f"1st class passengers tended to be older."
        ),
        "image_b64": _chart({
            "type": "box", "column": "Age", "hue": "Pclass",
            "title": "Age Distribution by Passenger Class",
            "xlabel": "Passenger Class", "ylabel": "Age (years)",
        }),
        "image_caption": "Age Distribution by Passenger Class",
    }

    male_n = int((df["Sex"] == "male").sum())
    female_n = total - male_n
    male_pct = round(male_n / total * 100, 1)
    female_pct = round(100 - male_pct, 1)
    PREBUILT["What percentage of passengers were male on the Titanic?"] = {
        "answer": (
            f"**{male_pct}%** of passengers were male ({male_n} out of {total}). "
            f"The remaining **{female_pct}%** ({female_n} passengers) were female."
        ),
        "image_b64": _chart({
            "type": "pie", "column": "Sex",
            "title": "Passenger Gender Breakdown",
        }),
        "image_caption": "Passenger Gender Breakdown",
    }

    avg_fare = round(float(df["Fare"].mean()), 2)
    min_fare = round(float(df["Fare"].min()), 2)
    max_fare = round(float(df["Fare"].max()), 2)
    median_fare = round(float(df["Fare"].median()), 2)
    PREBUILT["What was the average ticket fare?"] = {
        "answer": (
            f"The average ticket fare was **${avg_fare}**. "
            f"Fares ranged from **${min_fare}** to **${max_fare}**, "
            f"with a median of **${median_fare}**. "
            f"The wide range reflects the large price gap between passenger classes."
        ),
        "image_b64": _chart({
            "type": "histogram", "column": "Fare", "bins": 30,
            "title": "Distribution of Ticket Fares",
            "xlabel": "Fare ($)", "ylabel": "Passengers",
        }),
        "image_caption": "Distribution of Ticket Fares",
    }


try:
    _build_prebuilt()
except Exception as exc:
    print(f"[TitanicBot] Warning: pre-built responses failed to generate: {exc}")


prompt = PromptTemplate.from_template("""
You are TitanicBot, a friendly assistant for exploring the Titanic passenger dataset.

Columns: PassengerId, Survived (0=no,1=yes), Pclass (1/2/3), Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked (S/C/Q)

Rules:
- Use query_data for real numbers; never make up stats
- Use create_chart when asked for a chart/histogram/plot
- Be concise and friendly

{tools}

Format:
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the input
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
        llm = ChatXAI(
            model="grok-3-mini",
            temperature=0,
            xai_api_key=os.getenv("XAI_API_KEY"),
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


@app.get("/prebuilt")
def get_prebuilt():
    return PREBUILT


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