# agent.py
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import csv
from helpers import get_llm
from report_pdf import generate_pdf_report

class State(TypedDict):
    message: str
    dataset_info: str

NDJSON_SPEC = r"""
You are a senior Python data analyst.

You must output EXACTLY 8 lines of NDJSON (one JSON object per line), covering 4 sections with 2 items each:

SECTIONS & ITEMS (exactly 8 total; return in any order):

1) Overview (2 items)
   a) Dataset Overview (narrative_only=true)
      - goal:generate 2–3 sentences explaining why an overview matters.
      - description_template:generate description include {total_papers}, {year_min}, {year_max}, {num_conferences}
      - compute_only_code: compute METRICS only (no figure).
   b) Conference Distribution (barh, top N but show all if fewer exist)
      - goal:generate 2 sentences on dissemination channels / venue structure.
      - description_template:generate description include {total_papers}, {top_conf}, {top_conf_count}, {k_used}
      - code: single horizontal bar chart; handle long labels safely.

2) Temporal Trends (2 items)
   a) Yearly Publication Trend (line)
      - goal:generate 2 sentences about growth and cycles.
      - description_template:generate description include {year_min}, {year_max}, {peak_year}, {peak_total}
   b) Yearly Awarded Papers Trend (line)
      - goal:generate 2 sentences on recognition over time.
      - description_template:generate description include {award_year_min}, {award_year_max}, {award_peak_year}, {award_peak_total}

3) Top Entities (2 items)
   a) Top 10 Prolific Authors (barh)
      - goal:generate 2 sentences on core contributors and collaboration potential.
      - description_template:generate description include {top_author}, {top_author_count}
   b) Top Keywords (barh; split by comma ",", case-insensitive; drop NaN/empty)
      - goal:generate 2 sentences on dominant themes and vocabulary.
      - description_template:generate description include {kw1}, {kw2}, {kw3}

4) Cross-Metric Relationships (2 items)
   a) Citations vs Downloads (scatter)
      - goal:generate 2 sentences on whether download interest aligns with scholarly impact.
      - description_template:generate description include {n_points}, {spearman_rho:.2f}
      - code: one scatter using numeric 'CitationCount_CrossRef' and 'Downloads_Xplore'; compute Spearman via pandas corr(method='spearman'); drop NaNs; set METRICS.
   b) Award Share by Conference (dot plot; share = awarded/total per conference; point size = total)
      - goal:generate 2 sentences on which venues have higher recognition intensity.
      - description_template:generate description include {k_used}, {top_share_conf}, {top_share:.0%}. From the figure above, we observe visible differences across venues.
      - code: define awarded if 'Award' non-empty after strip; groupby Conference → total & awarded → share = awarded/total; optionally filter small total (e.g., total>=3) for stability; sort by share desc; take top N (≤10); draw a dot plot (sns.scatterplot) with x=share, y=conference, size=total, sizes=(50,500), legend='brief'; set METRICS including k_used, top_share_conf, top_share.

JSON schema for each line:
{
  "section": "<Overview | Temporal Trends | Top Entities | Cross-Metric Relationships>",
  "title": "<short title>",
  "goal": "<2–3 sentences>",
  "description_template": "<1–2 sentences with {placeholders} computed by METRICS>",
  "narrative_only": <true|false>,
  "compute_only_code": "<python>",   // required if narrative_only=true
  "code": "<python>"                 // required if narrative_only=false, must create exactly one chart and set METRICS
}

NON-NEGOTIABLE OUTPUT RULES:
- Return NDJSON ONLY: exactly 8 lines, no markdown, no extra prose.
- For narrative_only=true, you MUST provide valid compute_only_code that sets METRICS.
- For narrative_only=false, you MUST provide valid code that:
  (1) creates exactly ONE figure, (2) sets METRICS used in description_template, and
  (3) ends with plt.tight_layout(); plt.show()
- Never print() anything.

CODE CONSTRAINTS (apply to compute_only_code and code):
- Use ONLY pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns). NO other libraries. NO file I/O.
- A DataFrame named df is already provided.
- Always start with: df.columns = df.columns.str.strip()
- Convert numeric columns you use: df[col] = pd.to_numeric(df[col], errors='coerce')
- Always set METRICS = {...} with every key referenced in description_template.
- If data is insufficient, draw a placeholder figure (still one figure):
  plt.figure(figsize=(8,5)); plt.text(0.5,0.5,'Insufficient data for <title>', ha='center'); plt.axis('off'); plt.tight_layout(); plt.show()
- Avoid custom colors; rely on seaborn defaults. Prefer horizontal bars for long labels.
- Keep description placeholders minimal (2–4 keys). Add a short transition sentence like “From the figure above, we observe …” where appropriate.

DATA-QUALITY GUIDELINES (follow the spirit precisely):
[Authors split]
- Treat 'AuthorNames-Deduped' as semicolon-separated; split by ';', explode, strip spaces/quotes; ignore empty or literal 'nan' tokens.

[Keywords cleaning]
- 'AuthorKeywords' is comma-separated; drop missing first; treat literal 'nan' (any case) as missing; split by ','; for each token strip quotes/whitespace, collapse multiple spaces to one, lowercase; drop empty tokens before counting.

[Awards]
- A row is “awarded” if 'Award' is non-empty after strip; count per numeric Year.

[Relationships specifics]
- Citations vs Downloads: use 'CitationCount_CrossRef' and 'Downloads_Xplore'; drop NaNs; compute Spearman with pandas.
- Award Share by Conference (Dot Plot): share = awarded/total; optionally filter total>=3; sort by share desc; top ≤10; use sns.scatterplot with x=share, y=index, size=total, sizes=(50,500), legend='brief'; set k_used, top_share_conf, top_share.

METRICS CLARITY:
- k_used = number of items actually shown (≤10), NOT a percentage.
- Use integers for counts; cast years to integers where possible.
"""

def build_system_prompt(dataset_info: str) -> str:
    return f"""{NDJSON_SPEC}

Dataset context (columns and a sample row):
{dataset_info}
"""

def generate_msg(state: State):
    sys_prompt = build_system_prompt(state["dataset_info"])
    llm = get_llm(temperature=0, max_tokens=3000)
    response = llm.invoke(
        [SystemMessage(content=sys_prompt),
         HumanMessage(content="Produce EXACTLY 8 lines of NDJSON now. No extra text.")]
    )
    raw = getattr(response, "content", response)
    try:
        with open("last_llm_output.txt", "w", encoding="utf-8") as f:
            f.write(str(raw))
    except Exception:
        pass
    return {"message": raw}

def create_workflow():
    g = StateGraph(State)
    g.add_node("generate_msg", generate_msg)
    g.add_edge(START, "generate_msg")
    g.add_edge("generate_msg", END)
    return g.compile()

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()

    def initialize_state_from_csv(self) -> dict:
        path = "./dataset.csv"
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            try:
                first_row = next(reader)
            except StopIteration:
                first_row = [""] * len(header)
        attributes = ", ".join(header)
        example_values = "\t".join(first_row)
        example_input = f"""
There is a dataset in CSV.
It has {len(header)} columns: {attributes}
First row example:
{example_values}
File name: dataset.csv
"""
        return {"dataset_info": example_input}

    def decode_output(self, output: dict):
        generate_pdf_report({"message": output.get("message", "")}, "output.pdf")

    def process(self):
        if not self.workflow:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        state = self.initialize_state_from_csv()
        out = self.workflow.invoke(state)
        print("NDJSON received. Writing PDF…")
        self.decode_output(out)
        return out