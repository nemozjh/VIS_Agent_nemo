# report_pdf.py
import io, os, json, textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

# ---------- Theme ----------
def _apply_global_plot_theme():
    sns.set_theme(context="notebook", style="whitegrid", palette="Set2", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.figsize": (9, 4.8),
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": False,
    })
_apply_global_plot_theme()

# ---------- Canvas with page numbers ----------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        self.report_title = kwargs.pop("report_title", "")
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved_page_states)
        for st in self._saved_page_states:
            self.__dict__.update(st)
            self._header_footer(n)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _header_footer(self, page_count):
        w, h = letter
        if self.report_title:
            self.setFont("Helvetica", 9)
            self.setFillColor(colors.grey)
            self.drawCentredString(w/2.0, h-0.5*inch, self.report_title[:120])
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.grey)
        self.drawCentredString(w/2.0, 0.4*inch, f"Page {self._pageNumber} of {page_count}")
        self.setFillColor(colors.black)

# ---------- Styles ----------
def build_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["Title"] = ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold",
                                     fontSize=20, alignment=TA_LEFT, spaceAfter=8, leading=24)
    styles["SubTitle"] = ParagraphStyle("SubTitle", parent=base["Normal"], fontName="Helvetica",
                                        fontSize=10.5, alignment=TA_LEFT, textColor=colors.grey, spaceAfter=14)
    styles["H1"] = ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold",
                                  fontSize=16, alignment=TA_LEFT, spaceBefore=12, spaceAfter=8, leading=20)
    styles["H2"] = ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold",
                                  fontSize=13, alignment=TA_LEFT, spaceBefore=8, spaceAfter=4, leading=16)
    styles["Body"] = ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica",
                                    fontSize=10.5, alignment=TA_JUSTIFY, leading=14, spaceAfter=6)
    styles["Caption"] = ParagraphStyle("Caption", parent=base["BodyText"], fontName="Helvetica-Oblique",
                                       fontSize=9, textColor=colors.grey, alignment=TA_CENTER,
                                       spaceBefore=2, spaceAfter=8, leading=12)
    styles["Small"] = ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica",
                                     fontSize=9, textColor=colors.grey, spaceAfter=4, leading=12)
    return styles

# ---------- Helpers ----------
def dedent_code(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return textwrap.dedent(code).strip("\n")

def capture_figs(max_w=6.3*inch, max_h=4.0*inch):
    flow = []
    for num in list(plt.get_fignums()):
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
        buf.seek(0)
        plt.close(fig)
        img = Image(buf)
        iw, ih = img.imageWidth, img.imageHeight
        ratio = min(max_w/iw, max_h/ih)
        img.drawWidth = iw*ratio
        img.drawHeight = ih*ratio
        flow += [img, Spacer(1, 0.12*inch)]
    return flow

def safe_format(tmpl: str, metrics: dict) -> str:
    from string import Formatter
    class _Missing: pass
    class _Fmt(Formatter):
        def get_value(self, key, args, kwargs):
            if isinstance(key, str) and (key in kwargs): return kwargs[key]
            return _Missing()
        def format_field(self, val, fmt):
            needs_num = any(ch in fmt for ch in "eEfFgG%")
            if isinstance(val, _Missing): val = 0 if needs_num else ""
            if needs_num and isinstance(val, str):
                try: val = float(val)
                except: val = 0
            try: return super().format_field(val, fmt)
            except: return str(val) if not needs_num else "0"
    return _Fmt().vformat(tmpl or "", (), metrics or {})

def parse_ndjson(text: str):
    items = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            try:
                items.append(json.loads(s))
            except:
                pass
    return items

# ---------- Pre-clean helpers ----------
def _preclean_keywords_column(df: pd.DataFrame) -> pd.DataFrame:
    """Robustly sanitize AuthorKeywords **before** LLM code runs, to avoid blank bars."""
    if "AuthorKeywords" not in df.columns:
        return df
    s = df["AuthorKeywords"].astype(str)

    # Normalize obvious "nan"/"none"/empty to NaN
    mask_nan_literal = s.str.strip().str.lower().isin(["", "nan", "none", "null"])
    s = s.mask(mask_nan_literal, np.nan)

    # Remove quotes; normalize whitespace; trim ends
    s = (s
         .str.replace(r"[\"']", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip())

    # Normalize comma separators; trim leading/trailing commas
    s = (s
         .str.replace(r"\s*,\s*", ",", regex=True)
         .str.replace(r"(,){2,}", ",", regex=True)
         .str.replace(r"^,|,$", "", regex=True))

    s = s.replace("", np.nan)

    df = df.copy()
    df["AuthorKeywords"] = s
    return df

# ---------- Main ----------
def generate_pdf_report(output_state: dict, output_path: str):
    raw = output_state.get("message", "") or ""
    if not isinstance(raw, str):
        raw = str(raw)
    nditems = parse_ndjson(raw)

    styles = build_styles()
    report_title = "Structured AI Agent–Generated Visualization Report"
    subtitle_text = ("This report was automatically generated by an AI agent. "
                     "All quantitative values in the narrative are computed from code, not inferred.")

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.8*inch, bottomMargin=0.7*inch,
        leftMargin=0.8*inch, rightMargin=0.8*inch,
        title=report_title, author="Agent"
    )
    story = []

    # 顶部标题 + 简要说明
    story.append(Paragraph(report_title, styles["Title"]))
    story.append(Paragraph(subtitle_text, styles["SubTitle"]))
    story.append(Spacer(1, 0.15*inch))

    # 预加载 df + 关键词预清洗
    try:
        shared_df = pd.read_csv("dataset.csv")
        shared_df = _preclean_keywords_column(shared_df)
    except Exception:
        shared_df = None

    # 将 NDJSON 分组为章节
    sections_map = {}
    for obj in nditems:
        sec = (obj.get("section") or "Section").strip()
        sections_map.setdefault(sec, []).append(obj)

    # 固定章节顺序；未列出的其余章节按出现顺序追加
    preferred_order = [
        "Overview",
        "Temporal Trends",
        "Top Entities",
        "Cross-Metric Relationships",
    ]
    ordered_sections = []
    seen = set()
    for s in preferred_order:
        if s in sections_map:
            ordered_sections.append((s, sections_map[s]))
            seen.add(s)
    for s, items in sections_map.items():
        if s not in seen:
            ordered_sections.append((s, items))

    fig_counter = 1

    # 逐章输出
    for sec_name, items in ordered_sections:
        story.append(Paragraph(sec_name, styles["H1"]))
        story.append(Spacer(1, 0.06*inch))

        for obj in items:
            title2 = obj.get("title") or "Untitled"
            goal = obj.get("goal") or ""
            desc_tmpl = obj.get("description_template") or obj.get("description") or ""
            narrative_only = bool(obj.get("narrative_only"))
            compute_only_code = obj.get("compute_only_code") or ""
            code = obj.get("code") or ""

            # 小节标题 + 目标
            story.append(Paragraph(title2, styles["H2"]))
            if goal:
                story.append(Paragraph(goal, styles["Body"]))

            METRICS = {}
            ns = {"pd": pd, "np": np, "plt": plt, "sns": sns, "METRICS": METRICS}
            if shared_df is not None:
                ns["df"] = shared_df.copy()
            # 捕获图像：我们自己处理
            plt.show = lambda *a, **k: None

            try:
                cleaned = dedent_code(compute_only_code if narrative_only else code)
                if cleaned:
                    exec(cleaned, ns)
            except Exception as e:
                story.append(Paragraph(f"<font color='red'>Failed to execute item: {e}</font>", styles["Body"]))
                story.append(Spacer(1, 0.12*inch))
                continue

            # 非 narrative_only：插入图像，并加【编号 + 标题】图注
            if not narrative_only:
                figs = capture_figs()
                for img in figs:
                    story.append(img)
                    if isinstance(img, Image):
                        cap_title = title2 if title2 else "Untitled"
                        story.append(Paragraph(f"Figure {fig_counter}. {cap_title}", styles["Caption"]))
                        fig_counter += 1

            # 描述（基于 METRICS 安全格式化）
            if desc_tmpl:
                story.append(Paragraph(safe_format(desc_tmpl, ns.get("METRICS", {})), styles["Body"]))

            story.append(Spacer(1, 0.18*inch))

    doc.build(story, canvasmaker=lambda *a, **k: NumberedCanvas(*a, report_title=report_title, **k))