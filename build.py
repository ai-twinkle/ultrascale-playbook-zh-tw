#!/usr/bin/env python3
"""Ultra-Scale Playbook 中文版建置腳本：
content/*.md → webapp/chapters/*.html + webapp/index.html
重新執行即可整站重建。
"""
import html
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(ROOT, "content")
WEBAPP = os.path.join(ROOT, "webapp")
CHAPDIR = os.path.join(WEBAPP, "chapters")

CHAPTERS = [
    dict(id="ch00", no="導論", zh="導論與全書鳥瞰", en="Introduction & High-Level Overview",
         desc="為什麼要學大規模訓練、全書地圖與速查表。"),
    dict(id="ch01", no="第 1 章", zh="第一步：單 GPU 訓練", en="First Steps: Training on one GPU",
         desc="Transformer 記憶體剖析、活化重算與梯度累積。"),
    dict(id="ch02", no="第 2 章", zh="資料平行", en="Data Parallelism",
         desc="DP 通訊與計算重疊、全域批次大小，與 ZeRO 1/2/3 分片。"),
    dict(id="ch03", no="第 3 章", zh="張量平行", en="Tensor Parallelism",
         desc="行/列切分線性層、Transformer 區塊的 TP 與序列平行。"),
    dict(id="ch04", no="第 4 章", zh="上下文平行", en="Context Parallelism",
         desc="超長序列的活化記憶體、Ring Attention 與 Zig-Zag 平衡。"),
    dict(id="ch05", no="第 5 章", zh="管線平行", en="Pipeline Parallelism",
         desc="AFAB、1F1B、交錯階段到 Zero Bubble 與 DualPipe。"),
    dict(id="ch06", no="第 6 章", zh="專家平行", en="Expert Parallelism",
         desc="MoE 模型的專家分片與路由。"),
    dict(id="ch07", no="第 7 章", zh="5D 平行速覽", en="5D Parallelism in a Nutshell",
         desc="DP／TP／SP／CP／PP／EP 全家族的統一視角與組合原則。"),
    dict(id="ch08", no="第 8 章", zh="尋找最佳訓練配置", en="Finding the Best Training Configuration",
         desc="三步驟：塞進記憶體、達成目標批次、最大化吞吐；數千組實測的教訓。"),
    dict(id="ch09", no="第 9 章", zh="深入 GPU：架構與核心", en="Diving in the GPUs: Architecture & Kernels",
         desc="GPU 架構速成、執行緒與記憶體階層、寫出高效 kernel。"),
    dict(id="ch10", no="第 10 章", zh="融合核心、Flash Attention 與混合精度", en="Fused Kernels, Flash Attention & Mixed Precision",
         desc="kernel 融合、FA1-3 的記憶體魔法，與 FP32/BF16/FP8 訓練。"),
    dict(id="ch11", no="結語", zh="結語與展望", en="Conclusion",
         desc="全書回顧、下一步，與致謝。"),
    dict(id="appa", no="附錄 A0", zh="平行程式設計速成", en="A0: Parallel Programming Crash Course",
         desc="broadcast、all-reduce、scatter/gather 等集體通訊操作。"),
    dict(id="appb", no="附錄 A1", zh="分散式訓練效能分析", en="A1: Distributed Training Profiling",
         desc="用 profiler 看懂 kernel 與通訊的時間軸。"),
    dict(id="appc", no="附錄 A2", zh="LLM 訓練的典型尺度", en="A2: Typical Scales in LLM Training",
         desc="模型、批次、叢集規模的數量級速查。"),
    dict(id="appd", no="附錄 A3", zh="計算／通訊重疊的數學", en="A3: Math for Compute/Communication Overlap",
         desc="各平行策略下重疊條件的推導。"),
    dict(id="references", no="參考資源", zh="參考資源", en="References",
         desc="里程碑論文、訓練框架、除錯與硬體資源的註解書單。"),
]

PAGE_TMPL = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Ultra-Scale Playbook 中文版</title>
<link rel="stylesheet" href="../assets/katex/katex.min.css">
<link rel="stylesheet" href="../assets/style.css">
</head>
<body>
<header class="topbar">
  <a class="brand" href="../index.html">Ultra-Scale Playbook 中文版<span>在 GPU 叢集上訓練 LLM</span></a>
  <nav class="chapnav">{prev_top}<span class="chapnav-current">{no}</span>{next_top}</nav>
</header>
<div class="layout">
  <aside class="sidebar"><nav id="toc" class="toc"><h2>本章目錄</h2></nav></aside>
  <main class="main">
    <div id="lab-banner"></div>
    <article id="content" class="prose"></article>
    <section id="lab" class="lab" hidden>
      <h2 class="lab-title">🧪 互動實驗室</h2>
      <div id="lab-intro"></div>
      <div id="lab-root"></div>
    </section>
    <nav class="pager">{prev_card}{next_card}</nav>
    <footer class="pagefoot">本站為 <a href="https://github.com/ai-twinkle">Twinkle AI Community</a>（台灣）的<strong>非官方社群翻譯</strong> · 譯自 Hugging Face nanotron 團隊《<a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook</a>》· 原作依 Apache 2.0 釋出，譯文同授權，僅供學習研究使用。</footer>
  </main>
</div>
<script type="text/markdown" id="chapter-md">
{md}
</script>
<script src="../assets/marked.min.js"></script>
<script src="../assets/katex/katex.min.js"></script>
<script src="../assets/katex/auto-render.min.js"></script>
{widget_tag}<script src="../assets/app.js"></script>
</body>
</html>
"""

INDEX_TMPL = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ultra-Scale Playbook 中文版 — 在 GPU 叢集上訓練 LLM</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<section class="hero">
  <div class="eyebrow">繁體中文全譯本 · 互動版</div>
  <h1>超大規模訓練實戰手冊<br>The Ultra-Scale Playbook</h1>
  <p class="sub">Hugging Face nanotron 團隊著（Nouamane Tazi、Ferdinand Mom、Haojun Zhao、Phuc Nguyen、Mohamed Mekkouri、Leandro von Werra、Thomas Wolf）。從單 GPU 的記憶體剖析出發，一路走到資料／張量／上下文／管線／專家五維平行、GPU kernel 最佳化與混合精度——在 GPU 叢集上訓練大型語言模型所需的一切。每章附互動實驗。</p>
  <p class="sub" style="font-size:.82rem">由台灣 <a href="https://github.com/ai-twinkle">Twinkle AI Community</a> 翻譯維護的非官方社群翻譯版本。</p>
  <div class="meta">
    <div><b>{n_ch}</b>章節</div>
    <div><b>{n_app}</b>附錄</div>
    <div><b>{n_fig}</b>插圖</div>
    <div><b>{n_lab}</b>互動實驗</div>
  </div>
</section>
<div class="grid-wrap">
  <h2>章節</h2>
  <div class="chapter-grid">
{chapter_cards}
  </div>
  <h2>附錄與資源</h2>
  <div class="chapter-grid">
{appendix_cards}
  </div>
</div>
<footer class="foot">本站為 <a href="https://github.com/ai-twinkle">Twinkle AI Community</a>（台灣）的<strong>非官方社群翻譯</strong>（unofficial community translation）· 譯自 Hugging Face nanotron 團隊《<a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook: Training LLMs on GPU Clusters</a>》· 原作依 Apache 2.0 釋出，譯文同授權 · 僅供學習研究使用。</footer>
</body>
</html>
"""


def build_pages():
    os.makedirs(CHAPDIR, exist_ok=True)
    built = {c["id"]: os.path.exists(os.path.join(CONTENT, c["id"] + ".md")) for c in CHAPTERS}
    n_built = 0
    for i, ch in enumerate(CHAPTERS):
        if not built[ch["id"]]:
            continue
        with open(os.path.join(CONTENT, ch["id"] + ".md")) as f:
            md = f.read().replace("</script", "<\\/script")
        prev = next((c for c in reversed(CHAPTERS[:i]) if built[c["id"]]), None)
        nxt = next((c for c in CHAPTERS[i + 1:] if built[c["id"]]), None)
        prev_top = f'<a href="{prev["id"]}.html">← {prev["no"]}</a>' if prev else ""
        next_top = f'<a href="{nxt["id"]}.html">{nxt["no"]} →</a>' if nxt else ""
        prev_card = (f'<a class="prev" href="{prev["id"]}.html"><span class="dir">← 上一章</span>'
                     f'<div class="ttl">{prev["no"]}　{prev["zh"]}</div></a>') if prev else "<span></span>"
        next_card = (f'<a class="next" href="{nxt["id"]}.html"><span class="dir">下一章 →</span>'
                     f'<div class="ttl">{nxt["no"]}　{nxt["zh"]}</div></a>') if nxt else ""
        widget = os.path.join(WEBAPP, "assets", "widgets", ch["id"] + ".js")
        widget_tag = (f'<script src="../assets/widgets/{ch["id"]}.js"></script>\n'
                      if os.path.exists(widget) else "")
        page = PAGE_TMPL.format(
            title=f'{ch["no"]}　{ch["zh"]}', no=f'{ch["no"]}　{ch["zh"]}',
            prev_top=prev_top, next_top=next_top,
            prev_card=prev_card, next_card=next_card,
            md=md, widget_tag=widget_tag)
        with open(os.path.join(CHAPDIR, ch["id"] + ".html"), "w") as f:
            f.write(page)
        n_built += 1
    return built, n_built


def build_index(built):
    def card(ch):
        has_widget = os.path.exists(os.path.join(WEBAPP, "assets", "widgets", ch["id"] + ".js"))
        badge = '<span class="badge">互動實驗</span>' if has_widget else ""
        inner = (f'{badge}<span class="no">{html.escape(ch["no"])}</span>'
                 f'<span class="zh">{html.escape(ch["zh"])}</span>'
                 f'<span class="en">{html.escape(ch["en"])}</span>'
                 f'<span class="desc">{html.escape(ch["desc"])}</span>')
        if built.get(ch["id"]):
            return f'    <a class="card" href="chapters/{ch["id"]}.html">{inner}</a>'
        return f'    <div class="card disabled">{inner}<span class="badge">翻譯中</span></div>'

    main_ids = [c for c in CHAPTERS if c["id"].startswith("ch")]
    app_ids = [c for c in CHAPTERS if not c["id"].startswith("ch")]
    ch_cards = "\n".join(card(c) for c in main_ids)
    app_cards = "\n".join(card(c) for c in app_ids)
    figdir = os.path.join(WEBAPP, "assets", "images")
    n_fig = len(os.listdir(figdir)) if os.path.isdir(figdir) else 0
    widgets = os.path.join(WEBAPP, "assets", "widgets")
    n_lab = len([f for f in os.listdir(widgets) if f.endswith(".js")]) if os.path.isdir(widgets) else 0
    page = INDEX_TMPL.format(n_ch=12, n_app=4, n_fig=n_fig, n_lab=n_lab,
                             chapter_cards=ch_cards, appendix_cards=app_cards)
    with open(os.path.join(WEBAPP, "index.html"), "w") as f:
        f.write(page)


if __name__ == "__main__":
    built, n = build_pages()
    build_index(built)
    missing = [k for k, v in built.items() if not v]
    print(f"built {n} chapter pages; missing: {', '.join(missing) if missing else 'none'}")
