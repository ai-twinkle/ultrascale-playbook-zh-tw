/* 第 3 章：張量平行 —— 切分視覺化互動元件 */
(function () {
  'use strict';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var GPU = ['var(--accent)', 'var(--accent-2)', 'var(--link)', 'var(--fg-muted)'];

  function h(tag, attrs, parent, text) {
    var e = document.createElement(tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function s(tag, attrs, parent, text) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function tex(el, t) {
    if (window.katex) window.katex.render(t, el, { throwOnError: false });
    else el.textContent = t;
  }
  function arrow(p, x1, y1, x2, y2, o) {
    o = o || {};
    var a = { d: 'M' + x1 + ',' + y1 + ' L' + x2 + ',' + y2, fill: 'none', stroke: o.stroke || 'var(--fg-muted)', 'stroke-width': o.w || 1.5 };
    if (o.marker) a['marker-end'] = 'url(#' + o.marker + ')';
    if (o.cls) a['class'] = o.cls;
    if (o.dash) a['stroke-dasharray'] = o.dash;
    if (o.op != null) a.opacity = o.op;
    return s('path', a, p);
  }
  function markerDef(svg, id) {
    var mk = s('marker', { id: id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' }, s('defs', {}, svg));
    s('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'var(--fg-muted)' }, mk);
  }
  // 畫一個可切分的矩陣塊；n=0 表示不切分，dir='v' 沿寬切、'h' 沿高切
  function shards(parent, x, y, w, ht, n, dir, op) {
    var g = s('g', {}, parent);
    s('rect', { x: x, y: y, width: w, height: ht, rx: 3, fill: 'var(--code-bg)', stroke: 'var(--border)' }, g);
    for (var i = 0; i < n; i++) {
      var a = dir === 'v' ? { x: x + (w / n) * i + 1.5, y: y + 1.5, width: w / n - 3, height: ht - 3 }
                          : { x: x + 1.5, y: y + (ht / n) * i + 1.5, width: w - 3, height: ht / n - 3 };
      a.rx = 2; a.fill = GPU[i]; a['fill-opacity'] = op || 0.55;
      s('rect', a, g);
    }
    return g;
  }
  function mini(p, x, y, w, ht, i, op) {
    return s('rect', { x: x, y: y, width: w, height: ht, rx: 2, fill: GPU[i], 'fill-opacity': op || 0.7, stroke: GPU[i] }, p);
  }

  /* ---------- 分頁 1：切矩陣 ---------- */
  var STEP_TXT = {
    col: ['模式：column-linear（行切分）。W 沿行（輸出維 n）切成 t 份色塊，常駐各 GPU。按「播放」或「下一步」開始。',
      '步驟 1／分發：X 以 broadcast 完整複製到每張 GPU（實際訓練中輸入通常已同步，可省略這次通訊）。',
      '步驟 2／本地矩陣乘：各 GPU 計算 Yᵢ = X·Wᵢ，得到 Y 沿 n 維的一段（m×n/t），完全不需通訊。',
      '步驟 3／合併：all-gather 把各段沿 n 維串接，還原完整 Y（若下一層接 row-linear，這步可以省略）。'],
    row: ['模式：row-linear（列切分）。W 沿列（輸入維 k）切成 t 份；X 也必須跟著沿 k 維切開。',
      '步驟 1／分發：X 以 scatter 沿內維 k 切開，GPU i 只拿到分片 Xᵢ（若上一層是 column-linear，輸入天然已是分片）。',
      '步驟 2／本地矩陣乘：各 GPU 計算 XᵢWᵢ，形狀已是完整的 m×n，但只是「部分和」，數值還不正確。',
      '步驟 3／合併：all-reduce 把各 GPU 的部分和逐元素加總，得到正確的完整 Y。'],
  };
  var EQ = {
    col: 'Y = X\\begin{bmatrix} W_1 & \\cdots & W_t \\end{bmatrix} = \\begin{bmatrix} XW_1 & \\cdots & XW_t \\end{bmatrix} \\;\\;(\\text{all-gather})',
    row: 'Y = \\begin{bmatrix} X_1 & \\cdots & X_t \\end{bmatrix}\\begin{bmatrix} W_1 \\\\ \\vdots \\\\ W_t \\end{bmatrix} = \\textstyle\\sum_{i=1}^{t} X_i W_i \\;\\;(\\text{all-reduce})',
  };

  function drawScene(svg, st) {
    var col = st.mode === 'col', tp = st.tp, step = st.step, i;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    markerDef(svg, 'ch03-arrA');
    s('text', { x: 95, y: 22, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--fg)' }, svg, 'X（輸入 m×k）');
    shards(svg, 50, 32, 90, 54, col ? 0 : tp, 'v');
    s('text', { x: 158, y: 68, 'font-size': 18, fill: 'var(--fg)' }, svg, '·');
    s('text', { x: 235, y: 22, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--fg)' }, svg, 'W（權重 k×n）');
    shards(svg, 180, 32, 110, 76, tp, col ? 'v' : 'h');
    if (step >= 1) s('text', { x: 95, y: 104, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--accent)' }, svg,
      col ? 'broadcast：複製 ×' + tp : 'scatter：切成 ' + tp + ' 份');
    var y0 = 150, ph = 78, gap = 14, pw = (680 - (tp - 1) * gap) / tp;
    for (i = 0; i < tp; i++) {
      var px = 20 + i * (pw + gap), cx = px + pw / 2, bx = px + 16, cy = y0 + 44;
      var pg = s('g', step === 2 ? { 'class': 'ch03-pulse' } : {}, svg);
      s('rect', { x: px, y: y0, width: pw, height: ph, rx: 8, fill: 'var(--panel)', stroke: GPU[i], 'stroke-width': step === 2 ? 2.5 : 1.2 }, pg);
      s('text', { x: px + 10, y: y0 + 16, 'font-size': 11, 'font-weight': 600, fill: GPU[i] }, pg, 'GPU ' + i);
      if (col) {
        shards(pg, bx, cy - 11, 26, 22, 0);
        s('text', { x: bx + 31, y: cy + 4, 'font-size': 12, fill: 'var(--fg)' }, pg, '·');
        mini(pg, bx + 40, cy - 15, 12, 30, i);
        s('text', { x: bx + 57, y: cy + 4, 'font-size': 12, fill: 'var(--fg)' }, pg, '=');
        mini(pg, bx + 68, cy - 11, 12, 22, i);
      } else {
        mini(pg, bx, cy - 11, 16, 22, i);
        s('text', { x: bx + 21, y: cy + 4, 'font-size': 12, fill: 'var(--fg)' }, pg, '·');
        mini(pg, bx + 30, cy - 6, 26, 12, i);
        s('text', { x: bx + 61, y: cy + 4, 'font-size': 12, fill: 'var(--fg)' }, pg, '=');
        s('rect', { x: bx + 72, y: cy - 11, width: 26, height: 22, rx: 2, fill: GPU[i], 'fill-opacity': 0.3, stroke: GPU[i], 'stroke-dasharray': '3 3' }, pg);
      }
      s('text', { x: cx, y: y0 + ph - 8, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--fg-muted)' }, pg,
        col ? 'Yᵢ = X·Wᵢ（m×n/t）' : 'XᵢWᵢ（m×n，部分和）');
      if (step >= 1) {
        var flow = step === 1 ? 'ch03-flow' : '', op1 = step === 1 ? 1 : 0.3;
        arrow(svg, col ? 95 : 50 + (90 / tp) * (i + 0.5), 90, cx - 8, y0 - 4, { marker: 'ch03-arrA', cls: flow, op: op1 });
        arrow(svg, col ? 180 + (110 / tp) * (i + 0.5) : 235 + (i - (tp - 1) / 2) * 9, 112, cx + 10, y0 - 4,
          { stroke: GPU[i], dash: '3 3', op: op1 });
      }
      if (step >= 3) arrow(svg, cx, y0 + ph + 4, col ? 300 + 120 * (i + 0.5) / tp : 360 + (i - (tp - 1) / 2) * 12, 250,
        { marker: 'ch03-arrA', cls: 'ch03-flow' });
    }
    s('text', { x: 292, y: 284, 'text-anchor': 'end', 'font-size': 12, fill: 'var(--fg)' }, svg, 'Y =');
    if (step >= 3) {
      if (col) shards(svg, 300, 254, 120, 52, tp, 'v');
      else {
        s('rect', { x: 300, y: 254, width: 120, height: 52, rx: 3, fill: 'var(--accent)', 'fill-opacity': 0.5, stroke: 'var(--border)' }, svg);
        s('text', { x: 360, y: 284, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--fg)' }, svg, 'Σᵢ XᵢWᵢ');
      }
      s('text', { x: 432, y: 284, 'font-size': 11, fill: 'var(--accent)' }, svg, col ? 'all-gather：沿 n 維串接' : 'all-reduce：逐元素加總');
    } else {
      s('rect', { x: 300, y: 254, width: 120, height: 52, rx: 3, fill: 'none', stroke: 'var(--border)', 'stroke-dasharray': '4 4' }, svg);
      s('text', { x: 360, y: 284, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--fg-muted)' }, svg, '（待合併）');
    }
  }

  function renderTab1(panel) {
    var st = { mode: 'col', tp: 2, step: 0, timer: null };
    var row = h('div', { 'class': 'widget-row' }, panel);
    var modeSel = h('select', { 'aria-label': '切分模式' }, row);
    h('option', { value: 'col' }, modeSel, 'column-linear（沿行切 W）');
    h('option', { value: 'row' }, modeSel, 'row-linear（沿列切 W）');
    var tpSel = h('select', { 'aria-label': 'TP 度數' }, row);
    h('option', { value: '2' }, tpSel, 'TP = 2');
    h('option', { value: '4' }, tpSel, 'TP = 4');
    var playBtn = h('button', {}, row, '▶ 播放三步驟');
    var stepBtn = h('button', {}, row, '下一步');
    var svg = s('svg', { viewBox: '0 0 720 312', width: '100%', role: 'img', 'aria-label': '矩陣切分示意圖' }, panel);
    var cap = h('div', { 'class': 'ch03-cap' }, panel);
    var eq = h('div', { style: 'margin-top:.4rem;overflow-x:auto;' }, panel);
    function stop() { if (st.timer) { clearInterval(st.timer); st.timer = null; } playBtn.textContent = '▶ 播放三步驟'; }
    function update() { drawScene(svg, st); cap.textContent = STEP_TXT[st.mode][st.step]; tex(eq, EQ[st.mode]); }
    modeSel.addEventListener('change', function () { stop(); st.mode = modeSel.value; st.step = 0; update(); });
    tpSel.addEventListener('change', function () { stop(); st.tp = +tpSel.value; st.step = 0; update(); });
    stepBtn.addEventListener('click', function () { stop(); st.step = (st.step + 1) % 4; update(); });
    playBtn.addEventListener('click', function () {
      if (st.timer) { stop(); return; }
      st.step = 0; update(); playBtn.textContent = '⏸ 暫停';
      st.timer = setInterval(function () { st.step++; update(); if (st.step >= 3) stop(); }, 1400);
    });
    update();
  }

  /* ---------- 分頁 2：一層 Transformer 的 TP+SP ---------- */
  var INFO = {
    f: 'f（TP 過渡）：前向 = identity（no-op，輸入已在各 TP rank 複製）；反向 = all-reduce（同步輸入梯度）。f 與 f* 互為共軛。',
    fs: 'f*（TP 過渡）：前向 = all-reduce（加總 row-linear 的部分和以確保正確性）；反向 = identity（no-op）。這次 all-reduce 落在關鍵路徑上，難以與計算重疊。',
    g: 'g（SP→TP 過渡）：前向 = all-gather（沿 seq 維把各卡的 s/t 片段拼回完整序列，供 column-linear 使用）；反向 = reduce-scatter。g 與 g* 互為共軛。',
    gs: 'g*（TP→SP 過渡）：前向 = reduce-scatter（一邊完成 row-linear 部分和的加總、一邊沿 seq 維切分）；反向 = all-gather。避免了會推高峰值記憶體的 all-reduce。',
    ln: 'LayerNorm 需要完整 hidden 維度 h 才能計算平均值與變異數，因此不能沿 h 切分；SP 改沿 seq 維切，讓這一段的活化也降為 1/t。',
    col: 'column-linear：權重沿輸出維切分——Attention 的 Q/K/V 投影（每卡負責部分 head）、MLP 的第一層。輸出的 h 維被分片為 h/t。',
    row: 'row-linear：權重沿輸入維切分——Attention 的輸出投影、MLP 的第二層。輸出形狀完整（h），但只是部分和，需 f*／g* 完成加總。',
    drop: 'Dropout：純 TP 下各卡看到相同的完整活化，必須同步隨機種子；TP+SP 下各卡處理不同的 seq 片段，活化降為 1/t。',
  };
  function drawFlow(svg, sp, setInfo) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    markerDef(svg, 'ch03-arrB');
    var bands = [[12, 195, !sp], [207, 357, true], [564, 184, !sp]];
    bands.forEach(function (b, bi) {
      var mid = bi === 1;
      s('rect', { x: b[0], y: 16, width: b[1], height: 130, rx: 8, fill: mid ? 'var(--accent)' : (sp ? 'var(--accent-2)' : 'var(--code-bg)'), 'fill-opacity': mid || sp ? 0.1 : 1 }, svg);
      s('text', { x: b[0] + b[1] / 2, y: 34, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 600, fill: mid ? 'var(--accent)' : (sp ? 'var(--accent-2)' : 'var(--fg-muted)') }, svg,
        mid ? 'TP 區（沿 hidden 切分）' : (sp ? 'SP 區（沿 seq 切分）' : '未切分（完整活化）'));
    });
    [[108, 191], [222, 246], [378, 400], [532, 549], [579, 602], [688, 722]].forEach(function (a) {
      arrow(svg, a[0], 86, a[1], 86, { marker: 'ch03-arrB' });
    });
    s('text', { x: 726, y: 90, 'font-size': 10, fill: 'var(--fg-muted)' }, svg, '下一子層');
    var boxes = [
      { x: 22, w: 86, id: 'ln', l1: 'LayerNorm', shape: sp ? '(b, s/t, h)' : '(b, s, h)' },
      { x: 246, w: 132, id: 'col', l1: 'Attn／MLP 前半', l2: 'column-linear', shape: '(b, s, h/t)' },
      { x: 400, w: 132, id: 'row', l1: 'Attn／MLP 後半', l2: 'row-linear', shape: '(b, s, h) 部分和' },
      { x: 602, w: 86, id: 'drop', l1: 'Dropout', shape: sp ? '(b, s/t, h)' : '(b, s, h)' },
    ];
    boxes.forEach(function (b) {
      var g = s('g', { 'class': 'ch03-op' }, svg);
      s('rect', { x: b.x, y: 64, width: b.w, height: 44, rx: 6, fill: 'var(--panel)', stroke: 'var(--border)' }, g);
      s('text', { x: b.x + b.w / 2, y: b.l2 ? 83 : 90, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--fg)' }, g, b.l1);
      if (b.l2) s('text', { x: b.x + b.w / 2, y: 98, 'text-anchor': 'middle', 'font-size': 9.5, fill: 'var(--fg-muted)' }, g, b.l2);
      s('text', { x: b.x + b.w / 2, y: 132, 'text-anchor': 'middle', 'font-size': 10.5, fill: 'var(--fg-muted)' }, svg, b.shape);
      hook(g, b.id, setInfo);
    });
    [[207, sp ? 'g' : 'f'], [564, sp ? 'gs' : 'fs']].forEach(function (o) {
      var g = s('g', { 'class': 'ch03-op' }, svg);
      s('circle', { cx: o[0], cy: 86, r: 15, fill: 'var(--bg)', stroke: 'var(--accent)', 'stroke-width': 1.8 }, g);
      s('text', { x: o[0], y: 91, 'text-anchor': 'middle', 'font-size': 12, 'font-style': 'italic', fill: 'var(--accent)' }, g,
        o[1] === 'g' ? 'g' : o[1] === 'gs' ? 'g*' : o[1] === 'f' ? 'f' : 'f*');
      hook(g, o[1], setInfo);
    });
  }
  function hook(g, id, setInfo) {
    g.addEventListener('pointerenter', function () { setInfo(id); });
    g.addEventListener('pointerleave', function () { setInfo(null); });
    g.addEventListener('click', function () { setInfo(id); });
  }
  function renderTab2(panel) {
    var sp = false;
    var row = h('div', { 'class': 'widget-row ch03-seg' }, panel);
    var bTP = h('button', { 'aria-pressed': 'true' }, row, '純 TP（f／f*）');
    var bSP = h('button', { 'aria-pressed': 'false' }, row, 'TP + SP（g／g*）');
    var svg = s('svg', { viewBox: '0 0 760 152', width: '100%', role: 'img', 'aria-label': 'Transformer 子層的 TP 與 SP 流程圖' }, panel);
    var info = h('div', { 'class': 'ch03-info' }, panel);
    var DEF = '將滑鼠移到（或點擊）f／g 圓圈與各方塊，查看該操作的前向／反向通訊原語與形狀變化。';
    function setInfo(id) { info.textContent = INFO[id] || DEF; }
    function draw() {
      bTP.setAttribute('aria-pressed', String(!sp));
      bSP.setAttribute('aria-pressed', String(sp));
      drawFlow(svg, sp, setInfo);
      setInfo(null);
    }
    bTP.addEventListener('click', function () { sp = false; draw(); });
    bSP.addEventListener('click', function () { sp = true; draw(); });
    h('div', { 'class': 'ch03-cap' }, panel,
      '「column 再 row」讓 MLP／Attention 內部免去中間通訊。純 TP 前向每層 2 次 all-reduce；TP+SP 為 2 次 all-gather＋2 次 reduce-scatter——因 all-reduce ≡ all-gather＋reduce-scatter，總通訊量等價，卻讓 SP 區的活化也降為 1/t。前向與反向互為共軛：no-op ↔ all-reduce、all-gather ↔ reduce-scatter。');
    draw();
  }

  /* ---------- 動態解讀 ---------- */
  function renderInsight(root) {
    var panel = h('div', { 'class': 'widget-panel', style: 'margin-top:1rem;' }, root);
    h('div', { style: 'font-weight:600;margin-bottom:.5rem;' }, panel, '動態解讀：拉高 TP 度數會發生什麼事？');
    var row = h('div', { 'class': 'widget-row' }, panel);
    h('span', {}, row, 'TP 度數');
    var rng = h('input', { type: 'range', min: 0, max: 3, step: 1, value: 1, 'aria-label': 'TP 度數' }, row);
    var val = h('strong', {}, row, '');
    function bar(label) {
      var r = h('div', { 'class': 'widget-row', style: 'margin-top:.5rem;flex-wrap:nowrap;' }, panel);
      h('span', { style: 'flex:0 0 11em;font-size:.9em;' }, r, label);
      var t = h('div', { 'class': 'ch03-bar' }, r);
      var f = h('i', {}, t);
      var pct = h('span', { style: 'flex:0 0 4.5em;font-size:.85em;color:var(--fg-muted);' }, r, '');
      return { f: f, pct: pct };
    }
    var bw = bar('每卡權重／梯度／優化器狀態');
    var ba = bar('每卡活化（TP 區；搭配 SP 後全段）');
    var msg = h('div', { 'class': 'ch03-cap' }, panel);
    var warn = h('div', { 'class': 'ch03-info', style: 'margin-top:.4rem;' }, panel);
    function upd() {
      var t = [2, 4, 8, 16][+rng.value];
      val.textContent = 'TP = ' + t;
      bw.f.style.width = 100 / t + '%'; bw.pct.textContent = '1/' + t + ' ≈ ' + (100 / t).toFixed(1) + '%';
      ba.f.style.width = 100 / t + '%'; ba.pct.textContent = '1/' + t + ' ≈ ' + (100 / t).toFixed(1) + '%';
      msg.textContent = 'TP=' + t + '：每卡權重與活化都降為 1/' + t + '，但每個子層前向仍有 2 次通訊（純 TP：all-reduce ×2；TP+SP：all-gather ×2＋reduce-scatter ×2）落在關鍵路徑上，無法完全與計算重疊——TP 越高，暴露的通訊佔比越重。';
      warn.textContent = t > 8
        ? '⚠ TP = ' + t + ' 已超出單一節點的 8 張 GPU：通訊必須跨節點走較慢的網路（而非節點內 NVLink），吞吐量會大幅下滑。經驗法則：TP 不出節點（TP ≤ 8）。'
        : '✓ TP = ' + t + ' ≤ 8：通訊留在節點內高速 NVLink 上，開銷相對可控。';
      warn.style.borderLeft = '3px solid ' + (t > 8 ? 'var(--accent-2)' : 'var(--accent)');
    }
    rng.addEventListener('input', upd);
    upd();
  }

  var CSS = '.ch03-tabs{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}' +
    '.ch03-tabs button[aria-selected="true"],.ch03-seg button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);font-weight:600}' +
    '.ch03-flow{stroke-dasharray:6 5;animation:ch03-dash .7s linear infinite}' +
    '@keyframes ch03-dash{to{stroke-dashoffset:-11}}' +
    '.ch03-pulse{animation:ch03-pulse 1.1s ease-in-out infinite}' +
    '@keyframes ch03-pulse{50%{opacity:.55}}' +
    '.ch03-cap{margin-top:.6rem;color:var(--fg-muted);font-size:.92em;line-height:1.6;min-height:2.6em}' +
    '.ch03-info{margin-top:.5rem;padding:.5rem .75rem;background:var(--code-bg);border-radius:6px;font-size:.9em;line-height:1.6;min-height:3em}' +
    '.ch03-op{cursor:pointer}.ch03-op:hover rect,.ch03-op:hover circle{stroke:var(--accent-2)}' +
    '.ch03-bar{height:10px;border-radius:5px;background:var(--accent-soft);flex:1;min-width:60px;overflow:hidden}' +
    '.ch03-bar>i{display:block;height:100%;background:var(--accent);border-radius:5px;transition:width .3s}';

  window.ChapterWidget = {
    title: '張量平行切分視覺化',
    intro: '親手切一次矩陣：看 column-linear／row-linear 需要哪些通訊原語，再看一層 Transformer 在純 TP 與 TP+SP 下，f/f* 與 g/g* 過渡操作與張量形狀如何變化。',
    render: function (rootEl) {
      var style = document.createElement('style');
      style.textContent = CSS;
      rootEl.appendChild(style);
      var tabs = h('div', { 'class': 'ch03-tabs', role: 'tablist' }, rootEl);
      var b1 = h('button', { role: 'tab', 'aria-selected': 'true' }, tabs, '① 切矩陣');
      var b2 = h('button', { role: 'tab', 'aria-selected': 'false' }, tabs, '② 一層 Transformer 的 TP+SP');
      var p1 = h('div', { 'class': 'widget-panel' }, rootEl);
      var p2 = h('div', { 'class': 'widget-panel' }, rootEl);
      p2.hidden = true;
      function sel(i) {
        b1.setAttribute('aria-selected', String(i === 0));
        b2.setAttribute('aria-selected', String(i === 1));
        p1.hidden = i !== 0; p2.hidden = i !== 1;
      }
      b1.addEventListener('click', function () { sel(0); });
      b2.addEventListener('click', function () { sel(1); });
      renderTab1(p1);
      renderTab2(p2);
      renderInsight(rootEl);
    },
  };
})();
