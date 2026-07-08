(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';

  // ---- 三大挑戰（術語對齊 ch00：記憶體用量／運算效率／通訊負擔）----
  var CHALLENGES = {
    memory:  { label: '記憶體用量', color: 'var(--accent)',
      hint: '記憶體用量是硬性限制：一個訓練步驟放不進記憶體，訓練就無法進行。高亮的策略把權重、激活值或訓練狀態切開，直接降低單顆 GPU 的負擔。' },
    compute: { label: '運算效率', color: 'var(--accent-2)',
      hint: '我們希望硬體把大部分時間花在運算上。高亮的策略能擴增可用算力、或減少冗餘計算，藉此提高整體吞吐量。' },
    comm:    { label: '通訊負擔', color: 'var(--link)',
      hint: '通訊會讓 GPU 閒置。高亮的策略通訊量較小、或容易與計算重疊，適合放在較慢的節點間（inter-node）連線上。' }
  };

  // ---- 六個策略節點 ----
  var STRATS = {
    dp: { name: '資料平行 DP', dim: '切批次', helps: ['compute', 'comm'], side: 'L', cy: 78, to: [196, 80],
      principle: '每顆 GPU 保有一份完整模型副本，各自處理全域批次中不同的微批次；反向傳播後以 all-reduce 平均梯度，讓所有副本保持同步。',
      solve: '把一個步驟要處理的資料量分攤到多顆 GPU，直接提高吞吐量；梯度同步還能與反向傳播的計算重疊。',
      cost: '完全不省單卡記憶體——每顆 GPU 仍要放下整個模型與訓練狀態；全域批次大小也不能無限加大。',
      ch: ['ch02.html', '第 2 章　資料平行'] },
    tp: { name: '張量平行 TP', dim: '切權重矩陣', helps: ['memory', 'compute'], side: 'R', cy: 78, to: [505, 168],
      principle: '把每一層的權重矩陣沿行或列切開，多顆 GPU 各算一部分矩陣乘法，再把部分結果合併起來。',
      solve: '單層的權重與激活值都被切小，突破「單卡放不下一層」的記憶體限制，同時多卡合力計算同一層。',
      cost: '每層前後都要 all-reduce／all-gather，非常吃頻寬，一般只放在節點內（intra-node）的高速互連上。',
      ch: ['ch03.html', '第 3 章　張量平行'] },
    cp: { name: '上下文平行 CP', dim: '切序列', helps: ['memory'], side: 'L', cy: 182, to: [230, 142],
      principle: '把長序列沿上下文（序列）維度切成數段，分給不同 GPU；注意力所需的跨段資訊再透過通訊補齊。',
      solve: '激活值記憶體隨序列長度線性成長；切開序列後，每顆 GPU 只需扛一段上下文，長序列訓練才放得下。',
      cost: '注意力必須看見整條序列，得靠環狀注意力（ring attention）等機制在 GPU 間交換 K/V，增加通訊。',
      ch: ['ch04.html', '第 4 章　上下文平行'] },
    pp: { name: '管線平行 PP', dim: '切層', helps: ['memory', 'comm'], side: 'L', cy: 288, to: [196, 250],
      principle: '把模型的層依序切成多個階段（stage），每顆 GPU 只保存並執行自己負責的那幾層，激活值像接力棒一樣往下傳。',
      solve: '模型層數多到單卡放不下時，把層分攤出去；階段之間只傳激活值，通訊量小，適合節點間連線。',
      cost: '管線氣泡（bubble）：暖機與收尾時部分 GPU 閒置，傷害運算效率，需要微批次與精細排程來攤平。',
      ch: ['ch05.html', '第 5 章　管線平行'] },
    ep: { name: '專家平行 EP', dim: '切專家', helps: ['memory', 'compute'], side: 'R', cy: 182, to: [514, 286],
      principle: '在 MoE 模型中，把前饋層的不同專家（expert）放到不同 GPU，token 由路由器分派給對應的專家計算。',
      solve: '專家的參數量巨大、但每個 token 只會用到少數專家；把專家分散各卡，同時省下記憶體並提高效率。',
      cost: 'token 路由帶來 all-to-all 通訊，且各專家負載可能不均，通常需要與其他平行方式搭配使用。',
      ch: ['ch06.html', '第 6 章　專家平行'] },
    zero: { name: 'ZeRO', dim: '切訓練狀態', helps: ['memory'], side: 'R', cy: 288, to: [544, 345],
      principle: '沿資料平行維度把優化器狀態、梯度與參數分片（ZeRO-1／2／3 逐階切更多），計算時再臨時聚合回完整參數。',
      solve: '訓練狀態在每個 DP 副本上重複存放是最大宗的記憶體冗餘；分片後單卡記憶體隨 DP 度數下降。',
      cost: '計算前要先 all-gather 把分片聚合回來，引入額外通訊；能否與計算重疊掉，是成敗關鍵。',
      ch: ['ch02.html', '第 2 章　ZeRO 一節'] }
  };
  var ORDER = ['dp', 'tp', 'cp', 'pp', 'ep', 'zero'];

  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }
  function txt(parent, x, y, str, size, fill, anchor, weight) {
    var t = svgEl('text', { x: x, y: y, 'font-size': size, fill: fill, 'text-anchor': anchor || 'start' });
    if (weight) t.setAttribute('font-weight', weight);
    t.textContent = str;
    parent.appendChild(t);
    return t;
  }
  function injectStyle() {
    if (document.getElementById('usw5d-style')) return;
    var s = document.createElement('style');
    s.id = 'usw5d-style';
    s.textContent =
      '@keyframes usw5d-march{to{stroke-dashoffset:-13}}' +
      '@keyframes usw5d-pulse{0%,100%{opacity:1}50%{opacity:.5}}' +
      '.usw5d-cut{stroke:var(--accent);stroke-width:2.5;stroke-dasharray:7 6;stroke-linecap:round;animation:usw5d-march .8s linear infinite}' +
      '.usw5d-hl{animation:usw5d-pulse 1.1s ease-in-out 2}';
    document.head.appendChild(s);
  }

  function render(rootEl) {
    injectStyle();
    var state = { strat: 'dp', challenge: null };

    // ---- 挑戰切換列 ----
    var chalRow = document.createElement('div');
    chalRow.className = 'widget-row';
    chalRow.style.cssText = 'gap:.5rem;flex-wrap:wrap;align-items:center;margin-bottom:.4rem;';
    var chalLabel = document.createElement('span');
    chalLabel.textContent = '三大挑戰：';
    chalLabel.style.cssText = 'color:var(--fg-muted);font-size:.9rem;'; chalRow.appendChild(chalLabel);
    var chalBtns = {};
    [['all', '全部']].concat(Object.keys(CHALLENGES).map(function (k) { return [k, CHALLENGES[k].label]; })).forEach(function (f) {
      var b = document.createElement('button');
      if (f[0] !== 'all') {
        var dot = document.createElement('span');
        dot.style.cssText = 'display:inline-block;width:.6em;height:.6em;border-radius:50%;margin-right:.4em;background:' + CHALLENGES[f[0]].color + ';';
        b.appendChild(dot);
      }
      b.appendChild(document.createTextNode(f[1]));
      b.addEventListener('click', function () { state.challenge = f[0] === 'all' ? null : f[0]; update(); });
      chalBtns[f[0]] = b;
      chalRow.appendChild(b);
    });
    var hint = document.createElement('p');
    hint.style.cssText = 'margin:.1rem 0 .6rem;color:var(--fg-muted);font-size:.85rem;line-height:1.6;';

    // ---- SVG 地圖 ----
    var svgWrap = document.createElement('div');
    svgWrap.style.cssText = 'overflow-x:auto;padding-bottom:.25rem;';
    var svg = svgWrap.appendChild(svgEl('svg', { viewBox: '0 0 720 385', role: 'img', 'aria-label': '5D 平行策略互動地圖' }));
    svg.style.cssText = 'display:block;width:100%;min-width:600px;height:auto;';
    txt(svg, 360, 24, '一個訓練步驟（模型層 × 批次 × 序列）', 13, 'var(--fg)', 'middle', 700);

    var reg = {}, allCells = [];
    ORDER.forEach(function (k) { reg[k] = { cells: [], cuts: [], labels: [] }; });
    function cell(x, y, w, h) {
      var r = svgEl('rect', { x: x, y: y, width: w, height: h, rx: 3, fill: 'var(--panel-2)', stroke: 'var(--border)', 'stroke-width': 1 });
      r.style.transition = 'fill .25s,fill-opacity .25s'; svg.appendChild(r); allCells.push(r); return r;
    }
    function cut(key, x1, y1, x2, y2) {
      var l = svgEl('line', { x1: x1, y1: y1, x2: x2, y2: y2, 'class': 'usw5d-cut' });
      l.style.display = 'none'; svg.appendChild(l); reg[key].cuts.push(l);
    }
    function shardLab(key, x, y, str, anchor, size) {
      var t = txt(svg, x, y, str, size || 9, 'var(--accent)', anchor, 700);
      t.style.display = 'none'; reg[key].labels.push(t);
    }
    function pill(text, color) {
      var p = document.createElement('span');
      p.textContent = text;
      p.style.cssText = 'display:inline-block;padding:0 .55em;margin-right:.4em;border-radius:999px;font-size:.75rem;border:1px solid ' + color + ';color:' + color + ';';
      return p;
    }

    // 中央：輸入批次（DP 切列＝樣本、CP 切欄＝序列段）
    txt(svg, 200, 56, '輸入批次（樣本 × 序列 token）', 10, 'var(--fg-muted)');
    for (var r = 0; r < 3; r++) for (var c = 0; c < 6; c++) {
      var b = cell(200 + 35 * c, 62 + 25 * r, 32, 22);
      reg.dp.cells.push({ el: b, shard: r });
      reg.cp.cells.push({ el: b, shard: Math.floor(c / 2) });
    }
    cut('dp', 196, 85.5, 411, 85.5); cut('dp', 196, 110.5, 411, 110.5);
    [76, 101, 126].forEach(function (y, i) { shardLab('dp', 413, y, 'GPU ' + i); });
    cut('cp', 233.5, 58, 233.5, 138); cut('cp', 303.5, 58, 303.5, 138);
    [233, 303, 373].forEach(function (x, i) { shardLab('cp', x, 149, 'GPU ' + i, 'middle'); });

    // 中央：模型層堆疊（PP 切層）
    txt(svg, 200, 166, '模型（Transformer 層堆疊）', 10, 'var(--fg-muted)');
    for (var i = 0; i < 4; i++) {
      var bar = cell(200, 172 + 30 * i, 207, 24);
      reg.pp.cells.push({ el: bar, shard: i });
      txt(svg, 303, 188 + 30 * i, '層 ' + (i + 1), 10, 'var(--fg-muted)', 'middle');
      shardLab('pp', 402, 188 + 30 * i, 'GPU ' + i, 'end');
      if (i > 0) cut('pp', 196, 169 + 30 * i, 411, 169 + 30 * i);
    }
    // 放大鏡：單層內部（TP 切權重欄、EP 切專家）
    svg.appendChild(svgEl('line', { x1: 407, y1: 206, x2: 430, y2: 172, stroke: 'var(--border)', 'stroke-dasharray': '3 4' }));
    svg.appendChild(svgEl('line', { x1: 407, y1: 222, x2: 430, y2: 308, stroke: 'var(--border)', 'stroke-dasharray': '3 4' }));
    txt(svg, 430, 166, '單層內部：權重矩陣', 10, 'var(--fg-muted)');
    for (var mr = 0; mr < 3; mr++) for (var mc = 0; mc < 3; mc++)
      reg.tp.cells.push({ el: cell(430 + 27 * mc, 172 + 27 * mr, 24, 24), shard: mc });
    cut('tp', 455.5, 168, 455.5, 254); cut('tp', 482.5, 168, 482.5, 254);
    [442, 469, 496].forEach(function (x, i) { shardLab('tp', x, 188, 'G' + i, 'middle', 8.5); });
    txt(svg, 430, 262, 'FFN 專家（MoE）', 10, 'var(--fg-muted)');
    for (var e = 0; e < 4; e++) {
      var ex = 430 + (e % 2) * 42, ey = 268 + Math.floor(e / 2) * 22;
      reg.ep.cells.push({ el: cell(ex, ey, 38, 18), shard: e });
      txt(svg, ex + 19, ey + 13, '專家 ' + (e + 1), 8.5, 'var(--fg-muted)', 'middle');
    }
    cut('ep', 470, 264, 470, 312); cut('ep', 426, 288, 514, 288);

    // 中央：訓練狀態列（ZeRO 沿 DP 維度分片）
    txt(svg, 200, 326, '訓練狀態（每個副本都要保存）', 10, 'var(--fg-muted)');
    [['參數', 200, 110], ['梯度', 313, 110], ['優化器狀態', 426, 114]].forEach(function (seg) {
      var w3 = (seg[2] - 2) / 3;
      for (var s = 0; s < 3; s++) reg.zero.cells.push({ el: cell(seg[1] + s * (w3 + 1), 332, w3, 26), shard: s });
      txt(svg, seg[1] + seg[2] / 2, 349, seg[0], 10, 'var(--fg)', 'middle');
      cut('zero', seg[1] + w3 + 0.5, 328, seg[1] + w3 + 0.5, 362);
      cut('zero', seg[1] + 2 * w3 + 1.5, 328, seg[1] + 2 * w3 + 1.5, 362);
    });
    shardLab('zero', 540, 326, '每種狀態沿 DP 維度切成 3 份', 'end');

    // ---- 周圍策略節點與連接線 ----
    var nodes = {};
    ORDER.forEach(function (key) {
      var st = STRATS[key], nx = st.side === 'L' ? 18 : 564, cx = nx + 69;
      var link = svg.appendChild(svgEl('line', { x1: st.side === 'L' ? 156 : 564, y1: st.cy, x2: st.to[0], y2: st.to[1], stroke: 'var(--border)', 'stroke-width': 1.5 }));
      var g = svgEl('g', { role: 'button', tabindex: 0, 'aria-label': st.name + '（' + st.dim + '）' });
      g.style.cursor = 'pointer';
      var box = svgEl('rect', { x: nx, y: st.cy - 23, width: 138, height: 46, rx: 10, fill: 'var(--panel)', stroke: 'var(--border)', 'stroke-width': 1.5 });
      box.style.transition = 'stroke .2s,fill .2s'; g.appendChild(box);
      var nameT = txt(g, cx, st.cy - 3, st.name, 12.5, 'var(--fg)', 'middle', 700);
      var dimT = txt(g, cx, st.cy + 14, st.dim, 9.5, 'var(--fg-muted)', 'middle');
      function pick() { state.strat = key; update(); }
      g.addEventListener('click', pick);
      g.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(); } });
      svg.appendChild(g);
      nodes[key] = { g: g, box: box, nameT: nameT, dimT: dimT, link: link };
    });

    // ---- 策略卡片 ----
    var card = document.createElement('div');
    card.className = 'widget-panel';
    var cardHead = document.createElement('div');
    cardHead.style.cssText = 'display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.4rem;';
    var cardName = document.createElement('span');
    cardName.style.cssText = 'font-weight:700;font-size:1.05rem;color:var(--fg);';
    var cardDim = document.createElement('span');
    cardDim.style.cssText = 'padding:.1em .7em;border-radius:999px;font-size:.75rem;color:var(--accent);border:1px solid var(--accent);background:var(--accent-soft);';
    cardHead.appendChild(cardName); cardHead.appendChild(cardDim); card.appendChild(cardHead);
    function makeRow(label) {
      var row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:.6rem;margin:.35rem 0;line-height:1.7;';
      var lb = document.createElement('span');
      lb.textContent = label;
      lb.style.cssText = 'flex:none;min-width:4.2em;color:var(--fg-muted);font-size:.85rem;padding-top:.1em;';
      var val = document.createElement('span');
      val.style.cssText = 'color:var(--fg);font-size:.95rem;';
      row.appendChild(lb); row.appendChild(val); card.appendChild(row);
      return val;
    }
    var vPrin = makeRow('原理'), vSolve = makeRow('解決瓶頸'), vCost = makeRow('代價'), vLink = makeRow('深入閱讀');
    var foot = document.createElement('p');
    foot.style.cssText = 'margin:.6rem 0 0;color:var(--fg-muted);font-size:.85rem;';
    foot.appendChild(document.createTextNode('這些維度可以彼此組合——它們如何拼成完整的 5D 平行，詳見 '));
    var footA = document.createElement('a');
    footA.href = 'ch07.html'; footA.textContent = '第 7 章　5D 平行速覽';
    foot.appendChild(footA); foot.appendChild(document.createTextNode('。'));

    // ---- 更新邏輯 ----
    function applyHighlight(key) {
      allCells.forEach(function (el) {
        el.setAttribute('fill', 'var(--panel-2)'); el.setAttribute('fill-opacity', '1');
        el.setAttribute('stroke', 'var(--border)'); el.setAttribute('stroke-width', '1');
        el.classList.remove('usw5d-hl');
      });
      ORDER.forEach(function (k) {
        reg[k].cuts.concat(reg[k].labels).forEach(function (el) { el.style.display = k === key ? '' : 'none'; });
      });
      reg[key].cells.forEach(function (c) { // 相鄰分片交替兩種色調，代表不同 GPU 各持一份
        var tone = c.shard % 2 === 0 ? 'var(--accent)' : 'var(--accent-2)';
        c.el.setAttribute('fill', tone); c.el.setAttribute('fill-opacity', '0.45');
        c.el.setAttribute('stroke', tone); c.el.setAttribute('stroke-width', '1.5');
        c.el.classList.add('usw5d-hl');
      });
    }
    function update() {
      Object.keys(chalBtns).forEach(function (k) {
        var active = (state.challenge || 'all') === k;
        chalBtns[k].style.outline = active ? '2px solid var(--accent)' : '';
        chalBtns[k].style.background = active ? 'var(--accent-soft)' : '';
        chalBtns[k].setAttribute('aria-pressed', active);
      });
      hint.textContent = state.challenge ? CHALLENGES[state.challenge].hint
        : '點擊周圍節點，看每種策略在中央示意圖上「切」哪個維度；或選一項挑戰，看哪些策略能幫上忙。';
      ORDER.forEach(function (key) {
        var n = nodes[key], sel = key === state.strat;
        var helped = state.challenge && STRATS[key].helps.indexOf(state.challenge) !== -1;
        n.box.setAttribute('stroke', sel ? 'var(--accent)' : helped ? CHALLENGES[state.challenge].color : 'var(--border)');
        n.box.setAttribute('stroke-width', sel || helped ? 2.5 : 1.5);
        n.box.setAttribute('fill', sel ? 'var(--accent-soft)' : 'var(--panel)');
        n.g.setAttribute('opacity', state.challenge && !helped && !sel ? 0.35 : 1);
        n.dimT.setAttribute('fill', sel ? 'var(--accent)' : 'var(--fg-muted)');
        n.link.setAttribute('stroke', sel ? 'var(--accent)' : 'var(--border)');
        n.link.setAttribute('stroke-width', sel ? 2.5 : 1.5);
      });
      applyHighlight(state.strat);
      var st = STRATS[state.strat];
      cardName.textContent = st.name;
      cardDim.textContent = st.dim;
      vPrin.textContent = st.principle;
      vCost.textContent = st.cost;
      vSolve.textContent = '';
      st.helps.forEach(function (h) { vSolve.appendChild(pill(CHALLENGES[h].label, CHALLENGES[h].color)); });
      vSolve.appendChild(document.createTextNode(st.solve));
      vLink.textContent = '';
      var a = document.createElement('a');
      a.href = st.ch[0]; a.textContent = st.ch[1];
      vLink.appendChild(a);
    }

    [chalRow, hint, svgWrap, card, foot].forEach(function (el) { rootEl.appendChild(el); });
    update();
  }

  window.ChapterWidget = {
    title: '5D 平行策略互動地圖',
    intro: '超大規模訓練的所有技術都在回答同一個問題：一個訓練步驟該沿哪個維度切開？點擊六個策略節點，觀察它們各自切分批次、序列、層、權重矩陣、專家或訓練狀態，並用「三大挑戰」切換列找出能幫上忙的策略。',
    render: render
  };
})();
