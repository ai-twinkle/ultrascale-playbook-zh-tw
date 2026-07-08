/* 第 4 章互動元件：Ring Attention 環狀傳遞動畫 + Zig-Zag 負載平衡對比 */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var GPU_COLORS = ['var(--accent)', 'var(--link)', 'var(--fg-muted)', 'var(--fg)'];

  function el(tag, cls, parent, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  function svg(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  /* ============ 分頁 1：環狀傳遞 ============ */
  function buildRingTab(box) {
    var CX = 220, CY = 168, R = 105, RK = 60, N = 4;
    var step = 0, animating = false, playing = false, raf = 0, timer = 0;
    var CAPTIONS = [
      '步驟 1/4：各 GPU 對「自己的分塊」計算注意力分數（如 GPU 1 計算 Q1×KV1），同時以非阻塞方式把 KV 送往下一顆 GPU。',
      '步驟 2/4：KV 分塊沿環前進一格，GPU 1 現在計算 Q1×KV4；上一輪的傳送已在計算期間悄悄完成，GPU 幾乎不必等待。',
      '步驟 3/4：KV 再前進一格（GPU 1 計算 Q1×KV3），「邊算邊傳」持續重疊進行，通訊成本被計算時間吸收。',
      '步驟 4/4：最後一輪（GPU 1 計算 Q1×KV2），KV 已繞行一整圈、不需再傳送。四步之後每顆 GPU 都看過所有 KV，注意力計算完成！'
    ];
    function ang(i) { return -Math.PI / 2 + i * Math.PI / 2; }
    function pos(r, a) { return [CX + r * Math.cos(a), CY + r * Math.sin(a)]; }

    var view = svg('svg', { viewBox: '0 0 440 348', width: '100%', role: 'img',
      'aria-label': 'Ring Attention 四顆 GPU 環狀傳遞 KV 分塊示意圖' }, box);
    svg('circle', { cx: CX, cy: CY, r: R, fill: 'none', stroke: 'var(--border)',
      'stroke-dasharray': '4 5', 'stroke-width': 1.5 }, view);

    var arrows = [];
    for (var i = 0; i < N; i++) {
      var a1 = ang(i) + 0.5, a2 = ang(i + 1) - 0.5;
      var p1 = pos(R, a1), p2 = pos(R, a2);
      var path = svg('path', { d: 'M' + p1[0] + ' ' + p1[1] + ' A ' + R + ' ' + R + ' 0 0 1 ' + p2[0] + ' ' + p2[1],
        fill: 'none', stroke: 'var(--border)', 'stroke-width': 2.5 }, view);
      var t = a2 + Math.PI / 2, tip = [p2[0] + 10 * Math.cos(t), p2[1] + 10 * Math.sin(t)];
      var b1 = pos(R - 5.5, a2), b2 = pos(R + 5.5, a2);
      var head = svg('polygon', { points: tip + ' ' + b1 + ' ' + b2, fill: 'var(--border)' }, view);
      arrows.push([path, head]);
    }

    var nodes = [];
    for (i = 0; i < N; i++) {
      var p = pos(R, ang(i));
      var g = svg('g', {}, view);
      var rect = svg('rect', { x: p[0] - 42, y: p[1] - 27, width: 84, height: 54, rx: 9,
        fill: 'var(--panel)', stroke: GPU_COLORS[i], 'stroke-width': 2 }, g);
      svg('text', { x: p[0], y: p[1] - 8, 'text-anchor': 'middle', 'font-size': 13,
        'font-weight': 700, fill: 'var(--fg)' }, g).textContent = 'GPU ' + (i + 1);
      svg('text', { x: p[0], y: p[1] + 14, 'text-anchor': 'middle', 'font-size': 12,
        'font-weight': 600, fill: GPU_COLORS[i] }, g).textContent = 'Q' + (i + 1) + '（固定）';
      nodes.push(rect);
    }

    var stepText = svg('text', { x: CX, y: CY - 4, 'text-anchor': 'middle', 'font-size': 17,
      'font-weight': 700, fill: 'var(--fg)' }, view);
    var subText = svg('text', { x: CX, y: CY + 18, 'text-anchor': 'middle', 'font-size': 11,
      fill: 'var(--fg-muted)' }, view);
    subText.textContent = 'KV 分塊沿環流動';

    var chips = [];
    for (i = 0; i < N; i++) {
      g = svg('g', {}, view);
      svg('rect', { x: -23, y: -12, width: 46, height: 24, rx: 6, fill: GPU_COLORS[i],
        stroke: 'var(--bg)', 'stroke-width': 1.5 }, g);
      svg('text', { x: 0, y: 4.5, 'text-anchor': 'middle', 'font-size': 12,
        'font-weight': 700, fill: 'var(--bg)' }, g).textContent = 'KV' + (i + 1);
      chips.push(g);
    }
    function placeChip(k, a) {
      var p = pos(RK, a);
      chips[k].setAttribute('transform', 'translate(' + p[0] + ' ' + p[1] + ')');
    }

    var row = el('div', 'widget-row', box);
    var btnNext = el('button', null, row, '下一步 ▶');
    var btnPlay = el('button', null, row, '自動播放');
    var btnReset = el('button', 'secondary', row, '重播 ⟲');
    var status = el('div', null, box);
    status.style.cssText = 'margin-top:.6rem;font-size:.85rem;font-weight:600;color:var(--accent);min-height:1.6em;';
    var caption = el('p', null, box);
    caption.style.cssText = 'margin:.3rem 0 0;font-size:.9rem;color:var(--fg);';
    var note = el('p', null, box,
      '📝 旁註：計算與通訊重疊——send 是非阻塞的，GPU 在等待下一份 KV 抵達時，並不會停下手邊的注意力計算。');
    note.style.cssText = 'margin:.5rem 0 0;font-size:.82rem;color:var(--fg-muted);';

    function setHot(on) {
      nodes.forEach(function (r) { r.setAttribute('fill', on ? 'var(--accent-soft)' : 'var(--panel)'); });
      arrows.forEach(function (a) {
        a[0].setAttribute('stroke', on ? 'var(--accent)' : 'var(--border)');
        a[1].setAttribute('fill', on ? 'var(--accent)' : 'var(--border)');
      });
      status.textContent = on ? '⚡ 同時進行：計算目前分塊的注意力分數 ＋ 傳送 KV → 下一顆 GPU（非阻塞）' : '';
    }
    function updateUI() {
      stepText.textContent = '步驟 ' + (step + 1) + '/' + N;
      caption.textContent = CAPTIONS[step];
      btnNext.disabled = animating || step >= N - 1;
      if (!animating) for (var k = 0; k < N; k++) placeChip(k, ang(k + step));
    }
    function advance(done) {
      if (animating || step >= N - 1) return;
      animating = true; setHot(true); btnNext.disabled = true;
      var t0 = performance.now(), dur = 900;
      (function frame(now) {
        var pr = Math.min(1, (now - t0) / dur);
        var e = pr < 0.5 ? 2 * pr * pr : 1 - Math.pow(-2 * pr + 2, 2) / 2;
        for (var k = 0; k < N; k++) {
          var a0 = ang(k + step);
          placeChip(k, a0 + (Math.PI / 2) * e);
        }
        if (pr < 1) { raf = requestAnimationFrame(frame); }
        else { animating = false; setHot(false); step++; updateUI(); if (done) done(); }
      })(t0);
    }
    function stopPlay() {
      playing = false; clearTimeout(timer); btnPlay.textContent = '自動播放';
    }
    function reset() {
      stopPlay(); cancelAnimationFrame(raf); animating = false;
      setHot(false); step = 0; updateUI();
    }
    btnNext.addEventListener('click', function () { stopPlay(); advance(); });
    btnPlay.addEventListener('click', function () {
      if (playing) { stopPlay(); return; }
      if (step >= N - 1) reset();
      playing = true; btnPlay.textContent = '⏸ 暫停';
      (function tick() {
        if (!playing) return;
        if (step >= N - 1) { stopPlay(); return; }
        advance(function () { timer = setTimeout(tick, 350); });
      })();
    });
    btnReset.addEventListener('click', reset);
    updateUI();
    return { stop: reset };
  }

  /* ============ 分頁 2：Zig-Zag 平衡 ============ */
  function buildZigzagTab(box) {
    var T = 8, MAX = 15;
    var SCHEMES = {
      naive: {
        label: 'naive 順序切分',
        owner: function (r) { return Math.floor(r / 2); },
        tokens: ['token 1–2', 'token 3–4', 'token 5–6', 'token 7–8'],
        work: [3, 7, 11, 15],
        text: '順序切分：GPU 1 分到最前面的 token 1–2，因果遮罩下只需計算 3 格；GPU 4 分到 token 7–8，' +
          '卻要計算 15 格——足足是 GPU 1 的 5 倍。整體速度被最忙的 GPU 4 拖住，其他 GPU 的閒置算力' +
          '（工作量條上的橘色段）全都浪費掉了。'
      },
      zigzag: {
        label: 'Zig-Zag 交錯切分',
        owner: function (r) { return r < 4 ? r : 7 - r; },
        tokens: ['token 1・8', 'token 2・7', 'token 3・6', 'token 4・5'],
        work: [9, 9, 9, 9],
        text: 'Zig-Zag 交錯切分：每顆 GPU 改拿「頭尾各一段」token——GPU 1 拿 token 1 與 8、GPU 2 拿 ' +
          'token 2 與 7……數一數色塊就會發現，每顆 GPU 都恰好計算 9 格，工作量完全齊平，沒有任何 GPU 閒置等待。'
      }
    };
    var scheme = 'naive';

    var toggles = el('div', 'widget-row', box);
    var btnN = el('button', null, toggles, 'naive 順序切分');
    var btnZ = el('button', 'secondary', toggles, 'Zig-Zag 交錯切分');

    var gridWrap = el('div', null, box);
    gridWrap.style.cssText = 'margin-top:.8rem;max-width:330px;';
    var legend = el('div', null, box);
    legend.style.cssText = 'display:flex;flex-wrap:wrap;gap:.4rem 1rem;font-size:.8rem;color:var(--fg-muted);margin-top:.4rem;';
    var caption = el('p', null, box);
    caption.style.cssText = 'margin:.7rem 0 0;font-size:.9rem;color:var(--fg);';

    var barsWrap = el('div', null, box);
    barsWrap.style.cssText = 'display:grid;gap:.8rem;margin-top:1rem;';
    var panels = {};
    ['naive', 'zigzag'].forEach(function (key) {
      var p = el('div', 'widget-panel', barsWrap);
      el('div', null, p, SCHEMES[key].label + '：各 GPU 工作量').style.cssText =
        'font-size:.85rem;font-weight:700;margin-bottom:.5rem;';
      for (var g = 0; g < 4; g++) {
        var r = el('div', null, p);
        r.style.cssText = 'display:flex;align-items:center;gap:.5rem;margin:.28rem 0;';
        el('span', null, r, 'GPU ' + (g + 1)).style.cssText =
          'flex:0 0 3.3em;font-size:.78rem;color:var(--fg-muted);';
        var track = el('div', null, r);
        track.style.cssText = 'flex:1;height:14px;background:var(--code-bg);border-radius:7px;overflow:hidden;display:flex;';
        var work = SCHEMES[key].work[g];
        var seg = el('div', null, track);
        seg.style.cssText = 'width:' + (work / MAX * 100) + '%;background:' + GPU_COLORS[g] + ';opacity:.9;';
        if (work < MAX) {
          var idle = el('div', null, track);
          idle.style.cssText = 'width:' + ((MAX - work) / MAX * 100) + '%;background:var(--accent-2);opacity:.45;';
          idle.title = '閒置：等待最慢的 GPU';
        }
        el('span', null, r, work + ' 格').style.cssText =
          'flex:0 0 3em;font-size:.78rem;color:var(--fg-muted);text-align:right;';
      }
      if (key === 'naive') el('div', null, p, '■ 橘色＝閒置（等待最慢的 GPU 4）').style.cssText =
        'font-size:.75rem;color:var(--accent-2);margin-top:.4rem;';
      panels[key] = p;
    });

    function renderGrid() {
      var s = SCHEMES[scheme], C = 30, L = 26, TP = 24;
      gridWrap.innerHTML = '';
      var view = svg('svg', { viewBox: '0 0 ' + (L + T * (C + 3)) + ' ' + (TP + T * (C + 3)),
        width: '100%', role: 'img', 'aria-label': s.label + '的因果注意力遮罩分配圖' }, gridWrap);
      for (var c = 0; c < T; c++)
        svg('text', { x: L + c * (C + 3) + C / 2, y: TP - 8, 'text-anchor': 'middle',
          'font-size': 11, fill: 'var(--fg-muted)' }, view).textContent = 'K' + (c + 1);
      for (var r = 0; r < T; r++) {
        var o = s.owner(r);
        svg('text', { x: L - 6, y: TP + r * (C + 3) + C / 2 + 4, 'text-anchor': 'end',
          'font-size': 11, 'font-weight': 700, fill: GPU_COLORS[o] }, view).textContent = 'Q' + (r + 1);
        for (c = 0; c < T; c++) {
          svg('rect', { x: L + c * (C + 3), y: TP + r * (C + 3), width: C, height: C, rx: 4,
            fill: c <= r ? GPU_COLORS[o] : 'var(--code-bg)',
            'fill-opacity': c <= r ? 0.88 : 0.55,
            stroke: 'var(--border)', 'stroke-width': 0.6 }, view);
        }
      }
      legend.innerHTML = '';
      for (var g = 0; g < 4; g++) {
        var item = el('span', null, legend);
        item.style.cssText = 'display:inline-flex;align-items:center;gap:.35rem;';
        var sw = el('span', null, item);
        sw.style.cssText = 'width:12px;height:12px;border-radius:3px;background:' + GPU_COLORS[g] + ';';
        el('span', null, item, 'GPU ' + (g + 1) + '（' + s.tokens[g] + '）');
      }
      caption.textContent = s.text;
      btnN.className = scheme === 'naive' ? '' : 'secondary';
      btnZ.className = scheme === 'zigzag' ? '' : 'secondary';
      panels.naive.style.border = '1px solid ' + (scheme === 'naive' ? 'var(--accent)' : 'var(--border)');
      panels.zigzag.style.border = '1px solid ' + (scheme === 'zigzag' ? 'var(--accent)' : 'var(--border)');
    }
    btnN.addEventListener('click', function () { scheme = 'naive'; renderGrid(); });
    btnZ.addEventListener('click', function () { scheme = 'zigzag'; renderGrid(); });
    renderGrid();
  }

  /* ============ 元件入口 ============ */
  window.ChapterWidget = {
    title: '環狀注意力與 Zig-Zag 負載平衡',
    intro: '上下文平行把長序列切到多顆 GPU 上，唯有注意力層需要交換 KV 分塊。先看 Ring Attention 如何讓' +
      '「計算」與「KV 傳遞」重疊，再看 Zig-Zag 交錯切分如何解決因果遮罩造成的負載不均。',
    render: function (rootEl) {
      rootEl.innerHTML = '';
      var tabs = el('div', 'widget-row', rootEl);
      var tabBtns = [el('button', null, tabs, '① 環狀傳遞'), el('button', 'secondary', tabs, '② Zig-Zag 平衡')];
      var pages = [el('div', 'widget-panel', rootEl), el('div', 'widget-panel', rootEl)];
      pages.forEach(function (p) { p.style.marginTop = '.8rem'; });
      var ring = buildRingTab(pages[0]);
      buildZigzagTab(pages[1]);
      pages[1].hidden = true;
      tabBtns.forEach(function (b, i) {
        b.addEventListener('click', function () {
          if (i !== 0) ring.stop();
          tabBtns.forEach(function (bb, j) { bb.className = i === j ? '' : 'secondary'; });
          pages.forEach(function (p, j) { p.hidden = i !== j; });
        });
      });
    }
  };
})();
