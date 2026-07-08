/* 第 5 章互動元件：管線氣泡排程模擬器（AFAB vs 1F1B） */
(function () {
  'use strict';

  var TF = 1, TB = 2; // 前向 / 反向耗時（依原書假設 t_b ≈ 2 t_f）

  /* ---- 排程模擬：回傳各操作 {stage, type, mb, start, end} 與總時長 ---- */
  function simulate(p, m, sched) {
    var order = [], s, i;
    for (s = 0; s < p; s++) {
      var list = [];
      if (sched === 'afab') {
        for (i = 0; i < m; i++) list.push(['F', i]);
        for (i = 0; i < m; i++) list.push(['B', i]);
      } else { // 1F1B：warmup (p−1−s) 個前向 → 一前一後穩態 → cooldown 反向
        var w = Math.min(p - 1 - s, m);
        for (i = 0; i < w; i++) list.push(['F', i]);
        for (i = w; i < m; i++) { list.push(['F', i]); list.push(['B', i - w]); }
        for (i = m - w; i < m; i++) list.push(['B', i]);
      }
      order.push(list);
    }
    var fDone = [], bDone = [], idx = [], tFree = [], ops = [];
    for (s = 0; s < p; s++) { fDone.push([]); bDone.push([]); idx.push(0); tFree.push(0); }
    var progress = true;
    while (progress) {
      progress = false;
      for (s = 0; s < p; s++) {
        while (idx[s] < order[s].length) {
          var op = order[s][idx[s]], type = op[0], mb = op[1], ready;
          if (type === 'F') ready = (s === 0) ? 0 : fDone[s - 1][mb];
          else ready = (s === p - 1) ? fDone[s][mb] : bDone[s + 1][mb];
          if (ready === undefined) break; // 依賴尚未就緒
          var start = Math.max(tFree[s], ready);
          var end = start + (type === 'F' ? TF : TB);
          ops.push({ stage: s, type: type, mb: mb, start: start, end: end });
          if (type === 'F') fDone[s][mb] = end; else bDone[s][mb] = end;
          tFree[s] = end; idx[s]++; progress = true;
        }
      }
    }
    var total = 0;
    ops.forEach(function (o) { if (o.end > total) total = o.end; });
    return { ops: ops, total: total };
  }

  /* ---- 各 GPU 活化駐留峰值：F 開始 +1、B 結束 −1（同時刻先釋放） ---- */
  function peakInFlight(ops, p) {
    var peaks = [];
    for (var s = 0; s < p; s++) {
      var ev = [];
      ops.forEach(function (o) {
        if (o.stage !== s) return;
        if (o.type === 'F') ev.push([o.start, 1]); else ev.push([o.end, -1]);
      });
      ev.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
      var cur = 0, peak = 0;
      ev.forEach(function (e) { cur += e[1]; if (cur > peak) peak = cur; });
      peaks.push(peak);
    }
    return peaks;
  }

  function el(tag, attrs, parent) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'text') n.textContent = attrs[k]; else n.setAttribute(k, attrs[k]);
    });
    if (parent) parent.appendChild(n);
    return n;
  }
  function svgEl(tag, attrs, parent) {
    var n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }

  window.ChapterWidget = {
    title: '管線氣泡排程模擬器',
    intro: '調整管線階段數 p、微批次數 m 與排程方式（AFAB／1F1B），觀察甘特圖上的前向、反向與灰色氣泡如何變化，體會 r = (p−1)/m 與 1F1B 省下的活化記憶體。',
    render: function (rootEl) {
      var state = { p: 4, m: 4, sched: '1f1b', raf: 0 };

      var panel = el('div', { 'class': 'widget-panel' }, rootEl);
      var row = el('div', { 'class': 'widget-row' }, panel);

      function makeSlider(labelText, min, max, val) {
        var wrap = el('label', null, row);
        wrap.style.cssText = 'display:flex;flex-direction:column;gap:.25rem;font-size:.85rem;color:var(--fg-muted);min-width:150px;flex:1;';
        var top = el('span', null, wrap);
        top.innerHTML = labelText + '：<strong style="color:var(--fg)">' + val + '</strong>';
        var input = el('input', { type: 'range', min: min, max: max, step: 1, value: val }, wrap);
        return { input: input, top: top, label: labelText };
      }
      var pCtl = makeSlider('管線階段數 p', 2, 8, state.p);
      var mCtl = makeSlider('微批次數 m', 1, 16, state.m);

      var selWrap = el('label', null, row);
      selWrap.style.cssText = 'display:flex;flex-direction:column;gap:.25rem;font-size:.85rem;color:var(--fg-muted);';
      el('span', { text: '排程方式' }, selWrap);
      var sel = el('select', null, selWrap);
      el('option', { value: 'afab', text: 'AFAB（全前向全後向）' }, sel);
      el('option', { value: '1f1b', text: '1F1B（一前一後）' }, sel);
      sel.value = state.sched;
      var playBtn = el('button', { type: 'button', text: '▶ 播放' }, row);
      playBtn.style.alignSelf = 'flex-end';

      var chartBox = el('div', null, panel);
      chartBox.style.cssText = 'margin-top:1rem;overflow-x:auto;';
      var legend = el('div', null, panel);
      legend.style.cssText = 'display:flex;gap:1rem;flex-wrap:wrap;font-size:.8rem;color:var(--fg-muted);margin-top:.5rem;';
      [['var(--accent)', '前向（t_f）'], ['var(--link)', '反向（t_b = 2 t_f）'], ['var(--border)', '氣泡（閒置）']].forEach(function (it) {
        var s = el('span', null, legend);
        s.innerHTML = '<span style="display:inline-block;width:.85em;height:.85em;border-radius:2px;background:' + it[0] + ';vertical-align:-.1em;margin-right:.35em;"></span>' + it[1];
      });

      var statsRow = el('div', null, panel);
      statsRow.style.cssText = 'display:flex;gap:.75rem;flex-wrap:wrap;margin-top:1rem;';
      var barsBox = el('div', null, panel);
      barsBox.style.cssText = 'margin-top:.75rem;';
      var mathBox = el('div', null, panel);
      mathBox.style.cssText = 'margin-top:1rem;padding:.6rem .8rem;background:var(--code-bg);border:1px solid var(--border);border-radius:6px;overflow-x:auto;';
      var noteBox = el('div', null, panel);
      noteBox.style.cssText = 'margin-top:.75rem;font-size:.88rem;line-height:1.7;color:var(--fg-muted);';

      function statTile(title, value, sub) {
        var t = el('div', null, statsRow);
        t.style.cssText = 'flex:1;min-width:120px;background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:.55rem .7rem;';
        el('div', { text: title }, t).style.cssText = 'font-size:.75rem;color:var(--fg-muted);';
        el('div', { text: value }, t).style.cssText = 'font-size:1.15rem;font-weight:600;color:var(--accent-2);margin-top:.15rem;';
        if (sub) el('div', { text: sub }, t).style.cssText = 'font-size:.72rem;color:var(--fg-muted);margin-top:.15rem;';
      }

      var cells = [], sweepLine = null, totalTime = 0;

      function rebuild() {
        cancelAnimationFrame(state.raf); state.raf = 0; playBtn.textContent = '▶ 播放';
        var p = state.p, m = state.m, sched = state.sched;
        var sim = simulate(p, m, sched);
        totalTime = sim.total;
        cells = [];

        /* --- 甘特圖 --- */
        var labelW = 58, rightPad = 6, rowH = 26, gap = 6, axisH = 24, topPad = 4;
        var W = 780, chartW = W - labelW - rightPad;
        var H = topPad + p * (rowH + gap) + axisH;
        var unit = chartW / totalTime;
        chartBox.innerHTML = '';
        var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': '管線排程甘特圖' }, chartBox);
        svg.style.cssText = 'width:100%;min-width:300px;height:auto;display:block;';

        for (var s = 0; s < p; s++) {
          var y = topPad + s * (rowH + gap);
          svgEl('rect', { x: labelW, y: y, width: chartW, height: rowH, fill: 'var(--border)', opacity: 0.55, rx: 2 }, svg);
          var lb = svgEl('text', { x: labelW - 8, y: y + rowH / 2 + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--fg-muted)' }, svg);
          lb.textContent = 'GPU ' + (s + 1);
        }
        sim.ops.forEach(function (o) {
          var y = topPad + o.stage * (rowH + gap);
          var x = labelW + o.start * unit, w = (o.end - o.start) * unit;
          var g = svgEl('g', null, svg);
          svgEl('rect', {
            x: x + 0.5, y: y, width: Math.max(w - 1, 1), height: rowH, rx: 2,
            fill: o.type === 'F' ? 'var(--accent)' : 'var(--link)'
          }, g);
          if (w >= 13) {
            var t = svgEl('text', { x: x + w / 2, y: y + rowH / 2 + 3.5, 'text-anchor': 'middle', 'font-size': Math.min(10, w * 0.6), fill: 'var(--bg)', 'font-weight': 600 }, g);
            t.textContent = String(o.mb + 1);
          }
          cells.push({ el: g, start: o.start });
        });
        // 時間軸
        var axisY = topPad + p * (rowH + gap) + 4;
        svgEl('line', { x1: labelW, y1: axisY, x2: labelW + chartW, y2: axisY, stroke: 'var(--border)' }, svg);
        var step = Math.max(1, Math.ceil(totalTime / 10));
        for (var tt = 0; tt <= totalTime; tt += step) {
          var tx = labelW + tt * unit;
          svgEl('line', { x1: tx, y1: axisY, x2: tx, y2: axisY + 4, stroke: 'var(--border)' }, svg);
          var lab = svgEl('text', { x: tx, y: axisY + 15, 'text-anchor': 'middle', 'font-size': 9, fill: 'var(--fg-muted)' }, svg);
          lab.textContent = tt;
        }
        var axName = svgEl('text', { x: labelW + chartW, y: axisY + 15, 'text-anchor': 'end', 'font-size': 9, fill: 'var(--fg-muted)' }, svg);
        axName.textContent = '時間（單位 = t_f）→';
        sweepLine = svgEl('line', { x1: labelW, y1: topPad, x2: labelW, y2: axisY, stroke: 'var(--accent-2)', 'stroke-width': 2, opacity: 0 }, svg);
        sweepLine._x0 = labelW; sweepLine._unit = unit;

        /* --- 統計 --- */
        var rTheory = (p - 1) / m;
        var busy = m * (TF + TB); // 每卡計算時間相同
        var rMeasured = (totalTime - busy) / busy; // 實測 idle / 理想計算時間
        var idleShare = (totalTime - busy) / totalTime;
        var peaks = peakInFlight(sim.ops, p);
        var peakMax = Math.max.apply(null, peaks);
        statsRow.innerHTML = '';
        statTile('氣泡比例（理論）', (rTheory * 100).toFixed(1) + '%', 'r = (p−1)/m = ' + (p - 1) + '/' + m);
        statTile('氣泡比例（圖上實測）', (rMeasured * 100).toFixed(1) + '%', '每卡 idle ÷ 計算時間');
        statTile('氣泡佔總時長', (idleShare * 100).toFixed(1) + '%', '總時長 ' + totalTime + '，計算 ' + busy);
        statTile('活化駐留峰值', peakMax + ' 個微批次', sched === '1f1b' ? '1F1B：至多 min(p, m)' : 'AFAB：全部 m 個');

        barsBox.innerHTML = '';
        el('div', { text: '每卡「同時在途微批次數」峰值（活化記憶體示意）' }, barsBox).style.cssText = 'font-size:.78rem;color:var(--fg-muted);margin-bottom:.35rem;';
        peaks.forEach(function (pk, i) {
          var line = el('div', null, barsBox);
          line.style.cssText = 'display:flex;align-items:center;gap:.5rem;margin:.15rem 0;font-size:.75rem;color:var(--fg-muted);';
          el('span', { text: 'GPU ' + (i + 1) }, line).style.cssText = 'width:3.2em;flex:none;';
          var track = el('div', null, line);
          track.style.cssText = 'flex:1;height:.7em;background:var(--panel);border:1px solid var(--border);border-radius:4px;overflow:hidden;';
          var fill = el('div', null, track);
          fill.style.cssText = 'height:100%;width:' + (pk / m * 100) + '%;background:var(--accent-soft);border-right:2px solid var(--accent);';
          el('span', { text: pk + ' / ' + m }, line).style.cssText = 'width:3.5em;flex:none;text-align:right;';
        });

        /* --- 公式（KaTeX） --- */
        mathBox.innerHTML = '';
        var tex = 'r_{\\text{bubble}} = \\frac{(p-1)\\,(t_f+t_b)}{m\\,(t_f+t_b)} = \\frac{p-1}{m} = \\frac{' + (p - 1) + '}{' + m + '} \\approx ' + (rTheory * 100).toFixed(1) + '\\%';
        if (window.katex) window.katex.render(tex, mathBox, { throwOnError: false, displayMode: true });
        else mathBox.textContent = 'r_bubble = (p−1)/m = ' + (p - 1) + '/' + m + ' ≈ ' + (rTheory * 100).toFixed(1) + '%';

        /* --- 動態解讀 --- */
        var share = function (mm) { return Math.round((p - 1) / (mm + p - 1) * 100); };
        var notes = [];
        if (m < 16) notes.push('m 從 ' + m + ' → 16：氣泡佔總時長 ' + share(m) + '% → ' + share(16) + '%——增加微批次是攤薄氣泡的頭號手段。');
        else notes.push('m = 16 時氣泡僅佔總時長 ' + share(16) + '%；但 m 受全域批次大小限制，不可能無限增加。');
        if (sched === '1f1b') notes.push('1F1B 的氣泡與 AFAB 完全相同（r 都是 (p−1)/m），但活化記憶體峰值從 m = ' + m + ' 降到 min(p, m) = ' + Math.min(p, m) + ' 個微批次——early backward 提早釋放激活值，這正是能再加大 m 的本錢。');
        else notes.push('AFAB 必須保留全部 m = ' + m + ' 個微批次的激活值直到反向開始；切到 1F1B 可降到 min(p, m) = ' + Math.min(p, m) + ' 個，氣泡不變。');
        if (m <= p - 1) notes.push('目前 m ≤ p−1（' + m + ' ≤ ' + (p - 1) + '），管線大半時間在空轉——正是基準測試中效能低落的區間。');
        notes.push('想再縮小氣泡？交錯階段（interleaved）可將 r 進一步降為 (p−1)/(v·m)，代價是通訊量增加 v 倍。');
        noteBox.innerHTML = notes.map(function (n) { return '<div style="margin:.2rem 0;">・' + n + '</div>'; }).join('');
      }

      /* --- 播放：時間軸掃描動畫 --- */
      function play() {
        cancelAnimationFrame(state.raf);
        var dur = Math.min(8000, Math.max(2500, totalTime * 130));
        var t0 = performance.now();
        cells.forEach(function (c) { c.el.setAttribute('opacity', 0.12); });
        sweepLine.setAttribute('opacity', 1);
        playBtn.textContent = '■ 停止';
        function frame(now) {
          var k = Math.min(1, (now - t0) / dur), tCur = k * totalTime;
          var x = sweepLine._x0 + tCur * sweepLine._unit;
          sweepLine.setAttribute('x1', x); sweepLine.setAttribute('x2', x);
          cells.forEach(function (c) { if (c.start <= tCur) c.el.setAttribute('opacity', 1); });
          if (k < 1) state.raf = requestAnimationFrame(frame);
          else { sweepLine.setAttribute('opacity', 0); state.raf = 0; playBtn.textContent = '▶ 播放'; }
        }
        state.raf = requestAnimationFrame(frame);
      }
      function stop() {
        cancelAnimationFrame(state.raf); state.raf = 0;
        cells.forEach(function (c) { c.el.setAttribute('opacity', 1); });
        sweepLine.setAttribute('opacity', 0);
        playBtn.textContent = '▶ 播放';
      }

      pCtl.input.addEventListener('input', function () {
        state.p = +pCtl.input.value;
        pCtl.top.innerHTML = pCtl.label + '：<strong style="color:var(--fg)">' + state.p + '</strong>';
        rebuild();
      });
      mCtl.input.addEventListener('input', function () {
        state.m = +mCtl.input.value;
        mCtl.top.innerHTML = mCtl.label + '：<strong style="color:var(--fg)">' + state.m + '</strong>';
        rebuild();
      });
      sel.addEventListener('change', function () { state.sched = sel.value; rebuild(); });
      playBtn.addEventListener('click', function () { if (state.raf) stop(); else play(); });

      rebuild();
    }
  };
})();
