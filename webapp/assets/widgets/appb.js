/* 附錄 A1 互動元件：玩具 Profiler Trace 檢視器（vanilla JS，無外部相依） */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var LAYERS = 4, FWD = 8, BWD = 16, OPT = 12, AR = 12; // 玩具時長（ms）

  // 依「通訊重疊」與「CPU launch 瓶頸」兩個開關，產生一個訓練步的 trace
  function buildTrace(overlap, launchBound) {
    var split = launchBound ? 2 : 1;          // 瓶頸模式：kernel 未融合、被切成更小的 kernel
    var launchDur = launchBound ? 9 : 2;      // 瓶頸模式：每次 launch 的 CPU 開銷變大
    var specs = [], i, s;
    for (i = 1; i <= LAYERS; i++) for (s = 1; s <= split; s++)
      specs.push({ kind: 'fwd', layer: i, dur: FWD / split, name: '前向 kernel·第 ' + i + ' 層' + (split > 1 ? '（子 kernel ' + s + '/' + split + '）' : '') });
    for (i = LAYERS; i >= 1; i--) for (s = 1; s <= split; s++)
      specs.push({ kind: 'bwd', layer: i, dur: BWD / split, name: '反向 kernel·第 ' + i + ' 層' + (split > 1 ? '（子 kernel ' + s + '/' + split + '）' : '') });
    specs.push({ kind: 'opt', dur: OPT, name: 'optimizer step' });

    var cpu = [], gpu = [], comm = [], idle = [];
    var launchT = 0, gpuT = 0, optLaunchEnd = 0, gradReady = {};
    specs.forEach(function (sp) {
      cpu.push({ kind: 'launch', name: 'cudaLaunchKernel（' + sp.name + '）', start: launchT, dur: launchDur });
      launchT += launchDur;
      if (sp.kind === 'opt') { optLaunchEnd = launchT; return; }
      var st = Math.max(gpuT, launchT); // kernel 要等：前一個 kernel 結束 + 自己被 launch
      gpu.push({ kind: sp.kind, name: sp.name, start: st, dur: sp.dur });
      gpuT = st + sp.dur;
      if (sp.kind === 'bwd') gradReady[sp.layer] = gpuT; // 這一層梯度就緒的時間
    });
    var bwdEnd = gpuT, arT = 0;
    for (i = LAYERS; i >= 1; i--) { // bucket 依反向順序（第 4 層梯度最先算完）
      var as = Math.max(overlap ? gradReady[i] : bwdEnd, arT);
      comm.push({ kind: 'ar', name: 'all-reduce·bucket ' + i + '（第 ' + i + ' 層梯度）', start: as, dur: AR });
      arT = as + AR;
    }
    var optStart = Math.max(gpuT, arT, optLaunchEnd); // optimizer 必須等所有梯度同步完成
    gpu.push({ kind: 'opt', name: 'optimizer step', start: optStart, dur: OPT });
    var prev = 0; // 計算 stream 的空隙（idle）
    gpu.forEach(function (ev) {
      if (ev.start - prev > 0.9) idle.push({ kind: 'idle', name: 'GPU 閒置（idle）', start: prev, dur: ev.start - prev });
      prev = Math.max(prev, ev.start + ev.dur);
    });
    return { cpu: cpu, gpu: gpu, comm: comm, idle: idle, total: optStart + OPT };
  }

  var DETAILS = {
    fwd: { what: '前向傳播 kernel：這一層在 GPU 計算 stream 上執行矩陣乘法、啟動函數等運算。',
      why: '訓練步從前向傳播開始，逐層算出 loss；每一層對應一個（或多個）kernel，依序排在計算 stream 上。' },
    bwd: { what: '反向傳播 kernel：計算這一層的梯度，計算量約為前向的兩倍。',
      why: '反向從最後一層往回走（第 4 層 → 第 1 層）；某層梯度一算完，它的 all-reduce bucket 就能出發——這正是通訊重疊的關鍵。' },
    ar: { what: 'all-reduce：資料平行（DP）在各 GPU 之間平均這個 bucket 的梯度，跑在獨立的通訊 stream 上。',
      why: '若等反向全部結束才開始，通訊時間會完全暴露、直接加長訓練步；與反向重疊執行則能把它「藏」起來。' },
    opt: { what: 'optimizer step：用同步完成的梯度更新模型參數（例如 Adam）。',
      why: '必須等「所有」梯度 all-reduce 完成才能開始，所以最後一個 bucket 的通訊會直接卡住它。' },
    launch: { what: 'cudaLaunchKernel：CPU 把 kernel 排入 GPU 的執行佇列。CUDA 是非同步的，CPU 送出後不等結果就繼續。',
      why: '正常情況 CPU 跑得比 GPU 快，佇列裡永遠有工作；一旦 kernel 太小、太多，launch 開銷追不上，GPU 就會空轉等 CPU。' },
    idle: { what: 'GPU 閒置（idle）：計算 stream 上沒有工作可做的空隙。',
      why: '成因可能是等 CPU launch 下一個 kernel，或等通訊完成才能繼續。profiler 最重要的用途，就是找出這些空隙的成因並消除它。' }
  };

  function svgEl(name, attrs) {
    var n = document.createElementNS(SVGNS, name), k;
    for (k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  window.ChapterWidget = {
    title: '玩具 Profiler Trace 檢視器',
    intro: '模擬一個訓練步（4 層模型、DP 梯度同步）的 profiler trace：滑過事件條看時長、點擊看說明，並切換「通訊重疊」與「CPU launch 瓶頸」，觀察總 wall time 如何變化。',
    render: function (rootEl) {
      var VBW = 640, X0 = 70, X1 = 632, AXY = 132, VBH = 154;
      var lanes = [
        { key: 'cpu', label: 'CPU launch', y: 16, h: 24, barY: 19, barH: 18 },
        { key: 'gpu', label: 'GPU 計算', y: 50, h: 32, barY: 53, barH: 26 },
        { key: 'comm', label: 'GPU 通訊', y: 92, h: 28, barY: 95, barH: 22 }
      ];
      var FILL = { fwd: 'var(--accent)', bwd: 'var(--accent)', opt: 'var(--accent)', ar: 'var(--link)', launch: 'var(--accent-2)', idle: 'var(--fg-muted)' };
      var OPAC = { fwd: 0.6, bwd: 0.95, opt: 0.8, ar: 0.9, launch: 0.85, idle: 0.18 };
      var domain = 0;
      [[false, false], [true, false], [false, true], [true, true]].forEach(function (c) {
        domain = Math.max(domain, buildTrace(c[0], c[1]).total);
      });
      domain += 8;
      function sx(t) { return X0 + t / domain * (X1 - X0); }
      function sw(d) { return d / domain * (X1 - X0); }

      function swatch(style) { return '<span style="width:.8em;height:.8em;border-radius:3px;flex:none;' + style + '"></span>'; }
      function legendItem(style, label) { return '<span style="display:inline-flex;align-items:center;gap:.35rem">' + swatch(style) + label + '</span>'; }
      rootEl.innerHTML =
        '<div class="widget-panel">' +
          '<div class="widget-row">' +
            '<button type="button" data-ref="ov"></button>' +
            '<button type="button" data-ref="lb"></button>' +
            '<button type="button" data-ref="rp" class="secondary">▶ 重播動畫</button>' +
            '<div style="margin-left:auto;text-align:right">' +
              '<div style="font-size:.78rem;color:var(--fg-muted)">總 wall time</div>' +
              '<div style="line-height:1.2"><strong data-ref="tw" style="font-size:1.35rem;color:var(--accent)">0</strong> <span style="font-size:.8rem;color:var(--fg-muted)">ms</span></div>' +
              '<div data-ref="sv" style="font-size:.76rem;color:var(--fg-muted)"></div>' +
            '</div>' +
          '</div>' +
          '<div data-ref="chart" style="position:relative;margin-top:.9rem"></div>' +
          '<div class="widget-row" style="margin-top:.5rem;font-size:.78rem;color:var(--fg-muted);gap:.9rem">' +
            legendItem('background:var(--accent)', '計算 kernel') +
            legendItem('background:var(--link)', '通訊 all-reduce') +
            legendItem('background:var(--accent-2)', 'CPU launch') +
            legendItem('background:var(--fg-muted);opacity:.3', '空隙 = idle') +
          '</div>' +
        '</div>' +
        '<p data-ref="interp" style="margin:.8rem 0 .4rem;font-size:.88rem;line-height:1.7;color:var(--fg)"></p>' +
        '<div data-ref="detail" class="widget-panel" hidden style="margin-top:.5rem;font-size:.85rem;line-height:1.7"></div>';

      function ref(n) { return rootEl.querySelector('[data-ref="' + n + '"]'); }
      var ovBtn = ref('ov'), lbBtn = ref('lb'), rpBtn = ref('rp'), chart = ref('chart'),
          twEl = ref('tw'), svEl = ref('sv'), interpEl = ref('interp'), detailEl = ref('detail');

      var tip = document.createElement('div');
      tip.style.cssText = 'position:absolute;display:none;pointer-events:none;z-index:5;max-width:220px;' +
        'background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:.35rem .55rem;' +
        'font-size:.76rem;line-height:1.5;color:var(--fg);box-shadow:0 2px 8px rgba(0,0,0,.15)';

      var state = { ov: true, lb: false, trace: null, bars: [], selected: null, raf: 0, numRaf: 0, playing: false };

      function fmt(v) { return String(Math.round(v * 10) / 10); }

      function showTip(ev, e) {
        tip.innerHTML = '<strong>' + ev.name + '</strong><br>時長 ' + fmt(ev.dur) + ' ms（' + fmt(ev.start) + ' → ' + fmt(ev.start + ev.dur) + ' ms）';
        tip.style.display = 'block';
        moveTip(e);
      }
      function moveTip(e) {
        var r = chart.getBoundingClientRect();
        tip.style.left = Math.max(0, Math.min(e.clientX - r.left + 12, r.width - 170)) + 'px';
        tip.style.top = (e.clientY - r.top + 14) + 'px';
      }
      function showDetail(ev, rect) {
        if (state.selected) state.selected.setAttribute('stroke-width', '0');
        state.selected = rect;
        rect.setAttribute('stroke', 'var(--fg)');
        rect.setAttribute('stroke-width', '1.5');
        var d = DETAILS[ev.kind];
        detailEl.hidden = false;
        detailEl.innerHTML = '<strong>' + ev.name + '</strong>' +
          '<span style="color:var(--fg-muted)">（時長 ' + fmt(ev.dur) + ' ms，' + fmt(ev.start) + ' → ' + fmt(ev.start + ev.dur) + ' ms）</span>' +
          '<div style="margin-top:.35rem"><strong>這是什麼：</strong>' + d.what + '</div>' +
          '<div><strong>為何在這裡：</strong>' + d.why + '</div>';
      }

      function addBar(svg, ev, lane) {
        var rect = svgEl('rect', {
          x: sx(ev.start), y: lane.barY, width: Math.max(sw(ev.dur), 0.8), height: lane.barH,
          rx: 2, fill: FILL[ev.kind], 'fill-opacity': OPAC[ev.kind], 'stroke-width': 0, cursor: 'pointer'
        });
        rect.addEventListener('mouseenter', function (e) { rect.setAttribute('fill-opacity', Math.min(1, OPAC[ev.kind] + 0.2)); showTip(ev, e); });
        rect.addEventListener('mousemove', moveTip);
        rect.addEventListener('mouseleave', function () { rect.setAttribute('fill-opacity', OPAC[ev.kind]); tip.style.display = 'none'; });
        rect.addEventListener('click', function () { showDetail(ev, rect); });
        svg.appendChild(rect);
        state.bars.push({ el: rect, start: ev.start, dur: ev.dur });
      }

      function draw() {
        if (state.raf) { cancelAnimationFrame(state.raf); state.raf = 0; state.playing = false; rpBtn.disabled = false; }
        var t = buildTrace(state.ov, state.lb);
        state.trace = t; state.bars = []; state.selected = null;
        detailEl.hidden = true;
        chart.innerHTML = '';
        var svg = svgEl('svg', { viewBox: '0 0 ' + VBW + ' ' + VBH, role: 'img', 'aria-label': '訓練步 profiler 時間軸' });
        svg.style.cssText = 'display:block;width:100%;height:auto';
        lanes.forEach(function (ln) { // 泳道底（空隙處露出的就是 idle）
          svg.appendChild(svgEl('rect', { x: X0, y: ln.y, width: X1 - X0, height: ln.h, fill: 'var(--panel)', stroke: 'var(--border)', rx: 3 }));
          var lb = svgEl('text', { x: X0 - 6, y: ln.y + ln.h / 2 + 3.5, 'text-anchor': 'end', 'font-size': 10, fill: 'var(--fg-muted)' });
          lb.textContent = ln.label;
          svg.appendChild(lb);
        });
        for (var tk = 0; tk <= domain; tk += 25) { // 時間刻度
          svg.appendChild(svgEl('line', { x1: sx(tk), y1: 12, x2: sx(tk), y2: AXY, stroke: 'var(--border)', 'stroke-width': 0.6 }));
          var tl = svgEl('text', { x: sx(tk), y: AXY + 13, 'text-anchor': 'middle', 'font-size': 9, fill: 'var(--fg-muted)' });
          tl.textContent = tk;
          svg.appendChild(tl);
        }
        svg.appendChild(svgEl('line', { x1: X0, y1: AXY, x2: X1, y2: AXY, stroke: 'var(--border)' }));
        var unit = svgEl('text', { x: X1, y: AXY + 13, 'text-anchor': 'end', 'font-size': 9, fill: 'var(--fg-muted)' });
        unit.textContent = '時間（ms）';
        svg.appendChild(unit);
        t.idle.forEach(function (ev) { addBar(svg, ev, lanes[1]); });
        t.cpu.forEach(function (ev) { addBar(svg, ev, lanes[0]); });
        t.gpu.forEach(function (ev) { addBar(svg, ev, lanes[1]); });
        t.comm.forEach(function (ev) { addBar(svg, ev, lanes[2]); });
        svg.appendChild(svgEl('line', { x1: sx(t.total), y1: 10, x2: sx(t.total), y2: AXY, stroke: 'var(--accent-2)', 'stroke-dasharray': '3 3' })); // wall time 終點
        var mk = svgEl('text', { x: sx(t.total) + 3, y: 10, 'font-size': 9, fill: 'var(--accent-2)' });
        mk.textContent = fmt(t.total) + ' ms';
        svg.appendChild(mk);
        chart.appendChild(svg);
        chart.appendChild(tip);

        var baseline = buildTrace(false, state.lb).total; // 同一 launch 模式、未重疊的基準
        animateStats(t.total, (baseline - t.total) / baseline * 100);
        interpEl.textContent = (state.ov
          ? '通訊重疊開啟：all-reduce 與反向傳播同時進行，只剩最後一個 bucket 的通訊暴露在外——這就是 DP 把梯度同步「藏進」反向傳播的效果。'
          : '通訊重疊關閉：所有梯度 all-reduce 都排在反向傳播結束之後，通訊時間完全暴露，白白加長整個訓練步。') +
          (state.lb
            ? ' 此外 CPU launch 目前是瓶頸：kernel 被切得太小、太多，GPU 每算完一段就得空等下一次 launch（灰色空隙）——這正是 kernel 融合（fusion）與 CUDA Graphs 的動機。'
            : ' 目前 CPU launch 很快、跑在 GPU 前面，計算 stream 幾乎沒有空隙。');
      }

      function animateStats(totTo, pctTo) {
        if (state.numRaf) cancelAnimationFrame(state.numRaf);
        var totFrom = parseFloat(twEl.textContent) || 0, pctFrom = state.pctShown || 0, t0 = performance.now();
        state.pctShown = pctTo;
        function step(now) {
          var p = Math.min(1, (now - t0) / 500), e = 1 - Math.pow(1 - p, 3);
          twEl.textContent = fmt(totFrom + (totTo - totFrom) * e);
          var pct = pctFrom + (pctTo - pctFrom) * e;
          svEl.textContent = state.ov ? '較未重疊縮短 ' + pct.toFixed(1) + '%' : '基準（通訊未重疊）';
          state.numRaf = p < 1 ? requestAnimationFrame(step) : 0;
        }
        state.numRaf = requestAnimationFrame(step);
      }

      function replay() {
        if (state.playing) return;
        state.playing = true; rpBtn.disabled = true;
        var t = state.trace, t0 = performance.now(), DUR = 2400;
        function frame(now) {
          var ph = Math.min(1, (now - t0) / DUR) * t.total;
          state.bars.forEach(function (b) {
            b.el.setAttribute('width', Math.max(0, sw(Math.max(0, Math.min(b.dur, ph - b.start)))));
          });
          if (ph < t.total) { state.raf = requestAnimationFrame(frame); }
          else { state.raf = 0; state.playing = false; rpBtn.disabled = false; }
        }
        state.raf = requestAnimationFrame(frame);
      }

      function sync() {
        ovBtn.textContent = '通訊重疊：' + (state.ov ? '開啟' : '關閉');
        ovBtn.classList.toggle('secondary', !state.ov);
        lbBtn.textContent = 'CPU launch 瓶頸：' + (state.lb ? '開啟' : '關閉');
        lbBtn.classList.toggle('secondary', !state.lb);
        draw();
      }
      ovBtn.addEventListener('click', function () { state.ov = !state.ov; sync(); });
      lbBtn.addEventListener('click', function () { state.lb = !state.lb; sync(); });
      rpBtn.addEventListener('click', replay);
      sync();
    }
  };
})();
