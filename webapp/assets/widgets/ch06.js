/* 第 6 章互動元件：MoE 路由與專家平行（EP）視覺化 */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const TOKENS = ['今天', '天氣', '模型', '訓練', '貓咪', '跳舞', '量子', '翻譯'];
  // 寫死的偽 router 偏好：[首選專家, 次選專家]（專家 2 刻意較熱門，示範負載不均）
  const PREF = [[0, 1], [1, 2], [2, 3], [2, 4], [5, 0], [5, 6], [6, 7], [7, 2]];
  const N = 8, SLOT = 95, CX0 = 47.5; // 8 個欄位，token 與專家垂直對齊
  const state = { ep: 2, k: 2, noise: null, routed: false };
  let svg, statsEl, warnEl, interpEl;

  function sampleNoise() {
    state.noise = TOKENS.map(() => PREF.map(() => Math.random() * 1.5));
  }
  function scores(t) {
    return PREF.map((_, e) => {
      const base = PREF[t][0] === e ? 2.0 : PREF[t][1] === e ? 1.1 : 0;
      return base + state.noise[t][e];
    });
  }
  function topK(t) {
    const s = scores(t);
    return s.map((v, e) => [v, e]).sort((a, b) => b[0] - a[0]).slice(0, state.k).map((p) => p[1]);
  }
  const cx = (i) => CX0 + SLOT * i;                    // 第 i 欄的中心 x
  const gpuOf = (i) => Math.floor(i / (N / state.ep)); // token / 專家所在 GPU

  function S(tag, attrs, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function H(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const per = N / state.ep;
    // GPU 框
    for (let g = 0; g < state.ep; g++) {
      const x1 = cx(g * per) - 44, x2 = cx((g + 1) * per - 1) + 44;
      svg.appendChild(S('rect', { x: x1, y: 178, width: x2 - x1, height: 104, rx: 10, fill: 'var(--panel)', stroke: 'var(--border)', 'stroke-width': 1.5 }));
      svg.appendChild(S('text', { x: x1 + 10, y: 196, 'font-size': 12, 'font-weight': 700, fill: 'var(--fg-muted)' }, 'GPU ' + g));
    }
    // 路由結果
    let counts = new Array(N).fill(0), cross = 0, total = 0, assign = [];
    if (state.routed) {
      for (let t = 0; t < N; t++) topK(t).forEach((e) => {
        counts[e]++; total++;
        const isCross = gpuOf(t) !== gpuOf(e);
        if (isCross) cross++;
        assign.push([t, e, isCross]);
      });
    }
    // 連線（畫在專家節點之下、GPU 框之上）
    assign.forEach(([t, e, isCross]) => {
      const p = S('path', {
        d: 'M ' + cx(t) + ' 60 C ' + cx(t) + ' 120, ' + cx(e) + ' 150, ' + cx(e) + ' 202',
        fill: 'none', 'stroke-width': 1.6, opacity: 0.85,
        stroke: isCross ? 'var(--accent-2)' : 'var(--accent)'
      });
      if (isCross) p.setAttribute('stroke-dasharray', '5 4');
      svg.appendChild(p);
    });
    // token 卡（上方）
    for (let t = 0; t < N; t++) {
      svg.appendChild(S('rect', { x: cx(t) - 36, y: 16, width: 72, height: 44, rx: 8, fill: 'var(--panel)', stroke: 'var(--border)', 'stroke-width': 1.5 }));
      svg.appendChild(S('text', { x: cx(t), y: 36, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 600, fill: 'var(--fg)' }, TOKENS[t]));
      svg.appendChild(S('text', { x: cx(t), y: 52, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--fg-muted)' }, 'GPU ' + gpuOf(t)));
    }
    // 專家節點與負載條
    const avg = total / N;
    for (let e = 0; e < N; e++) {
      const hot = state.routed && counts[e] > 2 * avg && counts[e] >= 2;
      const col = hot ? 'var(--accent-2)' : 'var(--accent)';
      svg.appendChild(S('rect', { x: cx(e) - 29, y: 202, width: 58, height: 30, rx: 7, fill: 'var(--accent-soft)', stroke: state.routed && counts[e] ? col : 'var(--border)', 'stroke-width': hot ? 2.2 : 1.5 }));
      svg.appendChild(S('text', { x: cx(e), y: 222, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: 'var(--fg)' }, '專家' + e));
      svg.appendChild(S('rect', { x: cx(e) - 32, y: 246, width: 64, height: 9, rx: 4.5, fill: 'var(--code-bg)', stroke: 'var(--border)', 'stroke-width': 1 }));
      if (state.routed && counts[e] > 0) {
        svg.appendChild(S('rect', { x: cx(e) - 32, y: 246, width: 64 * Math.min(counts[e], N) / N, height: 9, rx: 4.5, fill: col }));
      }
      svg.appendChild(S('text', { x: cx(e), y: 272, 'text-anchor': 'middle', 'font-size': 11, fill: hot ? 'var(--accent-2)' : 'var(--fg-muted)', 'font-weight': hot ? 700 : 400 },
        state.routed ? counts[e] + ' 個' + (hot ? ' ⚠' : '') : '—'));
    }
    updateText(counts, cross, total, avg);
  }

  function updateText(counts, cross, total, avg) {
    if (!state.routed) {
      statsEl.textContent = '統計：尚未路由。';
      warnEl.hidden = true;
      interpEl.textContent = '按「路由」，讓每個 token 依（寫死的偽 router 分數＋隨機擾動）選出 top-' + state.k + ' 專家；再試著調整 EP 度與 k，觀察跨 GPU 通訊量與負載變化。';
      return;
    }
    const pct = total ? Math.round(100 * cross / total) : 0;
    statsEl.textContent = '統計：共 ' + total + ' 個 token→專家指派｜跨 GPU ' + cross + ' 個（' + pct + '%）｜平均負載 ' + avg.toFixed(1) + '、最大 ' + Math.max.apply(null, counts) + '。';
    const hotList = counts.map((c, e) => (c > 2 * avg && c >= 2) ? '專家' + e + '（' + c + ' 個）' : null).filter(Boolean);
    warnEl.hidden = hotList.length === 0;
    warnEl.textContent = '⚠ ' + hotList.join('、') + ' 收到的 token 超過平均的 2 倍：負載不均——這是 MoE 訓練的核心難題。若不加以平衡，熱門專家會拖慢整步訓練，其餘專家則閒置。';
    let msg;
    if (state.ep === 1) {
      msg = 'EP=1 時所有專家都在同一顆 GPU 上，路由完全不需跨 GPU 通訊；但這顆 GPU 得放下全部 8 個專家的前饋層參數，等於沒有分攤記憶體。';
    } else {
      msg = '目前 EP=' + state.ep + '，8 個專家分散在 ' + state.ep + ' 顆 GPU（每顆 ' + (N / state.ep) + ' 個）。本輪有 ' + pct + '% 的指派要把 token 的隱藏狀態送到別顆 GPU 上的專家，這正是 EP 需要 all-to-all 通訊的原因——EP 度越大、k 越大，跨 GPU 比例通常越高。';
    }
    msg += ' 由於各專家的前饋層彼此獨立，EP 不必像 TP 那樣切分矩陣乘法，只需把 token 路由給正確的專家，因此相對輕量；實務上會再搭配 DP 切分輸入批次。為了壓低通訊開銷，DeepSeek-V3 便在 router 中限制每個 token 至多送往 M 個節點（其設定為 4），盡量把 token 留在單一節點上。';
    interpEl.textContent = msg;
  }

  function render(rootEl) {
    const panel = H('div', 'widget-panel');
    // 控制列
    const row = H('div', 'widget-row');
    const epLabel = H('label', null, 'EP 度（GPU 數）：');
    const epSel = document.createElement('select');
    [1, 2, 4].forEach((v) => {
      const o = document.createElement('option');
      o.value = v; o.textContent = v + ' 顆 GPU'; if (v === state.ep) o.selected = true;
      epSel.appendChild(o);
    });
    epLabel.appendChild(epSel);
    const kLabel = H('label', null, 'top-k：');
    const kSel = document.createElement('select');
    [1, 2].forEach((v) => {
      const o = document.createElement('option');
      o.value = v; o.textContent = 'k = ' + v; if (v === state.k) o.selected = true;
      kSel.appendChild(o);
    });
    kLabel.appendChild(kSel);
    const routeBtn = H('button', null, '路由');
    const resampleBtn = H('button', 'secondary', '重新抽樣');
    row.appendChild(epLabel); row.appendChild(kLabel);
    row.appendChild(routeBtn); row.appendChild(resampleBtn);
    panel.appendChild(row);
    // SVG 畫布
    svg = S('svg', { viewBox: '0 0 760 292', role: 'img', 'aria-label': 'MoE 路由與專家平行示意圖' });
    svg.style.width = '100%';
    svg.style.height = 'auto';
    svg.style.display = 'block';
    svg.style.marginTop = '.8rem';
    panel.appendChild(svg);
    // 圖例
    const legend = H('div', 'widget-row');
    legend.style.cssText = 'margin-top:.5rem;font-size:.82rem;color:var(--fg-muted);gap:1.2rem';
    const mk = (borderStyle, color, label) => {
      const item = H('span');
      const line = H('span');
      line.style.cssText = 'display:inline-block;width:26px;border-top:2px ' + borderStyle + ' ' + color + ';vertical-align:middle;margin-right:.35rem';
      item.appendChild(line);
      item.appendChild(document.createTextNode(label));
      return item;
    };
    legend.appendChild(mk('solid', 'var(--accent)', 'GPU 內路由（免通訊）'));
    legend.appendChild(mk('dashed', 'var(--accent-2)', '跨 GPU 路由：需要 all-to-all 通訊'));
    panel.appendChild(legend);
    // 統計、警示、解讀
    statsEl = H('div');
    statsEl.style.cssText = 'margin-top:.6rem;font-size:.88rem;color:var(--fg)';
    warnEl = H('div');
    warnEl.style.cssText = 'margin-top:.45rem;font-size:.86rem;font-weight:600;color:var(--accent-2);border:1px solid var(--accent-2);border-radius:8px;padding:.5rem .7rem;background:var(--accent-soft)';
    warnEl.hidden = true;
    interpEl = H('div');
    interpEl.style.cssText = 'margin-top:.6rem;font-size:.86rem;line-height:1.7;color:var(--fg-muted);border-left:3px solid var(--accent);padding-left:.7rem';
    panel.appendChild(statsEl); panel.appendChild(warnEl); panel.appendChild(interpEl);
    rootEl.appendChild(panel);
    // 事件
    epSel.addEventListener('change', () => { state.ep = +epSel.value; draw(); });
    kSel.addEventListener('change', () => { state.k = +kSel.value; draw(); });
    routeBtn.addEventListener('click', () => { state.routed = true; draw(); });
    resampleBtn.addEventListener('click', () => { sampleNoise(); state.routed = true; draw(); });
    sampleNoise();
    draw();
  }

  window.ChapterWidget = {
    title: 'MoE 路由與專家平行（EP）',
    intro: '上方是一小批 token，下方是分散在多顆 GPU 上的 8 個專家。按「路由」看每個 token 被送往哪些專家：GPU 內是實線、跨 GPU 是虛線（需要 all-to-all 通訊）。切換 EP 度與 top-k，觀察通訊比例與專家負載如何變化。',
    render: render
  };
})();
