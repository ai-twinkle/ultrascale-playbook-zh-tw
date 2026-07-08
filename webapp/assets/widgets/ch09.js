/* 第 9 章互動元件：矩陣乘法 Tiling 記憶體流量模擬 與 warp 記憶體合併存取 */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const N = 8, CS = 20; // 矩陣邊長、分頁 1 的格子邊長

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
  function mkSelect(labelText, options, current, onChange) {
    const label = H('label', null, labelText);
    const sel = document.createElement('select');
    options.forEach(([v, txt]) => {
      const o = document.createElement('option');
      o.value = v; o.textContent = txt;
      if (String(v) === String(current)) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => onChange(sel.value));
    label.appendChild(sel);
    return label;
  }
  const CELL_STYLE = { // [fill, stroke, fill-opacity, stroke-width]
    idle:   ['var(--panel-2)', 'var(--border)', 1, 1],
    read:   ['var(--accent-2)', 'var(--accent-2)', 0.4, 1.6],
    shared: ['var(--accent)', 'var(--accent)', 0.3, 1.6],
    active: ['var(--accent-soft)', 'var(--accent-2)', 1, 2],
    done:   ['var(--accent)', 'var(--accent)', 0.7, 1],
    waste:  ['var(--accent-2)', 'var(--border)', 0.15, 1]
  };
  function setCell(rect, st) {
    const m = CELL_STYLE[st];
    rect.setAttribute('fill', m[0]); rect.setAttribute('stroke', m[1]);
    rect.setAttribute('fill-opacity', m[2]); rect.setAttribute('stroke-width', m[3]);
  }
  function mkStats(panel, bigLabelText) { // 大字節省倍數＋計數列＋公式列＋解讀列
    const row = H('div', 'widget-row');
    row.style.cssText = 'margin-top:.7rem;align-items:center;gap:1rem;flex-wrap:wrap';
    const bigWrap = H('div');
    bigWrap.style.cssText = 'text-align:center;padding:.4rem 1rem;border:1px solid var(--border);border-radius:10px;background:var(--panel-2)';
    const big = H('div');
    big.style.cssText = 'font-size:2rem;font-weight:800;color:var(--accent);line-height:1.1';
    const bigLabel = H('div', null, bigLabelText);
    bigLabel.style.cssText = 'font-size:.72rem;color:var(--fg-muted);margin-top:.2rem';
    bigWrap.append(big, bigLabel);
    const col = H('div');
    const cnt = H('div');
    cnt.style.cssText = 'font-size:.88rem;color:var(--fg)';
    const cmp = H('div');
    cmp.style.cssText = 'font-size:.82rem;color:var(--fg-muted);margin-top:.25rem';
    col.append(cnt, cmp);
    row.append(bigWrap, col);
    const interp = H('div');
    interp.style.cssText = 'margin-top:.6rem;font-size:.86rem;line-height:1.7;color:var(--fg-muted);border-left:3px solid var(--accent);padding-left:.7rem';
    panel.append(row, interp);
    return { big, cnt, cmp, interp };
  }
  function mkPlayer(row, state, advance, paint) { // 播放／單步／重設 三顆按鈕
    const playBtn = H('button', null, '播放');
    const stepBtn = H('button', 'secondary', '單步');
    const resetBtn = H('button', 'secondary', '重設');
    state.stopFn = () => { if (state.timer) clearInterval(state.timer); state.timer = null; playBtn.textContent = '播放'; };
    playBtn.addEventListener('click', () => {
      if (state.timer) { state.stopFn(); return; }
      if (state.step >= state.maxFn()) { state.step = 0; paint(); }
      playBtn.textContent = '暫停';
      state.timer = setInterval(advance, 700);
    });
    stepBtn.addEventListener('click', () => { state.stopFn(); advance(); });
    resetBtn.addEventListener('click', () => { state.stopFn(); state.step = 0; paint(); });
    row.append(playBtn, stepBtn, resetBtn);
  }

  /* ---------- 分頁 1：Naive vs Tiled ---------- */
  const t1 = { mode: 'naive', T: 4, step: 0, timer: null, stopFn: null,
    maxFn: () => (t1.mode === 'naive' ? t1.T * t1.T : 2 * (N / t1.T)) };

  function t1Build(panel) {
    const row = H('div', 'widget-row');
    row.appendChild(mkSelect('kernel：',
      [['naive', 'naive（直讀全域記憶體）'], ['tiled', 'tiled（經共享記憶體）']],
      t1.mode, (v) => { t1.mode = v; t1.stopFn(); t1.step = 0; paint(); }));
    row.appendChild(mkSelect('tile 大小：', [[2, '2×2'], [4, '4×4']],
      t1.T, (v) => { t1.T = +v; t1.stopFn(); t1.step = 0; paint(); }));
    panel.appendChild(row);

    const svg = S('svg', { viewBox: '0 0 372 388', role: 'img', 'aria-label': '矩陣乘法 tiling 記憶體流量示意圖' });
    svg.style.cssText = 'width:100%;height:auto;display:block;margin-top:.8rem';
    const grid = (x0, y0, label) => {
      svg.appendChild(S('text', { x: x0, y: y0 - 6, 'font-size': 12, 'font-weight': 700, fill: 'var(--fg-muted)' }, label));
      const cells = [];
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const rc = S('rect', { x: x0 + c * CS, y: y0 + r * CS, width: CS - 2, height: CS - 2, rx: 3 });
        setCell(rc, 'idle'); svg.appendChild(rc); cells.push(rc);
      }
      return cells;
    };
    const B = grid(202, 26, 'B（8×8）');
    const A = grid(10, 216, 'A（8×8）');
    const C = grid(202, 216, 'C ＝ A × B');
    [['read', '本步全域記憶體讀取'], ['shared', '駐留共享記憶體'], ['active', '計算中的 C tile'], ['done', '已完成']]
      .forEach(([st, txt], i) => {
        const r = S('rect', { x: 12, y: 44 + i * 26, width: 15, height: 15, rx: 3 });
        setCell(r, st); svg.appendChild(r);
        svg.appendChild(S('text', { x: 34, y: 56 + i * 26, 'font-size': 11, fill: 'var(--fg-muted)' }, txt));
      });
    panel.appendChild(svg);
    const ui = mkStats(panel, '全域流量節省');

    function paint() {
      const T = t1.T, s = t1.step, max = t1.maxFn();
      const stA = new Array(N * N).fill('idle'), stB = stA.slice(), stC = stA.slice();
      let g = 0, sh = 0;
      if (t1.mode === 'naive') { // 每步算 1 個輸出元素：讀整列 A ＋ 整行 B，各 N 次
        g = s * 2 * N;
        for (let e = 0; e < s; e++) stC[Math.floor(e / T) * N + (e % T)] = 'done';
        if (s > 0) {
          const e = s - 1, r = Math.floor(e / T), c = e % T;
          for (let i = 0; i < N; i++) { stA[r * N + i] = 'read'; stB[i * N + c] = 'read'; }
          stC[r * N + c] = 'active';
        }
      } else { // 每階段 2 步：載入 A/B tile（2T² 次全域讀取）→ 在共享記憶體上計算（2T³ 次共享讀取）
        g = Math.ceil(s / 2) * 2 * T * T;
        sh = Math.floor(s / 2) * 2 * T * T * T;
        if (s > 0) {
          const p = Math.ceil(s / 2) - 1, isLoad = s % 2 === 1;
          const st = isLoad ? 'read' : 'shared';
          const cSt = s === max ? 'done' : (s >= 2 ? 'active' : 'idle');
          for (let r = 0; r < T; r++) for (let c = 0; c < T; c++) {
            stA[r * N + p * T + c] = st;
            stB[(p * T + r) * N + c] = st;
            stC[r * N + c] = cSt;
          }
        }
      }
      A.forEach((rc, i) => setCell(rc, stA[i]));
      B.forEach((rc, i) => setCell(rc, stB[i]));
      C.forEach((rc, i) => setCell(rc, stC[i]));
      const totNaive = T * T * 2 * N, totTiled = 2 * N * T; // (N/T) 階段 × 2T² 次載入
      ui.big.textContent = (totNaive / totTiled) + '×';
      ui.cnt.textContent = '步驟 ' + s + '/' + max + '｜全域記憶體讀取：' + g + '/' +
        (t1.mode === 'naive' ? totNaive : totTiled) + ' 次｜共享記憶體讀取：' + sh + ' 次';
      ui.cmp.textContent = '每元素全域讀取：naive 2·N ＝ ' + 2 * N + ' 次 vs tiled 2·N/T ＝ ' + 2 * N / T +
        ' 次｜此 ' + T + '×' + T + ' tile 合計：' + totNaive + ' vs ' + totTiled + ' 次';
      const ok = s >= max ? '✓ 完成！' : '';
      ui.interp.textContent = t1.mode === 'naive'
        ? ok + 'naive kernel：每條執行緒為 1 個輸出元素從全域記憶體讀整列 A（8 個）＋整行 B（8 個）共 16 次；tile 內各元素讀的資料大量重疊，同一筆資料被重複讀了 ' + T + ' 遍，合計 ' + totNaive + ' 次高延遲的全域存取。'
        : ok + 'tiled kernel：每輪由 block 內執行緒協力把 A、B 各一個 ' + T + '×' + T + ' tile 載入共享記憶體（' + 2 * T * T + ' 次全域讀取），' + T * T + ' 個輸出元素接著都在快速的共享記憶體上重複使用它——讀取總次數沒少，但其中 ' + (N / T) * 2 * T * T * T + ' 次搬到了共享記憶體，全域流量縮小 ' + (totNaive / totTiled) + ' 倍（tile 越大省越多）。';
    }
    function advance() {
      if (t1.step < t1.maxFn()) { t1.step++; paint(); }
      if (t1.step >= t1.maxFn()) t1.stopFn();
    }
    mkPlayer(row, t1, advance, paint);
    paint();
  }

  /* ---------- 分頁 2：記憶體合併存取 ---------- */
  const t2 = { mode: 'coal', step: 0, timer: null, stopFn: null,
    maxFn: () => (t2.mode === 'coal' ? 1 : 8) };

  function t2Build(panel) {
    const row = H('div', 'widget-row');
    row.appendChild(mkSelect('存取模式：',
      [['coal', '合併（相鄰位址，跨步 1）'], ['stride', '非合併（跨步 8）']],
      t2.mode, (v) => { t2.mode = v; t2.stopFn(); t2.step = 0; paint(); }));
    panel.appendChild(row);

    const M = 40, X0 = 20, TY = 26, MY = 150;
    const svg = S('svg', { viewBox: '0 0 360 500', role: 'img', 'aria-label': 'warp 記憶體合併存取示意圖' });
    svg.style.cssText = 'width:100%;height:auto;display:block;margin-top:.8rem';
    svg.appendChild(S('text', { x: X0, y: 16, 'font-size': 12, 'font-weight': 700, fill: 'var(--fg-muted)' }, '一個 warp（8 條執行緒示意）'));
    const thr = [];
    for (let i = 0; i < 8; i++) {
      const r = S('rect', { x: X0 + i * M + 3, y: TY, width: M - 6, height: 28, rx: 6, fill: 'var(--panel)', stroke: 'var(--border)', 'stroke-width': 1.5 });
      svg.appendChild(r); thr.push(r);
      svg.appendChild(S('text', { x: X0 + i * M + M / 2, y: TY + 19, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: 'var(--fg)' }, 'T' + i));
    }
    [['done', '已使用的元素'], ['waste', '叢發抓回但未使用（浪費頻寬）']].forEach(([st, txt], i) => {
      const r = S('rect', { x: X0, y: 74 + i * 22, width: 14, height: 14, rx: 3 });
      setCell(r, st); svg.appendChild(r);
      svg.appendChild(S('text', { x: X0 + 21, y: 85 + i * 22, 'font-size': 11, fill: 'var(--fg-muted)' }, txt));
    });
    svg.appendChild(S('rect', { x: 236, y: 74, width: 26, height: 14, rx: 4, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': '5 3' }));
    svg.appendChild(S('text', { x: 269, y: 85, 'font-size': 11, fill: 'var(--fg-muted)' }, '一次叢發'));
    svg.appendChild(S('text', { x: X0, y: MY - 10, 'font-size': 12, 'font-weight': 700, fill: 'var(--fg-muted)' }, '全域記憶體（DRAM，列優先位址 0–63）'));
    const mem = [];
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      const rc = S('rect', { x: X0 + c * M + 2, y: MY + r * M + 2, width: M - 4, height: M - 4, rx: 4 });
      setCell(rc, 'idle'); svg.appendChild(rc); mem.push(rc);
      svg.appendChild(S('text', { x: X0 + c * M + M / 2, y: MY + r * M + M / 2 + 4, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--fg-muted)' }, String(r * 8 + c)));
    }
    const dyn = S('g', {});
    svg.appendChild(dyn);
    panel.appendChild(svg);
    const ui = mkStats(panel, '傳輸次數差距');
    const cx = (i) => X0 + i * M + M / 2;
    const burst = (rowIdx, cur) => S('rect', { x: X0 - 3, y: MY + rowIdx * M - 3, width: 8 * M + 6, height: M + 6, rx: 8, fill: 'none', 'stroke-dasharray': '6 4', stroke: cur ? 'var(--accent-2)' : 'var(--accent)', 'stroke-width': cur ? 2.5 : 1.8 });

    function paint() {
      while (dyn.firstChild) dyn.removeChild(dyn.firstChild);
      mem.forEach((rc) => setCell(rc, 'idle'));
      thr.forEach((r) => { r.setAttribute('stroke', 'var(--border)'); r.setAttribute('fill', 'var(--panel)'); });
      const s = t2.step, max = t2.maxFn();
      if (t2.mode === 'coal') { // T0…T7 → 位址 0…7：一次叢發全包
        if (s >= 1) {
          dyn.appendChild(burst(0, false));
          for (let i = 0; i < 8; i++) {
            setCell(mem[i], 'done');
            thr[i].setAttribute('stroke', 'var(--accent)'); thr[i].setAttribute('fill', 'var(--accent-soft)');
            dyn.appendChild(S('path', { d: 'M ' + cx(i) + ' ' + (TY + 28) + ' L ' + cx(i) + ' ' + (MY - 4), stroke: 'var(--accent)', 'stroke-width': 1.8, fill: 'none' }));
          }
        }
      } else { // 執行緒 k → 位址 k×8：每條執行緒各觸發一次叢發
        for (let k = 0; k < s; k++) {
          const cur = k === s - 1, col = cur ? 'var(--accent-2)' : 'var(--accent)';
          dyn.appendChild(burst(k, cur));
          setCell(mem[k * 8], cur ? 'read' : 'done');
          for (let c = 1; c < 8; c++) setCell(mem[k * 8 + c], 'waste');
          thr[k].setAttribute('stroke', col); thr[k].setAttribute('fill', 'var(--accent-soft)');
          dyn.appendChild(S('path', {
            d: 'M ' + cx(k) + ' ' + (TY + 28) + ' C ' + cx(k) + ' 120, ' + cx(0) + ' ' + (MY + k * M - 50) + ', ' + cx(0) + ' ' + (MY + k * M),
            stroke: col, 'stroke-width': cur ? 2 : 1.4, fill: 'none', opacity: cur ? 1 : 0.6
          }));
        }
      }
      const used = t2.mode === 'coal' ? (s ? 8 : 0) : s;
      const fetched = t2.mode === 'coal' ? (s ? 8 : 0) : s * 8;
      ui.big.textContent = '8×';
      ui.cnt.textContent = '傳輸次數：' + s + '/' + max + '｜叢發抓回元素：' + fetched + '｜實際使用：' + used +
        (fetched ? '（利用率 ' + Math.round(1000 * used / fetched) / 10 + '%）' : '');
      ui.cmp.textContent = '同樣讓 warp 讀 8 個元素：合併＝1 次叢發傳輸 vs 非合併＝8 次（相差 8 倍）';
      const ok = s >= max ? '✓ 完成！' : '';
      ui.interp.textContent = t2.mode === 'coal'
        ? ok + '相鄰執行緒存取相鄰位址（T0→0、T1→1…），硬體把 8 個請求合併成 1 次 DRAM 叢發傳輸，抓回的 8 個元素全部用上——本章把矩陣乘 kernel 的 x/y 索引對調後，同 warp 執行緒改讀 A 同一列的相鄰元素，正是達成這種合併存取。'
        : ok + '跨步存取（執行緒 i→位址 i×8，等於讀列優先矩陣的同一「行」）：每次叢發抓回 8 個連續元素卻只用到 1 個，得發 8 次傳輸、87.5% 的頻寬被浪費。本章 naive 矩陣乘 kernel 裡執行緒 (0,0) 與 (1,0) 各讀 A 不同列正是如此——改成合併存取後，記憶體吞吐量提升約 10 倍、執行時間縮為十分之一。';
    }
    function advance() {
      if (t2.step < t2.maxFn()) { t2.step++; paint(); }
      if (t2.step >= t2.maxFn()) t2.stopFn();
    }
    mkPlayer(row, t2, advance, paint);
    paint();
  }

  /* ---------- 分頁骨架 ---------- */
  function render(rootEl) {
    const wrap = H('div');
    const tabRow = H('div', 'widget-row');
    tabRow.style.cssText = 'gap:.5rem;margin-bottom:.6rem';
    const panels = [], btns = [];
    [['Naive vs Tiled', t1Build], ['記憶體合併', t2Build]].forEach(([name, build], i) => {
      const b = H('button', i === 0 ? null : 'secondary', name);
      b.setAttribute('aria-pressed', String(i === 0));
      btns.push(b); tabRow.appendChild(b);
      const p = H('div', 'widget-panel');
      p.hidden = i !== 0;
      build(p); panels.push(p);
      b.addEventListener('click', () => {
        t1.stopFn(); t2.stopFn();
        panels.forEach((pp, j) => { pp.hidden = j !== i; });
        btns.forEach((bb, j) => { bb.className = j === i ? '' : 'secondary'; bb.setAttribute('aria-pressed', String(j === i)); });
      });
    });
    wrap.appendChild(tabRow);
    panels.forEach((p) => wrap.appendChild(p));
    rootEl.appendChild(wrap);
  }

  window.ChapterWidget = {
    title: '深入 GPU：Tiling 與記憶體合併',
    intro: '分頁一模擬 8×8 矩陣乘法中 naive 與 tiled（共享記憶體）兩種 kernel 的全域記憶體讀取流量：按「播放」逐步計算 C 的一個 tile，觀察讀取計數如何暴增或被共享記憶體吸收。分頁二用一個 warp 的 8 條執行緒示意合併與跨步存取的 DRAM 叢發傳輸差異。',
    render: render
  };
})();
