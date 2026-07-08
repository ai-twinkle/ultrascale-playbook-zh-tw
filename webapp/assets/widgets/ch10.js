/* 第 10 章互動元件：浮點格式探索器（FP32 / BF16 / FP16 / FP8） */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const P2 = (n) => Math.pow(2, n);
  // 各格式的位元帳：max 為精確的最大有限值；E4M3 無 inf（全 1 樣式保留給 NaN）
  const FORMATS = [
    { id: 'fp32', name: 'FP32', e: 8, m: 23, bias: 127, max: (2 - P2(-23)) * P2(127) },
    { id: 'bf16', name: 'BF16', e: 8, m: 7, bias: 127, max: (2 - P2(-7)) * P2(127) },
    { id: 'fp16', name: 'FP16', e: 5, m: 10, bias: 15, max: 65504 },
    { id: 'e4m3', name: 'FP8-E4M3', e: 4, m: 3, bias: 7, max: 448, noInf: true },
    { id: 'e5m2', name: 'FP8-E5M2', e: 5, m: 2, bias: 15, max: 57344 }
  ];
  FORMATS.forEach((f) => {
    f.bits = 1 + f.e + f.m;
    f.minNormal = P2(1 - f.bias);            // 最小正規值 2^(1-bias)
    f.minSub = P2(1 - f.bias - f.m);         // 最小次正規值
    f.eps = P2(-f.m);                        // 1.0 之後第一個可表示數的間距
    f.range = Math.round(Math.log10(f.max / f.minSub));
  });
  const INTERP = {
    fp32: 'FP32 是訓練的安全基準：約 83 個數量級的範圍加上 1.19×10⁻⁷ 的 epsilon，範圍與精度都綽綽有餘——代價是每個數佔 4 位元組。混合精度訓練中，主權重與優化器狀態通常仍保留在 FP32。',
    bf16: 'BF16 的指數位與 FP32 一樣是 8 位，範圍幾乎相同——梯度再小也不易下溢，因此不需要損失縮放（loss scaling）。代價是尾數只剩 7 位、epsilon ≈ 7.8×10⁻³：精度粗得多，所以要靠 FP32 主權重與 FP32 累加來補救。',
    fp16: 'FP16 的指數只有 5 位，最小正規值約 6.1×10⁻⁵——而梯度值常小於這個下溢邊界，會被直接捨成 0。這就是需要損失縮放（loss scaling）的原因：反向傳播前先放大損失、之後再還原縮放；BF16 因為範圍夠大所以不用。',
    e4m3: 'FP8-E4M3 把較多位元給尾數（精度相對好），但最大值只有 448、1 到 2 之間僅 7 個可表示數。FP8 訓練最大的挑戰是穩定性：數值不穩定常讓損失發散，DeepSeek-V3 得靠逐 tile 縮放正規化等技巧才穩住大規模訓練。',
    e5m2: 'FP8-E5M2 把較多位元給指數，保住與 FP16 相近的範圍，但 1 到 2 之間只剩 3 個可表示數。FP8 訓練最大的挑戰是穩定性：低精度下數值不穩定常讓損失發散，難以追平高精度訓練的準確度。'
  };
  const TAIL_16 = 'FP8 的 GEMM 在 H100 上理論 FLOPS 是 BF16 的兩倍，誘因十足——但範圍與精度雙雙受限，穩定性是 FP8 預訓練最大的挑戰。';
  const TAIL_8 = '對照 16 位元：FP16 需要損失縮放來對抗梯度下溢，BF16 範圍夠大所以不用。';

  const state = { fmt: 'bf16', x: 0.0001 };
  let bitsSvg, formulaEl, cardsEl, tableWrap, noteEl, lineSvg, interpEl, inputEl;
  const btns = {};

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
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  const sup = (s) => String(s).split('').map((c) => SUP[c] || c).join('');
  function fmtNum(v) {
    if (!isFinite(v)) return v > 0 ? 'inf' : '-inf';
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) {
      const [mant, ex] = v.toExponential(2).split('e');
      return mant + ' × 10' + sup(String(+ex));
    }
    return String(parseFloat(v.toPrecision(5)));
  }

  // 把一個實數捨入到指定格式（就近捨入），回傳 {v, kind: ok|sub|under|over, err}
  function quantize(x, f) {
    if (x === 0 || !isFinite(x)) return { v: x, kind: 'ok', err: 0 };
    const sgn = Math.sign(x), a = Math.abs(x);
    let e = Math.floor(Math.log2(a));
    const emin = 1 - f.bias;
    if (e < emin) e = emin; // 次正規區用固定間距 2^(emin-m)
    const q = P2(e - f.m);
    const r = Math.round(a / q) * q;
    if (r > f.max) return { v: sgn * Infinity, kind: 'over', err: Infinity };
    if (r === 0) return { v: 0, kind: 'under', err: 1 };
    return { v: sgn * r, kind: r < f.minNormal ? 'sub' : 'ok', err: Math.abs(r - a) / a };
  }
  const cur = () => FORMATS.find((f) => f.id === state.fmt);

  function drawBits() {
    clear(bitsSvg);
    const f = cur(), W = 680, y = 36, h = 30;
    const cw = Math.min((W - 24) / f.bits, 46);
    const x0 = (W - cw * f.bits) / 2;
    const segs = [
      { n: 1, label: '符號', color: 'var(--accent-2)' },
      { n: f.e, label: '指數', color: 'var(--accent)' },
      { n: f.m, label: '尾數', color: 'var(--link)' }
    ];
    let bit = 0;
    segs.forEach((seg) => {
      for (let i = 0; i < seg.n; i++) {
        bitsSvg.appendChild(S('rect', {
          x: (x0 + (bit + i) * cw + 1).toFixed(1), y: y, width: (cw - 2).toFixed(1), height: h, rx: 3,
          fill: seg.color, 'fill-opacity': 0.22, stroke: seg.color, 'stroke-opacity': 0.7
        }));
      }
      bitsSvg.appendChild(S('text', {
        x: (x0 + (bit + seg.n / 2) * cw).toFixed(1), y: 26, 'text-anchor': 'middle',
        'font-size': 13, 'font-weight': 700, fill: seg.color
      }, seg.label + ' ' + seg.n + ' 位'));
      bit += seg.n;
    });
    bitsSvg.appendChild(S('text', { x: W / 2, y: 88, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--fg-muted)' },
      '共 ' + f.bits + ' 位元（' + f.bits / 8 + ' 位元組）　指數偏移 bias = ' + f.bias));
    // 公式
    clear(formulaEl);
    const tex = '(-1)^{s}\\times 1.\\text{尾數}\\times 2^{\\,\\text{指數}-' + f.bias + '}';
    if (window.katex) window.katex.render(tex, formulaEl, { throwOnError: false });
    else formulaEl.textContent = '值 = (−1)^符號 × 1.尾數 × 2^(指數 − ' + f.bias + ')';
  }

  function drawCards() {
    clear(cardsEl);
    const f = cur();
    const mk = (label, value, sub) => {
      const c = H('div');
      c.style.cssText = 'background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:.5rem .65rem';
      c.appendChild(H('div', null, label)).style.cssText = 'font-size:.75rem;color:var(--fg-muted)';
      c.appendChild(H('div', null, value)).style.cssText = 'font-size:.98rem;font-weight:700;color:var(--fg);margin:.15rem 0';
      c.appendChild(H('div', null, sub)).style.cssText = 'font-size:.72rem;color:var(--fg-muted)';
      return c;
    };
    cardsEl.appendChild(mk('最大值', fmtNum(f.max), f.noInf ? '再大就成 NaN（此格式無 inf）' : '再大就溢位成 inf'));
    cardsEl.appendChild(mk('最小正規值', fmtNum(f.minNormal), '更小就進入次正規區、終至下溢'));
    cardsEl.appendChild(mk('epsilon（1 之後的間距）', fmtNum(f.eps), '= 2' + sup('-' + f.m) + '，決定有效位數'));
    cardsEl.appendChild(mk('動態範圍', '約 ' + f.range + ' 個數量級', '含次正規數：' + fmtNum(f.minSub) + ' ～ ' + fmtNum(f.max)));
  }

  function drawTable() {
    clear(tableWrap);
    const x = state.x;
    if (!isFinite(x)) { noteEl.hidden = false; noteEl.textContent = '請輸入一個有效的數字。'; return; }
    const tbl = H('table');
    tbl.style.cssText = 'width:100%;min-width:430px;border-collapse:collapse;font-size:.85rem';
    const thr = H('tr');
    ['格式', '實際儲存值', '相對誤差', '狀態'].forEach((t) => {
      const th = H('th', null, t);
      th.style.cssText = 'text-align:left;padding:.35rem .5rem;color:var(--fg-muted);font-weight:600;border-bottom:1px solid var(--border)';
      thr.appendChild(th);
    });
    tbl.appendChild(thr);
    const bad = [];
    FORMATS.forEach((f) => {
      const r = quantize(x, f);
      const tr = H('tr');
      if (f.id === state.fmt) tr.style.background = 'var(--accent-soft)';
      const td = (text, css) => {
        const d = H('td', null, text);
        d.style.cssText = 'padding:.35rem .5rem;border-bottom:1px solid var(--border);color:var(--fg)' + (css || '');
        tr.appendChild(d);
      };
      td(f.name, f.id === state.fmt ? ';font-weight:700;color:var(--accent)' : '');
      const warn = ';color:var(--accent-2);font-weight:700';
      if (r.kind === 'over') {
        td((x < 0 ? '-' : '') + (f.noInf ? 'NaN' : 'inf'), warn);
        td('—');
        td('溢位 ⚠', warn);
        bad.push(f.name + '（溢位）');
      } else if (r.kind === 'under') {
        td('0', warn);
        td('100%');
        td('下溢 → 0 ⚠', warn);
        bad.push(f.name + '（下溢）');
      } else {
        td(fmtNum(r.v));
        td(r.err < 1e-12 ? '0（精確）' : fmtNum(r.err * 100) + '%');
        td(r.kind === 'sub' ? '次正規（精度更差）' : '正常', r.kind === 'sub' ? ';color:var(--accent)' : ';color:var(--fg-muted)');
      }
      tbl.appendChild(tr);
    });
    tableWrap.appendChild(tbl);
    noteEl.hidden = bad.length === 0;
    if (bad.length) noteEl.textContent = '⚠ ' + fmtNum(x) + ' 無法被 ' + bad.join('、') + ' 表示——訓練中一個這樣的值就可能讓 loss 變成 NaN 或讓權重卡在 0。';
  }

  function drawLine() {
    clear(lineSvg);
    const W = 760, padL = 92, padR = 14, LG0 = -46, LG1 = 40, axisY = 160;
    const X = (lg) => padL + ((lg - LG0) / (LG1 - LG0)) * (W - padL - padR);
    const l10 = Math.log10;
    // 典型梯度量級（約 10^-8 ～ 10^-3）
    lineSvg.appendChild(S('rect', { x: X(-8).toFixed(1), y: 18, width: (X(-3) - X(-8)).toFixed(1), height: axisY - 18, fill: 'var(--accent-2)', 'fill-opacity': 0.07 }));
    lineSvg.appendChild(S('text', { x: X(-5.5).toFixed(1), y: 13, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--accent-2)' }, '典型梯度量級'));
    // 座標軸與刻度
    lineSvg.appendChild(S('line', { x1: padL, y1: axisY, x2: W - padR, y2: axisY, stroke: 'var(--fg-muted)', 'stroke-width': 1 }));
    for (let lg = -40; lg <= 40; lg += 10) {
      lineSvg.appendChild(S('line', { x1: X(lg).toFixed(1), y1: 18, x2: X(lg).toFixed(1), y2: axisY + 4, stroke: 'var(--border)', 'stroke-width': 1 }));
      lineSvg.appendChild(S('text', { x: X(lg).toFixed(1), y: 176, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--fg-muted)' }, '10' + sup(lg)));
    }
    // 各格式的可表示範圍帶
    FORMATS.forEach((f, i) => {
      const sel = f.id === state.fmt, y = 24 + i * 26;
      const col = sel ? 'var(--accent)' : 'var(--link)';
      lineSvg.appendChild(S('text', { x: 8, y: y + 11, 'font-size': 12, 'font-weight': sel ? 700 : 500, fill: sel ? 'var(--accent)' : 'var(--fg-muted)' }, f.name));
      lineSvg.appendChild(S('rect', { x: X(l10(f.minSub)).toFixed(1), y: y, width: (X(l10(f.minNormal)) - X(l10(f.minSub))).toFixed(1), height: 14, fill: col, 'fill-opacity': sel ? 0.3 : 0.15 }));
      lineSvg.appendChild(S('rect', { x: X(l10(f.minNormal)).toFixed(1), y: y, width: (X(l10(f.max)) - X(l10(f.minNormal))).toFixed(1), height: 14, rx: 3, fill: col, 'fill-opacity': sel ? 0.8 : 0.35 }));
    });
    // 輸入值標記
    if (isFinite(state.x) && state.x !== 0) {
      const lg = l10(Math.abs(state.x));
      if (lg >= LG0 && lg <= LG1) {
        lineSvg.appendChild(S('line', { x1: X(lg).toFixed(1), y1: 18, x2: X(lg).toFixed(1), y2: axisY, stroke: 'var(--fg)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }));
        lineSvg.appendChild(S('text', { x: X(lg).toFixed(1), y: 194, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, fill: 'var(--fg)' }, '↑ 你輸入的數字'));
      }
    }
  }

  function refresh() {
    FORMATS.forEach((f) => {
      const b = btns[f.id], sel = f.id === state.fmt;
      b.style.background = sel ? 'var(--accent-soft)' : '';
      b.style.borderColor = sel ? 'var(--accent)' : '';
      b.style.color = sel ? 'var(--accent)' : '';
      b.style.fontWeight = sel ? '700' : '';
      b.setAttribute('aria-pressed', sel);
    });
    drawBits(); drawCards(); drawTable(); drawLine();
    interpEl.textContent = INTERP[state.fmt] + ' ' + (state.fmt === 'e4m3' || state.fmt === 'e5m2' ? TAIL_8 : TAIL_16);
  }

  function render(rootEl) {
    // 面板一：格式選擇、位元佈局與屬性卡
    const p1 = H('div', 'widget-panel');
    const row = H('div', 'widget-row');
    row.style.cssText = 'flex-wrap:wrap;gap:.4rem;margin-bottom:.6rem';
    FORMATS.forEach((f) => {
      const b = H('button', null, f.name + '　1+' + f.e + '+' + f.m);
      b.type = 'button';
      b.addEventListener('click', () => { state.fmt = f.id; refresh(); });
      btns[f.id] = b;
      row.appendChild(b);
    });
    p1.appendChild(row);
    bitsSvg = S('svg', { viewBox: '0 0 680 96', role: 'img', 'aria-label': '位元佈局圖' });
    bitsSvg.style.cssText = 'width:100%;height:auto;display:block';
    p1.appendChild(bitsSvg);
    formulaEl = H('div');
    formulaEl.style.cssText = 'text-align:center;font-size:.95rem;color:var(--fg);margin:.2rem 0 .6rem';
    p1.appendChild(formulaEl);
    cardsEl = H('div');
    cardsEl.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:.5rem';
    p1.appendChild(cardsEl);
    rootEl.appendChild(p1);
    // 面板二：輸入數字與各格式儲存結果
    const p2 = H('div', 'widget-panel');
    const irow = H('div', 'widget-row');
    irow.style.cssText = 'flex-wrap:wrap;gap:.5rem;align-items:center;margin-bottom:.5rem';
    const lab = H('label', null, '輸入一個數字：');
    lab.style.cssText = 'font-size:.88rem;color:var(--fg)';
    inputEl = H('input');
    inputEl.type = 'number'; inputEl.step = 'any'; inputEl.value = '0.0001';
    inputEl.style.width = '9.5rem';
    inputEl.setAttribute('aria-label', '要測試的數字');
    lab.appendChild(inputEl);
    irow.appendChild(lab);
    [['0.0001', '小梯度'], ['70000', '大激活值'], ['1e-7', '極小梯度']].forEach(([v, t]) => {
      const b = H('button', null, v + '（' + t + '）');
      b.type = 'button';
      b.style.fontSize = '.8rem';
      b.addEventListener('click', () => { inputEl.value = v; state.x = parseFloat(v); drawTable(); drawLine(); });
      irow.appendChild(b);
    });
    p2.appendChild(irow);
    tableWrap = H('div');
    tableWrap.style.cssText = 'overflow-x:auto';
    p2.appendChild(tableWrap);
    noteEl = H('div');
    noteEl.style.cssText = 'margin-top:.5rem;font-size:.85rem;font-weight:600;color:var(--accent-2);border:1px solid var(--accent-2);border-radius:8px;padding:.45rem .65rem;background:var(--accent-soft)';
    noteEl.hidden = true;
    p2.appendChild(noteEl);
    rootEl.appendChild(p2);
    // 面板三：對數刻度數線
    const p3 = H('div', 'widget-panel');
    const t3 = H('div', null, '各格式可表示範圍（對數刻度）');
    t3.style.cssText = 'font-size:.88rem;font-weight:700;color:var(--fg);margin-bottom:.4rem';
    p3.appendChild(t3);
    lineSvg = S('svg', { viewBox: '0 0 760 204', role: 'img', 'aria-label': '各浮點格式可表示範圍的對數數線' });
    lineSvg.style.cssText = 'width:100%;height:auto;display:block';
    p3.appendChild(lineSvg);
    const legend = H('div', null, '深色帶＝正規數範圍；淺色帶＝次正規數（精度更差）；虛線＝你輸入的數字。BF16 的帶和 FP32 幾乎等長（範圍相同、精度較粗），FP16 則窄得多（精度較細、範圍有限）。');
    legend.style.cssText = 'margin-top:.45rem;font-size:.8rem;line-height:1.6;color:var(--fg-muted)';
    p3.appendChild(legend);
    rootEl.appendChild(p3);
    // 動態解讀
    interpEl = H('div');
    interpEl.style.cssText = 'margin-top:.7rem;font-size:.86rem;line-height:1.75;color:var(--fg-muted);border-left:3px solid var(--accent);padding-left:.7rem';
    rootEl.appendChild(interpEl);
    inputEl.addEventListener('input', () => { state.x = parseFloat(inputEl.value); drawTable(); drawLine(); });
    refresh();
  }

  window.ChapterWidget = {
    title: '浮點格式探索器：FP32 / BF16 / FP16 / FP8',
    intro: '選一種浮點格式，看它的位元佈局（符號／指數／尾數）與最大值、最小正規值、epsilon 等關鍵屬性；輸入一個數字，看它在各格式下實際被存成什麼、誤差多大、何時溢位或下溢；下方的對數數線則一眼比較各格式的可表示範圍——理解為什麼 FP16 需要損失縮放、BF16 不用、FP8 又難在哪。',
    render: render
  };
})();
