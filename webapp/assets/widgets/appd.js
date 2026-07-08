/* 附錄 A3：計算／通訊重疊條件計算器（公式取自翻譯稿 appd.md） */
(function () {
  'use strict';

  var NVLINK = 450e9;      // 節點內 NVLink 頻寬（bytes/s）
  var IB = 50e9;           // 跨節點 InfiniBand 頻寬（bytes/s）
  var DPD = 8;             // 資料平行度（固定假設）
  var TPD = 8;             // 張量平行度（固定假設）
  var NUM_LAYERS = 32;     // DP 情境的層數（num_params = num_layers·16h²）
  var BUCKET = 25e6;       // DP 梯度桶大小 25 MB
  var PP_NEXT = 4;         // PP 下一階段層數 num_layers_in_next_pp

  function sig(x) { return Number(x.toPrecision(3)).toString(); }
  function fmtTime(s) {
    if (!isFinite(s) || s <= 0) return '—';
    if (s >= 1) return sig(s) + ' s';
    if (s >= 1e-3) return sig(s * 1e3) + ' ms';
    if (s >= 1e-6) return sig(s * 1e6) + ' µs';
    return sig(s * 1e9) + ' ns';
  }
  function texTime(s) {
    if (!isFinite(s) || s <= 0) return '\\text{—}';
    if (s >= 1) return sig(s) + '\\,\\text{s}';
    if (s >= 1e-3) return sig(s * 1e3) + '\\,\\text{ms}';
    if (s >= 1e-6) return sig(s * 1e6) + '\\,\\mu\\text{s}';
    return sig(s * 1e9) + '\\,\\text{ns}';
  }
  function sci(x) {
    var e = Math.floor(Math.log10(x));
    return Number((x / Math.pow(10, e)).toPrecision(3)) + '{\\times}10^{' + e + '}';
  }
  function tex(elm, src) {
    if (window.katex) window.katex.render(src, elm, { throwOnError: false, displayMode: true });
    else elm.textContent = src;
  }
  function el(tag, cls, parent, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    if (text != null) e.textContent = text;
    return e;
  }
  var SVGNS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  /* ── 四種情境：公式與計算皆對應翻譯稿 appd.md ── */
  var SCEN = {
    dp: {
      label: 'DP（all-reduce 梯度）',
      note: '固定假設：DP = 8、bucket = 25 MB、num_layers = 32；num_params = num_layers·16h²、num_tokens = seq·mbs。依翻譯稿，t_comm 取「單一桶」的 all-reduce 時間，t_compute 為整個反向傳播。',
      texComm: 't_{comm} = t_{comm\\_bucket} = \\frac{bucket\\_size \\cdot 2(DP-1)}{DP \\cdot peak\\_bw}',
      texCompute: 't_{compute} = \\frac{4 \\cdot num\\_tokens \\cdot num\\_params}{peak\\_flops}',
      texCond: '\\frac{t_{comm}}{t_{compute}} = \\frac{num\\_params}{2 \\cdot num\\_tokens} \\cdot \\frac{DP-1}{DP} \\cdot \\frac{peak\\_flops}{peak\\_bw} \\leq 1',
      calc: function (p) {
        var params = NUM_LAYERS * 16 * p.h * p.h, tokens = p.seq * p.mbs;
        var tc = BUCKET * 2 * (DPD - 1) / (DPD * p.bw);
        var tx = 4 * tokens * params / p.fl;
        return { tc: tc, tx: tx,
          subC: 't_{comm} = \\frac{25{\\times}10^{6} \\cdot 2 \\cdot 7}{8 \\cdot ' + p.bwTex + '} \\approx ' + texTime(tc),
          subX: 't_{compute} = \\frac{4 \\cdot ' + tokens + ' \\cdot ' + sci(params) + '}{' + p.flTex + '} \\approx ' + texTime(tx) };
      }
    },
    zero3: {
      label: 'ZeRO-3（all-gather 參數）',
      note: '固定假設：DP = 8。每個 transformer 區塊 16h² 位元組的參數需在前向時 all-gather；t_compute 為單一 decoder 層的前向計算。',
      texComm: 't_{comm} = 16h^2 \\cdot \\frac{DP-1}{DP \\cdot peak\\_bw}',
      texCompute: 't_{compute} = \\frac{32 \\cdot seq \\cdot mbs \\cdot h^2}{peak\\_flops}',
      texCond: '\\frac{t_{comm}}{t_{compute}} = \\frac{1}{2 \\cdot seq \\cdot mbs} \\cdot \\frac{DP-1}{DP} \\cdot \\frac{peak\\_flops}{peak\\_bw} \\leq 1',
      calc: function (p) {
        var tc = 16 * p.h * p.h * (DPD - 1) / (DPD * p.bw);
        var tx = 32 * p.seq * p.mbs * p.h * p.h / p.fl;
        return { tc: tc, tx: tx,
          subC: 't_{comm} = 16 \\cdot ' + p.h + '^2 \\cdot \\frac{7}{8 \\cdot ' + p.bwTex + '} \\approx ' + texTime(tc),
          subX: 't_{compute} = \\frac{32 \\cdot ' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + '^2}{' + p.flTex + '} \\approx ' + texTime(tx) };
      }
    },
    tp: {
      label: 'TP（all-gather 激活值）',
      note: '固定假設：TP = 8。分析某層的激活值 all-gather 能否藏進下一個線性層（參數量 h²）的計算。',
      texComm: 't_{comm} = \\frac{seq \\cdot mbs \\cdot h \\cdot (TP-1)}{TP \\cdot peak\\_bw}',
      texCompute: 't_{compute} = \\frac{2 \\cdot seq \\cdot mbs \\cdot h^2}{TP \\cdot peak\\_flops}',
      texCond: '\\frac{t_{comm}}{t_{compute}} = \\frac{TP-1}{2h} \\cdot \\frac{peak\\_flops}{peak\\_bw} \\leq 1',
      calc: function (p) {
        var tc = p.seq * p.mbs * p.h * (TPD - 1) / (TPD * p.bw);
        var tx = 2 * p.seq * p.mbs * p.h * p.h / (TPD * p.fl);
        return { tc: tc, tx: tx,
          subC: 't_{comm} = \\frac{' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + ' \\cdot 7}{8 \\cdot ' + p.bwTex + '} \\approx ' + texTime(tc),
          subX: 't_{compute} = \\frac{2 \\cdot ' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + '^2}{8 \\cdot ' + p.flTex + '} \\approx ' + texTime(tx) };
      }
    },
    pp: {
      label: 'PP（點對點激活值）',
      note: '固定假設：下一階段層數 num_layers_next = 4。分析階段間的 P2P 激活值傳輸能否藏進下一階段 transformer 區塊的計算。',
      texComm: 't_{comm} = \\frac{seq \\cdot mbs \\cdot h}{peak\\_bw}',
      texCompute: 't_{compute} = \\frac{32 \\cdot seq \\cdot mbs \\cdot h^2 \\cdot num\\_layers\\_next}{peak\\_flops}',
      texCond: '\\frac{t_{comm}}{t_{compute}} = \\frac{peak\\_flops}{32 \\cdot h \\cdot num\\_layers\\_next \\cdot peak\\_bw} \\leq 1',
      calc: function (p) {
        var tc = p.seq * p.mbs * p.h / p.bw;
        var tx = 32 * p.seq * p.mbs * p.h * p.h * PP_NEXT / p.fl;
        return { tc: tc, tx: tx,
          subC: 't_{comm} = \\frac{' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + '}{' + p.bwTex + '} \\approx ' + texTime(tc),
          subX: 't_{compute} = \\frac{32 \\cdot ' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + '^2 \\cdot 4}{' + p.flTex + '} \\approx ' + texTime(tx) };
      }
    }
  };

  function interpret(key, p, ratio) {
    var inter = p.bw === IB;
    if (key === 'tp') {
      if (ratio > 1 && inter) return '在跨節點 IB 頻寬（50 GB/s）下，TP 的 all-gather 藏不進下一個線性層的計算——這就是實務上「TP 不出節點」的數學原因。比值 (TP−1)/2h · peak_flops/peak_bw 只取決於 h、TP 與硬體，調 seq、mbs 都救不了。按「切回 NVLink 看對比」看看節點內的情況。';
      if (ratio > 1) return '即使在 NVLink 節點內頻寬下，比值仍大於 1：線性層計算量 ∝ h²、通訊量 ∝ h，h 太小時計算不足以掩蓋 all-gather。把 h 拉大可壓低比值（h ≳ (TP−1)·peak_flops / (2·peak_bw) 時才可重疊）。';
      return '在 NVLink 頻寬下，all-gather 能藏進下一個線性層的計算。注意：TP 的比值與 seq、mbs 完全無關——拖動這兩支滑桿，兩條時間等比縮放，比值不動；真正關鍵的是 h、TP 與 peak_flops/peak_bw。';
    }
    if (key === 'zero3') {
      if (ratio > 1) return (inter ? '跨節點 IB 頻寬下，' : '') + '下一層參數的 all-gather 藏不進當前層計算。ZeRO-3 的比值與 h 無關（分子分母的 h² 相消），唯一解方是加大 seq×mbs（每 GPU 的 token 數）或換更快的互連——試著拉高 seq 或 mbs。';
      return '下一層參數的 all-gather（預取）能藏在當前層的計算背後。比值 ∝ 1/(seq·mbs)：每 GPU 處理的 token 越多越容易重疊，且與 h 無關——調 h 時兩條時間等比縮放。';
    }
    if (key === 'pp') {
      if (ratio > 1) return 'P2P 傳輸藏不進下一階段的計算——通常是 h 太小或下一階段層數太少。實務上 PP 的通訊量（seq·mbs·h）是四種平行中最小的，很少成為瓶頸。';
      return 'P2P 只需在階段邊界傳一份激活值（seq·mbs·h），是四種平行中通訊量最小的；即使在跨節點 IB 頻寬下也能輕鬆重疊——這正是「PP、DP 跨節點，TP 留在節點內」佈局的數學基礎。與 TP 相同，比值與 seq、mbs 無關。';
    }
    if (ratio > 1) return '通訊追不上計算：每 GPU 的 token 數（seq·mbs）太少，反向傳播太快結束，梯度 all-reduce 來不及躲進去。增大 seq、mbs，或換更快的互連。';
    return '梯度在反向傳播進行的同時，以 25 MB 的桶為單位逐桶 all-reduce；單桶通訊時間遠小於整個反向傳播，幾乎總能完全重疊——這是 DP 成為最容易擴展的平行方式的原因。但左側條件式提醒：num_tokens 太小（模型大、批次小）時通訊仍會浮出檯面。';
  }

  function render(rootEl) {
    var state = { scen: 'tp', bw: IB, h: 8192, seqExp: 12, mbs: 1, tflops: 989 };

    /* ── 控制面板 ── */
    var ctrl = el('div', 'widget-panel', rootEl);
    var row1 = el('div', 'widget-row', ctrl);
    var selWrap = el('div', null, row1);
    selWrap.style.cssText = 'flex:1 1 220px;min-width:200px;';
    el('label', null, selWrap, '平行策略情境').style.cssText = 'display:block;font-size:.85rem;color:var(--fg-muted);margin-bottom:.25rem;';
    var sel = el('select', null, selWrap);
    sel.style.width = '100%';
    Object.keys(SCEN).forEach(function (k) {
      var o = el('option', null, sel, SCEN[k].label);
      o.value = k;
    });
    sel.value = state.scen;
    sel.addEventListener('change', function () { state.scen = sel.value; update(); });

    var bwWrap = el('div', null, row1);
    bwWrap.style.cssText = 'flex:1 1 260px;min-width:220px;';
    el('label', null, bwWrap, '互連頻寬 peak_bw').style.cssText = 'display:block;font-size:.85rem;color:var(--fg-muted);margin-bottom:.25rem;';
    var bwRow = el('div', null, bwWrap);
    bwRow.style.cssText = 'display:flex;gap:.5rem;flex-wrap:wrap;';
    var btnNv = el('button', null, bwRow, '節點內 NVLink 450 GB/s');
    var btnIb = el('button', null, bwRow, '跨節點 IB 50 GB/s');
    function setBw(bw) {
      state.bw = bw;
      btnNv.className = bw === NVLINK ? '' : 'secondary';
      btnIb.className = bw === IB ? '' : 'secondary';
      update();
    }
    btnNv.addEventListener('click', function () { setBw(NVLINK); });
    btnIb.addEventListener('click', function () { setBw(IB); });

    var row2 = el('div', 'widget-row', ctrl);
    row2.style.marginTop = '.8rem';
    function slider(labelText, min, max, step, value, fmt, apply) {
      var w = el('div', null, row2);
      w.style.cssText = 'flex:1 1 150px;min-width:130px;';
      var lab = el('label', null, w);
      lab.style.cssText = 'display:flex;justify-content:space-between;gap:.5rem;font-size:.82rem;color:var(--fg-muted);';
      el('span', null, lab, labelText);
      var val = el('span', null, lab);
      val.style.cssText = 'font-family:var(--mono,monospace);color:var(--fg);';
      var input = el('input', null, w);
      input.type = 'range'; input.min = min; input.max = max; input.step = step; input.value = value;
      input.style.width = '100%';
      input.addEventListener('input', function () {
        apply(Number(input.value));
        val.textContent = fmt(Number(input.value));
        update();
      });
      val.textContent = fmt(value);
    }
    slider('隱藏維度 h', 1024, 16384, 128, state.h, function (v) { return String(v); }, function (v) { state.h = v; });
    slider('序列長度 seq', 11, 17, 1, state.seqExp, function (v) { return (Math.pow(2, v) / 1024) + 'k'; }, function (v) { state.seqExp = v; });
    slider('微批次 mbs', 1, 8, 1, state.mbs, function (v) { return String(v); }, function (v) { state.mbs = v; });
    slider('peak_flops（bf16）', 100, 2000, 1, state.tflops, function (v) { return v + ' TFLOPs'; }, function (v) { state.tflops = v; });

    /* ── 結果面板：雙條 SVG 橫條圖 + 判定 + 解讀 ── */
    var res = el('div', 'widget-panel', rootEl);
    res.style.marginTop = '1rem';
    function barRow(name) {
      var head = el('div', null, res);
      head.style.cssText = 'display:flex;justify-content:space-between;gap:.5rem;font-size:.85rem;margin-top:.3rem;';
      el('span', null, head, name).style.color = 'var(--fg-muted)';
      var val = el('span', null, head);
      val.style.cssText = 'font-family:var(--mono,monospace);color:var(--fg);';
      var svg = svgEl('svg', { viewBox: '0 0 100 10', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
      svg.style.cssText = 'width:100%;height:18px;display:block;margin:.25rem 0 .5rem;';
      res.appendChild(svg);
      svgEl('rect', { x: 0, y: 0, width: 100, height: 10, rx: 1.2, fill: 'var(--border)', opacity: 0.35 }, svg);
      var bar = svgEl('rect', { x: 0, y: 0, width: 0, height: 10, rx: 1.2, fill: 'var(--accent)' }, svg);
      return { val: val, bar: bar };
    }
    var rowComm = barRow('通訊時間 t_comm');
    var rowComp = barRow('計算時間 t_compute');

    var verdictRow = el('div', null, res);
    verdictRow.style.cssText = 'display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;margin-top:.3rem;';
    var badge = el('span', null, verdictRow);
    badge.style.cssText = 'border:1.5px solid currentColor;border-radius:999px;padding:.18rem .75rem;font-size:.85rem;font-weight:600;';
    var flipBtn = el('button', 'secondary', verdictRow);
    flipBtn.addEventListener('click', function () { setBw(state.bw === IB ? NVLINK : IB); });
    var reading = el('p', null, res);
    reading.style.cssText = 'margin:.7rem 0 0;font-size:.88rem;line-height:1.7;color:var(--fg);';

    /* ── 公式面板：KaTeX 公式與代入結果 ── */
    var fpanel = el('div', 'widget-panel', rootEl);
    fpanel.style.marginTop = '1rem';
    el('div', null, fpanel, '公式（取自附錄 A3）與代入結果').style.cssText = 'font-size:.85rem;font-weight:600;color:var(--fg-muted);margin-bottom:.3rem;';
    function texBlock() {
      var d = el('div', null, fpanel);
      d.style.cssText = 'overflow-x:auto;padding:.15rem 0;';
      return d;
    }
    var fComm = texBlock(), fCompute = texBlock(), fCond = texBlock(), fSub = texBlock();
    var noteEl = el('p', null, fpanel);
    noteEl.style.cssText = 'margin:.5rem 0 0;font-size:.78rem;line-height:1.6;color:var(--fg-muted);';

    function update() {
      var s = SCEN[state.scen];
      var p = {
        h: state.h, seq: Math.pow(2, state.seqExp), mbs: state.mbs,
        bw: state.bw, fl: state.tflops * 1e12,
        bwTex: (state.bw / 1e9) + '{\\times}10^{9}', flTex: state.tflops + '{\\times}10^{12}'
      };
      var r = s.calc(p);
      var ratio = r.tc / r.tx;
      var ok = ratio <= 1;
      var color = ok ? 'var(--accent)' : 'var(--accent-2)';

      var max = Math.max(r.tc, r.tx);
      rowComm.bar.setAttribute('width', Math.max(0.6, 100 * r.tc / max));
      rowComm.bar.setAttribute('fill', color);
      rowComm.val.textContent = '≈ ' + fmtTime(r.tc);
      rowComp.bar.setAttribute('width', Math.max(0.6, 100 * r.tx / max));
      rowComp.bar.setAttribute('fill', 'var(--fg-muted)');
      rowComp.val.textContent = '≈ ' + fmtTime(r.tx);

      badge.style.color = color;
      badge.textContent = ok
        ? '可重疊：t_comm / t_compute ≈ ' + sig(ratio) + ' ≤ 1'
        : '無法完全重疊——通訊成為瓶頸（t_comm / t_compute ≈ ' + sig(ratio) + '）';
      flipBtn.textContent = state.bw === IB ? '切回 NVLink 看對比' : '切到跨節點 IB 看對比';
      reading.textContent = interpret(state.scen, p, ratio);

      tex(fComm, s.texComm);
      tex(fCompute, s.texCompute);
      tex(fCond, s.texCond);
      tex(fSub, r.subC + ',\\qquad ' + r.subX + ',\\qquad \\frac{t_{comm}}{t_{compute}} \\approx ' + sig(ratio));
      noteEl.textContent = s.note;
    }

    setBw(IB); // 預設：TP + 跨節點頻寬（最有教學價值的警示案例），setBw 內含首次 update()
  }

  window.ChapterWidget = {
    title: '計算／通訊重疊條件計算器',
    intro: '選擇平行策略與硬體參數，即時比較 t_comm 與 t_compute：只有當比值 t_comm/t_compute ≤ 1，通訊才能完全藏進計算——這條不等式決定了每種平行方式該部署在節點內還是跨節點。',
    render: render
  };
})();
