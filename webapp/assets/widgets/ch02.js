/* 第 2 章互動元件：ZeRO 記憶體分片視覺化（vanilla JS，無外部相依） */
(function () {
  'use strict';

  var MODELS = [
    { label: '1B', psi: 1e9 },
    { label: '7B', psi: 7e9 },
    { label: '70B', psi: 70e9 }
  ];
  var K = 12;      // Adam 優化器狀態倍率（fp32 參數 4Ψ + 動量 4Ψ + 變異數 4Ψ）
  var H100 = 80;   // 單張 H100 容量（GB）
  var STAGES = ['無 ZeRO', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3'];
  var STAGE_DESC = ['完整複製所有狀態', '切分優化器狀態', '＋切分梯度', '＋切分參數（FSDP）'];
  var COMM = [
    '梯度 all-reduce（通訊量 ≈ 2Ψ），可與反向傳播重疊。',
    '梯度 reduce-scatter ＋ 優化器步驟後 bf16 參數 all-gather（合計 ≈ 2Ψ）；reduce-scatter 比 all-reduce 快約一倍。',
    '梯度 reduce-scatter ＋ 參數 all-gather（≈ 2Ψ），與原生 DP 等價——相較 ZeRO-1 幾乎沒有額外開銷。',
    '前向逐層 all-gather 參數（Ψ）＋ 反向再一輪 all-gather（Ψ）＋ 梯度 reduce-scatter（Ψ）＝ 3Ψ，約為 ZeRO-1/2 的 1.5 倍；且一步多出 2·num_layers−1 次 all-gather，需靠預取（prefetching）重疊。'
  ];

  function fmtGB(x) {
    if (x >= 100) return String(Math.round(x));
    if (x >= 10) return String(Math.round(x * 10) / 10);
    return String(Math.round(x * 100) / 100);
  }
  function tex(el, s) {
    if (window.katex) window.katex.render(s, el, { throwOnError: false });
    else el.textContent = s;
  }
  function niceCeil(target, marks) {
    var pow = Math.pow(10, Math.floor(Math.log10(target)));
    for (var i = 0; i < marks.length; i++) if (marks[i] * pow >= target) return marks[i] * pow;
    return 10 * pow;
  }

  window.ChapterWidget = {
    title: 'ZeRO 記憶體分片視覺化',
    intro: '調整模型參數量 Ψ、資料平行度 N_d 與 ZeRO 階段，觀察每張 GPU 上 bf16 參數（2Ψ）、bf16 梯度（2Ψ）與 fp32 優化器狀態（kΨ=12Ψ）如何被分片，以及對應的通訊代價。',
    render: function (rootEl) {
      var state = { model: 1, ndExp: 3, stage: 1 }; // 預設 7B、N_d=8、ZeRO-1

      rootEl.innerHTML =
        '<div class="widget-panel">' +
        '  <div class="widget-row" style="row-gap:.7rem">' +
        '    <label style="display:flex;align-items:center;gap:.5rem;font-size:.9rem">模型參數量 Ψ' +
        '      <select data-ref="model"><option value="0">1B</option><option value="1" selected>7B</option><option value="2">70B</option></select></label>' +
        '    <label style="display:flex;align-items:center;gap:.5rem;flex:1;min-width:180px;font-size:.9rem">DP 度' +
        '      <input data-ref="nd" type="range" min="0" max="6" step="1" value="3" style="flex:1;min-width:90px">' +
        '      <span data-ref="ndLabel" style="font-family:var(--mono);font-size:.85rem;min-width:5.5em"></span></label>' +
        '  </div>' +
        '  <div class="widget-row" data-ref="stageRow" style="margin-top:.8rem;gap:.5rem"></div>' +
        '  <div data-ref="stageDesc" style="margin-top:.4rem;font-size:.82rem;color:var(--fg-muted)"></div>' +
        '</div>' +
        '<div class="widget-panel" style="margin-top:1rem">' +
        '  <div class="widget-row" style="justify-content:space-between;row-gap:.3rem">' +
        '    <div><span style="font-size:.85rem;color:var(--fg-muted)">每 GPU 記憶體：</span>' +
        '      <span data-ref="total" style="font-size:1.35rem;font-weight:700;font-family:var(--mono)"></span></div>' +
        '    <div data-ref="fit" style="font-size:.85rem;font-weight:600"></div>' +
        '  </div>' +
        '  <div data-ref="chart" style="margin-top:.5rem"></div>' +
        '  <div class="widget-row" data-ref="legend" style="margin-top:.4rem;gap:.6rem 1.2rem;font-size:.82rem"></div>' +
        '</div>' +
        '<div class="widget-panel" style="margin-top:1rem">' +
        '  <div style="font-size:.85rem;color:var(--fg-muted)">記憶體公式（每 GPU）</div>' +
        '  <div data-ref="formula" style="margin:.4rem 0;overflow-x:auto"></div>' +
        '  <div style="overflow-x:auto"><span data-ref="subLabel" style="font-size:.85rem;color:var(--fg-muted)"></span> <span data-ref="subst"></span></div>' +
        '  <div style="margin-top:.7rem;padding-top:.7rem;border-top:1px dashed var(--border);font-size:.85rem">' +
        '    <strong>📡 通訊代價：</strong><span data-ref="comm"></span></div>' +
        '  <div data-ref="interp" style="margin-top:.6rem;padding:.6rem .8rem;background:var(--accent-soft);border-radius:8px;font-size:.88rem"></div>' +
        '</div>';

      var refs = {};
      rootEl.querySelectorAll('[data-ref]').forEach(function (el) { refs[el.getAttribute('data-ref')] = el; });

      // 階段切換按鈕
      STAGES.forEach(function (name, i) {
        var b = document.createElement('button');
        b.type = 'button';
        b.textContent = name;
        b.addEventListener('click', function () { state.stage = i; update(false); });
        refs.stageRow.appendChild(b);
      });

      // 圖例
      var SEG_NAMES = ['bf16 參數（2Ψ）', 'bf16 梯度（2Ψ）', 'fp32 優化器狀態（12Ψ）'];
      var SEG_COLORS = ['var(--accent)', 'var(--link)', 'var(--fg-muted)'];
      var legendVals = SEG_NAMES.map(function (name, i) {
        var item = document.createElement('span');
        item.style.cssText = 'display:inline-flex;align-items:center;gap:.35rem';
        var chip = document.createElement('span');
        chip.style.cssText = 'width:11px;height:11px;border-radius:3px;display:inline-block;background:' + SEG_COLORS[i];
        var txt = document.createElement('span');
        txt.textContent = name + '：';
        var val = document.createElement('strong');
        val.style.fontFamily = 'var(--mono)';
        item.appendChild(chip); item.appendChild(txt); item.appendChild(val);
        refs.legend.appendChild(item);
        return val;
      });

      // SVG 圖表
      var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      var CH = 132;
      svg.setAttribute('height', CH);
      svg.setAttribute('width', '100%');
      svg.style.display = 'block';
      refs.chart.appendChild(svg);
      var chartW = 640;
      var disp = { p: 0, g: 0, o: 0, max: 100 };
      var warn = false, raf = null;

      function draw() {
        var W = chartW, mL = 8, mR = 8, barY = 34, barH = 42, axisY = barY + barH;
        var plotW = Math.max(10, W - mL - mR);
        function X(gb) { return mL + plotW * (gb / disp.max); }
        var s = '';
        // 刻度格線
        var step = niceCeil(disp.max / 4, [1, 2, 2.5, 5, 10]);
        for (var gb = 0; gb <= disp.max * 1.001; gb += step) {
          var x = X(gb);
          s += '<line x1="' + x + '" y1="' + (barY - 6) + '" x2="' + x + '" y2="' + axisY + '" stroke="var(--border)" stroke-width="1"/>';
          s += '<text x="' + x + '" y="' + (axisY + 16) + '" text-anchor="middle" font-size="11" fill="var(--fg-muted)" font-family="var(--sans)">' + fmtGB(gb) + (gb + step > disp.max * 1.001 ? ' GB' : '') + '</text>';
        }
        // 底軌
        s += '<rect x="' + mL + '" y="' + barY + '" width="' + plotW + '" height="' + barH + '" rx="6" fill="var(--panel)" stroke="var(--border)"/>';
        // 堆疊段
        var vals = [disp.p, disp.g, disp.o], cx = 0;
        for (var i = 0; i < 3; i++) {
          var w = plotW * (vals[i] / disp.max);
          if (w > 0.2) s += '<rect x="' + (mL + cx) + '" y="' + barY + '" width="' + w + '" height="' + barH + '" fill="' +
            (warn ? 'var(--accent-2)' : SEG_COLORS[i]) + '" fill-opacity="' + (warn ? [1, 0.68, 0.42][i] : 0.9) + '" stroke="var(--panel)" stroke-width="1"/>';
          cx += w;
        }
        // H100 80GB 參考線
        if (H100 <= disp.max) {
          var hx = X(H100), left = hx > W * 0.72;
          s += '<line x1="' + hx + '" y1="14" x2="' + hx + '" y2="' + (axisY + 4) + '" stroke="var(--accent-2)" stroke-width="1.5" stroke-dasharray="4 3"/>';
          s += '<text x="' + (hx + (left ? -6 : 6)) + '" y="12" text-anchor="' + (left ? 'end' : 'start') + '" font-size="11" font-weight="600" fill="var(--accent-2)" font-family="var(--sans)">H100 80GB</text>';
        }
        svg.setAttribute('viewBox', '0 0 ' + W + ' ' + CH);
        svg.innerHTML = s;
      }

      function animateTo(t) {
        if (raf) cancelAnimationFrame(raf);
        var from = { p: disp.p, g: disp.g, o: disp.o, max: disp.max };
        var t0 = performance.now(), dur = 450;
        function tick(now) {
          var u = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - u, 3);
          ['p', 'g', 'o', 'max'].forEach(function (k) { disp[k] = from[k] + (t[k] - from[k]) * e; });
          draw();
          if (u < 1) raf = requestAnimationFrame(tick);
        }
        raf = requestAnimationFrame(tick);
      }

      function update(instant) {
        var m = MODELS[state.model], nd = Math.pow(2, state.ndExp), st = state.stage;
        var unit = 2 * m.psi / 1e9; // 2Ψ 的 GB 數
        var p = st >= 3 ? unit / nd : unit;
        var g = st >= 2 ? unit / nd : unit;
        var o = st >= 1 ? (K / 2) * unit / nd : (K / 2) * unit;
        var total = p + g + o, base = 8 * unit; // 16Ψ
        var fits = total <= H100;
        var max = niceCeil(Math.max(total * 1.12, H100 * 1.15), [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]);

        // 控制列狀態
        refs.ndLabel.textContent = 'N_d = ' + nd;
        Array.prototype.forEach.call(refs.stageRow.children, function (b, i) {
          b.className = i === st ? '' : 'secondary';
        });
        refs.stageDesc.textContent = STAGES[st] + '：' + STAGE_DESC[st];

        // 讀數、圖例與警示
        refs.total.textContent = fmtGB(total) + ' GB';
        refs.total.style.color = fits ? 'var(--fg)' : 'var(--accent-2)';
        refs.fit.textContent = fits ? '✔ 可放入單張 H100（80GB）' : '⚠ 超出單張 H100（80GB）';
        refs.fit.style.color = fits ? 'var(--accent)' : 'var(--accent-2)';
        [p, g, o].forEach(function (v, i) { legendVals[i].textContent = fmtGB(v) + ' GB'; });

        // 公式與代入數值
        var f2 = fmtGB(unit), f12 = fmtGB((K / 2) * unit);
        var FORM = [
          '2\\Psi+2\\Psi+k\\Psi = 16\\Psi \\quad (k=12)',
          '2\\Psi+2\\Psi+\\frac{k\\Psi}{N_d}',
          '2\\Psi+\\frac{2\\Psi+k\\Psi}{N_d}',
          '\\frac{2\\Psi+2\\Psi+k\\Psi}{N_d}'
        ];
        var SUBST = [
          f2 + '+' + f2 + '+' + f12 + ' = ' + fmtGB(total) + '\\ \\text{GB}',
          f2 + '+' + f2 + '+\\tfrac{' + f12 + '}{' + nd + '} = ' + f2 + '+' + f2 + '+' + fmtGB(o) + ' = ' + fmtGB(total) + '\\ \\text{GB}',
          f2 + '+\\tfrac{' + f2 + '+' + f12 + '}{' + nd + '} = ' + f2 + '+' + fmtGB(g + o) + ' = ' + fmtGB(total) + '\\ \\text{GB}',
          '\\tfrac{' + f2 + '+' + f2 + '+' + f12 + '}{' + nd + '} = \\tfrac{' + fmtGB(base) + '}{' + nd + '} = ' + fmtGB(total) + '\\ \\text{GB}'
        ];
        tex(refs.formula, FORM[st]);
        refs.subLabel.textContent = '代入 Ψ=' + m.label + '、N_d=' + nd + '：';
        tex(refs.subst, SUBST[st]);
        refs.comm.textContent = COMM[st];

        // 動態解讀
        var msg = m.label + ' 模型在 ' + STAGES[st] + '、N_d=' + nd + ' 之下，每卡需 ' + fmtGB(total) + ' GB';
        if (st === 3) msg += '——但每層前向都要 all-gather 參數、反向再一輪（可用預取重疊，經驗法則 DP ≲ 512）';
        if (fits) msg += '，放得進單張 H100，還剩約 ' + fmtGB(H100 - total) + ' GB 給激活值與緩衝區。';
        else msg += '，超出 H100 容量 ' + fmtGB(total - H100) + ' GB——需要更高的 ZeRO 階段、更大的 N_d，或其他平行化維度。';
        if (st > 0 && nd === 1) msg += ' 注意：N_d=1 時分片沒有任何效果，公式退化為 16Ψ。';
        else if (st > 0) msg += ' 相較無 ZeRO（' + fmtGB(base) + ' GB）節省 ' + (base / total).toFixed(1) + ' 倍；但 ZeRO 無法分片激活值記憶體。';
        refs.interp.textContent = '💡 ' + msg;

        warn = !fits;
        if (instant) { disp = { p: p, g: g, o: o, max: max }; draw(); }
        else animateTo({ p: p, g: g, o: o, max: max });
      }

      refs.model.addEventListener('change', function () { state.model = +refs.model.value; update(false); });
      refs.nd.addEventListener('input', function () { state.ndExp = +refs.nd.value; update(false); });

      // 響應寬度變化
      function measure() { chartW = refs.chart.clientWidth || 640; }
      if (typeof ResizeObserver === 'function') {
        new ResizeObserver(function () { measure(); draw(); }).observe(refs.chart);
      }
      measure();
      update(true);
    }
  };
})();
