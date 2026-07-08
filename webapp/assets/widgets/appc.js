/* 附錄 A2 — LLM 訓練數量級計算器（依翻譯稿 appc.md 的粗略估算式） */
(function () {
  'use strict';

  var PRESETS = [
    { label: '1B 風格', h: 2048, L: 16, V: 128256, seq: 4096, mbs: 1 },
    { label: 'Llama-3-8B 風格', h: 4096, L: 32, V: 128256, seq: 8192, mbs: 1 },
    { label: '70B 風格', h: 8192, L: 80, V: 128256, seq: 8192, mbs: 1 }
  ];
  var DEFAULT = 1;      // 預設載入 Llama-3-8B 風格
  var H100_GB = 80;     // H100 記憶體上限（GB；本元件 1 GB = 10^9 bytes）

  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'text') node.textContent = attrs[k];
      else if (k === 'style') node.setAttribute('style', attrs[k]);
      else node[k] = attrs[k];
    });
    for (var i = 2; i < arguments.length; i++) node.appendChild(arguments[i]);
    return node;
  }
  function trim(x) { return x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2); }
  function sep(n) { return Math.round(n).toLocaleString('en-US'); }
  function fmtCount(n) { // 參數／元素個數：K / M / B / T
    if (n >= 1e12) return trim(n / 1e12) + ' T';
    if (n >= 1e9) return trim(n / 1e9) + ' B';
    if (n >= 1e6) return trim(n / 1e6) + ' M';
    if (n >= 1e3) return trim(n / 1e3) + ' K';
    return String(Math.round(n));
  }
  function fmtBytes(b) {
    if (b >= 1e12) return trim(b / 1e12) + ' TB';
    if (b >= 1e9) return trim(b / 1e9) + ' GB';
    if (b >= 1e6) return trim(b / 1e6) + ' MB';
    return trim(b / 1e3) + ' KB';
  }
  function fmtFlops(f) {
    if (f >= 1e15) return trim(f / 1e15) + ' PFLOPs';
    if (f >= 1e12) return trim(f / 1e12) + ' TFLOPs';
    return trim(f / 1e9) + ' GFLOPs';
  }
  function tex(s) {
    var box = el('div', { style: 'overflow-x:auto;margin:.5em 0 .1em;font-size:.9em;color:var(--fg);' });
    if (window.katex) window.katex.render(s, box, { throwOnError: false });
    else box.textContent = s;
    return box;
  }
  function note(t) {
    return el('div', { text: t, style: 'font-size:.76rem;color:var(--fg-muted);margin-top:.3em;line-height:1.6;' });
  }
  function card(title, value) {
    var c = el('div', { style: 'background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:.85em 1em;' },
      el('div', { text: title, style: 'font-size:.8rem;font-weight:700;color:var(--fg-muted);' }),
      el('div', { text: value, style: 'font-size:1.22rem;font-weight:700;color:var(--accent);margin:.12em 0;font-variant-numeric:tabular-nums;' }));
    for (var i = 2; i < arguments.length; i++) c.appendChild(arguments[i]);
    return c;
  }
  function bullet(t, strong) {
    return el('li', { text: t, style: 'margin:.35em 0;line-height:1.65;font-size:.85rem;color:var(--fg);' +
      (strong ? 'font-weight:600;' : '') });
  }

  window.ChapterWidget = {
    title: 'LLM 訓練數量級計算器',
    intro: '把附錄 A2 的粗略估算式變成即時計算器：輸入模型結構超參數，立刻看到參數量、bf16 權重與梯度、' +
      'fp32 優化器狀態、活化值與一步 FLOPs 的數量級——以及這一切放不放得進一張 80 GB 的 H100。',
    render: function (root) {
      var controls = el('div', { className: 'widget-panel' });
      var presetBtns = [];
      var presetRow = el('div', { className: 'widget-row', style: 'margin-bottom:.7em;' },
        el('span', { text: '預設模型：', style: 'font-size:.85rem;color:var(--fg-muted);' }));
      PRESETS.forEach(function (p) {
        var b = el('button', { type: 'button', text: p.label });
        b.addEventListener('click', function () { setValues(p); update(); });
        presetBtns.push(b);
        presetRow.appendChild(b);
      });
      controls.appendChild(presetRow);

      var fields = {};
      var inputRow = el('div', { className: 'widget-row' });
      [['h', '隱藏維度 h', '7em'], ['L', '層數 L', '5.5em'], ['V', '詞彙量 V', '8em'],
       ['seq', '序列長 seq', '7em'], ['mbs', '微批次 mbs', '5.5em']].forEach(function (f) {
        var input = el('input', { type: 'number', min: '1', step: '1', style: 'width:' + f[2] + ';', 'aria-label': f[1] });
        input.addEventListener('input', update);
        fields[f[0]] = input;
        inputRow.appendChild(el('label', { className: 'widget-row', style: 'gap:.4em;' },
          el('span', { text: f[1], style: 'font-size:.85rem;color:var(--fg-muted);' }), input));
      });
      controls.appendChild(inputRow);

      var grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:.8rem;margin-top:1rem;' });
      var verdictBox = el('div', { style: 'margin-top:1rem;' });
      var interp = el('div', { className: 'widget-panel', style: 'margin-top:1rem;' });

      function setValues(p) {
        fields.h.value = p.h; fields.L.value = p.L; fields.V.value = p.V;
        fields.seq.value = p.seq; fields.mbs.value = p.mbs;
      }
      function readVals() {
        var v = {}, ok = true;
        ['h', 'L', 'V', 'seq', 'mbs'].forEach(function (k) {
          var n = parseInt(fields[k].value, 10);
          if (!isFinite(n) || n < 1) ok = false;
          v[k] = n;
        });
        return ok ? v : null;
      }

      function update() {
        var v = readVals();
        PRESETS.forEach(function (p, i) {
          var match = v && v.h === p.h && v.L === p.L && v.V === p.V && v.seq === p.seq && v.mbs === p.mbs;
          presetBtns[i].className = match ? '' : 'secondary';
        });
        grid.textContent = ''; verdictBox.textContent = ''; interp.textContent = '';
        if (!v) {
          grid.appendChild(el('p', { text: '請在所有欄位輸入正整數（h、L、V、seq、mbs）。',
            style: 'color:var(--accent-2);margin:0;grid-column:1/-1;' }));
          return;
        }
        // ---- 依 A2 公式計算 ----
        var attn = 4 * v.h * v.h * v.L;              // 注意力：QKV 3h² + 輸出投影 h²，共 4h²·L
        var mlp = 12 * v.h * v.h * v.L;              // GLU MLP：gate+up 8h² + down 4h²，共 12h²·L
        var other = 2 * v.V * v.h;                   // 輸入嵌入 Vh + LM head Vh（未綁定共用）
        var N = attn + mlp + other;                  // N = 16h²L + 2Vh
        var wBytes = 2 * N, gBytes = 2 * N;          // bf16 權重、梯度：各 2 bytes／參數
        var optBytes = 12 * N;                       // fp32 動量 + 變異數 + 主權重：各 4 bytes／參數
        var staticBytes = wBytes + gBytes + optBytes; // = 16N bytes
        var actElems = v.seq * v.mbs * v.h;          // 單層隱藏狀態元素數
        var tokens = v.seq * v.mbs;                  // 每步處理的 token 數
        var flops = 6 * tokens * N;                  // 前向 2·tokens·N + 反向 4·tokens·N
        var flopsAttn = 12 * v.L * v.h * v.seq * v.seq * v.mbs; // 注意力二次項（精確式的補正）
        var attnShare = flopsAttn / (flops + flopsAttn) * 100;

        // ---- 結果卡片 ----
        var c1 = card('模型參數量 N', fmtCount(N) + ' 參數',
          tex('N = \\underbrace{4h^2L}_{\\text{注意力}} + \\underbrace{12h^2L}_{\\text{GLU MLP}} + \\underbrace{2Vh}_{\\text{嵌入+LM head}}'),
          note('= 16×' + v.h + '²×' + v.L + ' + 2×' + v.V + '×' + v.h + ' = ' + sep(N)),
          note('注意力 4h²L = ' + fmtCount(attn) + '｜GLU MLP 12h²L = ' + fmtCount(mlp) +
            '｜嵌入+LM head 2Vh = ' + fmtCount(other)),
          note('依 A2：QKV 3h² + 輸出投影 h²；gate/up 8h² + down 4h²；LM head 未與嵌入綁定；RoPE 類位置編碼無參數。'));
        var c2 = card('bf16 權重 + 梯度', fmtBytes(wBytes + gBytes),
          tex('M_{w+g} = (2 + 2)\\,\\mathrm{bytes} \\times N = 4N\\ \\mathrm{bytes}'),
          note('= 4 × ' + sep(N) + ' B ≈ ' + fmtBytes(wBytes + gBytes) +
            '（權重 ' + fmtBytes(wBytes) + ' + 梯度 ' + fmtBytes(gBytes) + '，bf16 每元素 2 bytes）'));
        var c3 = card('fp32 優化器狀態（Adam）', fmtBytes(optBytes),
          tex('M_{\\mathrm{opt}} = (4 + 4 + 4)\\,\\mathrm{bytes} \\times N = 12N\\ \\mathrm{bytes}'),
          note('= 12 × ' + sep(N) + ' B ≈ ' + fmtBytes(optBytes) + '，動量 + 變異數 + fp32 主權重各 4 bytes／參數'),
          note('對應 A2：每個 h² 權重矩陣的優化器狀態約 6h² 個（bf16 等效）元素（2×2h² + 2h²）。'));
        var c4 = card('單層活化值（隱藏狀態）', fmtCount(actElems) + ' 元素 ≈ ' + fmtBytes(2 * actElems),
          tex('A_{\\text{層}} = seq \\cdot mbs \\cdot h'),
          note('= ' + v.seq + ' × ' + v.mbs + ' × ' + v.h + ' = ' + sep(actElems) + ' 元素，bf16（2 bytes）≈ ' +
            fmtBytes(2 * actElems)),
          note('全模型 ' + v.L + ' 層的隱藏狀態合計 ≈ ' + fmtBytes(2 * actElems * v.L) +
            '——僅計隱藏狀態張量，未含注意力分數等中間活化。'));
        var c5 = card('一步 FLOPs（前向 + 反向）', fmtFlops(flops),
          tex('F \\approx 6 \\cdot seq \\cdot mbs \\cdot N'),
          note('= 6 × ' + sep(tokens) + ' × ' + sep(N) + ' ≈ ' + fmtFlops(flops) +
            '（前向 2·tokens·N + 反向 4·tokens·N；每步 tokens = seq×mbs = ' + sep(tokens) + '）'),
          tex('F_{\\text{精確}} = 6\\,seq\\,mbs\\,N + 12\\,L\\,h\\,seq^2\\,mbs'),
          note('注意力二次項 12·L·h·seq²·mbs ≈ ' + fmtFlops(flopsAttn) + '（佔總量 ' + trim(attnShare) +
            '%）；A2 假設 seq² ≪ h 時可忽略此項。'));
        [c1, c2, c3, c4, c5].forEach(function (c) { grid.appendChild(c); });

        // ---- 「放得進一張 H100 嗎？」判定 ----
        var pct = staticBytes / (H100_GB * 1e9) * 100;
        var fits = pct <= 100;
        var tone = fits ? 'var(--accent)' : 'var(--accent-2)';
        var need = Math.ceil(staticBytes / (H100_GB * 1e9));
        var verdict = el('div', { style: 'border:1px solid ' + tone + ';border-left:5px solid ' + tone +
          ';border-radius:10px;padding:.85em 1em;background:var(--panel);' },
          el('div', { text: '這放得進一張 H100（' + H100_GB + ' GB）嗎？——' + (fits ? '放得下' : '放不下'),
            style: 'font-weight:700;color:' + tone + ';' }),
          el('div', { text: '權重 ' + fmtBytes(wBytes) + ' + 梯度 ' + fmtBytes(gBytes) + ' + 優化器 ' +
            fmtBytes(optBytes) + ' = ' + fmtBytes(staticBytes) + '（16N bytes），為 ' + H100_GB + ' GB 的 ' +
            trim(pct) + '%', style: 'font-size:.82rem;color:var(--fg-muted);margin-top:.35em;' }),
          el('div', { style: 'margin-top:.55em;height:.8em;border:1px solid var(--border);border-radius:5px;background:var(--panel-2);overflow:hidden;' },
            el('div', { style: 'width:' + Math.min(100, pct).toFixed(1) + '%;height:100%;background:' + tone + ';' })),
          note(fits
            ? '靜態記憶體塞得進單卡，但別忘了活化值與框架開銷還要另外吃記憶體——實務上仍可能需要重算（recomputation）或梯度累積。'
            : '光是靜態記憶體就需要至少 ' + need + ' 張 ' + H100_GB + ' GB GPU 分攤（還沒算活化值），' +
              '這正是第 3 章起各種平行化與 ZeRO 分片登場的原因。'));
        verdictBox.appendChild(verdict);

        // ---- 動態解讀 ----
        var ul = el('ul', { style: 'margin:.4em 0 0;padding-left:1.2em;' });
        ul.appendChild(bullet('訓練全家桶 = 權重(1) + 梯度(1) + 優化器(6) ≈ 每參數 8 個 bf16 等效元素 = 16 bytes——' +
          '參數記憶體 ×8 於參數量（bf16 訓練全家桶），這就是 A2 的經驗法則。本組：' + fmtCount(N) + ' 參數 → ' +
          fmtBytes(staticBytes) + '，恰為 bf16 權重（' + fmtBytes(wBytes) + '）的 8 倍。', true));
        ul.appendChild(bullet('參數組成：GLU MLP 佔 ' + trim(mlp / N * 100) + '%、注意力 ' + trim(attn / N * 100) +
          '%、嵌入+LM head ' + trim(other / N * 100) + '%。' +
          (other / N > 0.25
            ? '模型越小，詞彙表相關參數的占比越驚人——這也是小模型常把嵌入與 LM head 綁定共用的原因。'
            : '每個 transformer 區塊 16h²（GLU）中，MLP 以 12h² 對 4h² 穩居大宗。')));
        ul.appendChild(bullet(attnShare >= 5
          ? '此 seq = ' + v.seq + ' 下，注意力二次項已佔一步 FLOPs 約 ' + trim(attnShare) +
            '%，「seq² ≪ h」的簡化假設開始失真——長上下文訓練不能只用 6·tokens·N 估算。'
          : '注意力二次項僅佔一步 FLOPs 約 ' + trim(attnShare) + '%，6·tokens·N 的近似在此設定下夠用。'));
        interp.appendChild(el('div', { text: '動態解讀', style: 'font-weight:700;margin-bottom:.3em;' }));
        interp.appendChild(ul);
        interp.appendChild(note('單位約定：1 GB = 10⁹ bytes、參數量 B = 10⁹；所有數字皆為 A2 的數量級估算，' +
          '實際模型（GQA、非 4h 的 MLP 中間維度等）會略有出入。'));
      }

      root.appendChild(controls);
      root.appendChild(grid);
      root.appendChild(verdictBox);
      root.appendChild(interp);
      setValues(PRESETS[DEFAULT]);
      update();
    }
  };
})();
