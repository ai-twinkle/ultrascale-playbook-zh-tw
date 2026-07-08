/* 第 8 章互動元件：最佳配置三步驟精靈 */
(function () {
  'use strict';

  var CSS = [
    '.ch08-wiz .field { display: flex; flex-direction: column; gap: .35rem; min-width: 130px; flex: 1; }',
    '.ch08-wiz .field-wide { flex-basis: 100%; }',
    '.ch08-wiz .val { font-weight: 600; color: var(--accent); font-variant-numeric: tabular-nums; }',
    '.ch08-wiz .actions { margin-top: 1rem; display: flex; gap: .6rem; }',
    '.ch08-wiz .step-card { margin-top: 1rem; opacity: 0; transform: translateY(10px); transition: opacity .45s ease, transform .45s ease; }',
    '.ch08-wiz .step-card.show { opacity: 1; transform: none; }',
    '.ch08-wiz .step-head { display: flex; align-items: center; gap: .6rem; margin-bottom: .5rem; }',
    '.ch08-wiz .step-num { flex: none; width: 1.7rem; height: 1.7rem; border-radius: 50%; background: var(--accent); color: var(--bg); font: 700 .85rem/1.7rem sans-serif; text-align: center; }',
    '.ch08-wiz .step-title { font-weight: 700; color: var(--fg); }',
    '.ch08-wiz .combo { padding: .55rem .8rem; background: var(--accent-soft); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; font-size: .92rem; color: var(--fg); margin: .5rem 0; }',
    '.ch08-wiz .warn { padding: .5rem .8rem; border: 1px dashed var(--accent-2); border-radius: 8px; font-size: .84rem; color: var(--fg); margin: .5rem 0; }',
    '.ch08-wiz .math { font-family: ui-monospace, monospace; font-size: .84rem; background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; padding: .55rem .8rem; margin: .5rem 0; overflow-x: auto; white-space: nowrap; color: var(--fg); }',
    '.ch08-wiz ul.tips { margin: .4rem 0 .2rem 1.2rem; padding: 0; font-size: .88rem; color: var(--fg); }',
    '.ch08-wiz ul.tips li { margin: .3rem 0; }',
    '.ch08-wiz details { margin-top: .6rem; border: 1px solid var(--border); border-radius: 8px; background: var(--panel); }',
    '.ch08-wiz details summary { cursor: pointer; padding: .45rem .8rem; font-size: .82rem; font-weight: 600; color: var(--link); }',
    '.ch08-wiz details p { margin: 0; padding: .1rem .9rem .7rem; font-size: .84rem; color: var(--fg-muted); line-height: 1.7; }',
    '.ch08-wiz .foot { margin-top: .6rem; font-size: .76rem; color: var(--fg-muted); }',
    '.ch08-wiz .desc { font-size: .86rem; color: var(--fg-muted); margin: .1rem 0 .4rem; }'
  ].join('\n');

  var MODELS = { '1': '1B', '8': '8B', '70': '70B', '405': '405B' };
  var DEFAULTS = { model: '8', gpuExp: 6, gbs: '1048576', seq: '4096' };

  function fmt(n) { return n.toLocaleString('en-US'); }

  /* ---------- 步驟一：塞進記憶體（依本章規則） ---------- */
  function planStep1(p, gpus, seq) {
    var tp = 1, pp = 1, zero = 1, combo, reason, warn = null;
    if (p < 10) {
      if (gpus >= 512) {
        tp = 8; zero = 1;
        combo = 'TP=8（節點內）＋ DP（ZeRO-1）';
        reason = '模型小於 10B，本可用單卡＋ZeRO 或 TP=8 單一技術解決；但到了 512+ GPU 的規模，純 DP／ZeRO-3 會因通訊成本變得沒有效率，故結合節點內 TP。';
      } else if (p <= 1) {
        tp = 1; zero = 3;
        combo = '單卡可容納 ＋ 純 DP（ZeRO-3，搭配完整重算）';
        reason = '10B 以下的模型可以只用單一平行化技術：搭配完整重算的 ZeRO-3/DP，或改在 8 張 GPU 上使用 TP=8。';
      } else {
        tp = 8; zero = 1;
        combo = 'TP=8（單節點）；或改用 ZeRO-3/DP ＋ 完整重算';
        reason = '10B 以下的模型可以只用單一平行化技術：在 8 張 GPU 上使用張量平行（TP=8），或搭配完整重算的 ZeRO-3/DP。';
      }
    } else if (p < 100) {
      if (gpus >= 16) {
        tp = 8; pp = 2; zero = 1;
        combo = 'TP=8 ＋ PP=2（每份模型實例佔 16 張 GPU）';
        reason = '10B–100B 的模型需要超過 8 張 GPU：可選 TP=8＋PP、TP=8＋ZeRO-3，或純 ZeRO-3。這裡以 TP=8＋PP 為主要建議。';
      } else {
        tp = 8; zero = 3;
        combo = 'TP=8 ＋ ZeRO-3';
        reason = '10B–100B 的模型需要超過 8 張 GPU；GPU 不足 16 張時，以 TP=8 結合 ZeRO-3 分攤參數。';
        warn = '8 張 GPU 對這個大小相當吃緊（GPU 匱乏情境）：建議啟用完整激活值重算、增加梯度累積，或增加 GPU。';
      }
    } else {
      if (gpus >= 64) {
        tp = 8; pp = 8; zero = 1;
        combo = 'TP=8 ＋ PP=8（每份模型實例佔 64 張 GPU）';
        reason = '超過 100B 的模型，在 TP=8 之外需要加大 PP 深度來分攤參數與記憶體。';
      } else {
        tp = 8; pp = Math.max(1, gpus / 8); zero = 3;
        combo = 'TP=8 ＋ PP=' + pp + ' ＋ ZeRO-3（勉強嘗試）';
        reason = '超過 100B 的模型建議 TP=8 並加大 PP。';
        warn = '目前 GPU 數不足以妥善容納 405B（建議至少 64 張：TP=8 × PP=8）。請啟用完整重算並增加 GPU。';
      }
    }
    if (gpus >= 1024 && pp === 1) {
      pp = 2; zero = 2;
      combo += '；1024+ GPU 規模下再加上 PP（建議設定：TP=8 ＋ ZeRO-2 ＋ PP）';
    }
    var cp = 1, cpNote = null;
    if (seq >= 32768 && gpus / (tp * pp) >= 2) {
      cp = 2;
      cpNote = '序列長度達 ' + fmt(seq) + '：建議跨節點加上上下文平行（CP=2）以分攤長序列的激活值記憶體。';
    }
    return { tp: tp, pp: pp, cp: cp, zero: zero, combo: combo, reason: reason, warn: warn, cpNote: cpNote };
  }

  /* ---------- 步驟二：達到目標 gbs（token 帳） ---------- */
  function planStep2(s1, gpus, gbsTok, seq) {
    var dp = gpus / (s1.tp * s1.pp * s1.cp);
    var samples = gbsTok / seq;                      // 目標 gbs 換算成序列數
    var mbs = s1.tp * s1.pp === 1 ? 4 : (s1.pp >= 8 ? 1 : 2); // 依模型佔用粗估
    if (seq >= 8192) mbs = Math.max(1, mbs >> 1);
    if (seq >= 32768) mbs = 1;
    var warn = null, ga;
    if (dp > samples) {
      mbs = 1; ga = 1;
      warn = 'DP=' + dp + ' 已超過目標 gbs 所需的序列數（' + fmt(samples) + '）：實際 gbs 會是 ' + fmt(dp * seq) +
        ' tokens，超出目標。依本章建議「縮減資料平行、改採其他平行化策略」（加大 TP/PP，長序列則調整 CP），或改選較大的目標 gbs。';
    } else {
      var perRank = samples / dp;                    // 每個 DP rank 每步要吃的序列數
      if (mbs > perRank) mbs = perRank;
      ga = Math.max(1, Math.round(perRank / mbs));
    }
    var actual = dp * mbs * ga * seq;
    return { dp: dp, mbs: mbs, ga: ga, samples: samples, actual: actual, hit: actual === gbsTok, warn: warn };
  }

  /* ---------- 步驟三：吞吐量提示 ---------- */
  function planStep3(s1, s2, gpus) {
    var tips = [];
    tips.push(s1.tp < 8
      ? '將張量平行從 TP=' + s1.tp + ' 擴大到節點內上限（TP=8），利用節點內 NVLink 高速頻寬，減少對其他平行化的需求。'
      : 'TP 已達節點大小（TP=8）：跨節點的 TP 通訊昂貴，不建議再放大，改調整其他維度。');
    tips.push('嘗試多種微批次大小：從 mbs=' + s2.mbs + ' 逐步加大直到逼近 OOM，攤薄每步開銷；mbs 加倍時把梯度累積減半（' + s2.ga + ' → ' + Math.max(1, s2.ga >> 1) + '）即可維持相同 gbs。');
    if (s1.zero === 3) {
      tips.push('在維持目標 gbs 的前提下，增加使用 ZeRO-3 的資料平行；當 DP 通訊開始成為瓶頸時，轉而使用管線平行（PP）。');
    } else {
      tips.push('DP 擴展的通訊代價：目前 DP=' + s2.dp + '，梯度同步／ZeRO 通訊量隨 DP 上升' + (gpus >= 512 ? '——在 512+ GPU 規模下尤其明顯' : '') + '；當 DP 通訊成為瓶頸，改把資源投入 PP。');
    }
    tips.push('逐一嘗試擴大各種平行化，在 gbs、模型大小、計算與通訊之間找最佳平衡——最終仍以實測吞吐量（如 MFU）為準。');
    return tips;
  }

  /* ---------- 卡片渲染 ---------- */
  function card(num, title, bodyHTML, whyHTML) {
    var div = document.createElement('div');
    div.className = 'widget-panel step-card';
    div.innerHTML =
      '<div class="step-head"><span class="step-num">' + num + '</span><span class="step-title">' + title + '</span></div>' +
      bodyHTML +
      '<details><summary>為什麼？（本章的取捨）</summary><p>' + whyHTML + '</p></details>' +
      '<div class="foot">⚠️ 以上為粗略指南，實際仍需基準測試（見本章「對數千種配置進行基準測試」）。</div>';
    return div;
  }

  function render(rootEl) {
    var style = document.createElement('style');
    style.textContent = CSS;
    rootEl.appendChild(style);

    var wiz = document.createElement('div');
    wiz.className = 'ch08-wiz';
    wiz.innerHTML =
      '<div class="widget-panel">' +
        '<div class="desc">設定你的模型與叢集條件，依本章「步驟一 → 二 → 三」的決策流程產生一份起步配置建議。</div>' +
        '<div class="widget-row">' +
          '<label class="field">模型大小' +
            '<select data-k="model"><option value="1">1B</option><option value="8" selected>8B</option><option value="70">70B</option><option value="405">405B</option></select></label>' +
          '<label class="field">目標 gbs（tokens）' +
            '<select data-k="gbs"><option value="1048576" selected>1M（1,048,576）</option><option value="4194304">4M（4,194,304）</option></select></label>' +
          '<label class="field">序列長度' +
            '<select data-k="seq"><option value="4096" selected>4k（4,096）</option><option value="8192">8k（8,192）</option><option value="32768">32k（32,768）</option></select></label>' +
          '<label class="field field-wide">GPU 數（2 的冪）：<span class="val" data-k="gpuVal">64</span>' +
            '<input type="range" min="3" max="10" step="1" value="6" data-k="gpuExp"></label>' +
        '</div>' +
        '<div class="actions"><button data-k="go">產生建議</button><button class="secondary" data-k="reset">重設</button></div>' +
      '</div>' +
      '<div data-k="out"></div>';
    rootEl.appendChild(wiz);

    var $ = function (k) { return wiz.querySelector('[data-k="' + k + '"]'); };
    var out = $('out');

    $('gpuExp').addEventListener('input', function () {
      $('gpuVal').textContent = fmt(Math.pow(2, +this.value));
    });

    function run() {
      var p = +$('model').value;
      var gpus = Math.pow(2, +$('gpuExp').value);
      var gbsTok = +$('gbs').value;
      var seq = +$('seq').value;

      var s1 = planStep1(p, gpus, seq);
      var s2 = planStep2(s1, gpus, gbsTok, seq);
      var tips = planStep3(s1, s2, gpus);

      out.innerHTML = '';
      var body1 =
        '<div class="combo">建議組合：<strong>' + s1.combo + '</strong>（ZeRO-' + s1.zero + (s1.cp > 1 ? '，CP=' + s1.cp : '') + '）</div>' +
        '<div class="desc">' + s1.reason + '</div>' +
        (s1.cpNote ? '<div class="combo">' + s1.cpNote + '</div>' : '') +
        (s1.warn ? '<div class="warn">😭 ' + s1.warn + '</div>' : '');
      var why1 = '先確保「一份完整的模型實例」放得進 GPU：GPU 充裕時依模型大小挑平行化組合；GPU 匱乏時可用完整激活值重算以計算換記憶體（訓練稍慢），或增加梯度累積在有限記憶體下處理更大批次。本章亦提醒：混合專家（MoE）架構可跨節點使用專家平行（EP）。';

      var acct = 'gbs = dp × mbs × grad_acc × seq = ' + s2.dp + ' × ' + s2.mbs + ' × ' + s2.ga + ' × ' + fmt(seq) + ' = ' + fmt(s2.actual) + ' tokens';
      var body2 =
        '<div class="combo">建議：<strong>mbs = ' + s2.mbs + '，grad_acc = ' + s2.ga + '</strong>（DP=' + s2.dp + '）</div>' +
        '<div class="math">' + acct + '</div>' +
        '<div class="desc">目標 gbs = ' + fmt(gbsTok) + ' tokens（即每步 ' + fmt(s2.samples) + ' 條長度 ' + fmt(seq) + ' 的序列）' +
        (s2.hit ? ' → ✅ 帳面剛好吻合。' : ' → 與實際值有出入，見下方提醒。') + '</div>' +
        (s2.warn ? '<div class="warn">' + s2.warn + '</div>' : '');
      var why2 = '步驟一結束後的 mbs 與 DP 未必剛好湊出目標批次大小。要「加大」gbs：擴大資料平行或增加梯度累積步數，長序列則可利用上下文平行；要「縮小」gbs：縮減資料平行、改採其他平行化策略，或降低上下文平行的程度。';

      var body3 = '<ul class="tips">' + tips.map(function (t) { return '<li>' + t + '</li>'; }).join('') + '</ul>';
      var why3 = '模型與批次大小的大方向配置跑起來之後，剩下的問題是「用最快的方式訓練」：只要記憶體與通訊還不是瓶頸，就優先吃滿節點內高速頻寬（TP 靠近節點大小）、增加 ZeRO-3 的 DP、DP 通訊成瓶頸時轉 PP，並嘗試多種 mbs。本章實測也顯示效能高度取決於實作品質（TP 與 PP 的相對快慢曾因程式碼最佳化而互換）。';

      var cards = [
        card(1, '塞進記憶體（Fitting in Memory）', body1, why1),
        card(2, '達到目標全域批次大小（gbs）', body2, why2),
        card(3, '最佳化訓練吞吐量', body3, why3)
      ];
      cards.forEach(function (c, i) {
        out.appendChild(c);
        setTimeout(function () { c.classList.add('show'); }, 120 + i * 260);
      });
    }

    function reset() {
      $('model').value = DEFAULTS.model;
      $('gbs').value = DEFAULTS.gbs;
      $('seq').value = DEFAULTS.seq;
      $('gpuExp').value = DEFAULTS.gpuExp;
      $('gpuVal').textContent = fmt(Math.pow(2, DEFAULTS.gpuExp));
      out.innerHTML = '';
    }

    $('go').addEventListener('click', run);
    $('reset').addEventListener('click', reset);
  }

  window.ChapterWidget = {
    title: '最佳配置三步驟精靈',
    intro: '依本章的決策流程——步驟一「塞進記憶體」、步驟二「達到目標 gbs」、步驟三「最佳化吞吐量」——輸入模型大小、GPU 數、目標批次與序列長度，產生一份可作為起點的平行化配置建議。',
    render: render
  };
})();
