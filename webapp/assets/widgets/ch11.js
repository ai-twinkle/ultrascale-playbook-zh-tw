/* 第 11 章互動元件：全書回顧小測驗（10 題單選） */
(function () {
  'use strict';

  // ---------- 章節資料（推薦重讀連結用） ----------
  const CHAPTERS = {
    ch01: '第 1 章 第一步：單 GPU 訓練',
    ch02: '第 2 章 資料平行',
    ch03: '第 3 章 張量平行',
    ch04: '第 4 章 上下文平行',
    ch05: '第 5 章 管線平行',
    ch06: '第 6 章 專家平行',
    ch10: '第 10 章 融合核心、Flash Attention 與混合精度',
  };

  // ---------- 題庫（ans 為正解索引；ch 為對應章節） ----------
  const QUESTIONS = [
    {
      q: '在 bf16 混合精度搭配 Adam 訓練時，「優化器狀態」（fp32 主權重＋一階動量＋二階動量）每個參數共需多少位元組？',
      opts: ['4 位元組', '8 位元組', '12 位元組', '16 位元組'],
      ans: 2, ch: 'ch02',
      why: 'fp32 主權重、一階動量、二階動量各佔 4 位元組，合計 12Ψ；再加上 bf16 參數 2Ψ 與 bf16 梯度 2Ψ，整體約 16Ψ（不含 fp32 梯度累積）。',
    },
    {
      q: '使用「完全」活化重算（full activation recomputation）節省記憶體，主要代價是什麼？',
      opts: [
        'GPU 之間的通訊量大幅增加',
        '反向傳播時要重跑一次前向，計算時間增加約 30–40%',
        '梯度會變得不精確，必須調低學習率',
        '優化器狀態的記憶體佔用加倍',
      ],
      ans: 1, ch: 'ch01',
      why: '重算是用計算換記憶體：只存少數檢查點、反向時即時重算活化。選擇性重算更划算——GPT-3 只花 2.7% 計算就省下 70% 活化記憶體。',
    },
    {
      q: 'ZeRO-2 沿資料平行維度分片（shard）了哪些東西？',
      opts: [
        '只有優化器狀態',
        '優化器狀態＋梯度',
        '優化器狀態＋梯度＋模型參數',
        '模型參數＋激活值',
      ],
      ans: 1, ch: 'ch02',
      why: 'ZeRO-1 切優化器狀態；ZeRO-2 再加上梯度（把 all-reduce 換成 reduce-scatter）；ZeRO-3／FSDP 才進一步切參數。激活值是 ZeRO 切不了的。',
    },
    {
      q: '張量平行（TP）為什麼通常限制在單一節點內（例如 TP ≤ 8）？',
      opts: [
        '跨節點時矩陣切分無法對齊，計算結果會出錯',
        'NCCL 不支援跨節點的 all-reduce 操作',
        '跨節點會讓激活值記憶體爆增',
        'TP 每層都要通訊且難與計算重疊，跨節點頻寬遠低於 NVLink，會嚴重拖慢',
      ],
      ans: 3, ch: 'ch03',
      why: 'TP 的通訊位於關鍵路徑、每層前後都要做，得靠節點內 NVLink 的高頻寬才撐得住；一跨到節點間網路，從 TP=8 到 TP=16 吞吐量就大幅下滑。',
    },
    {
      q: 'AFAB／1F1B 管線排程中，有 p 個管線階段、m 個微批次時，氣泡時間佔理想計算時間的比例是？',
      opts: ['p − 1', '(p − 1) / m', 'm / (p − 1)', '(m − 1) / p'],
      ans: 1, ch: 'ch05',
      why: '氣泡固定為 (p−1)·(t_f+t_b)，理想計算時間為 m·(t_f+t_b)，故比例為 (p−1)/m——增加微批次數 m 可以把氣泡攤薄。',
    },
    {
      q: 'Ring Attention 中，各 GPU 在「環」上依序傳遞給下一張 GPU 的是什麼？',
      opts: ['查詢（Q）分塊', '鍵與值（K/V）分塊', '注意力分數矩陣', '各層的梯度'],
      ans: 1, ch: 'ch04',
      why: '每張 GPU 保留自己的 Q 分塊不動，一邊計算局部注意力、一邊把 K/V 傳給環上的下一張 GPU，讓通訊與計算重疊起來。',
    },
    {
      q: '訓練 MoE 模型時，專家平行（EP）的 all-to-all 通訊發生在哪裡？',
      opts: [
        'MoE 層前後：router 把 token 分派給各 GPU 上的專家，算完再收回',
        '注意力層內部，用來交換各頭的 K/V',
        '優化器步驟時，用來同步優化器狀態',
        '每個 epoch 結束時，用來重新平衡專家',
      ],
      ans: 0, ch: 'ch06',
      why: 'token 由 router 動態指派給散在各 GPU 上的專家：先 all-to-all 分發（dispatch），專家算完再 all-to-all 收回結果（combine）。',
    },
    {
      q: '同時使用梯度累積與資料平行時，全域批次大小 gbs 等於？',
      opts: ['mbs × grad_acc', 'mbs × dp', 'mbs × grad_acc × dp', 'mbs + grad_acc + dp'],
      ans: 2, ch: 'ch02',
      why: 'gbs ＝ 微批次大小 × 梯度累積步數 × DP 度。固定目標 gbs 時，這三個旋鈕可以互相取捨（例如加大 dp 就能減少 grad_acc）。',
    },
    {
      q: 'Flash Attention 之所以又快又省記憶體，關鍵在於？',
      opts: [
        '改用線性注意力近似，把複雜度降到 O(n)',
        '分塊計算、避免把注意力分數矩陣 S 具現化到 HBM，盡量在 SRAM 內完成',
        '以 fp8 低精度計算 softmax 來減少運算量',
        '跳過因果遮罩下三角以外的所有計算',
      ],
      ans: 1, ch: 'ch10',
      why: '樸素實作得把巨大的 S、P 矩陣寫回慢速 HBM 再讀回；Flash Attention 分塊計算、只保留 softmax 所需統計量，是精確計算而非近似。',
    },
    {
      q: '與 fp16 相比，bf16 的主要優勢是什麼？',
      opts: [
        '尾數位更多，數值精度比 fp16 高',
        '佔用記憶體只有 fp16 的一半',
        '在所有 GPU 上運算速度都是 fp16 的兩倍',
        '指數位和 fp32 一樣是 8 位，動態範圍與 fp32 相同，較不易溢位／下溢',
      ],
      ans: 3, ch: 'ch10',
      why: 'bf16 犧牲尾數（7 位，精度其實低於 fp16 的 10 位）換取 fp32 等級的動態範圍，因此通常不需要 loss scaling 就能穩定訓練。',
    },
  ];

  // ---------- 小工具 ----------
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'text') n.textContent = attrs[k];
      else if (k === 'html') n.innerHTML = attrs[k];
      else if (k === 'style') n.style.cssText = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    (children || []).forEach((c) => n.appendChild(c));
    return n;
  }
  // 簡單 LCG 偽隨機洗牌
  function shuffled(arr) {
    let seed = (Date.now() % 2147483647) || 42;
    const rnd = () => (seed = (seed * 48271) % 2147483647) / 2147483647;
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  function chapterLink(ch) {
    return el('a', { href: ch + '.html', text: CHAPTERS[ch] });
  }

  // ---------- 元件本體 ----------
  window.ChapterWidget = {
    title: '全書回顧小測驗',
    intro: '10 題單選題，涵蓋單 GPU 訓練到 5D 平行與 GPU 核心的核心概念。每題作答後立即顯示解析；全部答完會依錯題推薦值得重讀的章節。',
    render(root) {
      const state = { order: [], idx: 0, score: 0, wrong: [] };

      // 進度列
      const progText = el('div', { style: 'font-size:.88rem; color:var(--fg-muted); margin-bottom:.35rem; font-variant-numeric:tabular-nums;' });
      const progFill = el('div', { style: 'height:100%; width:0; background:var(--accent); border-radius:99px; transition:width .25s;' });
      const progBar = el('div', { style: 'height:8px; background:var(--panel-2); border:1px solid var(--border); border-radius:99px; overflow:hidden;' }, [progFill]);
      const progPanel = el('div', { style: 'margin-bottom:1rem;' }, [progText, progBar]);

      // 題目面板
      const qPanel = el('div', { class: 'widget-panel' });
      root.appendChild(progPanel);
      root.appendChild(qPanel);

      function updateProgress() {
        const done = state.idx;
        progText.textContent = '進度 ' + Math.min(done, QUESTIONS.length) + ' / ' + QUESTIONS.length + ' 題 · 目前答對 ' + state.score + ' 題';
        progFill.style.width = (done / QUESTIONS.length * 100) + '%';
      }

      function showQuestion() {
        updateProgress();
        qPanel.textContent = '';
        const item = state.order[state.idx];
        qPanel.appendChild(el('div', { style: 'font-size:.8rem; color:var(--accent); font-weight:600; margin-bottom:.4rem;', text: '第 ' + (state.idx + 1) + ' 題' }));
        qPanel.appendChild(el('div', { style: 'font-weight:600; margin-bottom:.8rem; line-height:1.6;', text: item.q }));

        const optWrap = el('div', { style: 'display:flex; flex-direction:column; gap:.5rem;' });
        const btns = item.opts.map((opt, i) => {
          const b = el('button', { class: 'secondary', style: 'width:100%; text-align:left; white-space:normal; line-height:1.5;', text: String.fromCharCode(65 + i) + '. ' + opt });
          b.addEventListener('click', () => answer(item, i, btns));
          optWrap.appendChild(b);
          return b;
        });
        qPanel.appendChild(optWrap);
      }

      function answer(item, picked, btns) {
        const correct = picked === item.ans;
        if (correct) state.score++;
        else state.wrong.push(item);

        btns.forEach((b, i) => {
          b.disabled = true;
          if (i === item.ans) {
            b.style.borderColor = 'var(--accent)';
            b.style.background = 'var(--accent-soft)';
            b.style.color = 'var(--fg)';
            b.style.fontWeight = '600';
            b.textContent = '✓ ' + b.textContent;
          } else if (i === picked) {
            b.style.opacity = '.6';
            b.style.textDecoration = 'line-through';
            b.textContent = '✗ ' + b.textContent;
          } else {
            b.style.opacity = '.55';
          }
        });

        const fb = el('div', { style: 'margin-top:.9rem; padding:.7rem .9rem; border-left:3px solid var(--accent); background:var(--accent-soft); border-radius:0 8px 8px 0; font-size:.9rem; line-height:1.65;' });
        fb.appendChild(el('div', { style: 'font-weight:700; margin-bottom:.25rem;', text: correct ? '✔ 答對了！' : '✘ 答錯了，正解是 ' + String.fromCharCode(65 + item.ans) + '。' }));
        const detail = el('div', { text: item.why + '（見' });
        detail.appendChild(chapterLink(item.ch));
        detail.appendChild(document.createTextNode('）'));
        fb.appendChild(detail);
        qPanel.appendChild(fb);

        const last = state.idx === QUESTIONS.length - 1;
        const nextBtn = el('button', { style: 'margin-top:.9rem;', text: last ? '查看總結 →' : '下一題 →' });
        nextBtn.addEventListener('click', () => {
          state.idx++;
          if (last) showSummary(); else showQuestion();
        });
        qPanel.appendChild(nextBtn);
        state.idx++; updateProgress(); state.idx--; // 進度以「已作答數」計
      }

      function showSummary() {
        updateProgress();
        qPanel.textContent = '';
        const s = state.score, n = QUESTIONS.length;
        const msg = s === n ? '滿分！你已經把 Ultra-Scale Playbook 融會貫通了 🎉'
          : s >= 8 ? '非常扎實！只差一點點就全對了。'
          : s >= 6 ? '基礎不錯，針對錯題章節再複習一輪吧。'
          : '別氣餒——分散式訓練本來就環環相扣，照下面的清單重讀最有效率。';
        qPanel.appendChild(el('div', { style: 'font-size:.8rem; color:var(--accent); font-weight:600; margin-bottom:.4rem;', text: '測驗總結' }));
        qPanel.appendChild(el('div', { style: 'font-size:2rem; font-weight:700; font-variant-numeric:tabular-nums;', text: s + ' / ' + n }));
        qPanel.appendChild(el('div', { style: 'color:var(--fg-muted); margin:.3rem 0 1rem; line-height:1.6;', text: msg }));

        if (state.wrong.length) {
          qPanel.appendChild(el('div', { style: 'font-weight:600; margin-bottom:.4rem;', text: '建議重讀的章節：' }));
          const seen = [];
          state.wrong.forEach((it) => { if (!seen.includes(it.ch)) seen.push(it.ch); });
          seen.sort();
          const ul = el('ul', { style: 'margin:0 0 1rem 1.2rem; line-height:1.9;' });
          seen.forEach((ch) => {
            const cnt = state.wrong.filter((it) => it.ch === ch).length;
            const li = el('li');
            li.appendChild(chapterLink(ch));
            li.appendChild(el('span', { style: 'color:var(--fg-muted); font-size:.85rem;', text: '（錯 ' + cnt + ' 題）' }));
            ul.appendChild(li);
          });
          qPanel.appendChild(ul);
        }

        const row = el('div', { class: 'widget-row' });
        const retry = el('button', { text: '↻ 重新測驗（重新洗牌）' });
        retry.addEventListener('click', start);
        row.appendChild(retry);
        qPanel.appendChild(row);
      }

      function start() {
        state.order = shuffled(QUESTIONS);
        state.idx = 0; state.score = 0; state.wrong = [];
        showQuestion();
      }
      start();
    },
  };
})();
