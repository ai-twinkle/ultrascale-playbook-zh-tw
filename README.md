# Ultra-Scale Playbook 中文版 — 超大規模訓練實戰手冊（互動版）

> **非官方社群翻譯（Unofficial Community Translation）**
> 本專案為 Hugging Face nanotron 團隊《[The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)》的繁體中文（zh-TW）全譯本，由台灣 [Twinkle AI Community](https://github.com/ai-twinkle) 翻譯維護。原作者：Nouamane Tazi、Ferdinand Mom、Haojun Zhao、Phuc Nguyen、Mohamed Mekkouri、Leandro von Werra、Thomas Wolf。原作依 Apache 2.0 釋出，譯文同授權。

**📖 線上閱讀：<https://apps.twinkleai.tw/ultrascale-playbook-zh-tw/>**

從單 GPU 的記憶體剖析出發，一路走到資料／張量／上下文／管線／專家五維平行、GPU kernel 最佳化與混合精度訓練。**每一章都配備互動實驗**——原網頁的互動圖表精神在中文版以自製元件重現。

## 互動實驗一覽

| 章節 | 實驗 |
|---|---|
| 導論 | 5D 平行策略互動地圖 |
| 第 1 章 單 GPU 訓練 | 記憶體計算器（權重/梯度/優化器/活化 + 重算策略） |
| 第 2 章 資料平行 | ZeRO 1/2/3 記憶體分片視覺化 |
| 第 3 章 張量平行 | column/row-linear 切分動畫 + TP/SP 流程 |
| 第 4 章 上下文平行 | Ring Attention 動畫 + Zig-Zag 負載平衡 |
| 第 5 章 管線平行 | AFAB vs 1F1B 氣泡排程模擬器 |
| 第 6 章 專家平行 | MoE 路由與 all-to-all 視覺化 |
| 第 7 章 5D 平行速覽 | 5D 配置探索器（GPU 格點圖 + 合理性檢查） |
| 第 8 章 最佳配置 | 三步驟配置精靈 |
| 第 9 章 深入 GPU | 矩陣乘 tiling 與記憶體合併動畫 |
| 第 10 章 FA 與混合精度 | 浮點格式探索器（FP32/BF16/FP16/FP8） |
| 結語 | 全書回顧小測驗 |
| 附錄 A0–A3 | 集體通訊動畫台／玩具 profiler trace／數量級計算器／重疊條件計算器 |

## 目錄結構與使用

```
├── content/          # 逐章翻譯 Markdown（12 章 + 4 附錄 + 參考資源）
├── webapp/           # 互動網站（純靜態、離線可用）
├── src/              # 原文 HTML 與 bibliography（對照用）
├── ultra_blog.md     # 原文 markdown 來源
└── build.py          # 建置：content/*.md → webapp
```

```bash
cd webapp && python3 -m http.server 8643   # 瀏覽 http://localhost:8643
```

修改譯稿或元件後重跑 `python3 build.py`。

## 翻譯說明

- 以原始 `ultra_blog.md` 為主要底本、`src/index.html`（distill 原始碼）為對照，還原轉檔遺失的數學定界符、修復損毀的程式碼區塊
- 原網頁互動圖表以「🔬」註記標明並連回原文；有靜態版本者一併附上
- 平行策略縮寫（DP/TP/PP/CP/EP/SP）、ZeRO、集體通訊操作名（all-reduce 等）保留英文
- 134 張插圖取自原始 assets（非 PDF 抽取）

## Citation

引用原作：

```bibtex
@misc{ultrascale_playbook,
  title={The Ultra-Scale Playbook: Training LLMs on GPU Clusters},
  author={Nouamane Tazi and Ferdinand Mom and Haojun Zhao and Phuc Nguyen and Mohamed Mekkouri and Leandro Werra and Thomas Wolf},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/spaces/nanotron/ultrascale-playbook}
}
```
