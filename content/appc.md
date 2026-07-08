# 附錄 A2　LLM 訓練的典型尺度（Typical Scales in LLM Training）

> 譯自 Hugging Face nanotron 團隊《The Ultra-Scale Playbook: Training LLMs on GPU Clusters》（Apache 2.0），原文為 [Hugging Face Space](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。

讓我們感受一下 LLM 訓練中各種東西的典型尺度。當我們談論記憶體或計算量時，通常是在數「元素」（element）的個數——你可以把它們想成張量中的數值。要換算成實際的記憶體位元組數，還需要乘上每個數值所佔的大小（例如 bf16 為 2 個位元組、fp32 為 4 個位元組）。

以下是一些快速的粗略估算數字：

- **輸入 token**：每個批次我們會處理 $seq \cdot mbs$ 個 token，其中 $mbs$ 是微批次大小（micro-batch size），$seq$ 是序列長度。
- **激活值（隱藏狀態）**：對單一層而言，隱藏狀態張量的大小為 $seq \cdot mbs \cdot h$ 個元素。
- **模型權重與梯度**：模型中的每個權重矩陣（例如線性層中的權重）大約有 $h^2$ 個元素。這是以單一權重矩陣計。梯度的大小與權重相同。
- **優化器狀態**：對每個權重矩陣（$h^2$ 個元素）而言，如果你使用像 Adam 這類優化器搭配混合精度訓練，它會以 fp32 精度保存動量（momentum）與變異數（variance）狀態（$2 \times 2h^2$），再加上 fp32 的主權重（master weights）（$2h^2$）。因此，每個權重矩陣的優化器狀態總計約為 $6h^2$。
- **模型參數總量**：每個 transformer 區塊會儲存：
  - 注意力參數：
    - QKV 投影：$3h^2$ 個參數
    - 輸出投影：$h^2$ 個參數
  - 使用 GLU（Gated Linear Unit）的 MLP 參數：
    - Gate 與 up 投影：$8h^2$ 個參數（2 個大小為 $h \times 4h$ 的矩陣）
    - Down 投影：$4h^2$ 個參數（1 個大小為 $4h \times h$ 的矩陣）
  - 每個區塊總計：使用 GLU MLP 時為 $16h^2$，不使用 GLU 時為 $12h^2$
  - 完整模型：$16h^2 \cdot num\_layers$（使用 GLU 時）
  - 額外參數：
    - 輸入嵌入（input embeddings）：$vocab\_size \cdot h$
    - LM head：$vocab\_size \cdot h$（若未與輸入嵌入綁定共用）
    - 位置嵌入（若有使用）：$max\_seq\_len \cdot h$
- **前向與反向傳播計算量（FLOPs）**：前向傳播的 FLOPs 有一個非常粗略的估算式：$2 \cdot num\_tokens \cdot num\_params$。反向傳播的計算量則是它的兩倍：$4 \cdot num\_tokens \cdot num\_params$。

> 📝 **註**：更精確的前向＋反向傳播 FLOPs 公式應為 $6 \cdot seq\_len \cdot num\_params + 12 \cdot num\_layers \cdot h \cdot seq\_len^2$，它把注意力運算在整條序列上的二次方擴展也納入考量；但為了簡化數學，我們假設 $seq\_len^2 \ll h$。
