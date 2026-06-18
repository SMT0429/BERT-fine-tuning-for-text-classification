# BERT Fine-tuning for Text Classification

使用 **BERT** 在 [IMDB Large Movie Review Dataset](https://aclanthology.org/P11-1015.pdf) 上做微調 (fine-tuning),完成電影評論的**二元情緒分類** (positive / negative)。

> 輸入一句話 → 模型辨識其情緒傾向。本作業著重於熟悉 PyTorch 的訓練流程與 Transformer 家族中 BERT 的應用。

## 任務說明

- **資料集**:IMDB,含 25,000 筆訓練 + 25,000 筆測試的高度極性電影評論
- **模型**:`bert-base-uncased`(透過 Hugging Face `transformers` 載入後再用資料微調)
- **類別**:`neg (0)` / `pos (1)`

## 實作流程

| 步驟 | 內容 | 對應 |
|------|------|------|
| 1. 工具函數 | `get_pred()` 取 argmax 預測類別;`cal_metrics()` 用 sklearn 算 acc / f1 / recall / precision | TODO1, TODO2 |
| 2. 資料準備 | 合併 IMDB 的 train + test,以 **8 : 1 : 1** 切成 train / val / test,存成 `.tsv` | TODO3 |
| 3. 自訂 Dataset | `CustomDataset` 把 tokenize 包進去:`encode_plus()` 加 `[CLS]/[SEP]`、截斷、padding 到 `max_len`,回傳 `input_ids` / `attention_mask` / `token_type_ids` | TODO4 |
| 4. 建立模型 | `BertClassifier`:**BERT → Dropout → Linear**,僅取 BERT 的句子表示 (pooled `[CLS]` output) 做分類,輸出 logits | TODO5 |
| 5. 訓練迴圈 | `zero_grad → forward → CrossEntropyLoss → backward → step`,每個 epoch 後在 val 評估並記錄指標 | TODO6 |
| 6. 預測 | `predict_one()` 單句預測 (tokenize → forward → softmax → argmax);`predict()` 批次預測並計算 Test Accuracy | TODO7 |

## 模型架構

```
input text
   │
   ▼
[CLS] tok1 tok2 ... [SEP]   ← tokenizer.encode_plus
   │
   ▼
   BERT  ──►  pooled output ([CLS] sentence representation)
   │
   ▼
 Dropout
   │
   ▼
 Linear (hidden_size → num_class)
   │
   ▼
 logits ──► CrossEntropyLoss / argmax
```

## 超參數

| 參數 | 值 |
|------|-----|
| config | `bert-base-uncased` |
| learning_rate | `2e-5` |
| epochs | `5` |
| max_len | `512` |
| batch_size | `32` |
| dropout | `0.2` |
| optimizer | Adam |
| loss | CrossEntropyLoss |

> 為節省記憶體與加速,訓練時可只 sample 部分資料(範例:train 4000 / val 500 / test 500)。

## 使用方式

1. 開啟 `HW2_BERT_110401544 (1).ipynb`(建議在 GPU 環境,如 Google Colab)
2. 依序執行各 cell:安裝套件 → 載入資料 → 建立 Dataset/模型 → 訓練 → 評估
3. 訓練完成後權重會存成 `bert.pt`,預測結果存成 `result.tsv`

```bash
pip install datasets transformers torch scikit-learn matplotlib pandas
```

## 輸出

- `bert.pt` — 微調後的模型權重
- `result.tsv` — 測試集預測結果
- learning curve 圖表 (loss / acc / f1 / recall / precision)
- **Test Accuracy**

## 參考

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://github.com/google-research/bert)
- [Hugging Face Transformers](https://huggingface.co/)
- [Learning Word Vectors for Sentiment Analysis (IMDB)](https://aclanthology.org/P11-1015.pdf)
