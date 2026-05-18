
### video_inference_yolo.inynb 我放置在 "\ultralytics"
> clone 專案連結: https://github.com/ultralytics/ultralytics
> 一樣按照原作者說的流程安裝必要的套件或東西。
* 仔細看我上面附的，有一個檔案是甚麼 osnet..... 那個要下載並且放到相對路徑: "ultralytics\weights，連結 https://drive.google.com/drive/folders/1_DH-CefgahzC_vOtrwo2_xswfcSC7ERU?usp=sharing"
* 裡面還有其他的檔案需要用到的權重也可以一併下載。
* YOLO 權重用 我給的那個 yolo11x.pt 路徑一樣 ultralytics\weights
  
### 2d_detection.py 我放在 "ViTPose\demo\2d_detection.py"
> 2d_detection.py 那個是我目前用別的 2d 檢測模型去跑的，你們要先 clone 到本機端，網址是 https://github.com/ViTAE-Transformer/ViTPose/tree/main
> 流程就按照作者上面所說的就好，指令則是照我的 python demo/2d_detection.py
>
> 權重放置路徑 ViTPose\weights
> 
### inference.py 我放在 "ViTPose\mmpose\apis" 覆蓋原本的就可以了


---

# 接下來是關於 posec3d 遷移訓練要注意的事項，這邊會列出來所有我改過的檔案，如果到時候訓練出問題、有錯、你們可以去一個一個爬說問題可能得出處，每個程式碼的功用建議你們開 AI，把 config 檔丟給他並且告訴他問題，再問他說 可能是哪一個環節有問題。

# PoseC3D 架構修改清單與除錯指南

## 架構總覽

```
pkl 資料
  ↓
DualPoseDataset (label_upper + label_lower → label=[u, l])
  ↓
Pipeline: UniformSampleFrames → PoseDecode → PoseCompact → Resize → Crop → [Flip] → SG → GeneratePoseTarget → FormatShape → PackActionInputs
  ↓
CustomDualRecognizer
  ├─ backbone: ResNet3dSlowOnly (官方，未改)
  ├─ neck: DualWindowGatingNeck (自建)
  └─ head: DualI3DHead → DualBaseHead (自建)
  ↓
DualAccMetric (自建)
```

---

## 檔案對照表

### 1. Config

| 自建/修改 | 對應官方原版 | 改動重點 |
|-----------|-------------|---------|
| `configs/.../slowonly_r50_...-keypoint.py` | 同名官方 config | model type 改 CustomDualRecognizer；新增 neck；head 改 DualI3DHead；pipeline 加 SG、調整順序；dataset 改 DualPoseDataset；evaluator 改 DualAccMetric；新增 load_from 和 paramwise_cfg |

### 2. Recognizer（資料流的總控制器）

| 自建/修改 | 對應官方原版 | 改動重點 |
|-----------|-------------|---------|
| `mmaction/models/recognizers/custom_recognizer.py` | `mmaction/models/recognizers/recognizer3d.py` + `base.py` | 覆寫 extract_feat：所有 neck 呼叫都補傳 data_samples；覆寫 predict：補傳 data_samples 給 extract_feat |

**除錯關鍵字：** `data_samples 為 None`、`sg_features` 找不到 → 來這裡看 neck 有沒有收到 data_samples

### 3. Neck（門控融合，核心新增模組）

| 自建/修改 | 對應官方原版 | 改動重點 |
|-----------|-------------|---------|
| `mmaction/models/necks/dual_window_gating.py` | 無（全新模組） | 從 backbone feature map 拆出 F_L、F_S；從 data_samples 取出 sg_features；MLP 門控產生 alpha/beta；融合輸出 F_fused |

**除錯關鍵字：** 通道數不符、crop_ratio、sg_feat_dim=136、batch 不一致 → 來這裡

### 4. Head（雙預測頭）

| 自建/修改 | 對應官方原版 | 改動重點 |
|-----------|-------------|---------|
| `mmaction/models/heads/multi_i3d_head.py` | `mmaction/models/heads/i3d_head.py` | 單一 fc_cls 拆成 fc_cls_upper + fc_cls_lower；forward 回傳 tuple |
| `mmaction/models/heads/dual_base.py` | `mmaction/models/heads/base.py` | num_classes 拆成 num_classes_upper + num_classes_lower；loss_by_feat 拆 labels[:,0] 和 labels[:,1] 分別算 loss；predict_by_feat 用 set_field 寫入 pred_score_upper/lower |

**除錯關鍵字：** labels shape 異常、label 越界、loss_cls_upper/lower、NaN → 來這裡

### 5. Dataset（資料讀取）

| 自建/修改 | 對應官方原版 | 改動重點 |
|-----------|-------------|---------|
| `mmaction/datasets/dual_pose_dataset.py` | `mmaction/datasets/pose_dataset.py` + `base.py` | 覆寫 get_data_info：從 pkl 讀 label_upper + label_lower 組合成 label=[u, l]；邊界檢查（硬編碼類別數上限） |

**除錯關鍵字：** label 缺失、label 負值/越界、KeyError: 'label' → 來這裡和 pkl 結構

### 6. Pipeline Transform（S-G 濾波）

| 自建/修改 | 對應官方原版 | 改動重點 |
|-----------|-------------|---------|
| `mmaction/datasets/transforms/SG_filter.py` | 無（全新模組） | 從 results['keypoint'] 算大小視窗的速度/加速度；輸出 sg_features (136 維) 存入 results |

**除錯關鍵字：** sg_features shape 錯誤、NaN velocity、window_length/polyorder → 來這裡

### 7. Metric（評估指標）

| 自建/修改 | 對應官方原版 | 改動重點 |
|-----------|-------------|---------|
| `mmaction/evaluation/metrics/dual_acc_metric.py` | `mmaction/evaluation/metrics/acc_metric.py` | 覆寫 process：從 data_sample 分別讀 pred_score_upper/lower 和 gt_label[0]/[1]；覆寫 compute_metrics：分別算 upper/lower 的 top-k accuracy |

**除錯關鍵字：** pred_score_upper 找不到、gt_label 形狀異常、top_k 錯誤 → 來這裡

---

## 官方未修改但需要理解的檔案

| 檔案 | 作用 | 什麼時候需要看它 |
|------|------|-----------------|
| `mmaction/models/recognizers/recognizer3d.py` | custom_recognizer 的父類別 | 想理解 extract_feat 原始邏輯時 |
| `mmaction/models/recognizers/base.py` | recognizer 最底層基類 | 想理解 loss/predict/forward 的呼叫鏈時 |
| `mmaction/models/heads/i3d_head.py` | DualI3DHead 的參考原版 | 比對 forward 的差異時 |
| `mmaction/models/heads/base.py` | DualBaseHead 的參考原版 | 比對 loss_by_feat / predict_by_feat 時 |
| `mmaction/datasets/pose_dataset.py` | DualPoseDataset 的父類別 | 資料載入或 split 出問題時 |
| `mmaction/datasets/transforms/formatting.py` | PackActionInputs | sg_features 沒被打包、inputs 格式錯誤時 |
| `mmaction/datasets/transforms/pose_transforms.py` | PoseDecode / GeneratePoseTarget / PoseCompact | heatmap 生成錯誤、keypoint shape 問題時 |
| `mmaction/structures/action_data_sample.py` | ActionDataSample 的 set_gt_label 等方法 | label 格式或 pred_score 存取出問題時 |
| `mmaction/evaluation/functional/accuracy.py` | top_k_accuracy 的實際計算 | metric 數值異常時 |

---

## 常見錯誤與排查路徑

### 錯誤：`AssertionError: data_samples 為 None`
**排查：** custom_recognizer.py → 確認 extract_feat 有傳 data_samples 給 neck

### 錯誤：`KeyError: 'sg_features'` 或 `AttributeError: sg_features`
**排查：**
1. SG_filter.py → 確認 results['sg_features'] 有被寫入
2. config pipeline → 確認 SG 在 PackActionInputs 之前
3. config PackActionInputs → 確認 algorithm_keys=('sg_features',)
4. formatting.py → 確認 set_field 有被呼叫

### 錯誤：`通道數不符` / shape mismatch
**排查：**
1. config → backbone out_indices / neck in_channels / head in_channels 三者一致 (512)
2. dual_window_gating.py → sg_feat_dim 是否和 SG_filter 輸出一致 (136)

### 錯誤：`labels shape 異常`
**排查：**
1. dual_pose_dataset.py → label=[upper, lower] 是否正確組合
2. pkl 檔案 → 確認每筆都有 label_upper 和 label_lower
3. dual_base.py → loss_by_feat 的 squeeze/reshape 邏輯

### 錯誤：`pred_score_upper 找不到`
**排查：**
1. dual_base.py → predict_by_feat 的 set_field 呼叫
2. dual_acc_metric.py → process 裡的讀取方式是否和寫入一致

### 錯誤：訓練 loss 不下降
**排查：**
1. config → load_from 路徑是否正確，checkpoint 是否存在
2. config → paramwise_cfg 的 lr_mult 設定
3. dual_window_gating.py → gating 權重是否退化（alpha ≈ 0.5 不動）
4. pkl → label 分佈是否極度不均衡

### 錯誤：val accuracy 和 train accuracy 差距極大
**排查：**
1. config → 三條 pipeline 的 SG 位置是否一致（都在空間變換之後）
2. config → val/test pipeline 的 GeneratePoseTarget 參數是否和 train 一致
3. pkl → train/val split 是否有資料洩漏（同影片的翻轉版出現在不同 split）

---

## 硬編碼值同步清單

以下數值散落在不同檔案，修改時必須同步：

| 參數 | 出現位置 | 目前值 |
|------|---------|-------|
| 上半身類別數 | config `num_classes_upper` / dual_pose_dataset.py `label_upper >= 2` | 2 |
| 下半身類別數 | config `num_classes_lower` / dual_pose_dataset.py `label_lower >= 6` | 6 |
| S-G 特徵維度 | config `sg_feat_dim` / SG_filter.py 的輸出邏輯 | 136 |
| 人數 | SG_filter.py `num_person` | 1 |
| 長視窗幀數 | config `clip_len` | 21 |
| 短視窗裁切量 | SG_filter.py `crop_margin` / neck `crop_ratio` | 3 / 0.7 |
| Backbone 輸出通道 | config backbone → neck `in_channels` → head `in_channels` | 512 |
