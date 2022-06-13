# AI_Project

## Overview

給定一個來源圖示主題和一個目標圖示主題，將來源圖示主題的風格轉換為目標圖示主題的風格。

## Prerequisite

```
pip3 install -r requirements.txt
```

## Usage

查看可用的選項：

```
python3 main.py --help
```

### 範例

#### 訓練

```
python3 main.py --model pix2pix --mode train -s ../dataset/Papirus/64x64/apps/ -t ../dataset/BeautyLine/BeautyLine/apps/scalable/ -o ../out --tmp ../tmp
```

- 使用 pix2pix 為基底的模型來訓練，來源圖示主題位於 `../dataset/Papirus/64x64/apps/`，目標風格圖示主題位於 `../dataset/BeautyLine/BeautyLine/apps/scalable/`，並且指定模型的權重輸出於 `../out`，暫存資料夾爲 `../tmp`
- 注意：使用者不必對圖示主題作處理，程式會自動處理圖示主題中的圖片，向量圖會被轉為點陣圖，並且會自動將圖片縮放爲模型輸入大小。使用者只需找到圖示主題中有一堆圖片的資料夾傳入給程式即可，剩下的程式會自己處理

#### 測試

```
python3 main.py --model pix2pix --mode test -w ../pix2pix_weight/ -s ../dataset/Papirus/64x64/apps/ -o ../out_img/ --tmp ../tmp/
```

- 訓練好的模型位於 `../pix2pix_weight/`，使用訓練好的模型將 `../dataset/Papirus/64x64/apps/` 中的圖片進行風格轉換，輸出於 `./out_img/`

## Hyperparameters
## Experiment results