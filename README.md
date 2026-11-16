# ComfyUI Launcher

一鍵安裝並啟動 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 的啟動器，以 Windows 為主，也支援 macOS 與 Linux。第一次執行會自動完成環境設定，之後雙擊即可啟動。

## 功能

- 自動建立 Python 虛擬環境（`./venv`）
- 自動偵測 NVIDIA 顯示卡：有 GPU 裝 CUDA 版 PyTorch，沒有就裝 CPU 版；CUDA 版本會依顯卡和 Python 版本自動選擇（RTX 50 系列用 CUDA 13.0）
- 自動下載 ComfyUI（`./ComfyUI`，git 失敗時自動重試，仍失敗改用 ZIP 下載）並安裝依賴
- 啟動服務後自動在瀏覽器開啟 `http://127.0.0.1:8188`
- 自動安裝 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)（自訂節點管理器），啟動後在 ComfyUI 介面裡點 Manager 按鈕就能安裝節點和模型
- 可選安裝加速函式庫：`--extras sageattention,flash-attn,llama-cpp,triton`（或寫在 `launcher.json` 的 `extras` 裡），已安裝的會自動跳過，安裝失敗只會警告、不會影響啟動
- 支援更新 ComfyUI、更換端口、傳遞額外啟動參數
- 設定可寫在 `launcher.json` 裡永久保存

## 需求

- Windows 10 / 11（macOS、Linux 也可使用）
- Python 3.10 以上（建議 3.11 或 3.12），安裝時勾選 **Add python.exe to PATH**
- Git（https://git-scm.com/downloads）
- NVIDIA 顯示卡（建議；沒有也能用 CPU 模式，只是生成速度慢很多）

## 快速開始

1. 下載或 clone 這個倉庫。
2. 雙擊 `start.bat`。
3. 第一次執行會下載並安裝環境，需要幾分鐘，請耐心等候。
4. 瀏覽器會自動開啟 ComfyUI。關閉時在啟動器視窗按 `Ctrl+C`。

macOS / Linux：

```bash
python3 launcher.py
```

## 命令列選項

| 選項 | 說明 |
| --- | --- |
| `--port 8189` | 更換監聽端口（預設 8188） |
| `--host 0.0.0.0` | 允許區域網路內其他裝置連線 |
| `--update` | 啟動前先更新 ComfyUI（`git pull`） |
| `--reinstall` | 重新安裝 Python 依賴 |
| `--cpu` | 強制使用 CPU 版 PyTorch |
| `--torch-index 網址` | 指定 PyTorch 下載來源，例如 `https://download.pytorch.org/whl/cu121`（舊顯卡用） |
| `--extras 清單` | 安裝可選加速函式庫，逗號分隔，例如 `--extras sageattention,triton`（可選：sageattention、flash-attn、llama-cpp、triton） |
| `--no-manager` | 不安裝 ComfyUI-Manager |
| `--no-browser` | 不自動開啟瀏覽器 |
| `--skip-install` | 跳過依賴安裝步驟 |
| `--comfyui-dir 路徑` | 使用已有的 ComfyUI 資料夾 |
| `-- --lowvram` | `--` 之後的參數會直接傳給 ComfyUI |

範例：

```bat
start.bat --port 8189 --update
```

## 設定檔

在 `launcher.json` 裡可以保存預設值（不存在時使用內建預設）：

```json
{
  "host": "127.0.0.1",
  "port": 8188,
  "extra_args": [],
  "auto_open_browser": true,
  "torch_index": null,
  "install_manager": true,
  "extras": []
}
```

## 常見問題

**提示找不到 Python？**
重新安裝 Python，安裝時務必勾選 **Add python.exe to PATH**，然後重新開啟命令提示字元。

**提示找不到 git？**
安裝 Git（https://git-scm.com/downloads），安裝時選擇讓 git 可在命令列中使用。

**端口 8188 已被佔用？**
先關掉佔用該端口的程式，或換一個端口：`start.bat --port 8189`。

**依賴安裝失敗？**
先試 `start.bat --reinstall`。若仍失敗，把視窗裡的錯誤訊息貼出來檢查。

**出現 No matching distribution found for torch？**
通常是 Python 版本和 PyTorch 下載來源不匹配：例如 Python 3.13 沒有 CUDA 12.1 的 PyTorch，而 RTX 50 系列顯卡必須用 CUDA 12.8 以上。啟動器會自動從最新的 CUDA 版本開始嘗試，失敗時依序換舊版本；也可以手動指定，例如 `start.bat --reinstall --torch-index https://download.pytorch.org/whl/cu130`。

**ComfyUI 說 PYTORCH CUDA VERSION BELOW 13.0？**
這是 ComfyUI 建議 RTX 50 系列顯卡用 CUDA 13.0 以上的 PyTorch 以獲得完整效能。執行 `start.bat --reinstall`，啟動器會重新安裝並優先使用 cu130 版本。

**git clone 失敗，出現 SSL/TLS connection failed？**
這是 git 連 GitHub 時的安全連線問題，通常是暫時的，重試一次大多能成功。啟動器會自動重試 3 次，失敗時改用直接下載 ZIP 的方式。如果一直失敗，檢查 VPN、代理、防火牆或防毒軟體是否擋住了 git，或確認系統時間正確。也可以手動從 https://github.com/comfyanonymous/ComfyUI 下載 ZIP，解壓到 ComfyUI 資料夾（讓 main.py 位於 `./ComfyUI/main.py`），再執行啟動器。

**如何更新 ComfyUI？**
執行 `start.bat --update`，或在 ComfyUI 資料夾裡執行 `git pull`。

**怎麼安裝 SageAttention、FlashAttention 這些加速函式庫？**
執行 `start.bat --extras sageattention,flash-attn,llama-cpp,triton`，或在 `launcher.json` 裡加上 `"extras": ["sageattention", "flash-attn", "llama-cpp", "triton"]`，啟動器就會在每次啟動時檢查並補裝缺少的項目。PyTorch 本身啟動器本來就會裝，不需要列在裡面。注意：flash-attn 在 Windows 上不一定有對應你 Python / PyTorch / CUDA 版本的預編譯包，找不到時只會警告並繼續啟動，不會嘗試從原始碼編譯（編譯要很久）。llama-cpp-python 預設裝的是 CPU 版，需要 CUDA 加速要另外用 `CMAKE_ARGS=-DGGML_CUDA=on` 自行編譯。

**怎麼用 Manager 安裝自訂節點或模型？**
啟動後點 ComfyUI 介面右側的 Manager 按鈕，裡面可以搜尋並安裝自訂節點、模型和工作流。啟動器會在第一次啟動時自動裝好 Manager；不想裝的話用 `start.bat --no-manager`。

**模型放哪裡？**
放在 `ComfyUI\models\` 下對應的子資料夾，例如 checkpoints 放在 `ComfyUI\models\checkpoints\`。

## 授權

MIT License。
