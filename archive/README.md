# Archive — 历史版本归档

本目录用于记录**已 superseded** 的本地文件夹命名与原型版本，避免与当前工作区 `rawlsian_project` 混淆。

当前机器上若已无下列旧文件夹，可通过 **Git** 恢复对应快照，无需保留多份重复代码。

---

## 曾使用的本地目录名（已归档 / 已合并）

| 旧文件夹名 | 对应原型 | 状态 |
|------------|----------|------|
| `rawlsian_marl_project` | v0.1 早期 | 已合并进主项目；无独立保留 |
| `rawlsian_marl_project.v1` | v0.1 | 已合并 |
| `rawlsian_marl_project.v2` | v0.2 副本 | 与 v0.2 重复，勿再使用 |
| `rawlsian_project-v0` | v0.1 | 已合并 |
| `rawlsian_project.v0.2` | v0.2–v0.3 开发期名称 | **已重命名为** `../rawlsian_project` |

**唯一工作区**：`thesis/rawlsian_project/`

---

## 从 GitHub 检出历史版本（只读参考）

仓库：https://github.com/Li2043/Graduation-Thesis

```powershell
# 在 thesis 下新建只读快照（示例：v0.2 tag）
cd C:\Users\HP\Desktop\thesis\archive
git clone --branch v0.2 --depth 1 https://github.com/Li2043/Graduation-Thesis.git v0.2_snapshot
```

检出后可将 `v0.2_snapshot` 保留作论文附录对照；**不要**在其中继续开发。

删除快照：

```powershell
Remove-Item -Recurse -Force v0.2_snapshot
```

---

## 完成重命名（`rawlsian_project.v0.2` → `rawlsian_project`）

若 Cursor/终端仍打开旧路径，Windows 可能无法直接重命名。请：

1. 关闭所有打开 `rawlsian_project.v0.2` 的编辑器窗口与终端。
2. 若存在目录联接 `rawlsian_project`（指向 v0.2），先删除联接：
   ```powershell
   cd C:\Users\HP\Desktop\thesis
   cmd /c rmdir rawlsian_project
   ```
3. 重命名真实目录：
   ```powershell
   Rename-Item rawlsian_project.v0.2 rawlsian_project
   ```
4. 确认：
   ```powershell
   cd rawlsian_project
   git status
   python --version
   ```

完成后只保留 `thesis/rawlsian_project/`，可删除本说明中的联接步骤。

---

## 原型功能对照（便于写论文）

| 版本 | 策略 | 主要脚本 |
|------|------|----------|
| v0.1 | Random | `run_random_*`, `evaluate_random`（仅 reward） |
| v0.2 | Random + fairness CSV/图 | + `metrics.py` |
| v0.3 | DQN trained | + `train_dqn_*`, `evaluate_trained.py` |

当前 `rawlsian_project` 包含 **v0.2 + v0.3** 全部脚本；v0.1 能力已被 v0.2 覆盖。
