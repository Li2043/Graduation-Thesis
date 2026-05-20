# Thesis — Rawlsian Highway Merge Project

## Workspace layout

```text
thesis/
├── README.md                 ← 本文件（工作区说明）
├── rawlsian_project/         ← 当前开发目录（主项目，含 v0.2 + v0.3）
└── archive/                  ← 历史版本说明与归档指引
```

## 当前主项目：`rawlsian_project`

所有日常开发、训练、评估请在此目录进行：

```powershell
cd C:\Users\HP\Desktop\thesis\rawlsian_project
.\.venv\Scripts\Activate.ps1
```

详见 [`rawlsian_project/README.md`](rawlsian_project/README.md)。

## 版本与 Git

| 原型 | 内容 | 获取方式 |
|------|------|----------|
| v0.1 | Random policy smoke test | GitHub 早期 commit / 见 `archive/README.md` |
| v0.2 | Fairness metrics + random 对比 | Git tag `v0.2` |
| v0.3 | DQN 训练 + trained 评估 | `rawlsian_project` 当前 `main` |

远程仓库：https://github.com/Li2043/Graduation-Thesis

## 文件夹重命名说明

当前可通过 **`rawlsian_project`** 进入项目（目录联接）。若仍看到 `rawlsian_project.v0.2`，请在关闭 IDE 后按 [`RENAME_TODO.txt`](RENAME_TODO.txt) 或 [`archive/README.md`](archive/README.md) 完成最终重命名，之后只保留 `rawlsian_project` 一个文件夹名。
