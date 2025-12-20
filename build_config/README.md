# 构建配置目录

本目录包含 PyInstaller 打包配置文件。

## 文件说明

| 文件 | 用途 |
|------|------|
| `build_gui.spec` | GUI 应用打包配置（推荐） |
| `build_exe.spec` | 命令行版本打包配置 |

## 打包命令

```bash
# 从项目根目录运行
pyinstaller build_config/build_gui.spec
```

生成的可执行文件位于 `dist/` 目录。
