# GitHub Token 配置指南

## 获取 GitHub Token

1. 访问 [https://github.com](https://github.com) 并登录
2. 点击头像 → **Settings** → **Developer settings**
3. 选择 **Personal access tokens** → **Tokens (classic)**
4. 点击 **Generate new token** → **Generate new token (classic)**
5. 配置权限（推荐勾选 `repo`）
6. 生成后立即复制 Token

## 全局配置命令

### 设置环境变量

```bash
# 临时设置（当前终端有效）
export GITHUB_TOKEN=your_github_token_here

# 永久设置（添加到 ~/.bashrc）
echo 'export GITHUB_TOKEN=your_github_token_here' >> ~/.bashrc
source ~/.bashrc
```

### 验证配置

```bash
# 检查环境变量是否设置成功
echo $GITHUB_TOKEN
```

## 注意事项

⚠️ Token 只在创建时显示一次，请妥善保管  
⚠️ 不要将 Token 提交到代码仓库

---
**项目**: gnn-vis | **更新**: 2026-02-08
