# GitHub Token 配置指南

## 获取 GitHub Token

1. 访问 [https://github.com](https://github.com) 并登录
2. 点击头像 → **Settings** → **Developer settings**
3. 选择 **Personal access tokens** → **Fine-grained tokens**
4. 点击 **Generate new token**
5. 在 **Repository access** 里选择 **Only select repositories**
6. 在仓库列表中仅勾选 `gnn-vis`
7. 按最小权限原则配置需要的权限（例如仅需拉取代码可给 `Contents: Read-only`）
8. 点击 **Generate token**，生成后立即复制并妥善保存

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

⚠️ 请使用 **Fine-grained personal access token**，不要再创建 `Tokens (classic)`  
⚠️ 在 **Repository access** 中务必选择 **Only select repositories**，并仅勾选 `gnn-vis`  
⚠️ Token 只在创建时显示一次，请妥善保管  
⚠️ 建议设置过期时间并定期轮换 Token  
⚠️ 不要将 Token 提交到代码仓库

---
**项目**: gnn-vis | **更新**: 2026-02-08
