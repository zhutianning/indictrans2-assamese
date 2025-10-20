# 分支使用指南

## 📋 概述

本项目采用多分支策略，针对不同开发环境和使用场景进行优化。每个分支都包含特定的配置和脚本，确保在不同环境下都能顺利运行。

## 🌿 分支说明

### 1. `main` 分支
**用途**: 主分支，包含完整项目和通用文档

**包含内容**:
- 完整的项目代码
- 通用文档和说明
- 核心脚本和工具
- 项目历史记录

**适用场景**:
- 项目概览和文档查阅
- 代码审查和版本管理
- 通用功能开发

### 2. `windows-vm` 分支
**用途**: Windows + 虚拟机专用分支

**包含内容**:
- `setup_vm_env.sh` - 虚拟机环境配置脚本
- `debug_model_generation.py` - 模型生成调试脚本
- `debug_model_simple.py` - 简化调试脚本
- Windows 特定的问题解决方案
- 虚拟机搭建指南

**适用场景**:
- 在 Windows 上使用虚拟机开发
- 本地调试和测试
- Windows 环境问题排查
- 快速原型开发

### 3. `school-server` 分支
**用途**: 学校服务器专用分支

**包含内容**:
- `setup_server_env.sh` - 服务器环境配置脚本
- `scripts/` - SLURM 作业脚本目录
  - `preprocess.sbatch` - 数据预处理作业
  - `finetune.sbatch` - 模型微调作业
  - `evaluate.sbatch` - 模型评估作业
  - `submit_jobs.sh` - 作业提交脚本
  - `monitor_jobs.sh` - 作业监控脚本
- GPU 集群优化配置
- 服务器特定环境设置

**适用场景**:
- 在学校服务器上运行训练任务
- 长时间训练作业
- GPU 集群资源利用
- 生产环境部署

## 🚀 使用流程

### 场景 1: 本地开发 (Windows + 虚拟机)

```bash
# 1. 克隆项目
git clone <your-git-repo-url> indictrans2-assamese
cd indictrans2-assamese

# 2. 切换到 Windows/VM 分支
git checkout windows-vm

# 3. 设置虚拟机环境
chmod +x setup_vm_env.sh
./setup_vm_env.sh

# 4. 激活环境
source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate

# 5. 开始开发
python organized_scripts/preprocess_indictrans2_fixed.py
```

### 场景 2: 服务器训练

```bash
# 1. 在服务器上克隆项目
git clone <your-git-repo-url> indictrans2-assamese
cd indictrans2-assamese

# 2. 切换到服务器分支
git checkout school-server

# 3. 设置服务器环境
chmod +x setup_server_env.sh
./setup_server_env.sh

# 4. 提交训练作业
chmod +x scripts/submit_jobs.sh
./scripts/submit_jobs.sh

# 5. 监控作业状态
chmod +x scripts/monitor_jobs.sh
./scripts/monitor_jobs.sh
```

### 场景 3: 跨分支协作

```bash
# 在 Windows/VM 分支上开发
git checkout windows-vm
# ... 进行开发 ...
git add .
git commit -m "Add new feature"
git push origin windows-vm

# 将更改合并到主分支
git checkout main
git merge windows-vm
git push origin main

# 将更改同步到服务器分支
git checkout school-server
git merge main
git push origin school-server
```

## 🔄 分支同步策略

### 开发流程

1. **在 `windows-vm` 分支进行开发**
   - 快速迭代和调试
   - 功能验证
   - 问题排查

2. **合并到 `main` 分支**
   - 功能稳定后合并
   - 代码审查
   - 版本标记

3. **同步到 `school-server` 分支**
   - 生产环境部署
   - 性能优化
   - 长期训练

### 同步命令

```bash
# 从 windows-vm 同步到 main
git checkout main
git merge windows-vm
git push origin main

# 从 main 同步到 school-server
git checkout school-server
git merge main
git push origin school-server

# 从 school-server 同步到 main (如果有服务器特定的优化)
git checkout main
git merge school-server
git push origin main
```

## 📁 分支文件结构

### `main` 分支
```
├── docs/                    # 通用文档
├── organized_scripts/       # 核心脚本
├── data/                   # 数据文件
├── outputs/                # 训练输出
├── downloads/              # 原始数据
├── IndicTrans2/            # 官方项目
└── README.md               # 项目说明
```

### `windows-vm` 分支
```
├── (包含 main 分支的所有内容)
├── setup_vm_env.sh         # VM 环境配置
├── debug_model_generation.py  # 调试脚本
├── debug_model_simple.py   # 简化调试脚本
└── docs/VM_SETUP_GUIDE.md  # VM 搭建指南
```

### `school-server` 分支
```
├── (包含 main 分支的所有内容)
├── setup_server_env.sh     # 服务器环境配置
├── scripts/                # SLURM 作业脚本
│   ├── preprocess.sbatch
│   ├── finetune.sbatch
│   ├── evaluate.sbatch
│   ├── submit_jobs.sh
│   └── monitor_jobs.sh
└── docs/SCHOOL_SERVER_MIGRATION_PLAN.md
```

## 🛠️ 最佳实践

### 1. 分支命名规范
- 使用描述性的分支名称
- 避免在分支名称中使用特殊字符
- 保持分支名称简洁明了

### 2. 提交信息规范
```bash
# 功能开发
git commit -m "Add new feature: model evaluation script"

# 问题修复
git commit -m "Fix: resolve model generation error"

# 环境配置
git commit -m "Config: add server environment setup"

# 文档更新
git commit -m "Docs: update installation guide"
```

### 3. 分支管理
- 定期同步分支
- 及时删除不需要的分支
- 保持分支历史清晰

### 4. 环境隔离
- 每个分支对应特定环境
- 避免跨环境配置混合
- 保持环境配置的一致性

## 🔧 故障排除

### 常见问题

1. **分支切换失败**
   ```bash
   # 检查当前状态
   git status
   
   # 保存当前更改
   git stash
   
   # 切换分支
   git checkout target-branch
   
   # 恢复更改
   git stash pop
   ```

2. **合并冲突**
   ```bash
   # 查看冲突文件
   git status
   
   # 手动解决冲突
   # 编辑冲突文件
   
   # 标记冲突已解决
   git add resolved-file
   
   # 完成合并
   git commit
   ```

3. **分支同步问题**
   ```bash
   # 强制同步
   git fetch origin
   git reset --hard origin/target-branch
   ```

## 📊 分支状态监控

### 检查分支状态
```bash
# 查看所有分支
git branch -a

# 查看分支差异
git diff main..windows-vm
git diff main..school-server

# 查看分支历史
git log --oneline --graph --all
```

### 分支同步状态
```bash
# 检查分支是否同步
git log --oneline main..windows-vm
git log --oneline main..school-server
```

---

**文档状态**: 持续更新  
**创建时间**: 2025-10-21  
**维护者**: AI Assistant
