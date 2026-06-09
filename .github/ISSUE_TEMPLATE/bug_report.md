name: Bug Report
description: Report a bug or unexpected behavior
labels: ["bug"]
body:
  - type: textarea
    id: description
    attributes:
      label: 问题描述
      description: 清晰描述你遇到的问题
      placeholder: |
        - 环境: (Python 版本, OS, GPU)
        - 命令: (完整 reproduce 命令)
        - 实际: (实际发生)
        - 预期: (期望发生)
    validations:
      required: true

  - type: textarea
    id: reproduce
    attributes:
      label: 复现步骤
      description: 复现问题的步骤
      placeholder: |
        1. 安装环境...
        2. 跑 python experiments/xxx.py
        3. 看到 ...
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: 错误日志
      description: 完整错误信息 / stack trace
      render: shell
    validations:
      required: false
