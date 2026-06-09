name: Feature Request
description: Suggest a new feature or experiment
labels: ["enhancement"]
body:
  - type: textarea
    id: motivation
    attributes:
      label: 动机
      description: 为什么需要这个 feature? 解决什么问题?
      placeholder: |
        例如: 想跑 8 qubit 看表达力上限
    validations:
      required: true

  - type: textarea
    id: proposal
    attributes:
      label: 提案
      description: 你建议怎么实现? 期望输出什么?
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: 替代方案
      description: 考虑过其他方案吗?
    validations:
      required: false
