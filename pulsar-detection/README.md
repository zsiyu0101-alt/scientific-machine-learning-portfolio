# 脉冲星识别

## 环境与运行

需要 Python 和 R。Python 依赖：`pip install -r requirements.txt`。R 脚本会尝试安装所需的 caret、MASS、rpart、kernlab、randomForest、ggplot2、pROC、e1071。

1. 将 `train_agnpsr_3fgl.csv` 与 `test_agnpsr_3fgl.csv` 放在本目录的 `data/` 下，或设置 `PULSAR_DATA_DIR` 指向数据目录。数据需包含 35 个数值预测变量和 `pulsarness` 标签（Pulsar / Non-Pulsar）。当前未提供 CSV。
2. 在本目录运行 `Rscript Pulsar_Detection_Part_A.R`，生成 `Part_A_results/`。
3. 运行 `python Pulsar_Detection_Part_B.py`，生成 `Part_B_results/`。Part B 比较阶段需要 Part A 的预测和指标文件。

输出包括指标 CSV、测试预测、模型文件和图表。依赖文件未锁定版本；请记录实际环境。代码保留原始固定样本数等断言，更换数据需检查相应假设。

参考资料由代码内链接提供；3FGL 星表本身不等于本项目使用的课程训练/测试划分。
