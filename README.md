## BiLSTM+CRF For NER Task
Environment: python3.7 + pytorch1.1.0


**执行前先设置工作路径**
```
export PYTHONPATH=`pwd`
```

###  先对原始文本进行预处理，拆分为句子粒度，同时重置标注下标
```
# 注意，数据中如果存在空格，则该条数据会被丢掉！
# 预处理后同时生成train:dev:test==8:1:1
python script/preprocess.py -i raw_data/data.json -o raw_data/sample.json
```

### 预处理样本，生成vocab并增加BIO标识
```
# 生成vocab，所在目录为配置中data_root
python main.py --exp_name medical --mode prep
```

### 执行训练
```
python main.py --exp_name medical --mode train
```

### 执行评估或预测
```
# 结果写入文件，配置文件中字段eval_result_file和predict_result_file指定
python main.py --exp_name medical --mode eval
python main.py --exp_name medical --mode predict
```
