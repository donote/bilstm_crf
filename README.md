## BiLSTM+CRF For NER Task
>「致谢」参考代码：`https://github.com/WindChimeRan/pytorch_multi_head_selection_re`

**Environment: python3.7 + pytorch1.1.0**

**执行前先设置工作路径**
```
export PYTHONPATH=`pwd`
```

###  先对原始文本进行预处理，拆分为句子粒度，同时重置标注下标
```
# 注意，数据中如果存在空格，则该条数据会被丢掉！todo...
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

### 样本数据格式
```
[
	{
	"urid":"1000";
	"text":"中国是世界上历史最悠久的国家之一，有着光辉灿烂的文化和光荣的革命传统",
	"annotation": [
	{
	"start": 0,
	"end": 2,
	"value":"中国"
	}
	]
	}
]
```
### TODO
- 增加多类型实体同时抽取能力

