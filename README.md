# ECG-QA: a Comprehensive Multi-modal Question Answering Dataset Combined With Electrocardiogram

ECG-QA is a public question answering dataset with ECG signals based on the existing ECG dataset, [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/).
This dataset includes various types of questions including questions involving with a single ECG and comparison questions between two different ECGs.
Furthermore, it covers a wide range of attribute types to be inquired such as SCP Code (Symptoms in ECG), Noise, Stage of Infarction, Extra Systole, Heart Axis, Numeric Feature.

# Dataset Description
The dataset is organized as follows:
```
ecgqa
├── answers_for_each_template.csv
├── answers.csv
├── test_ecgs.csv
├── train_ecgs.csv
├── valid_ecgs.csv
├── paraphrased
│    ├─ test.json
│    ├─ train.json
│    └─ valid.json
└── template
     ├─ test.json
     ├─ train.json
     └─ valid.json
```
* All the QA samples are saved in each .json file, where **paraphrased** directory indicates its questions are paraphrased and **template** directory indicates its questions are not paraphrased.
* Each json file contains a list of python dictionary where each key indicates:
    * template_id: a number indicating its template ID.
    * question_id: a number indicating its question ID. Different paraphrases from the same template question share the same question ID.
    * sample_id: a number indicating each QA sample for each split.
    * question_type: a string indicating its question type, which can be one of this list:
        * `single-verify`
        * `single-choose`
        * `single-query`
        * `comparison_consecutive-verify`
        * `comparison_consecutive-query`
        * `comparison_irrelevant-verify`
        * `comparison_irrelevant-query`
    * attribute_type: a string indicating its attribute type, which can be one of this list:
        * `scp_code`
        * `noise`
        * `stage_of_infarction`
        * `extra_systole`
        * `heart_axis`
        * `numeric_feature`
    * question: a question string
    * ecg_id: a list of ecg IDs of PTB-XL dataset. For comparison questions, it contains two corresponding ecg IDs. Otherwise, it has only one element.
    * attribute: a list of strings indicating the relevant attributes with the question. For comparison questions, it is set to `None` because the primary purpose of this information is aimed to the upperbound experiments where we need to convert each Single QA sample into appropriate ECG classification format.
* `answers_for_each_template.csv` provides the possible answer options for each template ID.
* `answers.csv` provides the whole answer options over all the QA samples.
* `*_ecgs.csv` indicate which ecg IDs of PTB-XL are included in each split.

# Usage Notes
You can easily open and read data by the following codelines.
```python
>>> import json
>>> with open("train.json", "r") as f:
...     data = json.load(f)
>>> len(data)
267539
>>> data[0]
{
    "template_id": 1,
    "question_id": 0,
    "sample_id": 0,
    "question_type": "single-verify",
    "attribute_type": "scp_code",
    "question": "Is non-diagnostic t abnormalities detectable from this ECG?",
    "answer": ["yes"],
    "ecg_id": [12662],
    "attribute": ["non-diagnostic t abnormalities"]
}
```

# Quick Start
We implemented all the experiment codes in the [fairseq-signals](https://github.com/Jwoo5/fairseq-signals) repostiory.  
For detailed implementations, please refer to [here](https://github.com/Jwoo5/fairseq-signals/tree/master/fairseq_signals/data/ecg_text/preprocess) (See ECG-QA section).

## Run QA Experiments
1. Install [fairseq-signals](https://github.com/Jwoo5/fairseq-signals) following the guidelines.
```shell script
$ git clone https://github.com/Jwoo5/fairseq-signals
$ cd fairseq-signals
$ pip install --editable ./
$ python setup.py build_ext --inplace
$ pip install scipy wfdb pyarrow transformers
```
2. Pre-process ECG-QA dataset.
```shell script
$ python fairseq_signals/data/ecg_text/preprocess/preprocess_ecgqa.py \
    --ptbxl-data-dir /path/to/ptbxl \
    --dest /path/to/output \
    --apply_paraphrase
```
Note that `--ptbxl-data-dir` should be set to the directory containing ptbxl ECG samples (i.e., `records500/...`).

3. Run experiments.
```shell script
$ fairseq-hydra-train task.data=/path/to/output/paraphrased \
    model.num_labels=103 \
    --config-dir examples/scratch/ecg_question_answering/$model_name \
    --config-name $model_config_name
```
$model_name: the name of the ECG-QA model (e.g., `ecg_transformer`)
$model_config_name: the name of the configuration file (e.g., `base`)

## Run Upperbound Experiments
1. Install [fairseq-signals](https://github.com/Jwoo5/fairseq-signals) as the same with the above.
2. Pre-process ECG-QA dataset to be compatible with upperbound experiments.
```shell script
$ python fairseq_signals/data/ecg_text/preprocess/preprocess_ecgqa_for_classification.py \
    /path/to/ecgqa \
    --ptbxl-data-dir /path/to/ptbxl \
    --dest /path/to/output
```
3. For W2V+CMSC+RLM:
```shell script
$ fairseq-hydra-train task.data=/path/to/output \
    model.num_labels=83 \
    model.model_path=/path/to/checkpoint.pt \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer/grounding_classification \
    --config-name base_total
```
Note that you need to pass the path to the pretrained model checkpoint through `model.model_path`.  
To pre-train the model, refer to [here](../../../../examples/w2v_cmsc/README.md).

4. For Resnet + Attention model:
```shell script
$ fairseq-hydra-train task.data=/path/to/output \
    model.num_labels=83 \
    --config-dir examples/scratch/ecg_classification/resnet \
    --config-name nejedly2021_total
```

5. For SE-WRN model:
```shell script
$ fairseq-hydra-train task.data=/path/to/output \
    model.num_labels=83 \
    --config-dir examples/scratch/ecg_classification/resnet \
    --config-name se_wrn_total
```