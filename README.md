# ECG-QA: a Comprehensive Multi-modal Question Answering Dataset Combined With Electrocardiogram

ECG-QA is a public question answering dataset with ECG signals based on the existing ECG dataset, [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/).
This dataset includes various types of questions including questions involving with a single ECG and comparison questions between two different ECGs.
Furthermore, it covers a wide range of attribute types to be inquired such as SCP Code (Symptoms in ECG), Noise, Stage of Infarction, Extra Systole, Heart Axis, Numeric Feature.

## Dataset Description
The dataset is organized as follows:
```
ecgqa
├── answers_for_each_template.csv
├── answers.csv
├── test_ecgs.csv
├── train_ecgs.csv
├── valid_ecgs.csv
├── paraphrased
│    ├─test.json
│    ├─train.json
│    └─valid.json
└── template
     ├─test.json
     ├─train.json
     └─valid.json
```
* All the QA samples are saved in each .json file, where **paraphrased** directory indicates its questions are paraphrased and **template** directory indicates its questions are not.
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
    * attribute: a list of strings indicating the relevant attributes with the question. We only provide this information for `verify` and `choose` question types.
* `answers_for_each_template.csv` provides the possible answer options for each template ID.
* `answers.csv` provides the whole answer options over all the QA samples.
* `*_ecgs.csv` indicate which ecg IDs of PTB-XL are included in each split.

## Usage Notes
You can easily open and read data by the following codelines.
```python
>>> import json
>>> with open("train.json", "r") as f:
...     data = json.load(f)
>>> data[0]
{
    "template_id": 1,
    "question_id": 0,
    "sample_id": 0,
    "question_type": "single-verify",
    "attribute_type": "scp_code",
    "question": "Is there evidence of non-diagnostic t abnormalities on this ECG?",
    "answer": ["yes"],
    "ecg_id": [12662],
    "attribute": ["non-diagnostic t abnormalities"]
}
```