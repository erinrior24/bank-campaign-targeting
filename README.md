# Bank Telemarketing Campaign App

Streamlit application for MSBX 5415 Mini-Project 4. It fits a logistic regression and an interpretable classification tree using the course-provided training data, evaluates both models on the course-provided test data, compares the four required logistic thresholds, and recommends a targeting rule.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The application expects these files in `data/`:

- `marketingcampaign_train.csv`
- `marketingcampaign_test.csv`

## Required deliverables covered

- Logistic regression summary and important predictors
- Displayed classification tree with an explanation of the main splits
- Five confusion matrices and a comparison of accuracy, precision, sensitivity, and specificity
- Preferred model and threshold with a business justification

