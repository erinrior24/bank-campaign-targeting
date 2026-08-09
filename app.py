from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


st.set_page_config(
    page_title="Bank Campaign Targeting",
    page_icon="☎️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1080px; padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background:#f5f8fb; border:1px solid #dce5ec; border-radius:14px; padding:15px;}
      [data-testid="stMetricValue"] {color:#123e5a;}
      .decision {background:linear-gradient(135deg,#123e5a,#167c80); color:white; padding:24px 28px; border-radius:18px; margin:8px 0 20px;}
      .decision h2 {color:white; margin:0 0 7px;}
      .decision p {margin:0; font-size:1.04rem;}
      .note {background:#eef7f7; color:#123e5a; border-left:5px solid #16858a; padding:14px 18px; border-radius:8px;}
      .small {color:#52606d; font-size:.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


FEATURES = [
    "age", "job", "marital", "education", "housing", "loan",
    "contact", "campaign", "previous", "poutcome",
]
THRESHOLDS = [0.50, 0.40, 0.30, 0.20]


@st.cache_data
def load_data():
    data_dir = Path(__file__).parent / "data"
    train = pd.read_csv(data_dir / "marketingcampaign_train.csv")
    test = pd.read_csv(data_dir / "marketingcampaign_test.csv")
    return train, test


@st.cache_resource
def fit_models(train: pd.DataFrame):
    categorical = [c for c in FEATURES if train[c].dtype == "object"]
    numeric = [c for c in FEATURES if c not in categorical]
    y_train = (train["y"] == "yes").astype(int)

    logistic_preprocessor = ColumnTransformer(
        [
            ("category", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
        ],
        verbose_feature_names_out=False,
    )
    logistic = Pipeline(
        [
            ("preprocessor", logistic_preprocessor),
            ("model", LogisticRegression(penalty=None, max_iter=1000)),
        ]
    ).fit(train[FEATURES], y_train)

    tree_preprocessor = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", "passthrough", numeric),
        ],
        verbose_feature_names_out=False,
    )
    tree = Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=100,
                    random_state=5415,
                ),
            ),
        ]
    ).fit(train[FEATURES], y_train)
    return logistic, tree


def performance_row(name: str, actual: pd.Series, predicted: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    total = tn + fp + fn + tp
    return {
        "Prediction set": name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": (tp + tn) / total,
        "Precision": tp / (tp + fp) if tp + fp else 0,
        "Sensitivity": tp / (tp + fn) if tp + fn else 0,
        "Specificity": tn / (tn + fp) if tn + fp else 0,
        "Clients called": int(tp + fp),
        "Call rate": (tp + fp) / total,
    }


train, test = load_data()
logistic_model, tree_model = fit_models(train)
y_test = (test["y"] == "yes").astype(int)
logistic_probability = logistic_model.predict_proba(test[FEATURES])[:, 1]

performance = []
prediction_lookup = {}
for threshold in THRESHOLDS:
    label = f"Logistic {threshold:.2f}"
    predicted = (logistic_probability >= threshold).astype(int)
    prediction_lookup[label] = predicted
    performance.append(performance_row(label, y_test, predicted))

tree_prediction = tree_model.predict(test[FEATURES])
prediction_lookup["Classification tree"] = tree_prediction
performance.append(performance_row("Classification tree", y_test, tree_prediction))
performance_df = pd.DataFrame(performance)

logistic_names = logistic_model.named_steps["preprocessor"].get_feature_names_out()
logistic_coefs = logistic_model.named_steps["model"].coef_[0]
coefficient_df = pd.DataFrame(
    {
        "Predictor level": logistic_names,
        "Coefficient": logistic_coefs,
        "Odds ratio": np.exp(logistic_coefs),
        "Absolute coefficient": np.abs(logistic_coefs),
    }
).sort_values("Absolute coefficient", ascending=False)

tree_names = tree_model.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame(
    {
        "Predictor level": tree_names,
        "Importance": tree_model.named_steps["model"].feature_importances_,
    }
).sort_values("Importance", ascending=False)
importance_df = importance_df[importance_df["Importance"] > 0]


st.title("Bank Telemarketing Campaign Targeting")
st.caption(
    "Logistic regression and classification tree results on 10,297 previously unseen clients"
)

overview_tab, logistic_tab, tree_tab, comparison_tab, recommendation_tab = st.tabs(
    ["Business Problem", "Logistic Model", "Tree Model", "Model Comparison", "Recommendation"]
)

with overview_tab:
    st.header("Why improve campaign targeting?")
    st.markdown(
        '<div class="note"><strong>Mission:</strong> Identify clients most likely to subscribe so the bank can reduce unproductive calls while retaining valuable prospects.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    a, b, c, d = st.columns(4)
    a.metric("Training clients", f"{len(train):,}")
    b.metric("Test clients", f"{len(test):,}")
    c.metric("Training subscription rate", f"{(train['y'] == 'yes').mean():.1%}")
    d.metric("Test subscription rate", f"{(test['y'] == 'yes').mean():.1%}")

    st.subheader("Modeling setup")
    st.write(
        "Both models use the same ten retained predictors. Categorical fields are converted to indicator variables. "
        "The logistic regression estimates subscription probabilities; the tree is limited to four levels and at "
        "least 100 training observations per leaf so it remains readable and resistant to very small, unstable splits."
    )
    st.warning(
        "Because only 11.3% of clients subscribe, a model could achieve 88.7% accuracy by predicting 'no' for everyone. "
        "The comparison therefore considers sensitivity, precision, and specificity—not accuracy alone."
    )
    st.caption(
        "Repeated rows were retained because the files contain no client identifier; identical recorded profiles may represent different clients or contacts."
    )

with logistic_tab:
    st.header("Logistic regression: what changes subscription odds?")
    st.write(
        "A positive coefficient raises estimated subscription odds relative to its reference category; a negative "
        "coefficient lowers them. Numeric predictors are standardized, so their coefficients represent a one-standard-deviation change."
    )
    strongest = coefficient_df.head(12).copy()
    display_coef = strongest[["Predictor level", "Coefficient", "Odds ratio"]].copy()
    st.dataframe(
        display_coef.style.format({"Coefficient": "{:.3f}", "Odds ratio": "{:.2f}×"}),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.success(
            "A successful previous campaign is the strongest positive signal: holding the other fields constant, "
            "its estimated odds are about 9.2 times the reference outcome (previous failure)."
        )
        st.write(
            "Student and retired clients also have higher estimated odds than the reference job category (admin.), "
            "while being contacted by telephone rather than cellular lowers estimated odds."
        )
    with right:
        chart = strongest.set_index("Predictor level")[["Coefficient"]].sort_values("Coefficient")
        st.bar_chart(chart, horizontal=True, color="#16858a")
        st.caption("Largest coefficients by absolute size; reference categories are omitted by indicator coding.")

with tree_tab:
    st.header("Classification tree: the dominant decision path")
    t1, t2, t3 = st.columns(3)
    top_importance = importance_df.reset_index(drop=True)
    t1.metric("Top split", "Previous success")
    t2.metric("Importance from top split", f"{top_importance.loc[0, 'Importance']:.1%}")
    t3.metric("Tree depth", "4")
    st.write(
        "The first split asks whether the previous campaign succeeded. That split dominates the tree. Among clients "
        "without a previous success, age, cellular contact, and student status provide the next most useful separation. "
        "Among prior successes, the number of earlier contacts and age refine the prediction."
    )

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        tree_model.named_steps["model"],
        feature_names=tree_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        proportion=True,
        precision=2,
        fontsize=7,
        ax=ax,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Variables used by the tree")
    st.dataframe(
        importance_df.style.format({"Importance": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Importance is the share of the tree's total impurity reduction attributed to each encoded predictor.")

with comparison_tab:
    st.header("How did the five prediction sets perform?")
    metric_view = performance_df[
        ["Prediction set", "Accuracy", "Precision", "Sensitivity", "Specificity", "Clients called", "Call rate"]
    ]
    st.dataframe(
        metric_view.style.format(
            {
                "Accuracy": "{:.1%}",
                "Precision": "{:.1%}",
                "Sensitivity": "{:.1%}",
                "Specificity": "{:.1%}",
                "Clients called": "{:,.0f}",
                "Call rate": "{:.1%}",
            }
        ).highlight_max(subset=["Accuracy", "Precision", "Sensitivity", "Specificity"], color="#d8efea"),
        use_container_width=True,
        hide_index=True,
    )

    selected = st.selectbox("Inspect a confusion matrix", list(prediction_lookup), index=3)
    selected_row = performance_df.loc[performance_df["Prediction set"] == selected].iloc[0]
    matrix = pd.DataFrame(
        [[selected_row["TN"], selected_row["FP"]], [selected_row["FN"], selected_row["TP"]]],
        index=["Actual: No", "Actual: Yes"],
        columns=["Predicted: No", "Predicted: Yes"],
    )
    left, right = st.columns([1, 1.4])
    with left:
        st.dataframe(matrix.style.format("{:,.0f}").background_gradient(cmap="Blues"), use_container_width=True)
    with right:
        x1, x2, x3 = st.columns(3)
        x1.metric("True positives", f"{int(selected_row['TP']):,}")
        x2.metric("False positives", f"{int(selected_row['FP']):,}")
        x3.metric("False negatives", f"{int(selected_row['FN']):,}")
        st.write(
            "Lowering the logistic threshold identifies more subscribers (higher sensitivity) but also sends more "
            "calls to non-subscribers (lower precision and specificity). This is the business trade-off management must choose."
        )

with recommendation_tab:
    preferred = performance_df.loc[performance_df["Prediction set"] == "Logistic 0.20"].iloc[0]
    baseline_calls = len(test)
    calls_avoided = baseline_calls - int(preferred["Clients called"])
    st.markdown(
        '<div class="decision"><h2>Recommendation: logistic regression at a 0.20 threshold</h2>'
        '<p>Use sensitivity as the primary selection metric, with precision and call volume as operating safeguards.</p></div>',
        unsafe_allow_html=True,
    )
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Subscribers identified", f"{int(preferred['TP']):,}", f"{preferred['Sensitivity']:.1%} sensitivity")
    r2.metric("Targeted clients", f"{int(preferred['Clients called']):,}", f"{preferred['Call rate']:.1%} of test set")
    r3.metric("Target-list precision", f"{preferred['Precision']:.1%}", "vs. 11.3% base rate")
    r4.metric("Calls avoided", f"{calls_avoided:,}", f"{calls_avoided / baseline_calls:.1%} fewer")

    st.subheader("Why this choice fits the campaign")
    st.write(
        "The bank wants to avoid wasting staff time, but an overly high threshold misses most potential subscribers. "
        "At 0.20, the model identifies 374 of 1,160 actual subscribers—more than any other required prediction set—"
        "while targeting only 750 of 10,297 clients. Nearly half of the targeted clients subscribe, compared with "
        "11.3% in the full test population."
    )
    st.info(
        "Operational implication: compared with calling everyone, the recommended rule avoids 9,547 calls and still "
        "captures 32.2% of subscribers. If management values fewer calls more than finding additional subscribers, "
        "the 0.40 threshold or the tree provides slightly higher precision but materially lower sensitivity."
    )
    st.caption(
        "This recommendation assumes the value of reaching additional subscribers outweighs the cost of the 376 false-positive calls. A known call cost and deposit value would allow a profit-maximizing threshold."
    )
