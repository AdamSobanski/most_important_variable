import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import gc

from pycaret import regression as reg
from pycaret import classification as clf

if 'top_feature' not in st.session_state:
    st.session_state.top_feature = None

# v0 Widocznosc opisu w zakladce karty strony
st.set_page_config(page_title="Most Important Variable", layout="wide")

# v0 Konfiguracja strony
st.header("Most Important Variable")
st.markdown(
    "Train models and evaluate them to discover which features have the greatest impact on your Target column."
)

# v1 – możliwość wczytania pliku CSV i wyświetlenie przykładowych danych

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    st.session_state.df = df

    st.success(f"Plik wczytany! Wierszy: {len(df)}, Kolumn: {len(df.columns)}")
    
else:
    st.warning("Proszę wczytać plik CSV")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.sample(5), hide_index=True)

st.subheader("Descriptive statistics")
st.dataframe(df.describe())

# v2 - wybór kolumny docelowej

previous_target = st.session_state.get('target_column', None)
target_column = st.selectbox("Select Target column:", df.columns)

st.session_state.target_column = target_column

# v3 - rozpoznanie problemu klasyfikacji/regresji i wyświetlenie informacji

st.subheader("Problem detection")

# Automatyczne wykrywanie typu problemu

def detect_problem_type(df, target):
    """Wykrywa czy problem to regresja czy klasyfikacja"""
    unique_values = df[target].nunique()
    total_values = len(df[target])
    
    # Jeśli kolumna to obiekt/string -> klasyfikacja

    if df[target].dtype == 'object':
        return 'classification'
    
    # Jeśli mało unikalnych wartości (< 20) -> klasyfikacja

    if unique_values < 20:
        return 'classification'
    
    # Jeśli stosunek unikalnych wartości do wszystkich > 0.05 -> regresja
    if unique_values / total_values > 0.05:
        return 'regression'
    
    # Domyślnie klasyfikacja
    return 'classification'

# Wykryj typ problemu
problem_type = detect_problem_type(df, target_column)

# Wyświetl info
st.info(f"Problem detected: **{problem_type.upper()}**")

# v4 - wygenerowanie modelu

st.subheader("Generate Model")

if 'best_model' not in st.session_state:
    st.session_state.best_model = None

if st.button("Setup ML Environment & Compare Models"):
    
    gc.collect()

    with st.spinner("Preparing ML environment..."):
        if problem_type == 'regression':
            reg.setup(data=df, target=target_column, session_id=123, verbose=False, html=False, n_jobs=1)
        else:
            clf.setup(data=df, target=target_column, session_id=123, verbose=False, html=False, n_jobs=1)

        st.session_state.problem_type = problem_type

    with st.spinner("Comparing models..."):
        if problem_type == 'regression':
            best_model = reg.compare_models(include=['et', 'rf', 'dt', 'lightgbm'])
        else:
            best_model = clf.compare_models(include=['et', 'rf', 'dt', 'lightgbm'])
        
        st.session_state.best_model = best_model
        st.session_state.results = reg.pull() if problem_type == 'regression' else clf.pull()

        model_name = type(best_model).__name__
        st.success(f"Best Model: **{model_name}**")

        results = reg.pull() if problem_type == 'regression' else clf.pull()
        # st.write("**All Models from Best to Worst:**")
        st.dataframe(results)

# v5 - wyświetlenie najważniejszych cech

if st.session_state.best_model is not None:
    st.subheader("Feature Importance - Most Important Variables")
    
    try:
        if problem_type == 'regression':
            reg.plot_model(st.session_state.best_model, plot='feature', save=True)
        else:
            clf.plot_model(st.session_state.best_model, plot='feature', save=True)

        # NOWY KOD
        try:
            importances = st.session_state.best_model.feature_importances_
        except AttributeError:
            try:
                importances = abs(st.session_state.best_model.coef_[0])
            except AttributeError:
                importances = None

        if importances is not None:
            try:
                if st.session_state.get('problem_type') == 'regression':
                    feature_names = reg.get_config('X_train').columns
                else:
                    feature_names = clf.get_config('X_train').columns
                min_len = min(len(importances), len(feature_names))
                fi = pd.Series(importances[:min_len], index=feature_names[:min_len])
                top_feature = fi.idxmax()
                st.session_state.top_feature = top_feature
            except Exception:
                st.session_state.top_feature = None
        else:
            st.session_state.top_feature = None

        st.image('Feature Importance.png')
        
    except Exception as e:
        st.warning(f"Feature importance not available for this model. Error: {e}")

# v6 - wygenerowanie opisu

st.subheader("Conlusion of Model Analysis")

st.markdown(f"Most Important Variable: **:blue[{st.session_state.top_feature}]**")

if st.session_state.best_model is None:
    st.markdown("Run the model first to see the analysis.")
    st.stop()

results = reg.pull() if problem_type == 'regression' else clf.pull()
best_model_name = results.index[0]

if problem_type == 'regression':
    r2_score = results.loc[best_model_name, 'R2']
    mae_score = results.loc[best_model_name, 'MAE']
    rmse_score = results.loc[best_model_name, 'RMSE']
    full_model_name = results.loc[best_model_name, 'Model']

    st.markdown(f"The Best Model is: **:blue[{full_model_name}]**.")

    st.markdown(f"""
            **{':green' if r2_score >= 0.75 else ':orange' if r2_score >= 0.5 else ':red'}[R2: {r2_score*100:.1f}%]** – how much of the variation in the target the model explains (max. 100% - perfect fit). {'Model explains most of the variation – reliable for deployment.' if r2_score >= 0.75 else 'Model explains moderate variation – consider further optimization.' if r2_score >= 0.5 else 'Model explains little variation – too risky for production.'}

            **{':green' if mae_score / df[target_column].mean() < 0.1 else ':orange' if mae_score / df[target_column].mean() < 0.25 else ':red'}[MAE: {mae_score:.2f}]** – average mistake the model makes in the same units as your target. {'Low average error – model predictions are close to reality.' if mae_score / df[target_column].mean() < 0.1 else 'Moderate average error – assess whether acceptable for your business.' if mae_score / df[target_column].mean() < 0.25 else 'High average error – model predictions may be too far from reality.'}

            **{':green' if rmse_score / df[target_column].mean() < 0.1 else ':orange' if rmse_score / df[target_column].mean() < 0.25 else ':red'}[RMSE: {rmse_score:.2f}]** – similar to MAE but penalizes large mistakes more heavily. {'Large errors are rare – model is consistent.' if rmse_score / df[target_column].mean() < 0.1 else 'Some large errors occur – review outlier cases.' if rmse_score / df[target_column].mean() < 0.25 else 'Large errors are frequent – model struggles with extreme cases.'}
             """)
else:
    accuracy = results.loc[best_model_name, 'Accuracy']
    precision = results.loc[best_model_name, 'Prec.']
    recall = results.loc[best_model_name, 'Recall']
    f1 = results.loc[best_model_name, 'F1']
    auc = results.loc[best_model_name, 'AUC']

    class_distribution = df[target_column].value_counts(normalize=True)
    is_balanced = class_distribution.min() > 0.3
    majority_class_pct = class_distribution.max() * 100

    full_model_name = results.loc[best_model_name, 'Model']
    st.markdown(f"The Best Model is: **:blue[{full_model_name}]**.")

    if is_balanced:
        st.markdown(f"""
            :green[**Classes are balanced**]: A class is a possible outcome you want to predict (e.g. 'buy/not buy', 'sick/healthy', '0/1'). Balanced means each class in the target column has more than 30% share in the data – the model has enough examples of each class to learn from.

            **{'​:green' if accuracy >= 0.75 else ':orange' if accuracy >= 0.5 else ':red'}[Accuracy: {accuracy*100:.1f}%]** – for every 100 cases the model makes {'less than ' + str(round((1-accuracy)*100)) if accuracy >= 0.75 else str(round((1-accuracy)*100))} mistakes. {'Solid foundation for deployment.' if accuracy >= 0.75 else 'Consider further optimization before deployment.' if accuracy >= 0.5 else 'Too high error risk for production environment.'} {' Model can support decision-making — monitor performance on new data regularly.' if accuracy >= 0.75 else ' Use for trend analysis only — collect more data and consult a data specialist before deployment.' if accuracy >= 0.5 else ' Do not use for critical decisions — expand the dataset (1000+ records) and add more variables.'}"

            **{'​:green' if f1 >= 0.75 else ':orange' if f1 >= 0.5 else ':red'}[F1 Score: {f1*100:.1f}%]** – measure combining detection rate and precision (max. 100% - model makes no mistakes). {"Model reliably detects cases (e.g. sick patients, fraud, churning customers) and rarely generates false alarms (predicts '1' when reality is '0' – e.g. flags a healthy person as sick)." if f1 >= 0.75 else "Model makes some mistakes – assess whether they are acceptable for your business." if f1 >= 0.5 else "Model makes too many mistakes – high number of false alarms or missed cases."}
            """)
    else:
        st.markdown(f"""
            :orange[**Classes are imbalanced**] – dominant class: {majority_class_pct:.1f}%. A class is a possible outcome you want to predict (e.g. 'buy/not buy', 'sick/healthy', '0/1'). Imbalanced means one class in the target column dominates the others – the model may ignore the minority class and predict mostly the dominant one.

            **{':green' if auc >= 0.75 else ':orange' if auc >= 0.5 else ':red'}[AUC: {auc*100:.1f}%]** – model's ability to distinguish one class from another. {'Model distinguishes classes very well – ready for deployment.' if auc >= 0.75 else 'Model distinguishes classes moderately – consider optimization.' if auc >= 0.5 else 'Model barely distinguishes classes – not suitable for deployment.'} {'Strong discriminatory power — model can be trusted to separate positive from negative cases.' if auc >= 0.75 else 'Moderate separation — validate on new data before business use.' if auc >= 0.5 else 'Poor class separation — do not rely on this model for business decisions.'}"

            **{':green' if recall >= 0.75 else ':orange' if recall >= 0.5 else ':red'}[Recall: {recall*100:.1f}%]** – for every 100 real cases the model detects {round(recall*100)}. {'Most critical cases will not be missed.' if recall >= 0.75 else 'Some critical cases will remain undetected.' if recall >= 0.5 else 'Most critical cases remain undetected – high operational risk.'} {'Low miss rate — suitable where catching every case matters (e.g. fraud, disease detection).' if recall >= 0.75 else 'Moderate miss rate — consider the cost of missed cases for your business.' if recall >= 0.5 else 'High miss rate — critical cases will be overlooked, unacceptable for high-stakes decisions.'}"
            
            **{'​:green' if f1 >= 0.75 else ':orange' if f1 >= 0.5 else ':red'}[F1 Score: {f1*100:.1f}%]** – measure combining detection rate and precision (max. 100% - model makes no mistakes). {"Model reliably detects cases (e.g. sick patients, fraud, churning customers) and rarely generates false alarms (predicts '1' when reality is '0' – e.g. flags a healthy person as sick)." if f1 >= 0.75 else "Model makes some mistakes – assess whether they are acceptable for your business." if f1 >= 0.5 else "Model makes too many mistakes – high number of false alarms or missed cases."}
            """)

# v9 - pobranie modelu

st.subheader("Download Model")

import pickle

if st.session_state.best_model is not None:
    model_bytes = pickle.dumps(st.session_state.best_model)
    st.download_button(
        label="Download Model (.pkl)",
        data=model_bytes,
        file_name="best_model.pkl",
        mime="application/octet-stream"
    )
else:
    st.markdown("Run the model first to download it.")

# v10 - instrukcja użycia modelu

st.subheader("How to Use the Model")

if st.session_state.best_model is not None:
    st.markdown("""
**How to use the downloaded model:**
```python
import pickle
import pandas as pd

# Load the model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare your new data (must have the same columns as training data, without target)
new_data = pd.DataFrame({
    "column_1": [value1],
    "column_2": [value2],
    # ... all your input columns
})

# Make predictions
predictions = model.predict(new_data)
print(predictions)
```
> Make sure your new data has exactly the same column names and data types as the file you used for training. You can use this model in **Jupyter Notebook**, a Python script, or any other Python environment.
    """)

else:
    st.markdown("Run the model first to see instructions.")