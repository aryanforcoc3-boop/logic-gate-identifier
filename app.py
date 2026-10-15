
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Page settings
st.set_page_config(
    page_title="Logic Gate Identifier",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Digital Logic Gate Identifier")
st.write(
    "Describe a logic gate's behaviour in English "
    "to identify the most likely logic gate."
)

# Train the machine learning model
@st.cache_resource
def train_model():
    dataset_path = Path(__file__).parent / "logic_gate_dataset.csv"
    data = pd.read_csv(dataset_path)

    required_columns = {"sentence", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            "CSV must contain columns named 'sentence' and 'label'."
        )

    data = data.dropna(subset=["sentence", "label"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    model.fit(
        data["sentence"].astype(str),
        data["label"].astype(str)
    )
    return model


# Load the model
try:
    model = train_model()
except FileNotFoundError:
    st.error(
        "Dataset not found. Keep logic_gate_dataset.csv "
        "in the same folder as app.py."
    )
    st.stop()
except (ValueError, KeyError) as error:
    st.error(f"Dataset problem: {error}")
    st.stop()


# Natural language prediction
st.subheader("🧠 Identify a Logic Gate")

sentence = st.text_area(
    "Enter the logic gate description:",
    placeholder="Example: Output is high when inputs are different"
)

if st.button("Identify Logic Gate"):
    if not sentence.strip():
        st.warning("Please enter a description first.")
    else:
        text = sentence.lower()

        same_words = [
            "same", "equal", "equals", "match",
            "identical"
        ]
        different_words = [
            "different", "unequal", "differs"
        ]

        if (
            any(word in text for word in same_words)
            and "high" in text
        ):
            prediction = "XNOR"

        elif (
            any(word in text for word in different_words)
            and "high" in text
        ):
            prediction = "XOR"

        else:
            prediction = model.predict([sentence])[0]

        st.success(f"Predicted Logic Gate: {prediction}")


# Truth tables
st.divider()
st.subheader("📘 Logic Gate Truth Tables")

truth_tables = {
    "AND":  [0, 0, 0, 1],
    "OR":   [0, 1, 1, 1],
    "NAND": [1, 1, 1, 0],
    "NOR":  [1, 0, 0, 0],
    "XOR":  [0, 1, 1, 0],
    "XNOR": [1, 0, 0, 1],
}

with st.expander("View all truth tables"):
    selected_gate = st.selectbox(
        "Select a logic gate",
        ["AND", "OR", "NOT", "NAND", "NOR", "XOR", "XNOR"],
        key="truth_table_gate"
    )

    if selected_gate == "NOT":
        table = pd.DataFrame({
            "Input A": [0, 1],
            "Output": [1, 0]
        })
    else:
        table = pd.DataFrame({
            "Input A": [0, 0, 1, 1],
            "Input B": [0, 1, 0, 1],
            "Output": truth_tables[selected_gate]
        })

    st.table(table)


# Binary logic calculator
st.divider()
st.header("🔢 Binary Logic Gate Calculator")
st.write("Select a gate and binary inputs to calculate the output.")

gate_choice = st.selectbox(
    "Choose a gate",
    ["AND", "OR", "NOT", "NAND", "NOR", "XOR", "XNOR"],
    key="calculator_gate"
)

input_a = st.selectbox(
    "Input A",
    [0, 1],
    key="calculator_a"
)

if gate_choice != "NOT":
    input_b = st.selectbox(
        "Input B",
        [0, 1],
        key="calculator_b"
    )
else:
    input_b = None

if gate_choice == "AND":
    output = input_a & input_b
elif gate_choice == "OR":
    output = input_a | input_b
elif gate_choice == "NOT":
    output = 1 - input_a
elif gate_choice == "NAND":
    output = 1 - (input_a & input_b)
elif gate_choice == "NOR":
    output = 1 - (input_a | input_b)
elif gate_choice == "XOR":
    output = input_a ^ input_b
else:
    output = 1 - (input_a ^ input_b)

if st.button("Calculate Output"):
    if gate_choice == "NOT":
        st.success(f"NOT({input_a}) = {output}")
    else:
        st.success(
            f"{gate_choice}({input_a}, {input_b}) = {output}"
        )
