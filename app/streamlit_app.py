import streamlit as st
from inference import InferencePipeline
import yaml


@st.cache_resource
def load_dataset_description():
    with open('data/README.md', 'r') as file:
        return file.read()


@st.cache_resource
def load_model():
    config = yaml.safe_load(open("config/config.yml"))
    inference_pipeline = InferencePipeline(config)
    return config, inference_pipeline


def predict_question_type(text):
    config, inference_pipeline = load_model()

    class_labels = [
        "ABBR: Abbreviation",
        "DESC: Description",
        "ENTY: Entity",
        "HUM: Human",
        "LOC: Location",
        "NUM: Numeric"
    ]

    prediction, confidence = inference_pipeline.run_onnx_session(text)

    return class_labels[prediction], confidence  # class_labels[predicted_class], predictions[0][predicted_class].item()


def main():
    st.title("TREC Question Classifier 🤖")

    tab1, tab2 = st.tabs(["Classifier", "Dataset Description"])

    with tab1:
        input_text = st.text_area("Enter your question:", height=150)

        if st.button("Classify Question"):
            if input_text:
                pred_class, confidence = predict_question_type(input_text)

                st.subheader("Classification Results")
                st.write(f"**Predicted Class:** {pred_class}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")

                st.progress(confidence)
            else:
                st.warning("Please enter a question to classify.")

        st.sidebar.title("Example Questions")
        examples = [
            "When was the Eiffel Tower built?",
            "Who wrote Romeo and Juliet?",
            "What is the capital of France?",
            "How tall is Mount Everest?",
            "Define machine learning"
        ]

        for example in examples:
            if st.sidebar.button(example):
                st.session_state.input_text = example
                pred_class, confidence = predict_question_type(example)

                st.subheader("Classification Results")
                st.write(f"**Predicted Class:** {pred_class}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.progress(confidence)

    with tab2:
        st.markdown(load_dataset_description(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
