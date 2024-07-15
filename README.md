# PROBLEM STATEMENT 16:
# Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU and fine-tuning of LLM Models using Intel® OpenVINO™
## Category:
Artificial Intelligence, Machine Learning, LLM, NLP
## Participants:
5th-8th Semester Students
## Pre-requisite:
- Understanding of Machine Learning Concepts.
- Programming skills (Python, NLP libraries like Hugging Face, transformers).
- Experience with natural language processing (NLP) and text-based AI models (e.g., language models, Chatbots).
## Description:
This problem statement is designed to introduce beginners to the exciting field of Generative Artificial Intelligence (GenAI) through a series of hands-on exercises. Participants will learn the basics of GenAI, perform simple Large Language Model (LLM) inference on a CPU, and explore the process of fine-tuning an LLM model to create a custom Chatbot.
## Major Challenges:
1. Pre-trained language models can have large file sizes, which may require significant storage space and memory to load and run.
2. Learn LLM inference on CPU.
3. Understanding the concept of fine-tuning and its importance in customizing LLMs.
4. Create a Custom Chatbot with Fine-tuned Pre-trained Large Language Models (LLMs) using Intel AI Tools.

## Outcomes:
1. Participants will gain a foundational understanding of Generative AI and its applications.
2. Participants will be able to perform simple LLM inference on a CPU and understand the process of fine-tuning LLMs for custom applications.
3. Create a 5-page report on Problem, Technical approach and results.

## Setup and Installation:

### Prerequisites

- Python 3 or higher
- Intel® OpenVINO™ toolkit
- Hugging Face transformers library
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ABINSABUPHILIP/TECHNOWIZZ.git
    cd <repository-directory>
    ```

2. Create a virtual environment:
    ```sh
    python -m venv openvino_env
    .\openvino_env\Scripts\Activate
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Install Intel® OpenVINO™ toolkit:
    Follow the instructions at [Intel OpenVINO Installation Guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino.html)

### Steps taken to create the chatbot.
1. For conversion and optimisation of the program run 
   ```sh
   python conversion_optimization.py
   ```

2. For cpu inferance run the code
    ```sh
    python cpu_inferance.py
    ```
3. For running the llm on the chatbot run the code
    ```sh
    python chatbot.py
    ```




## Usage

## Libraries used 
- `torch:` Used for creating and training deep learning models with GPU acceleration.
- `transformers:` Provides pre-trained models and tools for NLP tasks.
- `openvino:` Optimizes and accelerates deep learning models on Intel hardware.
- `optimum[openvino]:` Integrates Hugging Face models with OpenVINO for optimization.
- `nncf:` Applies compression techniques like quantization to neural networks.
- `gradio:` Builds interactive web interfaces for machine learning models.
- `psutil:` Monitors system resources and processes for performance analysis.
- `numpy:` Handles numerical operations and array manipulations.
- `matplotlib:` Creates visualizations for data analysis and model evaluation.
## Team Members

-   [Abin Sabu Philip](https://github.com/ABINSABUPHILIP)

-   [Abhinand M](https://github.com/aiswaryarahull)
# Demo
### Demo 1
https://github.com/ABINSABUPHILIP/TECHNOWIZZ/blob/main/demo%201.webm
### Demo 2
https://github.com/ABINSABUPHILIP/TECHNOWIZZ/blob/main/demo%202.webm
