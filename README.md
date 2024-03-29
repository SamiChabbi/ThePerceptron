# Perceptron Implementation from Scratch

This project involves the implementation of a perceptron (neuron) from scratch using Python and NumPy. The goal is to understand fundamental machine learning concepts and how neural networks operate.

## Content

- [Description](#description)
- [How to Use](#how-to-use)
- [Files](#files)
- [Notes](#notes)
- [Sources](#sources)

## Description

This project implements a perceptron for binary classification. It loads a dataset of images containing cats and other objects, normalizes the data, trains the perceptron, and evaluates its performance on a test set. The code is well-commented to explain each step of the process.

## How to Use
1. Clone the repository:

    ```bash
    git clone https://github.com/SamiChabbi/ThePerceptron.git
    ```

2. Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. Navigate to the project directory:

    ```bash
    cd ThePerceptron
    ```

4. Run the `perceptron.ipynb` notebook in a Jupyter Notebook environment or any other Python environment or type:
    ```bash
    python3 main.py
    ```

## Files

- `perceptron.ipynb`: The main Jupyter notebook containing code, comments, and explanatory plots.
- `perceptron.py`: A Python file containing the perceptron implementation as a class.
- `main.py`: An example demonstrating how to use the perceptron class in a main application.
- `cost_evolution_plot.py`: Python script showing the evolution of the cost (located in the `demo/` directory).
- `datasets/`: Folder containing the datasets (not included in the GitHub repository).

## Highlights

- The code uses vectorization to enhance performance and understanding.
- Plots are generated to visualize the cost evolution over iterations.
- The model is evaluated on a test set to measure its performance.
- Detailed comments in the code provide in-depth explanations.

## Achievements

- Developed a perceptron from scratch for binary classification.
- Demonstrated understanding of core machine learning concepts.
- Implemented vectorization for improved computational efficiency.
- Generated informative plots for visualization of the training process.

## Next Steps

- Explore and experiment with different datasets and model architectures.
- Further optimize and extend the perceptron for more complex tasks.
- Continue learning and applying machine learning principles.
