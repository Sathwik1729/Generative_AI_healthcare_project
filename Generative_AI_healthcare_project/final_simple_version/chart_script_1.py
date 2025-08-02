import plotly.graph_objects as go
import json

# Data from provided JSON
data = {
    "algorithms": ["Random Forest", "Naive Bayes", "Support Vector Machine", "Decision Tree", "Logistic Regression", "Neural Network"],
    "accuracy": [93.5, 86.5, 89.5, 82.5, 84.0, 91.5],
    "min_accuracy": [92, 85, 88, 80, 82, 90],
    "max_accuracy": [95, 88, 91, 85, 86, 93]
}

# Abbreviate algorithm names to fit 15 character limit
abbreviated_algorithms = [
    "Random Forest",
    "Naive Bayes", 
    "SVM",
    "Decision Tree",
    "Logistic Reg",
    "Neural Network"
]

# Calculate error bar values
error_y_minus = [acc - min_acc for acc, min_acc in zip(data["accuracy"], data["min_accuracy"])]
error_y_plus = [max_acc - acc for acc, max_acc in zip(data["accuracy"], data["max_accuracy"])]

# Brand colors
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C']

# Create bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=abbreviated_algorithms,
    y=data["accuracy"],
    error_y=dict(
        type='data',
        symmetric=False,
        array=error_y_plus,
        arrayminus=error_y_minus,
        visible=True
    ),
    marker_color=colors,
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title="ML Algo Performance - Medical Diagnosis",
    xaxis_title="Algorithm",
    yaxis_title="Accuracy (%)"
)

# Save the chart
fig.write_image("ml_algorithm_performance.png")