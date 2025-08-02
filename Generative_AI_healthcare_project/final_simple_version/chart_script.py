import plotly.graph_objects as go
import pandas as pd

# Load the provided data and create a comprehensive workflow dataset
workflow_data = []
y_pos = 0

data = {
    "phases": [
        {"name": "Data Collection", "steps": ["Dataset Sources", "Medical Databases", "Symptom-Disease Mapping"], "color": "#FF6B6B"}, 
        {"name": "Data Processing", "steps": ["Data Cleaning", "Feature Engineering", "Train/Test Split"], "color": "#4ECDC4"}, 
        {"name": "Model Development", "steps": ["Random Forest", "Naive Bayes", "SVM Training", "Performance Evaluation"], "color": "#45B7D1"}, 
        {"name": "Application Development", "steps": ["Streamlit Interface", "Symptom Input", "Prediction Engine", "Recommendations"], "color": "#96CEB4"}, 
        {"name": "Deployment", "steps": ["User Interface", "Real-time Predictions", "Medical Guidance"], "color": "#FFEAA7"}
    ]
}

# Use brand colors
brand_colors = ["#1FB8CD", "#DB4545", "#2E8B57", "#5D878F", "#D2BA4C"]

# Create workflow data for comprehensive visualization
for phase_idx, phase in enumerate(data["phases"]):
    for step_idx, step in enumerate(phase["steps"]):
        workflow_data.append({
            'phase': phase["name"][:15],
            'step': step[:15],  # Limit step names to 15 chars
            'phase_order': phase_idx,
            'step_order': step_idx,
            'x_pos': step_idx + 1,
            'y_pos': phase_idx,
            'color': brand_colors[phase_idx]
        })

df = pd.DataFrame(workflow_data)

# Create the flowchart visualization
fig = go.Figure()

# Add scatter plot for each phase showing individual steps
for phase_idx, phase in enumerate(data["phases"]):
    phase_data = df[df['phase_order'] == phase_idx]
    
    fig.add_trace(go.Scatter(
        x=phase_data['x_pos'],
        y=[phase_idx] * len(phase_data),
        mode='markers+text',
        marker=dict(
            size=60,
            color=brand_colors[phase_idx],
            symbol='square'
        ),
        text=phase_data['step'],
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        name=phase['name'][:15],
        showlegend=True,
        hovertemplate="<b>%{text}</b><br>Phase: " + phase['name'][:15] + "<extra></extra>"
    ))

# Add connecting lines between phases
for i in range(len(data["phases"]) - 1):
    fig.add_trace(go.Scatter(
        x=[2.5, 2.5],
        y=[i + 0.2, i + 0.8],
        mode='lines',
        line=dict(color='gray', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))

# Update layout
fig.update_layout(
    title="Medical Chatbot Workflow",
    xaxis_title="Process Steps",
    yaxis_title="Project Phases",
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(len(data["phases"]))),
        ticktext=[phase["name"][:15] for phase in data["phases"]]
    ),
    xaxis=dict(
        tickmode='array',
        tickvals=[1, 2, 3, 4],
        ticktext=['Step 1', 'Step 2', 'Step 3', 'Step 4']
    ),
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    showlegend=True
)

# Save the chart
fig.write_image("medical_chatbot_workflow.png")