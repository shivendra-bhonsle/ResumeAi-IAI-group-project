"""
ResumeAI - Streamlit Frontend
AI-Powered Resume Screening and Ranking System

This is the main Streamlit application that provides a user-friendly interface
for uploading resumes, job descriptions, and viewing ranked results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import tempfile
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import ResumePipeline
import config

# Page configuration
st.set_page_config(
    page_title="ResumeAI – Candidate Ranking",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .score-excellent { color: #28a745; font-weight: bold; }
    .score-good { color: #17a2b8; font-weight: bold; }
    .score-average { color: #ffc107; font-weight: bold; }
    .score-poor { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def get_score_class(score):
    """Get CSS class based on score."""
    if score >= 0.8:
        return "score-excellent"
    elif score >= 0.6:
        return "score-good"
    elif score >= 0.4:
        return "score-average"
    else:
        return "score-poor"


def get_score_label(score):
    """Get label based on score."""
    if score >= 0.8:
        return "Excellent Match"
    elif score >= 0.6:
        return "Good Match"
    elif score >= 0.4:
        return "Average Match"
    else:
        return "Poor Match"


# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Current Weights")
    st.metric("Skills", f"{config.WEIGHTS['skills']:.0%}")
    st.metric("Experience", f"{config.WEIGHTS['experience']:.0%}")
    st.metric("Semantic", f"{config.WEIGHTS['semantic']:.0%}")
    st.metric("Education", f"{config.WEIGHTS['education']:.0%}")
    st.metric("Location", f"{config.WEIGHTS['location']:.0%}" + (" (Disabled)" if config.WEIGHTS['location'] == 0 else ""))

    st.divider()

    st.subheader("Model Info")
    st.caption(f"**Parser**: {config.GEMINI_MODEL}")
    st.caption(f"**Embeddings**: {config.EMBEDDING_MODEL.split('/')[-1]}")

    st.divider()

    st.caption("💡 **Tip**: Adjust weights in `.env` file")


# Main app
st.markdown('<div class="main-header">📄 ResumeAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Resume Screening & Ranking System</div>', unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'output_path' not in st.session_state:
    st.session_state.output_path = None

# Input section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Job Description")
    job_text = st.text_area(
        "Paste the full job description here",
        value="",
        max_chars=10000,
        height=300,
        help="Copy and paste the complete job description including requirements, responsibilities, and qualifications.",
        placeholder="Example:\n\nSenior Software Engineer\n\nWe are looking for an experienced engineer with 5+ years in Python and cloud technologies...\n\nRequirements:\n- Bachelor's degree in Computer Science\n- 5+ years of software development\n- Strong Python, AWS, Docker skills\n..."
    )

with col2:
    st.subheader("📂 Resume Files")
    uploaded_files = st.file_uploader(
        "Upload resume files (.docx format only)",
        type=["docx"],
        accept_multiple_files=True,
        help="Select up to 10 resume files in .docx format. Multiple files can be selected at once."
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning(f"⚠️ You uploaded {len(uploaded_files)} files. Only the first 10 will be processed.")
            uploaded_files = uploaded_files[:10]

        with st.container():
            st.success(f"✅ **{len(uploaded_files)} file(s) ready:**")
            for i, file in enumerate(uploaded_files, 1):
                st.caption(f"{i}. {file.name}")

# Action buttons
st.divider()

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    rank_button = st.button(
        "🔎 Rank Candidates",
        type="primary",
        use_container_width=True,
        help="Process all resumes and generate rankings"
    )

with col2:
    save_csv = st.checkbox("Save as CSV", value=True)

with col3:
    save_json = st.checkbox("Save as JSON", value=False)

# Processing
if rank_button:
    # Validation
    if not job_text.strip():
        st.error("❌ Please paste a job description before ranking.")
    elif not uploaded_files or len(uploaded_files) == 0:
        st.error("❌ Please upload at least one resume (.docx) before ranking.")
    else:
        # Process resumes
        with st.spinner("🔄 Processing resumes... This may take 1-2 minutes..."):
            try:
                # Save uploaded files to temp directory
                temp_dir = tempfile.mkdtemp()
                resume_paths = []

                for file in uploaded_files:
                    temp_path = os.path.join(temp_dir, file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(file.getbuffer())
                    resume_paths.append(temp_path)

                # Initialize pipeline
                pipeline = ResumePipeline()

                # Run ranking
                results, output_path = pipeline.run(
                    job_text=job_text,
                    resume_files=resume_paths,
                    save_output=save_csv or save_json,
                    output_format='csv' if save_csv else 'json',
                    return_format='dataframe'
                )

                # Store in session state
                st.session_state.results = results
                st.session_state.output_path = output_path

                # Clean up temp files
                for path in resume_paths:
                    try:
                        os.remove(path)
                    except:
                        pass

                st.success("✅ Ranking complete!")

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

# Display results
if st.session_state.results is not None:
    st.divider()
    st.header("📊 Ranking Results")

    results = st.session_state.results

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Candidates", len(results))

    with col2:
        top_score = results.iloc[0]['final_score']
        st.metric("Top Score", f"{top_score:.1%}", delta=get_score_label(top_score))

    with col3:
        avg_score = results['final_score'].mean()
        st.metric("Average Score", f"{avg_score:.1%}")

    with col4:
        excellent_count = len(results[results['final_score'] >= 0.8])
        st.metric("Excellent Matches", excellent_count)

    st.divider()

    # Rankings table
    st.subheader("🏆 Candidate Rankings")

    # Format dataframe for display
    display_df = results[['rank', 'name', 'email', 'final_score', 'skills_score',
                          'experience_score', 'semantic_score', 'education_score']].copy()

    # Format scores as percentages
    for col in ['final_score', 'skills_score', 'experience_score', 'semantic_score', 'education_score']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")

    # Rename columns
    display_df.columns = ['Rank', 'Name', 'Email', 'Final Score', 'Skills', 'Experience', 'Semantic', 'Education']

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Detailed view
    st.divider()
    st.subheader("📋 Detailed Candidate View")

    # Select candidate
    selected_idx = st.selectbox(
        "Select a candidate to view details:",
        range(len(results)),
        format_func=lambda i: f"#{results.iloc[i]['rank']} - {results.iloc[i]['name']} ({results.iloc[i]['final_score']:.1%})"
    )

    if selected_idx is not None:
        candidate = results.iloc[selected_idx]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Candidate Info")
            st.markdown(f"**Name**: {candidate['name']}")
            st.markdown(f"**Email**: {candidate['email']}")
            st.markdown(f"**Rank**: #{candidate['rank']}")
            st.markdown(f"**Experience**: {candidate['years_experience']:.1f} years")
            st.markdown(f"**Skills**: {candidate['num_skills']} skills")
            st.markdown(f"**Location**: {candidate['location']}")

            score_class = get_score_class(candidate['final_score'])
            st.markdown(f"**Overall**: <span class='{score_class}'>{candidate['final_score']:.1%} - {get_score_label(candidate['final_score'])}</span>",
                       unsafe_allow_html=True)

        with col2:
            st.markdown("### Score Breakdown")

            # Create score breakdown chart
            scores = {
                'Skills': candidate['skills_score'],
                'Experience': candidate['experience_score'],
                'Semantic': candidate['semantic_score'],
                'Education': candidate['education_score']
            }

            weights = {
                'Skills': config.WEIGHTS['skills'],
                'Experience': config.WEIGHTS['experience'],
                'Semantic': config.WEIGHTS['semantic'],
                'Education': config.WEIGHTS['education']
            }

            fig = go.Figure()

            # Add bars for scores
            fig.add_trace(go.Bar(
                name='Score',
                x=list(scores.keys()),
                y=list(scores.values()),
                text=[f"{v:.1%}" for v in scores.values()],
                textposition='auto',
                marker_color='lightblue'
            ))

            # Add weight annotation
            annotations = []
            for i, (component, weight) in enumerate(weights.items()):
                annotations.append(
                    dict(
                        x=i,
                        y=scores[component] + 0.05,
                        text=f"Weight: {weight:.0%}",
                        showarrow=False,
                        font=dict(size=10, color='gray')
                    )
                )

            fig.update_layout(
                title="Component Scores",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                showlegend=False,
                height=400,
                annotations=annotations
            )

            st.plotly_chart(fig, use_container_width=True)

    # Download section
    st.divider()
    st.subheader("💾 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        if save_csv or save_json:
            if st.session_state.output_path:
                st.success(f"✅ Results saved to: `{st.session_state.output_path}`")

        # CSV download button
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"resume_rankings_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # JSON download button
        import json
        json_data = {
            'candidates': results.to_dict('records'),
            'metadata': {
                'total_count': len(results),
                'weights': config.WEIGHTS,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        json_str = json.dumps(json_data, indent=2)

        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"resume_rankings_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.divider()
st.caption("🤖 Powered by ResumeAI | Using Gemini API for parsing & Sentence Transformers for semantic similarity")
