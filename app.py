import streamlit as st
import pandas as pd
import os
import sys
import subprocess
import importlib.util
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import tempfile
import time
from datetime import datetime
import traceback

# Configure page
st.set_page_config(
    page_title="CrewAI ERP Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
    }
    
    .status-running {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }
    
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 CrewAI ERP Analytics Dashboard</h1>
    <p>Intelligent SQL Analysis with Automated Visualization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🛠️ Configuration")
    
    # OpenAI API Key input
    api_key = st.text_input("OpenAI API Key", type="password", 
                           help="Enter your OpenAI API key to run the analysis")
    
    # Database configuration
    st.subheader("Database Settings")
    db_host = st.text_input("Database Host", value="192.168.1.134")
    db_port = st.text_input("Database Port", value="3306")
    db_user = st.text_input("Database User", value="kavin")
    db_password = st.text_input("Database Password", type="password", value="Ispl@2025")
    db_name = st.text_input("Database Name", value="_b7843b7b27adc018")
    
    st.divider()
    
    # Analysis settings
    st.subheader("Analysis Settings")
    save_files = st.checkbox("Save Generated Files", value=True)
    show_code = st.checkbox("Show Generated Code", value=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Query Analysis")
    
    # Query input
    query = st.text_area(
        "Enter your business question:",
        value="Which employees have the highest number of leave applications this year?",
        height=100,
        help="Ask any business question about your ERP data"
    )

with col2:
    st.header("🎯 Quick Stats")
    
    # Status indicators
    if 'analysis_status' not in st.session_state:
        st.session_state.analysis_status = "Ready"
    
    st.metric("Status", st.session_state.analysis_status)
    
    if 'last_run_time' in st.session_state:
        st.metric("Last Run", st.session_state.last_run_time)

# Action buttons
col1, col2, col3 = st.columns(3)

with col1:
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

with col2:
    if st.button("🗑️ Clear Results", use_container_width=True):
        # Clear session state
        for key in list(st.session_state.keys()):
            if key.startswith(('analysis_', 'dataframe_', 'visualization_')):
                del st.session_state[key]
        st.rerun()

with col3:
    if st.button("📥 Download Files", use_container_width=True):
        if 'dataframe_code' in st.session_state or 'visualization_code' in st.session_state:
            st.info("Files are available in the Generated Code section below")

# Function to create and run CrewAI workflow
def create_and_run_crewai(query_text, api_key, db_config):
    """Create and execute the CrewAI workflow"""
    
    # Create the CrewAI script content
    crewai_script = f'''
import os
os.environ['OPENAI_API_KEY']="{api_key}"

import urllib.parse
from langchain_community.utilities import SQLDatabase

# Encode the password
password = urllib.parse.quote_plus("{db_config['password']}")

# Use the correct host and database name
DATABASE_URI = f"mysql+pymysql://{db_config['user']}:{{password}}@{db_config['host']}:{db_config['port']}/{db_config['name']}"

# Initialize the SQLDatabase
db = SQLDatabase.from_uri(DATABASE_URI)

from langchain.tools import Tool

sql_tool=Tool.from_function(
    name="sql_tool",
    func=db.run,
    description="A tool for running SQL queries against the database."
)

from crewai import LLM
llm=LLM(model="openai/gpt-4o-mini")

from crewai.tools import BaseTool
from pydantic import Field

class SQLTool(BaseTool):
    name: str = "sql_tool"
    description: str = "A tool for running SQL queries against the database. Use this to execute SELECT statements to retrieve data, INSERT to add records, UPDATE to modify existing data, DELETE to remove records, and other SQL operations. Provide the complete SQL query as input."
    
    db: any = Field(default_factory=lambda: db)
    
    def _run(self, query: str) -> str:
        """Execute the SQL query and return results"""
        try:
            return str(self.db.run(query))
        except Exception as e:
            return f"Error performing SQL query: {{str(e)}}"
            
from crewai import Agent, Task, Crew, Process

agent_1 = Agent(
   role="ERP SQL Database Assistant",
   goal="Understand user queries in natural language and provide accurate data insights that match ERP report results by analyzing MariaDB schema and executing appropriate SQL queries with proper business logic and filtering",
   backstory="You are an experienced ERP data analyst who specializes in translating business questions into SQL queries for MariaDB-based ERP systems. You have deep knowledge of ERP database structures, understand how ERP reports apply business rules and filters, and can quickly identify table relationships, column meanings, and data patterns. You ensure your SQL results match exactly what users see in their ERP reports by applying the same filtering logic, date contexts, and business rules. You communicate findings clearly in natural language while being precise about the data you're presenting and always cross-reference against expected ERP report behavior.",
   tools=[SQLTool()],
   llm=llm,
   verbose=True
)

task_1 = Task(
   description="Analyze the user's natural language query {{query}} and generate SQL that produces results identical to what they would see in their ERP reports. First explore the database schema to understand available tables and columns, then consider what business filters and logic the ERP system applies by default (such as enabled/disabled status, active records, date ranges). Generate and execute appropriate SQL queries that include proper joins, filtering, and calculations to match ERP report standards. Present the results in a clear, conversational manner that aligns with how data appears in the ERP interface. Handle follow-up questions and maintain context throughout the conversation while ensuring consistency with ERP report behavior.",
   expected_output="A natural language response that directly answers the user's question with relevant data insights that match ERP report results, including any clarifications about the data source, business logic applied, limitations, or additional context that would be helpful for ERP users.",
   agent=agent_1
)

agent_2 = Agent(
    role="Data Conversion Specialist",
    goal=(
        "Produce only the raw contents of a valid Python script (dataframe.py) that converts SQL query results "
        "from ERP SQL Database Assistant into pandas DataFrames and exports them as CSV, with no extra text, "
        "no markdown code fences, and no surrounding quotes."
    ),
    backstory=(
        "You are a data processing specialist who writes clean, production-ready Python scripts. "
        "Your scripts load SQL query results (assumed to be JSON/dictionary/list of dictionaries), convert them "
        "to pandas DataFrames with correct columns and data types, and export them as CSV files for analysis. "
        "You ensure the output is ONLY the code needed for dataframe.py, without any explanation or formatting."
    ),
    llm=llm,
    verbose=True
)

task_2 = Task(
    description=(
        "Take the output from agent_1 (ERP SQL Database Assistant) and produce ONLY the raw contents of a valid "
        "dataframe.py file. The file should contain complete, executable Python code that loads the given SQL "
        "query result (as a dictionary or list of dictionaries), converts it to a pandas DataFrame with correct "
        "column names and data types, and exports it as a CSV file. Do not include any markdown formatting "
        "(like ```python), no text before or after, and no quotes at the start or end. Output ONLY the code itself, "
        "so it can be saved directly as dataframe.py without modification."
    ),
    expected_output=(
        "The exact contents of a valid Python script (dataframe.py) containing only executable pandas code to "
        "convert SQL results into a DataFrame and export to CSV, with no extra text or formatting."
    ),
    agent=agent_2,
    output_file="dataframe.py",
    context=[task_1]
)

agent_3 = Agent(
    role="Data Visualization Specialist",
    goal=(
        "Produce only the raw contents of a valid Python script (diagram.py) that creates seaborn visualizations "
        "from pandas DataFrames, with no extra text, no markdown code fences, and no surrounding quotes."
    ),
    backstory=(
        "You are a data visualization expert who writes clear, production-ready Python scripts using seaborn "
        "and matplotlib. Your scripts analyze DataFrame columns to choose suitable plots (bar, line, scatter, heatmap, etc.), "
        "set proper titles, labels, legends, and save plots as high-quality images. You ensure the output is ONLY the code "
        "for diagram.py with no explanations or formatting."
    ),
    llm=llm,
    verbose=True
)

task_3 = Task(
    description=(
        "Take the pandas DataFrame output from agent_2 and produce ONLY the raw contents of a valid diagram.py file. "
        "The file should contain complete, executable Python code that uses seaborn and matplotlib to create appropriate "
        "plots based on the DataFrame's columns. Include clear titles, axis labels, legends, and save the plots as "
        "high-quality images. Output ONLY the code itself with no additional text, no markdown formatting (like ```python), "
        "and no quotes at the start or end. The code must be ready to save and run as a .py file without modification."
    ),
    expected_output=(
        "The exact contents of a valid Python script (diagram.py) containing only executable seaborn and matplotlib "
        "code to produce professional visualizations from the DataFrame and save them as images, with no extra text or formatting."
    ),
    agent=agent_3,
    output_file="diagram.py",
    context=[task_2]
)

crew = Crew(
    agents=[agent_1,agent_2,agent_3],
    tasks=[task_1,task_2,task_3],
    process=Process.sequential
)

result = crew.kickoff({{"query":"{query_text}"}})
print("CREWAI_RESULT:", result)
'''
    
    # Write and execute the script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(crewai_script)
        script_path = f.name
    
    try:
        # Execute the script and capture output
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        return result.stdout, result.stderr
    finally:
        # Clean up
        os.unlink(script_path)

# Execute analysis when button is clicked
if run_analysis:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar")
    elif not query.strip():
        st.error("Please enter a query to analyze")
    else:
        # Update status
        st.session_state.analysis_status = "Running"
        st.session_state.last_run_time = datetime.now().strftime("%H:%M:%S")
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Prepare database configuration
            db_config = {
                'host': db_host,
                'port': db_port,
                'user': db_user,
                'password': db_password,
                'name': db_name
            }
            
            status_text.markdown('<div class="status-running">🔄 Initializing CrewAI workflow...</div>', unsafe_allow_html=True)
            progress_bar.progress(10)
            
            # Run CrewAI analysis
            stdout, stderr = create_and_run_crewai(query, api_key, db_config)
            progress_bar.progress(50)
            
            if stderr and "Error" in stderr:
                st.session_state.analysis_status = "Error"
                status_text.markdown(f'<div class="status-error">❌ Error: {stderr}</div>', unsafe_allow_html=True)
            else:
                status_text.markdown('<div class="status-running">🔄 Processing results...</div>', unsafe_allow_html=True)
                progress_bar.progress(70)
                
                # Store results
                st.session_state.analysis_result = stdout
                st.session_state.analysis_status = "Complete"
                
                # Try to read generated files
                if os.path.exists('dataframe.py'):
                    with open('dataframe.py', 'r') as f:
                        st.session_state.dataframe_code = f.read()
                
                if os.path.exists('diagram.py'):
                    with open('diagram.py', 'r') as f:
                        st.session_state.visualization_code = f.read()
                
                progress_bar.progress(100)
                status_text.markdown('<div class="status-success">✅ Analysis completed successfully!</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.session_state.analysis_status = "Error"
            status_text.markdown(f'<div class="status-error">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            st.error(f"Error details: {traceback.format_exc()}")
        
        # Clear progress indicators after a delay
        time.sleep(2)
        progress_bar.empty()

# Display results if available
if 'analysis_result' in st.session_state:
    st.header("📋 Analysis Results")
    
    # Parse and display CrewAI output
    result_text = st.session_state.analysis_result
    if "CREWAI_RESULT:" in result_text:
        crew_result = result_text.split("CREWAI_RESULT:")[-1].strip()
        st.markdown("### 🤖 AI Analysis")
        st.info(crew_result)
    else:
        st.text_area("Raw Output", result_text, height=200)

# Display generated code
if show_code and ('dataframe_code' in st.session_state or 'visualization_code' in st.session_state):
    st.header("💻 Generated Code")
    
    tab1, tab2 = st.tabs(["📊 DataFrame Code", "📈 Visualization Code"])
    
    with tab1:
        if 'dataframe_code' in st.session_state:
            st.code(st.session_state.dataframe_code, language='python')
            
            # Execute dataframe code
            if st.button("🏃‍♂️ Execute DataFrame Code"):
                try:
                    exec(st.session_state.dataframe_code)
                    st.success("DataFrame code executed successfully!")
                    
                    # Try to load and display the CSV
                    if os.path.exists('output.csv'):
                        df = pd.read_csv('output.csv')
                        st.subheader("📊 Generated Data")
                        st.dataframe(df)
                        
                        # Provide download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name='analysis_results.csv',
                            mime='text/csv'
                        )
                        
                except Exception as e:
                    st.error(f"Error executing DataFrame code: {str(e)}")
        else:
            st.info("DataFrame code will appear here after analysis")
    
    with tab2:
        if 'visualization_code' in st.session_state:
            st.code(st.session_state.visualization_code, language='python')
            
            # Execute visualization code
            if st.button("🎨 Execute Visualization Code"):
                try:
                    exec(st.session_state.visualization_code)
                    st.success("Visualization code executed successfully!")
                    
                    # Display generated plots
                    plot_files = [f for f in os.listdir('.') if f.endswith(('.png', '.jpg', '.jpeg', '.svg'))]
                    if plot_files:
                        st.subheader("📊 Generated Visualizations")
                        for plot_file in plot_files:
                            st.image(plot_file, caption=plot_file)
                            
                            # Provide download button for each image
                            with open(plot_file, 'rb') as f:
                                st.download_button(
                                    label=f"📥 Download {plot_file}",
                                    data=f.read(),
                                    file_name=plot_file,
                                    mime='image/png'
                                )
                    
                except Exception as e:
                    st.error(f"Error executing visualization code: {str(e)}")
        else:
            st.info("Visualization code will appear here after analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 Powered by CrewAI | Built with Streamlit | ERP Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)
