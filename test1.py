# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from langchain_community.llms import Ollama

# --- Function to split code from LLM response ---
def split_code_sections(text):
    text = re.sub(r"```(?:python|sql)?", "", text)
    text = text.replace("```", "").replace("**", "").strip()

    cleaned_lines = []
    for line in text.splitlines():
        if re.match(r"^\s*(#|--|SQL Query|Seaborn Code|Matplotlib Code)", line):
            continue
        if 'sql_query' in line:
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Extract SQL
    sql_match = re.search(r"SELECT\s+\*\s+FROM\s+.*?;", cleaned_text, re.IGNORECASE | re.DOTALL)
    sql_query = sql_match.group(0).strip() if sql_match else None
    if sql_query:
        cleaned_text = cleaned_text.replace(sql_query, "")

    # Split seaborn
    parts = re.split(r"(?i)(?=import\s+seaborn|sns\.)", cleaned_text, maxsplit=1)
    if len(parts) == 2:
        pandas_code = parts[0].strip()
        seaborn_code = "import seaborn\n" + parts[1].strip()
    else:
        pandas_code = cleaned_text.strip()
        seaborn_code = None

    return pandas_code, sql_query, seaborn_code


# --- UI Setup ---
st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("🧠 AI Data Analyst with LLM + Pandas + SQL + Seaborn")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head(), use_container_width=True)

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    user_prompt = st.chat_input("💬 Ask a question about your data...")
    if user_prompt:
        st.chat_message("user").write(user_prompt)

        # Build LLM prompt
        head_str = df.head(5).to_string(index=False)
        prompt = f"""
        You are a data analyst working with the following pandas dataframe called `df`:

        {head_str}

        Return:
        1. Only the **Pandas code**, assigning the output to a variable called `result`
        2. A SQL query to do the same (using table_name)
        3. Python code to plot the result using seaborn, using the `result` DataFrame

        Do not redefine df or use sample data. Only use `df` that is already loaded. Use variable name `result` only.
        """

        llm = Ollama(model="llama3.1")
        response = llm.invoke(prompt)
        pandas_code, sql_query, seaborn_code = split_code_sections(response)

        # Execute code
        local_vars = {'df': df, 'sns': sns, 'plt': plt}
        result = None
        error = None

        try:
            if re.search(r"\w+\s*=", pandas_code):
                exec(pandas_code, {}, local_vars)
                result_var_name = [var for var in local_vars if isinstance(local_vars[var], pd.DataFrame) and var != 'df']
                result = local_vars[result_var_name[0]] if result_var_name else None
            else:
                result = eval(pandas_code, local_vars)
        except Exception as e:
            error = str(e)

        # Save interaction to history
        st.session_state.history.append({
            "prompt": user_prompt,
            "pandas_code": pandas_code,
            "sql_query": sql_query,
            "result": result,
            "chart_code": seaborn_code,
            "error": error
        })

    # Show all previous interactions
    for i, entry in enumerate(st.session_state.history):
        with st.chat_message("assistant"):
            st.markdown(f"**Prompt {i+1}:** {entry['prompt']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📊 Pandas Code**")
                st.code(entry['pandas_code'], language="python")
            with col2:
                st.markdown("**🗃️ SQL Query**")
                st.code(entry['sql_query'] or "No SQL", language="sql")

            st.markdown("**📋 Result**")
            if entry["error"]:
                st.error(f"Error: {entry['error']}")
            else:
                st.dataframe(entry["result"], use_container_width=True)

            st.markdown("**📈 Chart**")
            try:
                if entry["chart_code"]:
                    # exec(entry["chart_code"], {}, {'result': entry["result"], 'sns': sns, 'plt': plt})
                    exec(entry["chart_code"], {}, {
                        'result': entry["result"],
                        'df_male': entry["result"],
                        'df_female': entry["result"],  # also safe fallback
                        'sns': sns,
                        'plt': plt
                    })

                    st.pyplot(plt.gcf())
                    plt.clf()
                else:
                    st.warning("No chart code returned.")
            except Exception as e:
                st.error(f"Chart rendering error: {e}")



#     # Prompt input
#     user_prompt = st.chat_input("💬 Ask a question about your data...")
#     if user_prompt:
#         # Add to chat history
#         st.session_state.chat_history.append(user_prompt)

#         # Display prompt
#         st.chat_message("user").write(user_prompt)

#         # Build full LLM prompt
#         head_str = df.head(5).to_string(index=False)
#         prompt = f"""
# You are a data analyst working with the following pandas dataframe called `df`:

# {head_str}

# Write the pandas code and a SQL query to return this:
# {user_prompt}

# Return ONLY the code. First the pandas code, then the SQL query.
# Then return pure Python code to visualize the result using seaborn — no explanations or text, just valid Python code for plotting.
# """

#         # Get response from LLM
#         llm = Ollama(model="llama3.1")
#         response = llm.invoke(prompt)

#         pandas_code, sql_query, seaborn_code = split_code_sections(response)

#         # Execute code
#         local_vars = {'df': df, 'sns': sns, 'plt': plt}
#         result = None
#         error = None

#         try:
#             if re.search(r"\w+\s*=", pandas_code):
#                 exec(pandas_code, {}, local_vars)
#                 result_var_name = [var for var in local_vars if isinstance(local_vars[var], pd.DataFrame) and var != 'df']
#                 result = local_vars[result_var_name[0]] if result_var_name else None
#             else:
#                 result = eval(pandas_code, local_vars)
#         except Exception as e:
#             error = str(e)

#         # Layout for the 4-panel display
#         col1, col2 = st.columns(2)
#         with col1:
#             st.subheader("📊 Pandas Code")
#             st.code(pandas_code, language="python")
#         with col2:
#             st.subheader("🗃️ SQL Query")
#             st.code(sql_query if sql_query else "No valid SQL found", language="sql")

#         st.subheader("📋 Resulting DataFrame")
#         if error:
#             st.error(f"Error executing pandas code:\n{error}")
#         else:
#             st.dataframe(result, use_container_width=True)

#         st.subheader("📈 Visualization")
#         try:
#             if seaborn_code:
#                 local_vars.update({'result': result})
#                 exec(seaborn_code, {}, local_vars)
#                 st.pyplot(plt.gcf())
#                 plt.clf()
#             else:
#                 st.warning("No Seaborn code returned by the model.")
#         except Exception as e:
#             st.error(f"Error during chart rendering:\n{e}")
