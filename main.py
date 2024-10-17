import streamlit as st
import pandas as pd
import numpy as np


def calculate_adjoint_no_inverse(matrix):
    """Calculate the adjoint of a square matrix."""
    n = matrix.shape[0]
    adjoint_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Create minor matrix by removing the i-th row and j-th column
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            # Calculate the cofactor
            cofactor = ((-1) ** (i + j)) * np.linalg.det(minor)
            adjoint_matrix[j, i] = cofactor  # Note the transposition

    return adjoint_matrix


def calculate_adjoint_inverse(matrix):
    """Calculate the adjoint of a square matrix."""
    return np.linalg.inv(matrix) * np.linalg.det(matrix)


# Streamlit app
st.title("Matrix Adjoint Calculator")

# File upload
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load Excel file
    excel_data = pd.ExcelFile(uploaded_file)

    # Display sheet names
    sheet_names = excel_data.sheet_names
    selected_sheet = st.selectbox("Select a sheet", sheet_names)

    # Load the selected sheet into a DataFrame
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, )

    # Convert DataFrame to NumPy matrix
    try:
        matrix = df.to_numpy()
        st.write("Loaded matrix:")
        st.write(matrix)

        # Calculate adjoint
        adjoint_matrix_no_inv = calculate_adjoint_no_inverse(matrix)
        st.header("Adjoint of the matrix (no inverse algo):", divider="gray")
        st.write(adjoint_matrix_no_inv)

        adjoint_matrix_inv = calculate_adjoint_inverse(matrix)
        st.header("Adjoint of the matrix (inverse algo):", divider="gray")
        st.write(adjoint_matrix_inv)

    except Exception as e:
        st.error(f"Error processing the matrix: {e}")
