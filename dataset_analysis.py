import pandas as pd


def copy_predicted_label_to_label(csv_path, output_path=None):
    """
    Reads a CSV file, copies the 'Predicted_Label' column into 'Label',
    removes the 'Predicted_Label' column, and ensures the 'text' column is not modified.

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str, optional): Path to save the updated CSV file.
                                      If not provided, appends '_updated' to the original filename.

    Returns:
        pd.DataFrame: The updated DataFrame if successful, else None.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Display the first few rows (optional, for verification)
        print("Original DataFrame:")
        print(df.head())

        # Verify required columns exist
        required_columns = ['Text', 'Label', 'Predicted_Label']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in the CSV file.")
                return None

        # Make a copy of the 'text' column to ensure it remains unchanged
        original_text = df['Text'].copy()

        # Copy 'Predicted_Label' into 'Label'
        df['Label'] = df['Predicted_Label']

        # Remove 'Predicted_Label' column
        df.drop(columns=['Predicted_Label'], inplace=True)

        # Verify that the 'text' column is unchanged
        if df['Text'].equals(original_text):
            print("\n' Text' column remains unchanged.")
        else:
            print("\nWarning: 'text' column has been modified.")

        # Define the output path
        if output_path is None:
            if csv_path.lower().endswith('.csv'):
                output_path = csv_path[:-4] + '_updated.csv'
            else:
                output_path = csv_path + '_updated.csv'

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_path, index=False)
        print(f"\nUpdated CSV saved to: {output_path}")

        return df

    except FileNotFoundError:
        print(f"Error: File not found at path '{csv_path}'. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: The CSV file is malformed.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def compare_first_columns(csv1_path, csv2_path):
    """
    Compare the first columns of two CSV files to check if they are perfectly identical.

    Args:
        csv1_path (str): Path to the first CSV file.
        csv2_path (str): Path to the second CSV file.

    Returns:
        bool: True if the first columns are identical, False otherwise.
    """
    try:
        # Read only the first column from each file
        df1 = pd.read_csv(csv1_path, usecols=[0])
        df2 = pd.read_csv(csv2_path, usecols=[0])

        # Check if the number of rows is the same
        if len(df1) != len(df2):
            print("Primele coloane au un număr diferit de rânduri.")
            return False

        # Compare the values in the first column
        identical = df1.iloc[:, 0].equals(df2.iloc[:, 0])

        if identical:
            print("Primele coloane sunt perfect identice.")
        else:
            print("Primele coloane NU sunt identice.")

        return identical

    except FileNotFoundError as e:
        print(f"Fișierul nu a fost găsit: {e}")
        return False
    except pd.errors.EmptyDataError:
        print("Unul dintre fișierele CSV este gol.")
        return False
    except Exception as e:
        print(f"A apărut o eroare: {e}")
        return False


# Example usage
csv_file1 = 'test.csv'
csv_file2 = r'C:\Users\adi\Desktop\SII\tema_predictie\results\classification_result.csv'
csv_file3 = r'C:\Users\adi\Desktop\SII\tema_predictie\results\classification_result_updated.csv'

result = compare_first_columns(csv_file1, csv_file3)
print("Comparare rezultată:", result)


# updated_df = copy_predicted_label_to_label(csv_file2)
# if updated_df is not None:
#     print("\nLabel update was successful.")
# else:
#     print("\nLabel update failed.")
