
## Data Preprocessing

### Preprocess Class
1. **Initialization**: The class constructor (`__init__`) is set up to accept a list of input dataframes (`input`), a target dataframe (`target`), a list of days to look back (`back_day`), and the number of days to look forward for the target (`forward_day`).

2. **Data Shifting and Concatenation**: For each dataframe in the `input` list, the class performs data shifting based on the `back_day` list. This is useful for creating lagged features, a common practice in time-series modeling. After shifting the data, these are concatenated along their columns.
    - **Why Shift?**: Shifting is done to capture the historical context, which could be useful for predicting the future value of the target.

3. **Dimension Expansion**: The dimensions of the shifted data are expanded by one to prepare them for concatenation along a new axis. This is a preparatory step for model compatibility.

4. **Concatenation of Input Data**: All processed input data are concatenated along the last axis, essentially stacking them together as a multi-dimensional numpy array. This prepares the input data to be fed into machine learning models that might expect multi-dimensional input.

5. **NaN Handling for Input**: An index mask `idx1` is created to identify rows in the input data that do not contain any NaN values.
    - **Why Drop NaN?**: We drop rows with NaN values to ensure data integrity, as most machine learning models can't handle NaN values.

6. **Target Data Shifting**: The target variable is also shifted based on `forward_day`. This sets the target to predict for each corresponding input record.

7. **NaN Handling for Target**: A second index mask `idx2` is created to identify the rows in the target dataframe that are not NaN.

8. **Final Index Mask**: A final index mask `idx` is derived by logically combining `idx1` and `idx2`. This ensures that only rows with valid, non-NaN values in both the input and target data are retained.

9. **Final Data Preparation**: The input and target data are filtered based on the final index mask, and the indices are reset. These processed datasets are ready to be used in machine learning models.