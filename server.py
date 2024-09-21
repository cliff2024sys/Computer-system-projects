import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
app = Flask(__name__)
import matplotlib.pyplot as plt
import os
from flask import send_from_directory
import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import base64
from io import BytesIO



# Directory to store plots
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# Load the data into a DataFrame
data = pd.read_csv('pivoted_export_data_no_month.csv')

def analyze_data(years):
    """Perform data analysis for specific years."""
    results = {}
    for year in years:
        year_str = str(year)
        if year_str not in data.columns:
            results[year_str] = {"error": f"No data available for year {year_str}"}
            continue

        year_data = data[year_str]
        results[year_str] = {
            "mean": year_data.mean(),
            "max": year_data.max(),
            "min": year_data.min(),
            "standard_deviation": year_data.std()
        }
    return results
def create_histogram(year, data):
    """Generate a histogram for the given year."""
    year_str = str(year)
    if year_str not in data.columns:
        return None  # Ensure the year is valid
    plt.figure()
    data[year_str].hist(bins=20)
    histogram_path = os.path.join(PLOT_DIR, f"histogram_{year}.png")
    plt.title(f"Histogram of Data for {year}")
    plt.savefig(histogram_path)
    plt.close()
    return histogram_path


def create_scatter_plot(years, data):
    """Generate a scatter plot comparing two years."""
    if len(years) != 2 or not all(str(year) in data.columns for year in years):
        return None  # Ensure both years are valid and exactly two years are provided
    plt.figure()
    plt.scatter(data[str(years[0])], data[str(years[1])], alpha=0.5)
    scatter_path = os.path.join(PLOT_DIR, f"scatter_{years[0]}_{years[1]}.png")
    plt.title(f"Scatter Plot: {years[0]} vs {years[1]}")
    plt.xlabel(f"{years[0]} Data")
    plt.ylabel(f"{years[1]} Data")
    plt.savefig(scatter_path)
    plt.close()
    return scatter_path

@app.route('/plots/<path:filename>')
def download_plot(filename):
    """Serve a plot file to the client."""
    return send_from_directory(PLOT_DIR, filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    request_data = request.get_json()  # Get data from request
    years = request_data.get('years')
    if years and all(str(year).isdigit() for year in years):
        with ThreadPoolExecutor() as executor:
            analysis_results = executor.submit(analyze_data, years).result()
            # Generate histograms for each year
            for year in years:
                histogram_path = create_histogram(year, data)
                if histogram_path:  # Check if histogram was successfully created
                    analysis_results[str(year)]['histogram_url'] = request.host_url + 'plots/' + os.path.basename(histogram_path)

            # Generate scatter plot if exactly two years are specified
            if len(years) == 2:
                scatter_path = create_scatter_plot(years, data)
                if scatter_path:
                    analysis_results['scatter_url'] = request.host_url + 'plots/' + os.path.basename(scatter_path)
        return jsonify(analysis_results)
    else:
        return jsonify({"error": "Invalid years provided."}), 400


@app.route('/regression', methods=['POST'])
def perform_regression():
    request_data = request.get_json()
    response_year = request_data.get('response_year')
    if response_year and response_year.isdigit() and response_year in data.columns:
        try:
            predictor_years = [year for year in data.columns if year != response_year and year.isdigit()]

            # Prepare data for regression
            Y = data[response_year].values.reshape(-1, 1)
            X = data[predictor_years]
            X = sm.add_constant(X)  # Adds a constant term to the predictors

            # Perform the regression
            model = sm.OLS(Y, X)
            results = model.fit()

            # Prepare results
            regression_results = {
                "summary": results.summary().as_text(),
                "parameters": results.params.to_dict(),
                "pvalues": results.pvalues.to_dict(),
                "rsquared": results.rsquared,
            }
            return jsonify(regression_results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid or missing response year provided."}), 400


@app.route('/svr', methods=['POST'])
def perform_svr():
    request_data = request.get_json()
    response_year = request_data.get('response_year')
    
    try:
        if response_year and response_year.isdigit() and response_year in data.columns:
            predictor_years = [year for year in data.columns if year != response_year and year.isdigit()]

            # Prepare data
            Y = data[response_year].values
            X = data[predictor_years].values

            # Splitting the dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Support Vector Regression Model
            svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
            svr.fit(X_train_scaled, Y_train)

            # Predictions and evaluation
            Y_pred = svr.predict(X_test_scaled)
            mse = mean_squared_error(Y_test, Y_pred)
            rmse = np.sqrt(mse)

            # Optionally, plot results
            plt.scatter(Y_test, Y_pred)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            plot_data = base64.b64encode(buffer.getbuffer()).decode("ascii")

            return jsonify({"RMSE": rmse, "plot": plot_data})
        else:
            return jsonify({"error": "Invalid or missing response year provided."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)








