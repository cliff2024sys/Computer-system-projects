import requests

# The base URL where the server is running
BASE_URL = "http://127.0.0.1:5000"

def get_analysis(years):
    """Send a POST request to the server to analyze data for specific years."""
    url = f"{BASE_URL}/analyze"
    response = requests.post(url, json={'years': years})
    if response.status_code == 200:
        return response.json()
    else:
        return f"There was an error with your request: {response.text}"

def get_regression_analysis(response_year):
    """Send a POST request to the server to perform regression analysis for the specified response year."""
    url = f"{BASE_URL}/regression"
    response = requests.post(url, json={'response_year': response_year})
    if response.status_code == 200:
        return response.json()
    else:
        return f"There was an error with your request: {response.text}"

def get_svr_analysis(response_year):
    """Send a POST request to the server to perform SVR analysis for the specified response year."""
    url = f"{BASE_URL}/svr"
    response = requests.post(url, json={'response_year': response_year})
    if response.status_code == 200:
        return response.json()
    else:
        return f"There was an error with your request: {response.text}"

def main_menu():
    while True:
        print("\nPlease choose an option:")
        print("1. Analyze data for specific years.")
        print("2. Perform regression analysis.")
        print("3. Perform SVR analysis.")
        print("4. Exit.")
        
        choice = input("Enter choice (1-4): ")

        if choice == '1':
            years_to_analyze = input("Enter specific years for analysis separated by commas (e.g., 2013,2014): ")
            years_list = [int(year.strip()) for year in years_to_analyze.split(',') if year.strip().isdigit()]
            results = get_analysis(years_list)
            print(f"\nAnalysis Results for {years_list}:")
            print(results)
            if 'histogram_url' in results:
                print(f"Histogram available at: {results['histogram_url']}")
            if 'scatter_url' in results:
                print(f"Scatter Plot available at: {results['scatter_url']}")
        elif choice == '2':
            response_year = input("Enter a specific year to use as the response variable for regression analysis: ")
            results = get_regression_analysis(response_year)
            print(f"\nRegression Analysis Results for {response_year}:")
            print(results)
        elif choice == '3':
            response_year = input("Enter a specific year to use as the response variable for SVR analysis: ")
            results = get_svr_analysis(response_year)
            print(f"\nSVR Analysis Results for {response_year}:")
            if 'RMSE' in results:
                print(f"Root Mean Square Error: {results['RMSE']}")
            if 'plot' in results:
                print("Plot data received (display or use as needed).")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main_menu()
