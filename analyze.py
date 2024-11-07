import os
import shutil
import glob

def copy_and_rename_metrics(root_dir):
    # Find all request_metrics.csv files recursively
    for metrics_file in glob.glob(os.path.join(root_dir, "**/request_metrics.csv"), recursive=True):
        # Get the parent folder name which will be used in the new filename
        # Get the relative path from root_dir to the metrics file
        rel_path = os.path.relpath(metrics_file, root_dir)
        # Split path and take first directory after root as the subfolder name
        parent_folder = rel_path.split(os.sep)[0]
        
        # Create new filename with parent folder name
        new_filename = f"{parent_folder}_request_metrics.csv"
        new_filepath = os.path.join(root_dir, new_filename)
        
        # Copy the file to root directory with new name
        shutil.copy2(metrics_file, new_filepath)
        print(f"Copied {metrics_file} to {new_filepath}")

        
def plot_average_metrics(root_dir, column_name):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Get all CSV files in root directory that end with request_metrics.csv
    csv_files = glob.glob(os.path.join(root_dir, "*_request_metrics.csv"))
    
    x_labels = []
    y_values = []
    
    for csv_file in csv_files:
        # Extract subfolder name from filename (remove _request_metrics.csv)
        subfolder = os.path.basename(csv_file).replace("_request_metrics.csv", "").replace("T", "")
        
        # Read CSV and calculate average of specified column
        df = pd.read_csv(csv_file)
        avg_value = df[column_name].mean()
        
        x_labels.append(int(subfolder))
        y_values.append(avg_value)
    
    # Create bar plot
    plt.figure(figsize=(10,6))
    plt.plot(x_labels, y_values, marker='o', linestyle='-')
    plt.xlabel('T')
    plt.ylabel(f'Average {column_name}')
    plt.title(f'Average {column_name} ({os.path.basename(root_dir)})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(root_dir, f'average_{column_name}_plot.png'))
    plt.close()
    print(f"Generated plot for average {column_name}")

def plot_multiple_average_metrics(root_dirs, column_name):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,6))
    
    for root_dir in root_dirs:
        # Get all CSV files in root directory that end with request_metrics.csv
        csv_files = glob.glob(os.path.join(root_dir, "*_request_metrics.csv"))
        
        x_labels = []
        y_values = []
        
        for csv_file in csv_files:
            # Extract subfolder name from filename (remove _request_metrics.csv)
            subfolder = os.path.basename(csv_file).replace("_request_metrics.csv", "").replace("T", "")
            
            # Read CSV and calculate average of specified column
            df = pd.read_csv(csv_file)
            avg_value = df[column_name].mean()
            
            x_labels.append(int(subfolder))
            y_values.append(avg_value)
        
        # Plot line for this root_dir
        plt.plot(x_labels, y_values, marker='o', linestyle='-', label=os.path.basename(root_dir))
    
    plt.xlabel('T')
    plt.ylabel(f'Average {column_name}')
    plt.title(f'Average {column_name} Comparison')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'average_{column_name}_comparison.png')
    plt.close()
    print(f"Generated comparison plot for average {column_name}")


if __name__ == "__main__":
    dir_list = [
            "./simulator_output/vllm",
            "./simulator_output/sarathi",
            "./simulator_output/orca",
            "./simulator_output/faster_transformer",
        ]
    # for d in dir_list:
        # copy_and_rename_metrics(d)
        # plot_average_metrics(d, "request_e2e_time_normalized")
    
    # plot_multiple_average_metrics(dir_list, "request_e2e_time_normalized")
    # plot_multiple_average_metrics(dir_list, "request_execution_plus_preemption_time_normalized")
    plot_multiple_average_metrics(dir_list, "request_scheduling_delay")
