import ssl
import urllib.request
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Mute TensorFlow compilation warnings and info logs (Level '3' suppresses everything except critical errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Mute oneDNN specific log outputs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Bypass SSL for automated dataset downloads
ssl._create_default_https_context = ssl._create_unverified_context

from data_loader import load_dataset, iid_partition
from ppfl_cnn import PPFLWithCNN, print_benchmark

def generate_and_save_plots(stats, results_dir, timestamp, dataset_name):
    """Generates the two required performance charts and saves them to the results directory."""
    rounds = np.array([s.round_id for s in stats])
    accuracies = np.array([s.accuracy * 100 for s in stats])
    
    # Calculate cumulative training time in minutes
    train_times_sec = np.array([s.train_time_s for s in stats])
    cumulative_train_min = np.cumsum(train_times_sec) / 60.0

    # --- GRAPH 1: Accuracy vs Communication Rounds ---
    plt.clf()
    plt.plot(rounds, accuracies, marker='o', markersize=5, color='#1f77b4', linewidth=2, label='Accuracy')
    plt.xlabel('Number of Communication Rounds', fontsize=11, labelpad=8)
    plt.ylabel('Accuracy (%)', fontsize=11, labelpad=8)
    plt.title(f'Model Accuracy Convergence Profile ({dataset_name.upper()})', fontsize=12, fontweight='bold', pad=12)
    plt.xticks(np.arange(0, len(rounds) + 1, max(1, len(rounds) // 10)))
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    accuracy_plot_path = os.path.join(results_dir, f"ppfl_{dataset_name}_{timestamp}_accuracy.png")
    plt.savefig(accuracy_plot_path, dpi=300)
    print(f"   [SUCCESS] Accuracy chart saved to: {accuracy_plot_path}")

    # --- GRAPH 2: Training Time vs Communication Rounds ---
    plt.clf()
    plt.plot(rounds, cumulative_train_min, marker='o', markersize=5, color='#2ca02c', linewidth=2, label='Training Time')
    plt.xlabel('Number of Communication Rounds', fontsize=11, labelpad=8)
    plt.ylabel('Training Time (min)', fontsize=11, labelpad=8)
    plt.title('Training Time Accumulation Profile', fontsize=12, fontweight='bold', pad=12)
    plt.xticks(np.arange(0, len(rounds) + 1, max(1, len(rounds) // 10)))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    time_plot_path = os.path.join(results_dir, f"ppfl_{dataset_name}_{timestamp}_time.png")
    plt.savefig(time_plot_path, dpi=300)
    print(f"   [SUCCESS] Training time chart saved to: {time_plot_path}")

def save_benchmark_to_file(dataset_name, stats, params, timestamp):
    """Saves the benchmark report to a timestamped text file."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"{results_dir}/ppfl_{dataset_name}_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*66 + "\n")
        f.write(f"  PPFL EXPERIMENT REPORT - {dataset_name.upper()}\n")
        f.write(f"  Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*66 + "\n\n")
        
        # Write Hyperparameters
        f.write("  HYPERPARAMETERS:\n")
        for key, value in params.items():
            f.write(f"    {key}: {value}\n")
        f.write("\n")
        
        # Write Table Header
        f.write(f"{'Rnd':>4}   {'Enc':>8}   {'Dec':>8}   {'Train':>8}   {'Loss':>8}   {'Acc%':>6}   {'Hidden':>10}\n")
        f.write("-" * 66 + "\n")
        
        total_enc, total_dec, total_train = 0, 0, 0
        
        for s in stats:
            hidden_str = "YES" if s.masked_to_aggregator else "NO"
            f.write(f"{s.round_id:>4}   {s.enc_time_s:>8.3f}   {s.dec_time_s:>8.3f}   {s.train_time_s:>8.2f}   {s.loss:>8.4f}   {s.accuracy*100:>5.1f}%   {hidden_str:>10}\n")
            total_enc += s.enc_time_s
            total_dec += s.dec_time_s
            total_train += s.train_time_s

        total_crypto = total_enc + total_dec
        overhead = (total_crypto / (total_crypto + total_train)) * 100
        
        f.write("\n" + "-" * 66 + "\n")
        f.write(f"  Total Training Time: {total_train:.2f}s\n")
        f.write(f"  Total Crypto Time:   {total_crypto:.3f}s\n")
        f.write(f"  Crypto Overhead:     {overhead:.1f}%\n")
        f.write(f"  Final Accuracy:      {stats[-1].accuracy*100:.2f}%\n")
        f.write("="*66 + "\n")

    print(f"  [SUCCESS] Results report saved to: {filename}")
    
    # Generate and save the required plots into the same folder with the same timestamp sequence
    generate_and_save_plots(stats, results_dir, timestamp, dataset_name)

def main():
    # --- 1. SET EXPERIMENT PARAMETERS ---
    DATASET = "cifar100"  # Use "mnist", "svhn", or "cifar100"
    
    # Structural Federated Parameters
    N = 12  
    SAMPLES_PER_USER = 4000
    M = 12 
    T = 50
    K = 0 # Configured to look for top-k sparsified bounds
    
    # Local Optimizer & Dynamic Training Settings
    LR = 0.1
    LR_DECAY = 0.99
    LOCAL_EPOCHS = 10
    BATCH_SIZE = 256

    # --- 2. LOAD AND SHAPE DATA ---
    print(f"Loading {DATASET.upper()} dataset...")
    X_tr, y_tr, X_te, y_te, num_classes = load_dataset(DATASET)
    
    # input_shape is (H, W, C) for cnn_model.py
    channels = X_tr.shape[1]
    height = X_tr.shape[2]
    width = X_tr.shape[3]
    input_shape = (height, width, channels)

    # --- 3. PARTITION DATA ---
    shards = iid_partition(X_tr, y_tr, N=N, samples_per_user=SAMPLES_PER_USER)

    # Packaging for file saver (Uses the defined dynamic variables)
    params_to_save = {
        "Dataset": DATASET,
        "N": N,
        "M": M,
        "T": T,
        "SamplesPerUser": SAMPLES_PER_USER,
        "k": K,
        "lr": LR,
        "lr_decay": LR_DECAY,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "input_shape": input_shape,
        "num_classes": num_classes
    }

    # --- 4. INITIALIZE PPFL SYSTEM ---
    system = PPFLWithCNN(
        N=N,
        T=T,
        M=M,
        compress_k=K,        
        lr=LR,              
        lr_decay=LR_DECAY,       
        local_epochs=LOCAL_EPOCHS,      
        batch_size=BATCH_SIZE,
        input_shape=input_shape,
        num_classes=num_classes
    )

    # --- 5. EXECUTE TRAINING ---
    system.setup(shards, X_te, y_te)
    stats = system.train()

    # Create a uniform single timestamp for the file group
    experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 6. Print and Save results
    print_benchmark(stats)
    save_benchmark_to_file(DATASET, stats, params_to_save, experiment_timestamp)

if __name__ == "__main__":
    main()