import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import combinations

# Create visualizations directory
output_dir = 'metricsCreation/visualizations'
os.makedirs(output_dir, exist_ok=True)

# Load the data
definition_df = pd.read_csv('metricsCreation/definition_analysis_results.csv')
embedding_df = pd.read_csv('metricsCreation/embedding_drift_results.csv')
token_df = pd.read_csv('metricsCreation/token_analysis_results.csv')

# Set up plotting style
plt.style.use('default')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
teal_blue = '#008B8B'

# Bar graph showing average similarity for definition between model def and reference def
def plot_1_definition_similarity():
    model_cols = [col for col in definition_df.columns if col != 'word']
    avg_similarities = definition_df[model_cols].mean()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(avg_similarities)), avg_similarities.values, color=colors[:len(avg_similarities)])
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Average Similarity Score', fontsize=12)
    plt.title('Average Definition Similarity: Model vs Reference Definition', fontsize=14, fontweight='bold')
    plt.xticks(range(len(avg_similarities)), avg_similarities.index, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/definition_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()

# gpt models with years using token counts
def plot_2_gpt_token_line():
    # GPT models in chronological order with release years
    gpt_order = ['gpt2', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini']
    gpt_years = ['2019', '2022', '2023', '2024', '2024', '2024']
    gpt_labels = [f'{model}\n({year})' for model, year in zip(gpt_order, gpt_years)]
    
    # Filter available models
    available_gpt = [model for model in gpt_order if model in token_df.columns]
    available_labels = [gpt_labels[gpt_order.index(model)] for model in available_gpt]
    
    avg_tokens = token_df[available_gpt].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(avg_tokens)), avg_tokens.values, marker='o', linewidth=3, markersize=10, color='#2ca02c')
    plt.xlabel('GPT Models (Chronological Order)', fontsize=12)
    plt.ylabel('Average Token Count', fontsize=12)
    plt.title('Average Token Count Across GPT Models Over Time', fontsize=14, fontweight='bold')
    plt.xticks(range(len(avg_tokens)), available_labels, ha='center')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, value in enumerate(avg_tokens.values):
        plt.text(i, value + 0.05, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gpt_token_line.png', dpi=300, bbox_inches='tight')
    plt.close()

# two bar graphs containing avg token counts for all models (split in half)
def plot_3_token_counts_split():
    model_cols = [col for col in token_df.columns if col != 'word']
    avg_tokens = token_df[model_cols].mean().sort_values(ascending=False)
    
    mid = len(avg_tokens) // 2
    
    # First half
    plt.figure(figsize=(10, 6))
    first_half = avg_tokens[:mid]
    bars1 = plt.bar(range(len(first_half)), first_half.values, color=colors[:len(first_half)])
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Average Token Count', fontsize=12)
    plt.title('Average Token Count - Group 1', fontsize=14, fontweight='bold')
    plt.xticks(range(len(first_half)), first_half.index, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/token_counts_group1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Second half
    plt.figure(figsize=(10, 6))
    second_half = avg_tokens[mid:]
    bars2 = plt.bar(range(len(second_half)), second_half.values, color=colors[:len(second_half)])
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Average Token Count', fontsize=12)
    plt.title('Average Token Count - Group 2', fontsize=14, fontweight='bold')
    plt.xticks(range(len(second_half)), second_half.index, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/token_counts_group2.png', dpi=300, bbox_inches='tight')
    plt.close()

# embedding drifts for each of the five models, comparing one model to the other four
def plot_4_embedding_drift():
    models = ['HuggingFaceTB/SmolLM2-135M', 'gpt2', 'gemini', 'Qwen/Qwen3-0.6B', 'google/gemma-3-1b-it']
    
    for i, model in enumerate(models):
        plt.figure(figsize=(10, 6))
        
        # Find columns that contain this model
        model_cols = [col for col in embedding_df.columns if model in col and col != 'word']
        
        if model_cols:
            avg_drifts = embedding_df[model_cols].mean()
            bars = plt.bar(range(len(avg_drifts)), avg_drifts.values, color=teal_blue, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            plt.xlabel('Model Comparisons', fontsize=12)
            plt.ylabel('Average Embedding Drift', fontsize=12)
            plt.title(f'Embedding Drift: {model.split("/")[-1]} vs Other Models', fontsize=14, fontweight='bold')
            plt.xticks(range(len(avg_drifts)), 
                      [col.replace(f'{model}_vs_', '').replace(f'_vs_{model}', '').split('/')[-1] for col in avg_drifts.index], 
                      rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        safe_name = model.replace('/', '_').replace(' ', '_')
        plt.savefig(f'{output_dir}/embedding_drift_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

# gemini graph showing tokenizations and definition similarity for Gemini models
def plot_5_gemini_analysis():
    gemini_models = [col for col in token_df.columns if 'gemini' in col.lower()]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Token counts
    avg_tokens = token_df[gemini_models].mean()
    ax1.plot(range(len(avg_tokens)), avg_tokens.values, marker='o', linewidth=3, markersize=10, color='#ff7f0e')
    ax1.set_xlabel('Gemini Models', fontsize=12)
    ax1.set_ylabel('Average Token Count', fontsize=12)
    ax1.set_title('Gemini Models: Token Count Analysis', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(avg_tokens)))
    ax1.set_xticklabels(avg_tokens.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    for i, value in enumerate(avg_tokens.values):
        ax1.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Definition similarity
    gemini_def_models = [col for col in definition_df.columns if 'gemini' in col.lower()]
    if gemini_def_models:
        avg_similarity = definition_df[gemini_def_models].mean()
        ax2.plot(range(len(avg_similarity)), avg_similarity.values, marker='s', linewidth=3, markersize=10, color='#d62728')
        ax2.set_xlabel('Gemini Models', fontsize=12)
        ax2.set_ylabel('Average Definition Similarity', fontsize=12)
        ax2.set_title('Gemini Models: Definition Similarity Analysis', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(avg_similarity)))
        ax2.set_xticklabels(avg_similarity.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for i, value in enumerate(avg_similarity.values):
            ax2.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gemini_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# correlation between token count and definition similarity
def plot_6_correlation_analysis():
    model_cols = [col for col in definition_df.columns if col != 'word']
    token_cols = [col for col in token_df.columns if col != 'word']
    common_models = list(set(model_cols) & set(token_cols))
    
    if common_models:
        token_means = token_df[common_models].mean()
        def_means = definition_df[common_models].mean()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(token_means.values, def_means.values, s=100, alpha=0.7, color='#9467bd')
        
        for i, model in enumerate(common_models):
            plt.annotate(model.split('/')[-1], (token_means[model], def_means[model]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.xlabel('Average Token Count', fontsize=12)
        plt.ylabel('Average Definition Similarity', fontsize=12)
        plt.title('Correlation: Token Count vs Definition Similarity', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(token_means.values, def_means.values, 1)
        p = np.poly1d(z)
        plt.plot(token_means.values, p(token_means.values), "--", alpha=0.7, color='red', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_7_token_drift_heatmap():
    # Select models that appear in the token analysis
    available_models = [col for col in token_df.columns if col != 'word']
    
    # Limit to 10 models for readability (matching the image)
    selected_models = available_models[:10] if len(available_models) >= 10 else available_models
    
    # Create token drift matrix
    n_models = len(selected_models)
    drift_matrix = np.zeros((n_models, n_models))
    
    for i, model1 in enumerate(selected_models):
        for j, model2 in enumerate(selected_models):
            if i == j:
                drift_matrix[i, j] = 0.0  # No drift with itself
            else:
                # Calculate token drift for each word, then average
                model1_tokens = token_df[model1].dropna()
                model2_tokens = token_df[model2].dropna()
                
                # Find common indices (words that have data for both models)
                common_indices = model1_tokens.index.intersection(model2_tokens.index)
                
                if len(common_indices) > 0:
                    token_drifts = []
                    for idx in common_indices:
                        tokens1 = token_df.loc[idx, model1]
                        tokens2 = token_df.loc[idx, model2]
                        
                        if pd.notna(tokens1) and pd.notna(tokens2) and max(tokens1, tokens2) > 0:
                            # Token drift formula: 1 - |tokens1 - tokens2| / max(tokens1, tokens2)
                            drift = abs(tokens1 - tokens2) / max(tokens1, tokens2)
                            token_drifts.append(drift)
                    
                    if token_drifts:
                        drift_matrix[i, j] = np.mean(token_drifts)
                    else:
                        drift_matrix[i, j] = 0.0
                else:
                    drift_matrix[i, j] = 0.0
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    
    # Use a colormap similar to the image (yellow to red)
    cmap = plt.cm.YlOrRd
    
    im = plt.imshow(drift_matrix, cmap=cmap, aspect='equal', vmin=0, vmax=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Token Drift', rotation=270, labelpad=20, fontsize=12)
    
    # Set ticks and labels
    model_labels = [model.replace('/', '_').replace('-', '_') for model in selected_models]
    plt.xticks(range(n_models), model_labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(n_models), model_labels, fontsize=10)
    
    # Add title
    plt.title(f'Token Drift Matrix ({n_models}x{n_models})', fontsize=16, fontweight='bold', pad=20)
    
    # Add text annotations with drift values
    for i in range(n_models):
        for j in range(n_models):
            text = plt.text(j, i, f'{drift_matrix[i, j]:.2f}',
                           ha="center", va="center", 
                           color="black" if drift_matrix[i, j] < 0.15 else "white",
                           fontsize=9, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{output_dir}/token_drift_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# five bar graphs showing definition drift (absolute difference in similarity) for each model
def plot_8_definition_drift():
    model_cols = [col for col in definition_df.columns if col != 'word']
    
    print(f"Creating definition drift graphs for {len(model_cols[:5])} models...")
    
    # Calculate pairwise differences for each word
    for idx, target_model in enumerate(model_cols[:5]):  # Focus on first 5 models
        print(f"  Processing model {idx+1}/5: {target_model}")
        
        try:
            drift_data = []
            comparison_models = []
            
            for other_model in model_cols:
                if other_model != target_model:
                    # Get data for both models, handling NaN values
                    target_data = definition_df[target_model].dropna()
                    other_data = definition_df[other_model].dropna()
                    
                    # Find common indices (words that have data for both models)
                    common_indices = target_data.index.intersection(other_data.index)
                    
                    if len(common_indices) > 0:
                        # Calculate absolute difference for common words only
                        target_common = definition_df.loc[common_indices, target_model]
                        other_common = definition_df.loc[common_indices, other_model]
                        diffs = abs(target_common - other_common)
                        avg_drift = diffs.mean()
                        
                        drift_data.append(avg_drift)
                        comparison_models.append(other_model.split('/')[-1])
                    else:
                        print(f"    No common data between {target_model} and {other_model}")
            
            if drift_data:  # Only create graph if we have data
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(drift_data)), drift_data, color=teal_blue, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                plt.xlabel('Comparison Models', fontsize=12)
                plt.ylabel('Average Definition Drift', fontsize=12)
                plt.title(f'Definition Drift: {target_model.split("/")[-1]} vs Other Models', fontsize=14, fontweight='bold')
                plt.xticks(range(len(comparison_models)), comparison_models, rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + max(drift_data) * 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                safe_name = target_model.replace('/', '_').replace(' ', '_').replace('-', '_')
                plt.savefig(f'{output_dir}/definition_drift_{safe_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    ✓ Created definition_drift_{safe_name}.png")
            else:
                print(f"    ✗ No valid data for {target_model}")
                
        except Exception as e:
            print(f"    ✗ Error processing {target_model}: {e}")



def main():
    print("Creating visualizations...")
    
    plot_1_definition_similarity()
    print("✓ Definition similarity bar graph created")
    
    plot_2_gpt_token_line()
    print("✓ GPT token line graph with years created")
    
    plot_3_token_counts_split()
    print("✓ Token count split bar graphs created")
    
    plot_4_embedding_drift()
    print("✓ Embedding drift graphs created (teal blue)")
    
    plot_5_gemini_analysis()
    print("✓ Gemini analysis graph created")
    
    plot_6_correlation_analysis()
    print("✓ Correlation analysis scatter plot created")
    
    plot_7_token_drift_heatmap()
    print("✓ Token drift matrix heatmap created")
    
    plot_8_definition_drift()
    print("✓ Definition drift graphs created (5 models)")
    
    print(f"\nAll visualizations saved to {output_dir}/")

main()