import pandas as pd
import numpy as np
from metrics import count_tokens_transformers, count_tokens_openAI, semantic_drift, get_model_definition, compare_definitions


def add_metrics_columns(df, gemini_api_key=None, openai_model='gpt-4o'):
    df_copy = df.copy()

    # Initialize new columns, ensuring if an error occurs, the columns are still created
    df_copy['gpt2_token'] = np.nan
    df_copy['open_gpt_token'] = np.nan
    df_copy['embedding_similarity'] = np.nan
    df_copy['gpt2_definition_similarity'] = np.nan
    df_copy['opengpt_definition_similarity'] = np.nan
    
    print("Processing rows...")
    
    for idx, row in df_copy.iterrows():
        word = row['word']
        standard_def = row['standard_definition']
        
        # Start by calculating token counts for each model type
        try:
            gpt2_tokens = count_tokens_transformers(word, 'gpt2')
            df_copy.loc[idx, 'gpt2_token'] = gpt2_tokens
            print(f"  GPT2 tokens: {gpt2_tokens}")
        except Exception as e:
            print(f"Error counting GPT2 tokens: {e}")  
        try:
            # using gpt-4o as default OpenAI model
            openai_tokens = count_tokens_openAI(word, openai_model)
            df_copy.loc[idx, 'open_gpt_token'] = openai_tokens
            print(f"  OpenAI tokens: {openai_tokens}")
        except Exception as e:
            print(f"Error counting OpenAI tokens: {e}")
        
        # Caclulate embedding similarity between models
        # for initial testing purposes, we are using bert-base-uncased as the OpenAI model
        try:
            embedding_sim = semantic_drift('gpt2', 'bert-base-uncased', word)
            if embedding_sim is not None:
                # Convert drift score to similarity (drift = 1 - similarity)
                df_copy.loc[idx, 'embedding_similarity'] = embedding_sim
                print(f"Embedding similarity: {embedding_sim}")
        except Exception as e:
            print(f"Error calculating embedding similarity: {e}")
        

        # Calculate definition similarities between model definition and standard definition only if standard_definition is not null
        if pd.notna(standard_def):
            try:
                # getting cpt defintion
                gpt2_def = get_model_definition('gpt2', word)
                if gpt2_def is not None:
                    gpt2_def_sim = compare_definitions(gpt2_def, standard_def)
                    df_copy.loc[idx, 'gpt2_definition_similarity'] = gpt2_def_sim
                    print(f"GPT2 definition similarity: {gpt2_def_sim}")
                else:
                    print(f"Failed to get GPT2 definition")
            except Exception as e:
                print(f"Error calculating GPT2 definition similarity: {e}")
            
            try:
                # Using hugging face gpt neo model for text generation 
                openai_def = get_model_definition('EleutherAI/gpt-neo-1.3B', word)
                if openai_def is not None:
                    openai_def_sim = compare_definitions(openai_def, standard_def)
                    df_copy.loc[idx, 'opengpt_definition_similarity'] = openai_def_sim
                    print(f"OpenAI definition similarity: {openai_def_sim}")
                else:
                    print(f"Failed to get gpt neo defintion")
            except Exception as e:
                print(f"Error calculating gpt neo definition similarities: {e}")
        else:
            print(f"Skipping definition similarities due to standard_definition being null")
    return df_copy

df = pd.read_csv('standard_defined_terms_clean.csv')
df_with_metrics = add_metrics_columns(df)
df_with_metrics.to_csv('standard_defined_terms_with_metrics.csv', index=False)