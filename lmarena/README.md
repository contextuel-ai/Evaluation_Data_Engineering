This folder contains files for the processing of the **LMArena** dataset. Below is an explanation of the importance of each file. 

**augmented_lmarena.csv**
Contains all the same information as **lmarena.csv**, but was supplemented by a Claude API search to provide additional information on a one-sentence description, input_modality, output_modality, total_parameter_count, max_input_tokens, max_output_tokens, context_window_tokens, and the source of the information. This was done conservatively so as to avoid hallucinations. Some of the information can be inferred directly from the model name while other information is taken from HF. The additional information was not found for every model, so the data was only sparsely augmented where possible. 

**augmented_lmarena.ttl**
Triples for the **augmented_lmarena.csv** dataset. It contains all the same triples as **lmarena.ttl**, plus some additional ones for the information that was added. 

**generate_augmented_ttl_from_csv.py**
Generate **augmented_lmarena.ttl** from **augmented_lmarena.csv**.

**generated_trimmed_ttl_from_csv.py**
Generated **lmarena.ttl** from **lmarena.csv**.

**lmarena.csv**
The original dataset taken from the LMArena leaderboard. 

**lmarena.ttl**
Triples for the information contained in **lmarena.csv**. 

**lmarena.ttl.bak**
Backup file for **lmarena.ttl**.
