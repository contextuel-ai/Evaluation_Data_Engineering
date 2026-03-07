This folder handles the **Artifical Analysis** dataset. Below is an explanation of the importance of each file. 

**aa_csv_to_ttl.py**
Converts the information in **aa_models_with_pricing_speed_and_benchmarks.csv** into triples for the ontology. 

**aa_extraction.py**
Extracts the dataset from Artifical Analysis using an API key. It converts the dataset to **aa_models_with_pricing_speed_and_benchmarks.csv**.

**aa_models.ttl**
The triples, converted from the .csv, for use in Enki. 

**aa_models.ttl.bak**
A backup file of the aa_models.ttl. 

**aa_models_with_pricing_speed_and_benchmarks.csv**
The dataset in .csv format. It contains 406 models and evaluates latency, cost, and quality. 
