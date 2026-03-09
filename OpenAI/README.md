This folder uses OpenAI URL structure and patterns to parse models, collect information from its docs, and generate triples for its current selection of models.

**model_list_data.py** 
Python script that generates both JSON files using the OPENAI_API_KEY and by scraping data using cURL statements. 

**openai_models_raw.json**
This file contains only the information accessible from the OPENAI_API_KEY. 

**openai_models.json**
This file contains information scraped from the appropriate URL for each OpenAI model. Interesting data incldues: context window size, speed, knowledge cutoff, and more. 

