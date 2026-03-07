This folder contains the **Hugging Face** leaderboard data. Below is an explanation of the importance of each file. 

**hf_extaction.py**
The dataset is imported using the datasets library in Python. The information is then converted to a .csv file. 

**hf_leaderboard.ttl**
Triples of the dataset, for use in Enki. Note: The file is over 300,000 lines long. 

**hf_leaderboard.ttl.bak**
Backup for **hf_leaderboard.ttl**. 

**hf_leaderboard_csv_to_ttl.py**
Converts the information in **hf_leaderboard_long20260221.csv** into triples. 

**hf_leaderboard_long20260221.csv**
The dataset in .csv form. This version of the dataset was pulled on 02/21/2026. 
