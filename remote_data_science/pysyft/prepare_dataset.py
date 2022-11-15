# run this cell
try:
    import pandas as pd
    import json
    from collections import defaultdict
    
#     def make_Dictionaries(processed_data_file="data/preprocessed.csv",dictionary_storage_location="data"):
#         print("AAAAAHHHH")

#         Data=pd.read_csv(processed_data_file)

#         # Function to return a default
#         # values for keys that is not
#         # present
#         def def_value():
#             return "Not Present"

#         # Defining the dict
#         iid_dict = defaultdict(def_value)
#         uid_dict = defaultdict(def_value)
#         sort_movies=Data['MovieName'].unique()
#         sort_movies.sort()
#         for i in range(len(sort_movies)):
#             iid_dict[sort_movies[i]]=i

#         sort_users=Data['UserID'].unique()
#         sort_users.sort()
#         for i in range(len(sort_users)):
#             uid_dict[int(sort_users[i])]=i

#         with open(dictionary_storage_location+"iid_dict.json", "w") as outfile:
#             json.dump(iid_dict, outfile)
#         with open(dictionary_storage_location+"uid_dict.json", "w") as outfile:
#             json.dump(uid_dict, outfile)
        

#         return True
    
    dataset = pd.read_csv("preprocessed_10000_entries.csv")
    
    
    data = {'ID': ['011', '015', '022', '034'],
           'Age': [40, 39, 9, 8]}
    df = pd.DataFrame(data)

    print(df)
    
    newdict = {}

    print(dataset)
except Exception:
    print("Install the latest version of Pandas using the command: !pip install pandas")