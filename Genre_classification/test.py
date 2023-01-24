import pandas as pd
song_name = [1, 2, 3]
song_url = ['a', 'b', 'c']

# Convert lists to DataFrames
df1 = pd.DataFrame(song_name, columns=['Song Name'])
df2 = pd.DataFrame(song_url, columns=['Song Link'])

# Use the merge() function to merge the two DataFrames
merged_df = pd.merge(df1, df2, left_index=True, right_index=True)
recommendation= merged_df.to_string(index=False)

print(recommendation)
