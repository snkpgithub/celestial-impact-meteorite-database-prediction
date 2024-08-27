#!/usr/bin/env python
# coding: utf-8

# ## __________________________  [FINAL - PROJECT] _____________________________
# 
# ## Celestial Impact: Creating a Meteorite Database, Normalizing, and Predicting Fall Types

# ### About Data Sets: The Meteoritical Society collects data on meteorites that have fallen to Earth from outer space. This dataset includes the location, mass, composition, and fall year for over 45,000 meteorites that have struck our planet.
# - We have kept only 7000 rows to perform SQL operations and entire datasets for machine learning prediction.
# 
# 
# 
# ### _________ Team Members: Shashank Pandey, R.Sai Dinesh, Pratiksha S.J, Vinay C.S. _______________

# 
# ## MOTIVATION:
# ### The exploration of meteorite landings and their classification through making  machine learning brings forth exciting opportunities for scientific discovery and understanding. The successful development of a predictive model to differentiate between meteorites observed falling ("Fell") and those discovered after impact ("Found") opens up new avenues for researchers and enthusiasts alike.
# 

# ### - Our project focuses on organizing and normalizing this data into structured tables,allowing for efficient analysis. 
# 
# ### - Additionally, we applied machine learning algorithms to predict the outcomes of meteorite impacts-  whether they resulted in a find or a failure. 
# 
# ### - In essence, our project combines a passion for space exploration, database normalization, and machine learning to gain insights into the mysteries of meteorite landings on Earth.
# 

# ## About parsing data & normalization of table.
# ###  In our meteorite dataset, we've organized information into five normalized SQLite tables: Meteorites, Locations, Classifications, MassByReclass, Falls, and Types. Each table captures specific attributes like names, classifications, masses, locations, and fall types. 
# ### -The process involved parsing and storing data from a CSV file, establishing foreign key relationships for data integrity, and promoting efficient storage through normalization.
# ### -This structure minimizes redundancy, enhances data integrity, and facilitates streamlined queries for meteorite attributes. It aligns with normalization principles, optimizing storage and minimizing the risk of data anomalies.

# In[1]:


#IMPORTING USEFUL LIBRARIES, TO BE USE FOR THE INITIAL STEPS.
import csv
import sqlite3

#PARSING MY DATA FROM CSV FILE, I HAVE NOT USED PANDAS TO READ CSV FILE
file_path = 'MeteoriteLandings.csv'  
data = []

with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)

###########################################################################################
#                                 Create an SQLite database
###########################################################################################

conn = sqlite3.connect('meteorite_data_normalized_5_tables.db')
cursor = conn.cursor()

def create_table(conn, create_table_sql, drop_table_name=None):
    
    if drop_table_name:
        try:
            conn_curs = conn.cursor()
            conn_curs.execute("""DROP TABLE IF EXISTS %s""" % (drop_table_name))
        except Error as err:
            print(e)
    
    try:
        conn_curs = conn.cursor()
        conn_curs.execute(create_table_sql)
    except Error as err:
        print(err)
######################################################################################################       
#                           Creating tables for normalization
######################################################################################################

sql_query = '''CREATE TABLE IF NOT EXISTS Meteorites (
                    meteorite_id INTEGER PRIMARY KEY,
                    name TEXT,
                    id INTEGER,
                    mass INTEGER,
                    fall_id INTEGER,
                    year INTEGER,
                    location_id INTEGER,
                    classification_id INTEGER,
                    FOREIGN KEY (fall_id) REFERENCES Falls(fall_id),
                    FOREIGN KEY (location_id) REFERENCES Locations(location_id),
                    FOREIGN KEY (classification_id) REFERENCES Classifications(classification_id)
                    )'''

create_table(conn,sql_query,"Meteorites")

sql_query = '''CREATE TABLE IF NOT EXISTS Locations (
                    location_id INTEGER PRIMARY KEY,
                    reclat REAL,
                    reclong REAL,
                    GeoLocation TEXT
                    )'''
create_table(conn,sql_query,"Locations")


sql_query = '''CREATE TABLE IF NOT EXISTS Classifications (
                    classification_id INTEGER PRIMARY KEY,
                    recclass TEXT
                    )'''
create_table(conn,sql_query,"Classifications")

sql_query = '''CREATE TABLE IF NOT EXISTS MassByReclass (
                    mass_id INTEGER PRIMARY KEY,
                    mass INTEGER,
                    reclass_id INTEGER,
                    FOREIGN KEY (reclass_id) REFERENCES Classifications(classification_id)
                    )'''
create_table(conn, sql_query, "MassByReclass")


sql_query = '''CREATE TABLE IF NOT EXISTS Falls (
                    fall_id INTEGER PRIMARY KEY,
                    fall TEXT
                    )'''
create_table(conn,sql_query,"Falls")


##################################################################################################
                            # INSERTING PARSED DATA INTO THE NORMALISED TABLE
##################################################################################################

for row in data:
    # Insert into Falls table and get the last inserted row ID (fall_id)
    cursor.execute('''INSERT OR IGNORE INTO Falls (fall) VALUES (?)''', (row['fall'],))
    cursor.execute('''SELECT fall_id FROM Falls WHERE fall=?''', (row['fall'],))
    fall_id = cursor.fetchone()[0]  # Retrieve the fall_id

    # Insert into Locations table and get the last inserted row ID (location_id)
    cursor.execute('''INSERT OR IGNORE INTO Locations (reclat, reclong, GeoLocation)
                      VALUES (?, ?, ?)''',
                   (row['reclat'], row['reclong'], row['GeoLocation']))
    cursor.execute('''SELECT location_id FROM Locations WHERE reclat=? AND reclong=?''',
                   (row['reclat'], row['reclong']))
    location_id = cursor.fetchone()[0]  # Retrieve the location_id

    # Insert into Classifications table and get the last inserted row ID (classification_id)
    cursor.execute('''INSERT OR IGNORE INTO Classifications (recclass) VALUES (?)''', (row['recclass'],))
    cursor.execute('''SELECT classification_id FROM Classifications WHERE recclass=?''', (row['recclass'],))
    classification_id = cursor.fetchone()[0]  # Retrieve the classification_id

    # Insert into MassByReclass table and get the last inserted row ID (mass_id)
    cursor.execute('''INSERT OR IGNORE INTO MassByReclass (mass, reclass_id)
                      VALUES (?, ?)''', (row['mass (g)'], row['recclass']))
    cursor.execute('''SELECT mass_id FROM MassByReclass WHERE mass=? AND reclass_id=?''',
                   (row['mass (g)'], row['recclass']))
    mass_id = cursor.fetchone()[0]  # Retrieve the mass_id
    
    # Insert into Meteorites table with corresponding IDs
    cursor.execute('''INSERT INTO Meteorites 
                      (name, id, mass, fall_id, year, location_id, classification_id) 
                      VALUES (?, ?, ?, ?, ?, ?, ?)''',
                   (row['name'], row['id'], row['mass (g)'], fall_id, row['year'], location_id, classification_id))

conn.commit()


# In[39]:


#####################################################################################
#                     PRINTING THE CRAETED TABLE                    
####################################################################################
import sqlite3
def fetch_table_info(table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect('meteorite_data_normalized_5_tables.db')
    cursor = conn.cursor()

    # Fetch information from the specified table
    cursor.execute(f'SELECT * FROM {table_name}')
    table_data = cursor.fetchall()

    # Print the information
    print(f"\n{table_name} Table Information:")
    for row in table_data:
        print(row)
    # Close the connection
    conn.close()
    
fetch_table_info('Meteorites')


# In[3]:


fetch_table_info('Locations')


# In[4]:


fetch_table_info('Classifications')


# In[5]:


fetch_table_info('Falls')


# In[6]:


fetch_table_info('MassByReclass')


# ### Reconstructing the original data and loads it into a Pandas DataFrame. [Important Table & Columns]

# In[12]:


import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('meteorite_data_normalized_5_tables.db')

# SQL query with JOIN statements - Reconstructing original data (Modified)
join_sql_query = '''
    SELECT Meteorites.name, Meteorites.id, Meteorites.mass, Falls.fall, Meteorites.year,
           Locations.reclat, Locations.reclong, Meteorites.classification_id, Classifications.recclass
    FROM Meteorites
    INNER JOIN Falls ON Meteorites.fall_id = Falls.fall_id
    INNER JOIN Locations ON Meteorites.location_id = Locations.location_id
    INNER JOIN Classifications ON Meteorites.classification_id = Classifications.classification_id
'''

# Execute the SQL query and fetch data into a Pandas DataFrame
Meteorite_data = pd.read_sql_query(join_sql_query, conn)

# Close the database connection
conn.close()

# Display the Pandas DataFrame
Meteorite_data.head()


# ## QUERIES USING OUR CREATED DATABASE:
# ### These following queries are meant to be part from our crated table, its use and findings in our project.

# In[4]:


#Query 1 :  Provide the count of meteorites of fall

import sqlite3
import matplotlib.pyplot as plt
conn = sqlite3.connect('meteorite_data_normalized_5_tables.db')
cursor = conn.cursor()

# Query to get the count of meteorites for each fall type
query = '''
    SELECT f.fall AS FallType, COUNT(*) AS MeteoriteCount
    FROM Meteorites m
    INNER JOIN Falls f ON m.fall_id = f.fall_id
    GROUP BY f.fall
'''
cursor.execute(query)
results = cursor.fetchall()
conn.close()
# Display the results
for row in results:
    print(f"Fall Type: {row[0]}, Meteorite Count: {row[1]}")
# Plotting the graph
fall_types = [row[0] for row in results]
meteorite_counts = [row[1] for row in results]
colors = ['blue', 'green']
plt.bar(fall_types, meteorite_counts, color=colors)
plt.xlabel('Fall Type')
plt.ylabel('Meteorite Count')
plt.title('Count of Meteorites for Each Fall Type')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()
# Show the plot
plt.show()


# In[14]:


#Q2: Locations With Highest Crashes
import sqlite3

conn = sqlite3.connect('meteorite_data_normalized_5_tables.db')

#top 10 locations with the most meteorite landings
sqll_query = '''
    SELECT L.reclat, L.reclong, L.GeoLocation, COUNT(M.meteorite_id) AS meteorite_count
    FROM Locations L
    JOIN Meteorites M ON L.location_id = M.location_id
    GROUP BY L.location_id
    ORDER BY meteorite_count DESC
    LIMIT 10;
'''


cursor = conn.cursor()
cursor.execute(sqll_query)
top_locations = cursor.fetchall()

#results
print("Top 10 Locations with the Most Meteorite Landings:")
for location in top_locations:
    print(f"Latitude: {location[0]}, Longitude: {location[1]}, GeoLocation: {location[2]}, Meteorite Count: {location[3]}")


conn.close()


# In[15]:


# Q3: Top 5 Meteorite types with Largest Fall Count
import sqlite3


conn = sqlite3.connect('meteorite_data_normalized_5_tables.db')

try:
    #top 5 meteorite types with the most falls
    top_falls = '''
        SELECT CR.recclass, COUNT(M.meteorite_id) AS fall_count
        FROM Meteorites M
        JOIN Falls F ON M.fall_id = F.fall_id
        JOIN Classifications CR ON M.classification_id = CR.classification_id
        WHERE F.fall IS NOT NULL
        GROUP BY CR.recclass
        ORDER BY fall_count DESC
        LIMIT 5;
    '''


    cursor = conn.cursor()
    cursor.execute(top_falls)
    top_types_falls_count = cursor.fetchall()

    #results
    print("Top 5 Meteorite Types with the Most falls and their fall counts:")
    for meteorite_type in top_types_falls_count:
        print(f"Reclass: {meteorite_type[0]}, Fall Count: {meteorite_type[1]}")

except sqlite3.Error as e:
    print("SQLite Error:", e)

finally:

    conn.close()


# In[3]:


#Q4: List of highest and lowest meteorite-masses each year from 2001
import sqlite3
conn = sqlite3.connect('meteorite_data_normalized_5_tables.db')

try:
    mass_stats_query = '''
        SELECT year,
               MAX(mass) AS largest_mass,
               MIN(mass) AS smallest_mass,
               MAX(CASE WHEN mass = (SELECT MAX(mass) FROM Meteorites WHERE year = M.year) THEN name END) AS largest_mass_meteorite,
               MAX(CASE WHEN mass = (SELECT MIN(mass) FROM Meteorites WHERE year = M.year) THEN name END) AS smallest_mass_meteorite
        FROM Meteorites M
        WHERE year >= 2001
        GROUP BY year;
    '''

    cursor = conn.cursor()
    cursor.execute(mass_stats_query)
    mass_stats = cursor.fetchall()

    
    print("{:<10} {:<15} {:<25} {:<15} {:<25}".format("Year", "Largest Mass", "Largest Meteorite Name", "Smallest Mass", "Smallest Meteorite Name"))
    for stat in mass_stats:
        print("{:<10} {:<15} {:<25} {:<15} {:<25}".format(stat[0], stat[1], stat[3], stat[2], stat[4]))

except sqlite3.Error as e:
    print("SQLite Error:", e)

finally:
    conn.close()


# ## Problem: Predicting Meteorite Fall Type
# ### --------------------------------------------------------------------------------------------------------------------------------------------------
# ### The objective is to create a machine learning model that, using a variety of physical characteristics and landing data, can forecast the sort of meteorite that will fall. 
# - Considered: The fall column, which shows if the meteorite was seen falling (Fell) or found after impact (Found), is the target variable.

# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[17]:


# Load the original dataset kept from 
file_path = 'Meteorite_Landings.csv' # Data on over 45k meteorites that have struck Earth
df = pd.read_csv(file_path)


# In[18]:


# Select relevant features and target variable
features = ['mass (g)', 'nametype', 'year', 'reclat', 'reclong']
target_variable = 'fall'


# In[19]:


# Drop rows with missing values in the selected columns
df = df.dropna(subset=features + [target_variable])


# In[25]:


# Encode categorical variables
df_encoded = pd.get_dummies(df[features])


# In[28]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded,
    df[target_variable],
    test_size=0.20,
    random_state=42
)

# Choose a machine learning algorithm (Random Forest in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# In[29]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report_result)


# ### Accuracy: The accuracy of the model is approximately 98.62%, indicating that it correctly predicted the fall type for a large majority of the meteorites in the test set.
# 
# 
# ## Classification Report:
# ### Precision: Precision measures the accuracy of positive predictions of Fall.
# ### Recall (Sensitivity): Recall measures the proportion of actual positives correctly predicted Fall.
# ### F1-Score: The weighted average of falls - precision and recall.
# ### Support: The number of actual occurrences of each class in the specified target variable.
# 
# 

# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Fell', 'Predicted Found'], yticklabels=['Actual Fell', 'Actual Found'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Meteorite_Landings.csv'
meteorite_data = pd.read_csv(file_path)

print(meteorite_data.info())

plt.figure(figsize=(12, 6))
sns.scatterplot(x='reclong', y='reclat', data=meteorite_data, hue='fall', palette='viridis', marker='o', s=50)
plt.title('Geographical Distribution of Meteorite Landings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Fall Type')
plt.show()


# FUTURE SCOPE:
# - Enhanced Prediction Models
# - Geospatial Insights
# - Mobile App Development
# - Automation and Real-time Updates
# 
# Insights & Observations:
# - The project on meteorite datasets provides valuable insights into the geographical distribution and characteristics of meteorite landings on Earth. 
# - By leveraging database normalization and machine learning techniques, the project facilitates predictions of meteorite fall types. 
# - The visualizations showcase the spatial patterns of landings, aiding in understanding trends and potential correlations between geographical locations and meteorite characteristics. Additionally, the project lays the groundwork for further exploration, allowing for future analysis and discoveries in the field of meteoritics.
