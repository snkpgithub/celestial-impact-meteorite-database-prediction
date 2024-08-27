# celestial-impact-meteorite-database-prediction
A project focused on creating a meteorite database, normalizing it into structured tables, and applying machine learning to predict meteorite fall types.
Celestial Impact: Creating a Meteorite Database, Normalizing, and Predicting Fall Types
About the Dataset:
The dataset, provided by The Meteoritical Society, includes details on over 45,000 meteorites that have struck Earth. This data includes location, mass, composition, and fall year. We have selected 7,000 rows for SQL operations, while the entire dataset is used for machine learning predictions.

Team Members:

Shashank Pandey

R.Sai Dinesh

Pratiksha S.J

Vinay C.S.

Motivation:
Exploring meteorite landings and classifying them through machine learning brings exciting opportunities for scientific discovery. Our project organizes this data into structured tables and applies machine learning algorithms to predict meteorite impacts, differentiating between meteorites that were observed falling ("Fell") and those discovered post-impact ("Found").

Project Structure:
Data Parsing & Normalization:

The dataset is parsed and normalized into five SQLite tables: Meteorites, Locations, Classifications, MassByReclass, Falls, and Types. This structure minimizes redundancy, enhances data integrity, and facilitates streamlined queries.
SQL Queries & Data Analysis:

Several SQL queries are performed on the database, including:
Query 1: Count of meteorites by fall type.
Query 2: Locations with the highest number of meteorite crashes.
Query 3: Top 5 meteorite types with the largest fall count.
Query 4: List of highest and lowest meteorite masses each year from 2001.
Machine Learning - Predicting Meteorite Fall Types:

A Random Forest Classifier is applied to predict meteorite fall types using features like mass, year, location, and more.
Accuracy: Achieved approximately 98.62% accuracy on the test data.
Visualization:

A geographical distribution of meteorite landings is plotted, showcasing spatial patterns and aiding in understanding trends.
Future Scope:
Enhanced Prediction Models: Further improve the accuracy and robustness of the prediction models.
Geospatial Insights: Deeper analysis of meteorite landing locations.
Mobile App Development: Bring the data and insights to a wider audience.
Automation and Real-time Updates: Incorporate real-time data updates.
Insights & Observations:
The project offers valuable insights into the characteristics and geographical distribution of meteorite landings. By integrating database normalization with machine learning, the project allows for accurate predictions of meteorite fall types, contributing to the field of meteoritics.
