from dagster import asset
from pandas import DataFrame, read_html, get_dummies, to_numeric
from sklearn.linear_model import LinearRegression as Regression

@asset
def country_stats() -> DataFrame:
    """ Raw country population data """
    df = read_html("https://tinyurl.com/mry64ebh", flavor='html5lib')[0]
    df.columns = ["country", "pop_2022", "pop_2023", "pop_change", "continent", "region"]
    df["pop_change"] = ((to_numeric(df["pop_2023"]) / to_numeric(df["pop_2022"])) - 1) * 100
    return df

@asset
def change_model(country_stats: DataFrame) -> Regression:
    """ A regression model for each continent """
    data = country_stats.dropna(subset=["pop_change"])
    dummies = get_dummies(data[["continent"]])
    return Regression().fit(dummies, data["pop_change"])

@asset
def continent_stats(country_stats: DataFrame, change_model: Regression) -> DataFrame:
    """ Summary of continent populations and change """
    # total populations for 2022 and 2023 by continent
    continent_summary = country_stats.groupby("continent").agg(
        pop_2022_total=("pop_2022", "sum"),
        pop_2023_total=("pop_2023", "sum"),
        mean_pop_change=("pop_change", "mean")
    ).reset_index()
    
    # dummy values for the continent column
    dummies = get_dummies(continent_summary[["continent"]])
    
    # new column for % change
    continent_summary["predicted_pop_change"] = change_model.predict(dummies)

    # Calculate predicted population for 2024 based on the predicted population change
    continent_summary["predicted_pop_2024"] = continent_summary["pop_2023_total"] * (
        1 + continent_summary["predicted_pop_change"] / 100
    )
    return continent_summary

def run_model():
    raw_stats = country_stats()            # Get raw country statistics
    change = change_model(raw_stats)       # Apply the change model
    continent = continent_stats(raw_stats, change)  # Calculate continent stats
    print(continent)                       # Print the continent-level stats

# Run the model
run_model()
