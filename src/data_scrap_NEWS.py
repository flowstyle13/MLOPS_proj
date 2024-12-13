import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL of the news site (e.g., BBC News)
BASE_URL = 'https://www.bbc.com/news'

def fetch_news():
    """Fetch the latest news headlines from BBC News and return as a DataFrame."""
    try:
        # Send a request to the website
        response = requests.get(BASE_URL)
        response.raise_for_status()  # Raise error for bad status codes

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all article links (BBC News uses <a> tags with 'gs-c-promo-heading' class)
        articles = soup.find_all('a', class_='gs-c-promo-heading')

        # If no articles are found, return an empty DataFrame
        if not articles:
            print("No articles found!")
            return pd.DataFrame()

        # Extract the title and link for each article
        news_data = [{
            'title': article.get_text(),
            'url': article['href'] if article['href'].startswith('https') else 'https://www.bbc.com' + article['href']
        } for article in articles]

        # Convert to DataFrame and return
        return pd.DataFrame(news_data)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return pd.DataFrame()

def save_news(news_df):
    """Save the news data to a CSV file."""
    if not news_df.empty:
        news_df.to_csv('data/raw_news_bbc.csv', index=False)
        print("News data saved to data/raw_news_bbc.csv")
    else:
        print("No data to save.")

if __name__ == '__main__':
    news_df = fetch_news()  # Fetch the news
    save_news(news_df)  # Save it to CSV
