# Web Scraping

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import csv

# Getting HTML data
agent = webdriver.Firefox()
agent.get('https://romonitorstats.com/leaderboard/active/')

data = list()
page = 0
max_pages = 100 # Max games = 10 * max_pages (Max 'Max Pages' = 100)

while page < max_pages:  # Adjust this number to scrape more or fewer pages
    html = agent.page_source

    # Find the data
    soup = BeautifulSoup(html, 'lxml')
    games = soup.find('tbody')
    game_entries = games.find_all('tr')

    # Enter data of each game into a list, and add it into a main list
    for entry in game_entries:
        game = entry.find_all('td')
        game_data = list()
        for i in range(9):
            if i != 1:
                if i != 2:
                    game_data.append(game[i].text.replace(' ', ''))
                else:
                    game_data.append(game[i].text)
        data.append(game_data)
    page += 1

    # Click the "Next Page" button, so more game data can be scraped
    if page < max_pages:
            button = WebDriverWait(agent, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'btn btn-primary float-right') and text()=' Next Page ']"))
            )
            time.sleep(.75) # ensure there is no timeout by the website
            button.click()

# Create and write into CSV file
with open("roblox_games.csv", 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Rank", "Name", "Active", "Visits", "Favourites", "Likes", "Dislikes", "Rating"])
    for game in data:
        game[7] = game[7].replace("%", "")
        writer.writerow(game)

# Say goodbye
agent.quit()