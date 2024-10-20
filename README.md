# Roblox Explanatory Data Analysis

## Overview
An Explanatory Data Analysis project that uses machine learning algorithms to test hypotheses/questions.

This dataset was formed by a combination of **Selenium** and **BeautifulSoup**, and data was represented
and clean by **Pandas** and **Regular Expressions**.

The data found was sourced from **https://romonitorstats.com/**.

## Prerequisites
### Windows
Install Python via [this link.](https://www.python.org/downloads/windows/)

To run the program, you will need the following libraries/packages via `pip install [name]`: 
- Regular Expressions -> `regex`
- Pandas -> `pandas`
- Seaborn -> `seaborn`
- NumPy -> `numpy`
- Scikit-Learn -> `scikit-learn`
- BeautifulSoup -> `beautifulsoup4`
- Selenium -> `selenium`

### Linux

#### Ubuntu/Debian-based systems
- Update the package index: `sudo apt update`
- Install Python 3: `sudo apt install python3`
- For a specific version (e.g., Python 3.9): `sudo apt install python3.9`
- Verify the installation: `python3 --version`

#### Red Hat/CentOS/Fedora-based systems
- Update the package index: `sudo dnf update`
- Install Python 3: `sudo dnf install python3`
- For a specific version (e.g., Python 3.9): `sudo dnf install python39`
- Verify the installation: `vpython3 --versionv`

#### Arch Linux
- Update the package index: `sudo pacman -Syuv`
- Install Python 3: `sudo pacman -S python`
- For a specific version (e.g., Python 3.9): `sudo pacman -S python39`
- Verify the installation: `python3 --version`

## Questions / Hypotheses

- Games that include a reference to an update in the **start** of their title will have a greater ratio 
  of current players to total players than games that do not include a reference to an update at the 
  start of their title. This includes games that reference the update at the end of their title.

- There is a **quadratic** proportion between one of two variables in this dataset.

- Is the **length of a title** an indicator of the **rating of the game above 90%**?

- Are there **common words** used in games that have a **high all-time visits** and a 
  **high favourites** count?

- What games have **disproportionate** like and dislike counts, given their ratio?

- Do the **likes, dislikes and total visits** indicate whether or not a game's 
  **current players is above 10% of the number of favourites** on the game?

- Classify games by how many **unique** letters they have. Do the **general metrics** 
  to a game define the unique letters in the game?

## Tip
Run the **scrape.py** file as is to generate a fresh dataset. However, the dataset
may not align with the results found during development.

## Results
The results for each question/hypothesis are found in the **class method** that
relates to such question/hypothesis.

