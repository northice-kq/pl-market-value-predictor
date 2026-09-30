from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import re
import os

SEASON_URLS = {
    '2021-2022': 'https://www.whoscored.com/regions/252/tournaments/2/seasons/8618/stages/19793/playerstatistics/england-premier-league-2021-2022',
    '2022-2023': 'https://www.whoscored.com/regions/252/tournaments/2/seasons/9075/stages/20934/playerstatistics/england-premier-league-2022-2023',
    '2023-2024': 'https://www.whoscored.com/regions/252/tournaments/2/seasons/9618/stages/22076/playerstatistics/england-premier-league-2023-2024',
    '2024-2025': 'https://www.whoscored.com/regions/252/tournaments/2/seasons/10316/stages/23400/playerstatistics/england-premier-league-2024-2025',
    '2025-2026': 'https://www.whoscored.com/regions/252/tournaments/2/seasons/10743/stages/24533/playerstatistics/england-premier-league-2025-2026',
}

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
OUTPUT_DIR = os.path.join(project_root, 'data', 'raw', 'whoscored')

def setup_driver(headless=False):
    options = webdriver.ChromeOptions()

    if headless:
        options.add_argument('--headless')

    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def clean_player_data(text):
    lines = text.split('\n')
    if lines and lines[0].strip().isdigit():
        lines = lines[1:]
    player_name = lines[0].strip() if lines else ''
    team = ''
    age = ''
    position = ''
    if len(lines) > 1:
        info = lines[1].strip()
        parts = [p.strip() for p in info.split(',')]
        if len(parts) >= 1:
            team = parts[0]
        if len(parts) >= 2:
            age = parts[1]
        if len(parts) >= 3:
            position = ', '.join(parts[2:])
    return player_name, team, age, position


def get_table_headers(driver):
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#player-table-statistics-head"))
        )
        header_cells = driver.find_elements(By.CSS_SELECTOR, "#player-table-statistics-head th")
        headers = [cell.text.strip() for cell in header_cells]
        print(f"Table headers ({len(headers)}): {headers}")
        return headers
    except Exception as e:
        print(f"Could not get headers: {e}")
        return []


def scroll_to_load_all_rows(driver, max_scrolls=50):
    # Identify the scrollable container (usually the div containing the table)
    scrollable = None
    selectors = [
        "#player-table-statistics-container",
        ".table-container",
        "#statistics-table",
        "div[class*='table']",
        "body"
    ]
    for selector in selectors:
        try:
            elem = driver.find_element(By.CSS_SELECTOR, selector)
            if elem:
                scrollable = elem
                break
        except Exception:
            continue

    if scrollable is None:
        scrollable = driver.find_element(By.TAG_NAME, "body")

    print("Scrolling to load all rows...")
    last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable)
    scrolls_without_change = 0

    for _ in range(max_scrolls):
        # Scroll to bottom of the container
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable)
        time.sleep(2)  # Wait for lazy loading

        # Check if new content loaded
        new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable)
        if new_height > last_height:
            last_height = new_height
            scrolls_without_change = 0
            print(f"   New height: {new_height}")
        else:
            scrolls_without_change += 1
            if scrolls_without_change >= 3:
                print("   No more new rows loading.")
                break

    # After scrolling, count rows
    rows = driver.find_elements(By.CSS_SELECTOR, "#player-table-statistics-body tr")
    print(f"Total rows loaded: {len(rows)}")
    return rows


def scrape_passing_stats(url, headless=False):
    print(f"Scraping WhoScored passing stats...")
    print(f"URL: {url}")

    driver = setup_driver(headless=headless)

    try:
        driver.get(url)
        print("⏳ Waiting for page to load...")
        time.sleep(8)  # initial wait

        # If the page loaded with default summary tab,  click Passing tab
        # Check if URL already has statType parameter
        if 'statType' not in driver.current_url.lower():
            print("URL does not contain statType, attempting to click 'Passing' tab...")
            try:
                passing_tab = driver.find_element(By.LINK_TEXT, "Passing")
                passing_tab.click()
                time.sleep(5)  # wait for tab content to load
                print("Clicked Passing tab")
            except Exception as e:
                print(f"Could not click Passing tab: {e}")
                print("Trying alternative: looking for tab with data-stat-type='passing'")
                try:
                    passing_tab = driver.find_element(By.CSS_SELECTOR, "[data-stat-type='passing']")
                    passing_tab.click()
                    time.sleep(5)
                    print("Clicked Passing tab via data attribute")
                except Exception as e2:
                    print(f"Could not find Passing tab: {e2}")
                    print("   If the URL already shows passing stats, ignore this.")

        #  scroll to load all rows
        rows = scroll_to_load_all_rows(driver)

        if not rows:
            print("No rows found!")
            return None

        # Get headers after all rows loaded
        headers = get_table_headers(driver)

        all_players = []
        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 3:
                    player_text = cells[0].text.strip()
                    if player_text:
                        player_name, team, age, position = clean_player_data(player_text)
                        player_data = {
                            'Player': player_name,
                            'Team': team,
                            'Age': age,
                            'Position': position,
                        }
                        for i, cell in enumerate(cells[2:], start=2):
                            if i < len(headers) and headers[i]:
                                player_data[headers[i]] = cell.text.strip()
                            else:
                                player_data[f'Column_{i}'] = cell.text.strip()

                        # Try to get player ID
                        try:
                            link = cells[0].find_element(By.TAG_NAME, "a")
                            player_url = link.get_attribute("href")
                            match = re.search(r'/Players/(\d+)', player_url)
                            if match:
                                player_data['Player_ID'] = match.group(1)
                        except:
                            player_data['Player_ID'] = ''

                        all_players.append(player_data)
            except Exception as e:
                continue

        df = pd.DataFrame(all_players)
        print(f"Scraped {len(df)} players")
        return df

    except Exception as e:
        print(f"Error: {e}")
        return None

    finally:
        driver.quit()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for season, url in SEASON_URLS.items():
        output_file = os.path.join(OUTPUT_DIR, f'whoscored_{season}_passing.csv')

        if os.path.exists(output_file):
            print(f"\nSkipping {season} — file already exists: {output_file}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Season: {season}")
        print(f"{'=' * 60}")

        df = scrape_passing_stats(url, headless=False)

        if df is not None and len(df) > 0:
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
            print(f"Rows: {len(df)}")
        else:
            print(f"Failed to scrape {season}")

        time.sleep(5)

    print("\nAll seasons processed.")


if __name__ == "__main__":
    main()